// Copyright © 2026 Apple Inc.
// [agent-console patch K2] An A/B switch for, and an on-device check of, the fused decode expert
// kernels.

import Foundation
import MLX

/// Whether decode sends 2- and 3-bit affine experts to the fused kernels. On by default; off
/// restores the 4/8-bit-only gate, so one build can be measured both ways.
///
/// Decode reads it once per layer and step, so set it only while no generation runs (the app
/// sets it when it loads a model): a change mid-generation would mix the two paths.
public enum Inkling2FusedDecodeOptions {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var value = true

    public static var lowBitAffine: Bool {
        get { lock.withLock { value } }
        set { lock.withLock { value = newValue } }
    }
}

/// Runs the fused decode expert kernels (``inkling2ExpertGateUp(_:experts:indices:softcaps:)``
/// and ``inkling2ExpertDown(_:experts:indices:weights:residual:state:convolutionWeight:)``) and
/// the unfused path that decode otherwise takes (`gatherQuantizedMM`, the softcapped SwiGLU and
/// ``inkling2ExpertEpilogue(outputs:weights:residual:state:convolutionWeight:)``) on the same
/// random layer, and compares them.
///
/// The layer has Inkling 2 Mini's shapes by default and random weights quantized with MLX's own
/// `quantized`, so it checks the kernels' packing and indexing for each format; it does not
/// read the loaded model.
public enum Inkling2KernelSelfCheck {
    public struct Report: Sendable, CustomStringConvertible {
        public let gateUpBits: Int
        public let downBits: Int
        public let groupSize: Int
        /// Whether decode sends this pair of formats to the fused kernels.
        public let fusedInDecode: Bool
        /// The largest difference of the activations, and the largest unfused activation.
        public let activationDiff: Float
        public let activationScale: Float
        /// The same for the layer output (the residual is zero, so this is the expert sum after
        /// the short convolution).
        public let outputDiff: Float
        public let outputScale: Float
        public let seconds: Double

        /// Both differences within 2% of the output scale: bfloat16 roundings of the
        /// intermediates differ between the two paths, a packing or indexing error does not
        /// stay this small.
        public var passed: Bool {
            activationDiff <= 0.02 * activationScale + 1e-3 && outputDiff <= 0.02 * outputScale + 1e-3
        }

        public var description: String {
            String(
                format: "%@ gate/up %d-bit, down %d-bit, group %d%@: act Δ %.2g of %.2g, out Δ %.2g of %.2g (%.1f s)",
                passed ? "PASS" : "FAIL", gateUpBits, downBits, groupSize,
                fusedInDecode ? "" : " (not fused in decode)", activationDiff, activationScale,
                outputDiff, outputScale, seconds)
        }
    }

    /// The pairs of (gate/up bits, down bits, group size) that the Inkling 2 Mini conversions
    /// use, plus a 4-bit control that the fused kernels supported before 2/3-bit.
    public static let defaultCases: [(Int, Int, Int)] = [
        (2, 2, 64), (2, 3, 64), (3, 2, 64), (3, 3, 64), (2, 3, 128), (4, 4, 64),
    ]

    /// The peak bytes ``run(cases:hidden:intermediate:expertsPerToken:experts:softcaps:dtype:)``
    /// allocates, about 7 bytes per expert weight: the float32 layer (4), one case's bfloat16
    /// copy (2) and its quantized form (at most 4-bit with scales and biases, under 1). Inkling 2
    /// Mini's defaults: 31.9 M weights, 223 MB (127 MB float32 + 64 MB bfloat16 + ~20 MB packed).
    public static func peakBytes(hidden: Int = 2304, intermediate: Int = 1152, experts: Int = 4)
        -> Int
    {
        7 * experts * 3 * hidden * intermediate
    }

    /// Runs `cases` on a layer of `experts` random experts; see ``peakBytes(hidden:intermediate:experts:)``.
    public static func run(
        cases: [(Int, Int, Int)] = defaultCases, hidden: Int = 2304, intermediate: Int = 1152,
        expertsPerToken: Int = 8, experts: Int = 4, softcaps: (Float, Float) = (4, 25),
        dtype: DType = .bfloat16
    ) -> [Report] {
        let key = { (n: UInt64) in MLXRandom.key(0x1D_C0DE + n) }
        let gateUpDense = MLXRandom.normal([experts, 2 * intermediate, hidden], key: key(1)) * Float(0.05)
        let downDense = MLXRandom.normal([experts, hidden, intermediate], key: key(2)) * Float(0.05)
        let x = MLXRandom.normal([1, hidden], key: key(3)).asType(dtype)
        let indices = MLXArray(
            (0 ..< expertsPerToken).map { UInt32(($0 * 3 + 1) % experts) }, [1, expertsPerToken])
        let routing = MLXRandom.uniform(low: Float(0.1), high: Float(1), [1, expertsPerToken], key: key(4))
        let weights = routing / routing.sum(axis: -1, keepDims: true)
        let residual = MLXArray.zeros([1, hidden], dtype: .float32)
        let state = (MLXRandom.normal([1, 3, hidden], key: key(5)) * Float(0.1)).asType(dtype)
        let convolution = (MLXRandom.normal([hidden, 4], key: key(6)) * Float(0.5)).asType(dtype)
        let caps = MLXArray([softcaps.0, softcaps.1])
        eval(gateUpDense, downDense, x, indices, weights, residual, state, convolution, caps)

        return cases.map { testCase -> Report in
            let (gateUpBits, downBits, groupSize) = testCase
            let started = Date()
            let gateUp = tensors(gateUpDense.asType(dtype), bits: gateUpBits, groupSize: groupSize)
            let down = tensors(downDense.asType(dtype), bits: downBits, groupSize: groupSize)

            let fusedActivations = inkling2ExpertGateUp(
                x, experts: gateUp, indices: indices, softcaps: caps)
            let fused = inkling2ExpertDown(
                fusedActivations, experts: down, indices: indices, weights: weights,
                residual: residual, state: state, convolutionWeight: convolution)

            // The unfused decode path of `Inkling2Experts` for fewer than 64 indices.
            let projected = gatherQuantizedMM(
                MLX.expandedDimensions(x, axes: [-2, -3]), gateUp.weight, scales: gateUp.scales,
                biases: gateUp.biases, rhsIndices: indices, transpose: true,
                groupSize: groupSize, bits: gateUpBits, mode: .affine)
            let gate = projected[.ellipsis, .stride(from: 0, by: 2)].asType(.float32)
            let up = projected[.ellipsis, .stride(from: 1, by: 2)].asType(.float32)
            let activations = (tanh(gate / softcaps.0) * softcaps.0 * sigmoid(gate)
                * tanh(up / softcaps.1) * softcaps.1).asType(dtype)
            let outputs = gatherQuantizedMM(
                activations, down.weight, scales: down.scales, biases: down.biases,
                rhsIndices: indices, transpose: true, groupSize: groupSize, bits: downBits,
                mode: .affine)
            let unfused = inkling2ExpertEpilogue(
                outputs: outputs.reshaped(1, expertsPerToken, hidden), weights: weights,
                residual: residual, state: state, convolutionWeight: convolution)

            let unfusedActivations = activations.reshaped(fusedActivations.shape)
            return Report(
                gateUpBits: gateUpBits, downBits: downBits, groupSize: groupSize,
                fusedInDecode: gateUp.supportsFusedDecode(inputDims: hidden)
                    && down.supportsFusedDecode(inputDims: intermediate),
                activationDiff: maxDifference(fusedActivations, unfusedActivations),
                activationScale: maxMagnitude(unfusedActivations),
                outputDiff: maxDifference(fused.hidden, unfused.hidden),
                outputScale: maxMagnitude(unfused.hidden),
                seconds: Date().timeIntervalSince(started))
        }
    }

    /// One line per case, then the verdict.
    public static func summary(_ results: [Report]) -> String {
        let verdict = results.allSatisfy(\.passed) ? "all passed" : "FAILED"
        return (results.map(\.description) + [verdict]).joined(separator: "\n")
    }

    private static func tensors(_ dense: MLXArray, bits: Int, groupSize: Int)
        -> Inkling2ExpertTensors
    {
        let (weight, scales, biases) = MLX.quantized(
            dense, groupSize: groupSize, bits: bits, mode: .affine)
        eval(weight, scales)
        return Inkling2ExpertTensors(
            weight: weight, scales: scales, biases: biases, globalScale: nil,
            format: Inkling2ExpertFormat(
                mode: .affine, bits: bits, groupSize: groupSize, hasGlobalScale: false))
    }

    private static func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private static func maxMagnitude(_ a: MLXArray) -> Float {
        MLX.abs(a.asType(.float32)).max().item(Float.self)
    }
}
