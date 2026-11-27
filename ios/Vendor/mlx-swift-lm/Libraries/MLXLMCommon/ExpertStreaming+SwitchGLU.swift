// [agent-console patch C] Expert streaming for models whose experts are a `SwitchGLU`
// (`switch_mlp.{gate_proj,up_proj,down_proj}`: LFM2 MoE, Qwen MoE, ...). A model adapter calls
// `installStreamedSwitchGLU` per MoE layer from `prepareCheckpoint` and puts the returned module
// where its `switch_mlp` was.

import Foundation
import MLX
import MLXNN

/// A `SwitchGLU` whose expert tensors hold `S` slots of an `ExpertSlotCache`. Every call routes its
/// expert ids through the cache; a call that needs more distinct experts than slots (prefill) runs
/// in chunks of at most `S` experts, each evaluated before the next may evict its slots.
public final class StreamedSwitchGLU: SwitchGLU {
    package var stream: ExpertStreamLayer?

    public override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        guard let stream else { return super.callAsFunction(x, indices) }
        stream.runtime.noteLayer(stream.layer)
        switch stream.route(indices) {
        case .direct(let slots):
            return super.callAsFunction(x, slots)
        case .chunked(let ids):
            return chunked(x, ids: ids, shape: indices.shape, cache: stream.cache!)
        }
    }

    /// `[..., K, D]` like ``callAsFunction(_:_:)``, one row-expert pair per chunk row.
    private func chunked(_ x: MLXArray, ids: [UInt32], shape: [Int], cache: ExpertSlotCache)
        -> MLXArray
    {
        let k = shape.last ?? 1
        let rows = x.reshaped(-1, x.dim(-1))
        var outputs = [MLXArray]()
        var positions = [Int]()
        positions.reserveCapacity(ids.count)
        for chunk in cache.chunks(ids) {
            let slots = cache.ensure(chunk.map { ids[$0] })
            let input = rows[MLXArray(chunk.map { UInt32($0 / k) })]
            let out = super.callAsFunction(input, MLXArray(slots, [slots.count, 1]))
            let flat = out.reshaped(-1, out.dim(-1))
            eval(flat)
            outputs.append(flat)
            positions += chunk
        }
        var inverse = [UInt32](repeating: 0, count: ids.count)
        for (row, position) in positions.enumerated() { inverse[position] = UInt32(row) }
        let all = concatenated(outputs, axis: 0)[MLXArray(inverse)]
        return all.reshaped(shape + [all.dim(-1)])
    }
}

/// Builds the streamed `SwitchGLU` of one manifest layer: allocates `capacity`-slot tensors for
/// every field of `spec`, binds a slot cache to them and adds them to `weights` under
/// `<path>.<projection>.<part>`, so the checkpoint's quantization step and strict validation see
/// the usual tensors. Only affine experts with weight, scales and biases are supported.
package func installStreamedSwitchGLU(
    spec: ExpertStreamManifest.Layer, runtime: ExpertStreamRuntime, capacity: Int,
    inputDims: Int, hiddenDims: Int, path: String,
    quantization: (String) -> BaseConfiguration.Quantization?,
    weights: inout [String: MLXArray]
) throws -> StreamedSwitchGLU {
    var arrays = [String: MLXArray]()
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        guard let p = spec.projections[projection], p.mode == "affine" else {
            throw ExpertStreamingError.unsupported(
                "layer \(spec.layer) \(projection): needs affine experts")
        }
        if let q = quantization("\(path).\(projection)"), q.bits != p.bits || q.groupSize != p.groupSize {
            throw ExpertStreamingError.manifest(
                "layer \(spec.layer) \(projection) is \(p.bits)-bit/\(p.groupSize) but config.json "
                    + "says \(q.bits)-bit/\(q.groupSize)")
        }
        for part in ["weight", "scales", "biases"] {
            let name = "\(projection).\(part)"
            guard let field = spec.fields.first(where: { $0.name == name }),
                let dtype = field.dtypeValue
            else { throw ExpertStreamingError.unsupported("layer \(spec.layer) without \(name)") }
            arrays[name] = MLXArray.zeros([capacity] + field.shape, dtype: dtype)
        }
    }
    let cache = try ExpertSlotCache(spec: spec, capacity: capacity, arrays: arrays, runtime: runtime)
    try cache.bind()
    runtime.registerLayer(slots: capacity, experts: spec.experts, bytes: cache.slotBytes)
    for (name, array) in arrays { weights["\(path).\(name)"] = array }
    let module = StreamedSwitchGLU(
        inputDims: inputDims, hiddenDims: hiddenDims, numExperts: capacity, bias: false)
    module.stream = ExpertStreamLayer(layer: spec.layer, runtime: runtime, cache: cache)
    return module
}
