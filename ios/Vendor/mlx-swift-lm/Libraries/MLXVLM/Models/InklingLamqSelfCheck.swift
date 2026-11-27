// [agent-console patch D] On-device self-check of the lamq Metal kernels against the reference
// decoder's digests (ios/tools/lamq/lamq_digests.json, produced by crosscheck_lamq_torch.py with
// lamq_ref.py == LookAtMyQuant's own decoders, bit for bit).
//
// Needs only `luts.npz`, `layer_XX_meta.json` and `layer_XX_packed.npz` of the build the digests
// were made from (~190 MB for layer 7 of the 5.4 GB build, which uses every layer format of that build: rtr4, lm6 with an output rotation, ht2, ht3, rtr1), not the whole model.

import CryptoKit
import Foundation
import MLX

/// Digests of one layer's decoded weights (float32 little-endian bytes, SHA-256).
public struct InklingLamqExpectedLayer: Sendable {
    public let repo: String
    public let revision: String
    public let layer: Int
    public let luts: [String: String]
    public let linears: [String: String]
    public let gateUp: [String]
    public let down: [String]

    public init(repo: String, revision: String, layer: Int, luts: [String: String], linears: [String: String],
                gateUp: [String], down: [String]) {
        self.repo = repo
        self.revision = revision
        self.layer = layer
        self.luts = luts
        self.linears = linears
        self.gateUp = gateUp
        self.down = down
    }

    /// The files the check reads, relative to the repo root.
    public var files: [String] {
        ["luts.npz", String(format: "layer_%02d_meta.json", layer), String(format: "layer_%02d_packed.npz", layer)]
    }
}

public enum InklingLamqSelfCheck {
    public struct Result: Sendable {
        public var passed = true
        public var lines: [String] = []
        public var seconds: Double = 0
    }

    /// SHA-256 of the float32 bytes of `lamq_ref.hadamard_factors(n)` (A, B), which equal
    /// LookAtMyQuant's (crosscheck_lamq_torch.py).
    static let hadamardDigests: [String: String] = [
        "2304A": "db4e69c0ebea12b9b1156126f040b5670ed3afca3521b57031e0d67f66523b74",
        "2304B": "29375209b26bf9ceed6436a4723b2b94c4d6a59bdc8465e63955ea1d2b231447",
        "1152A": "c9878e24c0071879714bf80810c2b6b055bdba51c8ce3d740509b69cb30b133b",
        "1152B": "29375209b26bf9ceed6436a4723b2b94c4d6a59bdc8465e63955ea1d2b231447",
        "3072A": "ea5a4b8435c281847845ea7bdd695e25e3e7c1f1e61ad938c4d405934b1fde97",
        "3072B": "01756042c33b5aa1fd8e483a56bcecfefc7d03f98a6f3ad43b5232e1a5d47246",
    ]

    static func sha256(_ a: MLXArray) -> String {
        let v = a.asType(.float32).asArray(Float.self)
        let digest = v.withUnsafeBytes { SHA256.hash(data: $0) }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func sha256(_ d: Data) -> String {
        SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
    }

    /// Scalar Swift decode of one trellis row (spec §2), independent of the Metal kernels.
    static func cpuRow(words: ArraySlice<UInt32>, scale: Float16, lut: [Float16], n: Int,
                       format f: InklingLamqTrellisFormat) -> [Float] {
        let w = Array(words)
        var out = [Float](repeating: 0, count: n)
        for gi in 0 ..< n / f.g {
            let last = gi * f.g + f.g - 1
            let bp = (n - 1 - last) * f.k
            let wi = bp >> 5, off = bp & 31
            let lo = UInt64(w[wi])
            let hi = wi + 1 < w.count ? UInt64(w[wi + 1]) : 0
            let s = UInt32(truncatingIfNeeded: ((lo | (hi << 32)) >> UInt64(off)) & 4095)
            let idx = f.hashed ? Int((s &* 0x9E37_79B1) >> UInt32(32 - f.log2K)) : Int(s)
            for t in 0 ..< f.g { out[gi * f.g + t] = Float(lut[idx * f.g + t]) * Float(scale) }
        }
        return out
    }

    /// Runs the check. `experts`: which of the 128 experts to verify on both sides.
    public static func run(
        directory: URL, expected: InklingLamqExpectedLayer, experts: [Int] = [0, 1, 63, 127],
        log: (String) -> Void = { _ in }
    ) throws -> Result {
        let start = Date()
        var r = Result()
        func note(_ ok: Bool, _ line: String) {
            r.lines.append((ok ? "PASS " : "FAIL ") + line)
            if !ok { r.passed = false }
            log(r.lines.last!)
        }

        // 1. Lookup tables.
        let lutsNpz = try InklingLamqNpz(url: directory.appendingPathComponent("luts.npz"))
        for (name, want) in expected.luts.sorted(by: { $0.key < $1.key }) {
            let bytes = try lutsNpz.bytes(name)
            note(sha256(bytes) == want, "luts.npz \(name) bytes")
        }
        let tables = try InklingLamqTables(directory: directory)

        // 2. Attention linears: GPU dequant digest + 4 rows against the scalar Swift decode.
        let meta = try InklingLamqLayerMeta.load(directory.appendingPathComponent(expected.files[1]))
        let npz = try InklingLamqNpz(url: directory.appendingPathComponent(expected.files[2]))
        for name in inklingLamqLinearNames {
            guard let lm = meta.linears[name] else {
                note(false, "\(name): missing from meta")
                continue
            }
            let lin = try InklingLamqPackedLinear(name: name, meta: lm, npz: npz, tables: tables)
            let q = lin.dequantized(.float32)
            eval(q)
            note(sha256(q) == expected.linears[name], "\(name) [\(lm.fmt) \(lm.m)x\(lm.n)] dequant digest")
            if let f = InklingLamqTrellisFormat.all[lm.fmt] {
                let words = try npz.host("lin_\(lm.safe)_codes", of: UInt32.self)
                let scales = try npz.host("lin_\(lm.safe)_scales", of: Float16.self)
                let lut = tables.luts[lm.fmt]!.asArray(Float16.self)
                let wpr = lm.n * f.k / 32
                var same = true
                for row in [0, 1, lm.m / 2, lm.m - 1] {
                    let cpu = cpuRow(words: words[(row * wpr) ..< ((row + 1) * wpr)], scale: scales[row], lut: lut, n: lm.n, format: f)
                    let gpu = q[row].asArray(Float.self)
                    same = same && cpu.elementsEqual(gpu, by: { $0.bitPattern == $1.bitPattern })
                }
                note(same, "\(name): GPU rows == scalar Swift decode, bit for bit")
            }
            // Fused decode matvec vs dequant + matmul, on the SAME rotated input. The reference
            // runs one row at a time: MLX sends M = 1 to its full-float32 gemv, whereas an M >= 2
            // float32 GEMM runs at TF32 on M5 neural accelerators (MLX_ENABLE_TF32 defaults to 1),
            // which is what made the first device run read ~7e-4 here (ios/tools/lamq/rht_precision.py).
            let si = try npz.array("lin_\(lm.safe)_signs_in").asType(.float32)
            let xRot = try InklingLamqRotation.of(lm.n).input(MLXRandom.normal([2, lm.n]), signs: si)
            eval(xRot)
            let y = lin.rotatedProduct(xRot)
            let qT = q.transposed()
            let ref = concatenated(
                (0 ..< xRot.dim(0)).map { matmul(xRot[$0 ..< ($0 + 1)], qT) }, axis: 0)
            let err = (abs(y - ref).max() / (abs(ref).max() + 1e-12)).item(Float.self)
            note(err < 1e-5, "\(name): fused matvec vs dequant+gemv (f32), rel err \(err)")
        }

        // 3. Experts: assembled row groups, both sides. Besides `experts`, the first expert using
        // each format on each side, so every format of the layer runs.
        let packed = try InklingLamqPackedExperts(npz: npz, tables: tables)
        for (side, want) in [(packed.gateUp, expected.gateUp), (packed.down, expected.down)] {
            var chosen = experts
            var seen = Set<Int>()
            for e in 0 ..< side.experts {
                for r in 0 ..< side.groups {
                    let f = side.map[(e * side.groups + r) * 2]
                    if seen.insert(f).inserted, !chosen.contains(e) { chosen.append(e) }
                }
            }
            for e in chosen {
                let q = side.dequantized(expert: e, dtype: .float32)
                eval(q)
                note(sha256(q) == want[e], "\(side.side) expert \(e) [\(side.m)x\(side.n)] dequant digest")
                let x = MLXRandom.normal([1, side.n])
                let y = side.matvec(x, experts: MLXArray([Int32(e)]), xDiv: 1)
                let ref = matmul(x, q.transposed())
                let err = (abs(y - ref).max() / (abs(ref).max() + 1e-12)).item(Float.self)
                note(err < 1e-5, "\(side.side) expert \(e): fused matvec vs dequant+gemv (f32), rel err \(err)")
            }
        }

        // 4. Hadamard rotations.
        // (a) Exact: the factors are the reference's (SHA-256 of their float32 bytes, from
        //     lamq_ref.hadamard_factors == LookAtMyQuant's) and are Hadamard (M M^T = order I,
        //     checked on the host in exact arithmetic).
        for n in [2304, 1152, 3072] {
            let (a, b) = try InklingLamqHadamard.factors(n)
            for (label, m) in [("A", a), ("B", b)] {
                let order = m.dim(0)
                let v = m.asType(.float32).asArray(Float.self)
                var hadamard = v.allSatisfy { $0 == 1 || $0 == -1 }
                for i in 0 ..< order where hadamard {
                    for j in 0 ..< order {
                        var dot: Float = 0
                        for k in 0 ..< order { dot += v[i * order + k] * v[j * order + k] }
                        if dot != (i == j ? Float(order) : 0) { hadamard = false; break }
                    }
                }
                let digestOK = Self.hadamardDigests["\(n)\(label)"].map { $0 == sha256(m) } ?? true
                note(hadamard && digestOK, "Hadamard \(n) factor \(label) (\(order)x\(order)): exact, matches reference")
            }
        }
        // (b) The transform on the GPU vs the explicit R = kron(A, B) diag(s) / sqrt(n) applied
        //     through the full-float32 gemv. The two small factor matmuls are float32 GEMMs, which
        //     M5 runs at TF32 (inputs truncated to 10 mantissa bits): expect ~7e-4 relative, and
        //     ~1.5e-3 for the round trip (rht_precision.py reproduces the device's 5.2e-3 absolute).
        //     A wrong matrix, order or normalisation gives errors of order 1e-1 and up, so 1e-2
        //     separates the two. lamq's own runtime rounds rotated activations to float16 (~5e-4).
        let rot = try InklingLamqRotation.of(2304)
        let s = try npz.array("gu_signs_in").asType(.float32)
        let x = MLXRandom.normal([1, 2304])
        let (a, b) = try InklingLamqHadamard.factors(2304)
        let h = (a.reshaped(32, 1, 32, 1) * b.reshaped(1, 72, 1, 72)).reshaped(2304, 2304)
        let exact = matmul(x * s, h.transposed()) * (1 / Float(2304).squareRoot())
        let y = rot.input(x, signs: s)
        let transformErr = (abs(y - exact).max() / abs(exact).max()).item(Float.self)
        note(transformErr < 1e-2, "Hadamard 2304 transform vs explicit kron (gemv f32), rel err \(transformErr) (TF32 ~7e-4)")
        let back = rot.output(y, signs: s)
        let want = x * s * s
        let roundTrip = (abs(back - want).max() / abs(want).max()).item(Float.self)
        note(roundTrip < 1e-2, "Hadamard 2304 round trip, rel err \(roundTrip) (TF32 ~1.5e-3)")

        r.seconds = Date().timeIntervalSince(start)
        r.lines.append(String(format: "%@ in %.1f s (%@@%@, layer %d)", r.passed ? "ALL PASS" : "FAILED", r.seconds,
                              expected.repo, String(expected.revision.prefix(8)), expected.layer))
        return r
    }
}
