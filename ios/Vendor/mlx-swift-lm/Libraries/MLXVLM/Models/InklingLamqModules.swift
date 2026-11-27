// [agent-console patch D] LookAtMyQuant ("lamq") packed Inkling 2 Mini: packed layers built from
// the packed files. Not part of mlx-swift-lm PR #821. Spec: docs/LAMQ-FORMAT-SPEC.md §6-7, §9.
//
// Wired into `Inkling2.swift` by InklingLamqModel.swift (the `Linear`/`Embedding` subclasses and
// the experts box); the self-check exercises the same classes against the reference digests.

import Foundation
import MLX

/// Lookup tables of `luts.npz`, float16, flat `[K * g]`.
public struct InklingLamqTables: @unchecked Sendable {
    public let luts: [String: MLXArray]

    public init(directory: URL) throws {
        let npz = try InklingLamqNpz(url: directory.appendingPathComponent("luts.npz"))
        var luts: [String: MLXArray] = [:]
        for (name, e) in npz.entries {
            let a = try npz.array(name)
            luts[name] = e.descr == "<f2" ? a.reshaped([-1]) : a.asType(.float16).reshaped([-1])
        }
        self.luts = luts
    }

    func lut(_ fmt: String) throws -> MLXArray {
        guard let l = luts[fmt] else { throw InklingLamqError.file("luts.npz has no \(fmt)") }
        return l
    }
}

/// Decodes are below this many rows*slots routed through the fused matvec; above, prefill tiles.
let lamqMatvecMaxTokens = 8

// MARK: - Attention linear

/// One packed attention linear: `y = rht_out(rht_in(x, s_in) @ Q.T, s_out)` (spec §5-6).
public final class InklingLamqPackedLinear: @unchecked Sendable {
    public let name: String
    public let meta: InklingLamqLinearMeta
    let codes: MLXArray
    let scales: MLXArray
    let signsIn: MLXArray
    let signsOut: MLXArray?
    let levels: MLXArray?
    let lut: MLXArray?
    let format: InklingLamqTrellisFormat?
    let rotIn: InklingLamqRotation
    let rotOut: InklingLamqRotation?

    public init(name: String, meta: InklingLamqLinearMeta, npz: InklingLamqNpz, tables: InklingLamqTables) throws {
        self.name = name
        self.meta = meta
        let s = meta.safe
        scales = try npz.array("lin_\(s)_scales")
        signsIn = try npz.array("lin_\(s)_signs_in").asType(.float32)
        if meta.out_rot == true {
            signsOut = try npz.array("lin_\(s)_signs_out").asType(.float32)
            rotOut = try InklingLamqRotation.of(meta.m)
        } else {
            signsOut = nil
            rotOut = nil
        }
        rotIn = try InklingLamqRotation.of(meta.n)
        if meta.fmt.hasPrefix("lm") {
            codes = try npz.array("lin_\(s)_codes")
            levels = try npz.array("lin_\(s)_levels").asType(.float32)
            lut = nil
            format = nil
        } else {
            guard let f = InklingLamqTrellisFormat.all[meta.fmt] else {
                throw InklingLamqError.unsupported("format \(meta.fmt) of \(name)")
            }
            codes = try npz.array("lin_\(s)_codes", shape: [meta.m, meta.n * f.k / 32])
            levels = nil
            lut = try tables.lut(meta.fmt)
            format = f
        }
    }

    /// Dense `Q * scale` `[m, n]` (rotated basis): float32 for checks, float16 for prefill.
    public func dequantized(_ dtype: DType = .float32) -> MLXArray {
        if let format, let lut {
            return lamqTrellisDequant(
                codes: codes, scales: scales, lut: lut, locs: MLXArray([UInt32(0)]),
                rowsPerGroup: meta.m, n: meta.n, format: format, dtype: dtype)
        }
        return lamqLMDequant(
            codes: codes, scales: scales, levels: levels!,
            rowsSel: MLXArray((0 ..< meta.m).map { UInt32($0) }), n: meta.n, k: meta.k, dtype: dtype)
    }

    /// `Q @ x_rot` rows for rotated activations `[T, n]`, float32 `[T, m]`.
    func rotatedProduct(_ xRot: MLXArray) -> MLXArray {
        let t = xRot.dim(0)
        if t <= lamqMatvecMaxTokens {
            if let format, let lut {
                return lamqTrellisMatvec(
                    x: xRot, codes: codes, scales: scales, lut: lut,
                    sel: MLXArray.zeros([t, 1], dtype: .uint32), rowsPerGroup: meta.m, xDiv: 1,
                    n: meta.n, format: format)
            }
            return lamqLMMatvec(x: xRot, codes: codes, scales: scales, levels: levels!, m: meta.m, n: meta.n, k: meta.k)
        }
        let w = dequantized(.float16)
        return matmul(xRot.asType(.float16), w.transposed()).asType(.float32)
    }

    /// `x [..., n] -> [..., m]`, float32.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let lead = Array(x.shape.dropLast())
        let flat = x.reshaped([-1, meta.n])
        var y = rotatedProduct(rotIn.input(flat, signs: signsIn))
        if let rotOut, let signsOut { y = rotOut.output(y, signs: signsOut) }
        return y.reshaped(lead + [meta.m])
    }
}

// MARK: - Routed experts

/// One side (`gu` or `dn`) of a layer's 128 experts: per-format slabs and the row-group map.
public final class InklingLamqExpertSide: @unchecked Sendable {
    public let side: String
    /// Output rows (2 * 1152 for gu, 2304 for dn) and input columns.
    public let m: Int
    public let n: Int
    public let rowsPerGroup: Int
    public let groups: Int
    /// Host copy of the map: `map[(e * groups + r) * 2 + {0: fid, 1: loc}]`.
    public let map: [Int]
    /// The same on the GPU: fid int32 `[E, groups]`, loc uint32 `[E, groups]`.
    let mapFid: MLXArray
    let mapLoc: MLXArray
    /// fid -> (codes uint32 `[count * RG, n*k/32]`, scales f16 `[count * RG]`, lut, format).
    let slabs: [Int: (codes: MLXArray, scales: MLXArray, lut: MLXArray, format: InklingLamqTrellisFormat)]
    let signsIn: MLXArray
    let rot: InklingLamqRotation
    public let experts: Int

    init(side: String, m: Int, n: Int, npz: InklingLamqNpz, tables: InklingLamqTables) throws {
        self.side = side
        self.m = m
        self.n = n
        let key = "\(side)_rg_map"
        guard npz.has(key) else {
            throw InklingLamqError.unsupported("\(side): per-expert maps (8.9/11.2 GB builds); only rg128 builds are supported")
        }
        let e = try npz.entry(key)
        guard e.shape.count == 3, e.shape[2] == 2 else { throw InklingLamqError.file("\(key) shape \(e.shape)") }
        experts = e.shape[0]
        groups = e.shape[1]
        rowsPerGroup = m / groups
        guard rowsPerGroup * groups == m else { throw InklingLamqError.file("\(key): \(m) rows in \(groups) groups") }
        guard e.descr == "<i2" else { throw InklingLamqError.unsupported("\(key) dtype \(e.descr)") }
        let raw = try npz.host(key, of: Int16.self).map { Int($0) }
        map = raw
        var fid = [Int32](), loc = [UInt32]()
        fid.reserveCapacity(raw.count / 2)
        loc.reserveCapacity(raw.count / 2)
        for i in stride(from: 0, to: raw.count, by: 2) {
            fid.append(Int32(raw[i]))
            loc.append(UInt32(raw[i + 1]))
        }
        if fid.contains(7) { throw InklingLamqError.unsupported("NVFP4 inside a row-group map") }
        mapFid = MLXArray(fid, [experts, groups])
        mapLoc = MLXArray(loc, [experts, groups])
        var slabs: [Int: (codes: MLXArray, scales: MLXArray, lut: MLXArray, format: InklingLamqTrellisFormat)] = [:]
        for (id, name) in InklingLamqTrellisFormat.byID where id != 7 {
            let ck = "\(side)_\(name)_codes"
            guard npz.has(ck), let f = InklingLamqTrellisFormat.all[name] else { continue }
            let count = try npz.entry(ck).shape[0]
            slabs[id] = (
                try npz.array(ck, shape: [count * rowsPerGroup, n * f.k / 32]),
                try npz.array("\(side)_\(name)_scales", shape: [count * rowsPerGroup]),
                try tables.lut(name), f
            )
        }
        for f in Set(fid) where slabs[Int(f)] == nil {
            throw InklingLamqError.file("\(side) map references format id \(f) with no slab")
        }
        self.slabs = slabs
        signsIn = try npz.array("\(side)_signs_in").asType(.float32)
        rot = try InklingLamqRotation.of(n)
    }

    /// Dense `Q` of expert `e` `[m, n]` (rotated basis), assembled from its row groups.
    public func dequantized(expert e: Int, dtype: DType = .float32) -> MLXArray {
        var parts: [(fid: Int, rows: [Int], tile: MLXArray)] = []
        var byFid: [Int: [Int]] = [:]
        var order: [Int] = []
        for r in 0 ..< groups {
            let f = map[(e * groups + r) * 2]
            if byFid[f] == nil { order.append(f) }
            byFid[f, default: []].append(r)
        }
        for f in order {
            let rs = byFid[f]!
            let locs = MLXArray(rs.map { UInt32(map[(e * groups + $0) * 2 + 1]) })
            let slab = slabs[f]!
            parts.append((f, rs, lamqTrellisDequant(
                codes: slab.codes, scales: slab.scales, lut: slab.lut, locs: locs,
                rowsPerGroup: rowsPerGroup, n: n, format: slab.format, dtype: dtype)))
        }
        if parts.count == 1 { return parts[0].tile }
        // Put row groups back in order: position of row group r inside the concatenation.
        var position = [Int](repeating: 0, count: groups)
        var offset = 0
        for p in parts {
            for (i, r) in p.rows.enumerated() { position[r] = offset + i }
            offset += p.rows.count
        }
        var perm = [Int32]()
        perm.reserveCapacity(m)
        for r in 0 ..< groups {
            for i in 0 ..< rowsPerGroup { perm.append(Int32(position[r] * rowsPerGroup + i)) }
        }
        return concatenated(parts.map(\.tile), axis: 0)[MLXArray(perm)]
    }

    /// Fused decode: `[slots, m]` float32 for rotated inputs `xRot [slots / xDiv, n]` and expert
    /// ids `[slots]` (int32 or uint32). One launch per format on this side; summed.
    func matvec(_ xRot: MLXArray, experts ids: MLXArray, xDiv: Int) -> MLXArray {
        let fids = mapFid[ids]          // [slots, groups]
        let locs = mapLoc[ids]
        let sentinel = MLXArray(UInt32.max)
        var y: MLXArray?
        for (id, slab) in slabs {
            let sel = which(fids .== MLXArray(Int32(id)), locs, sentinel)
            let part = lamqTrellisMatvec(
                x: xRot, codes: slab.codes, scales: slab.scales, lut: slab.lut, sel: sel,
                rowsPerGroup: rowsPerGroup, xDiv: xDiv, n: n, format: slab.format)
            y = y.map { $0 + part } ?? part
        }
        return y!
    }
}

/// `tanh(g / 4) * 4 * sigmoid(g) * tanh(u / 25) * 25` of gate|up `[..., 2I]` (spec §6).
func lamqSoftcappedSwiGLU(_ gateUp: MLXArray, gateCap: Float = 4, upCap: Float = 25) -> MLXArray {
    let halves = split(gateUp.asType(.float32), parts: 2, axis: -1)
    let g = halves[0], u = halves[1]
    return (tanh(g / gateCap) * gateCap) * sigmoid(g) * (tanh(u / upCap) * upCap)
}

/// A layer's 128 routed experts (spec §6).
public final class InklingLamqPackedExperts: @unchecked Sendable {
    public let gateUp: InklingLamqExpertSide
    public let down: InklingLamqExpertSide

    public init(npz: InklingLamqNpz, tables: InklingLamqTables, hidden: Int = 2304, intermediate: Int = 1152) throws {
        gateUp = try InklingLamqExpertSide(side: "gu", m: 2 * intermediate, n: hidden, npz: npz, tables: tables)
        down = try InklingLamqExpertSide(side: "dn", m: hidden, n: intermediate, npz: npz, tables: tables)
    }

    /// `Σ_k w[t, k] * expert_{ids[t, k]}(x[t])`, float32 `[T, hidden]`.
    ///
    /// - Parameters:
    ///   - x: `[T, hidden]` (unrotated).
    ///   - ids: int32 `[T, K]` routed expert ids. weights: float32 `[T, K]` (normalised * route_scale).
    /// The unweighted output of each routed expert, float32 `[T, K, hidden]`, through the fused
    /// decode matvec (T <= `lamqMatvecMaxTokens`): one launch per format and side.
    public func slotOutputs(_ x: MLXArray, ids: MLXArray) -> MLXArray {
        let t = x.dim(0)
        let k = ids.dim(1)
        let xRot = gateUp.rot.input(x, signs: gateUp.signsIn)
        let flat = ids.reshaped([-1])
        let gu = gateUp.matvec(xRot, experts: flat, xDiv: k)                 // [T*K, 2I]
        let act = lamqSoftcappedSwiGLU(gu)
        let actRot = down.rot.input(act, signs: down.signsIn)
        let dn = down.matvec(actRot, experts: flat, xDiv: 1)                // [T*K, D]
        return dn.reshaped([t, k, down.m])
    }

    public func callAsFunction(_ x: MLXArray, ids: MLXArray, weights: MLXArray) -> MLXArray {
        let t = x.dim(0)
        let k = ids.dim(1)
        if t <= lamqMatvecMaxTokens {
            let outputs = slotOutputs(x, ids: ids)
            return sum(outputs * expandedDimensions(weights.asType(.float32), axis: -1), axis: 1)
        }
        let xRot = gateUp.rot.input(x, signs: gateUp.signsIn)
        // Prefill: one expert at a time over the tokens routed to it, float16 tiles.
        let hostIds = ids.asType(.int32).asArray(Int32.self)
        var tokensOf: [Int: [(token: Int, slot: Int)]] = [:]
        for (i, e) in hostIds.enumerated() { tokensOf[Int(e), default: []].append((i / k, i % k)) }
        var out = MLXArray.zeros([t, down.m], dtype: .float32)
        let w = weights.asType(.float32)
        for (e, uses) in tokensOf.sorted(by: { $0.key < $1.key }) {
            let tok = MLXArray(uses.map { Int32($0.token) })
            let flatSlot = MLXArray(uses.map { Int32($0.token * k + $0.slot) })
            let guW = gateUp.dequantized(expert: e, dtype: .float16)
            let gu = matmul(xRot[tok].asType(.float16), guW.transposed())
            let act = lamqSoftcappedSwiGLU(gu)
            let actRot = down.rot.input(act, signs: down.signsIn)
            let dnW = down.dequantized(expert: e, dtype: .float16)
            let y = matmul(actRot.asType(.float16), dnW.transposed()).asType(.float32)
            let we = w.reshaped([-1])[flatSlot]
            out = out.at[tok].add(y * expandedDimensions(we, axis: -1))
            eval(out)   // bound the graph and the tiles' lifetime to one expert
        }
        return out
    }
}

// MARK: - Embedding and head (spec §7)

/// The vocabulary buckets of `embed_packed.npz` or `head_packed.npz`.
public final class InklingLamqBuckets: @unchecked Sendable {
    public struct Bucket: @unchecked Sendable {
        let meta: InklingLamqBucketMeta
        let codes: MLXArray
        let scales: MLXArray
        let signs: MLXArray
        let levels: MLXArray?
        let lut: MLXArray?
        let format: InklingLamqTrellisFormat?
        let lo: Int
    }

    public let buckets: [Bucket]
    public let hidden: Int
    /// order[o] = token of bucket-ordered row o; inverse[token] = o.
    public let order: [Int32]
    public let inverse: [Int32]
    let inverseArray: MLXArray
    let rot: InklingLamqRotation

    public init(directory: URL, kind: String, tables: InklingLamqTables, hidden: Int = 2304) throws {
        self.hidden = hidden
        let metas = try JSONDecoder().decode(
            [InklingLamqBucketMeta].self,
            from: Data(contentsOf: directory.appendingPathComponent("\(kind)_meta.json")))
        let npz = try InklingLamqNpz(url: directory.appendingPathComponent("\(kind)_packed.npz"))
        order = try npz.host("order", of: Int32.self)
        var inv = [Int32](repeating: 0, count: order.count)
        for (o, tokenID) in order.enumerated() { inv[Int(tokenID)] = Int32(o) }
        inverse = inv
        inverseArray = MLXArray(inv)
        rot = try InklingLamqRotation.of(hidden)
        var lo = 0
        var out: [Bucket] = []
        for bm in metas {
            let kb = bm.kb
            let signs = try npz.array("b\(kb)_signs_in").asType(.float32)
            let scales = try npz.array("b\(kb)_scales")
            if bm.opt.hasPrefix("lm") {
                out.append(Bucket(
                    meta: bm, codes: try npz.array("b\(kb)_codes"), scales: scales, signs: signs,
                    levels: try npz.array("b\(kb)_levels").asType(.float32), lut: nil, format: nil, lo: lo))
            } else {
                guard let f = InklingLamqTrellisFormat.all[bm.opt], f.g == 1 else {
                    throw InklingLamqError.unsupported("\(kind) bucket format \(bm.opt)")
                }
                out.append(Bucket(
                    meta: bm, codes: try npz.array("b\(kb)_codes", shape: [bm.rows, hidden * f.k / 32]),
                    scales: scales, signs: signs, levels: nil, lut: try tables.lut(bm.opt), format: f, lo: lo))
            }
            lo += bm.rows
        }
        buckets = out
    }

    /// Rows `rows` (bucket-local) of bucket `b`, `Q * scale` in the rotated basis.
    func rows(_ b: Int, _ rows: [Int], dtype: DType = .float32) -> MLXArray {
        let bk = buckets[b]
        let sel = MLXArray(rows.map { UInt32($0) })
        if let f = bk.format, let lut = bk.lut {
            return lamqTrellisDequant(
                codes: bk.codes, scales: bk.scales, lut: lut, locs: sel, rowsPerGroup: 1, n: hidden,
                format: f, dtype: dtype)
        }
        return lamqLMDequant(
            codes: bk.codes, scales: bk.scales, levels: bk.levels!, rowsSel: sel, n: hidden, k: bk.meta.k, dtype: dtype)
    }

    func bucket(ofOrdered o: Int) -> Int {
        var b = 0
        while b + 1 < buckets.count && buckets[b + 1].lo <= o { b += 1 }
        return b
    }

    /// Embedding rows `[T, hidden]` float32 (before `embed_norm`) for host token ids.
    public func embed(_ tokens: [Int]) -> MLXArray {
        var groups: [Int: [(i: Int, r: Int)]] = [:]
        for (i, tok) in tokens.enumerated() {
            let o = Int(inverse[tok])
            let b = bucket(ofOrdered: o)
            groups[b, default: []].append((i, o - buckets[b].lo))
        }
        var parts: [MLXArray] = []
        var where_: [Int] = []
        for (b, items) in groups.sorted(by: { $0.key < $1.key }) {
            let q = rows(b, items.map(\.r))
            parts.append(rot.output(q, signs: buckets[b].signs))
            where_ += items.map(\.i)
        }
        var perm = [Int32](repeating: 0, count: tokens.count)
        for (pos, i) in where_.enumerated() { perm[i] = Int32(pos) }
        return concatenated(parts, axis: 0)[MLXArray(perm)]
    }

    /// Head logits `[T, V]` float32 before the soft cap, in token order.
    public func logits(_ h: MLXArray) -> MLXArray {
        let x = h.reshaped([-1, hidden])
        let t = x.dim(0)
        var parts: [MLXArray] = []
        for (b, bk) in buckets.enumerated() {
            let xr = rot.input(x, signs: bk.signs)
            let rowsB = bk.meta.rows
            if t <= lamqMatvecMaxTokens {
                if let f = bk.format, let lut = bk.lut {
                    parts.append(lamqTrellisMatvec(
                        x: xr, codes: bk.codes, scales: bk.scales, lut: lut,
                        sel: MLXArray.zeros([t, 1], dtype: .uint32), rowsPerGroup: rowsB, xDiv: 1, n: hidden, format: f))
                } else {
                    parts.append(lamqLMMatvec(x: xr, codes: bk.codes, scales: bk.scales, levels: bk.levels!, m: rowsB, n: hidden, k: bk.meta.k))
                }
            } else {
                // Prefill: 16K-row float16 tiles (~75 MB each).
                let xh = xr.asType(.float16)
                var r0 = 0
                while r0 < rowsB {
                    let r1 = min(rowsB, r0 + 16_384)
                    let w = rows(b, Array(r0 ..< r1), dtype: .float16)
                    let y = matmul(xh, w.transposed()).asType(.float32)
                    eval(y)
                    parts.append(y)
                    r0 = r1
                }
            }
        }
        return concatenated(parts, axis: 1).take(inverseArray, axis: 1)
    }
}
