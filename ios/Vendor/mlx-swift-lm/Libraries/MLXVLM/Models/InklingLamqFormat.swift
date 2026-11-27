// [agent-console patch D] LookAtMyQuant ("lamq") packed Inkling 2 Mini: formats, Hadamard
// rotations and the packed-file reader. Not part of mlx-swift-lm PR #821.
// Spec: docs/LAMQ-FORMAT-SPEC.md; reference: ios/tools/lamq/lamq_ref.py.

import Foundation
import MLX

// MARK: - Formats

/// A trellis code format (spec §2).
public struct InklingLamqTrellisFormat: Sendable, Hashable {
    public let name: String
    /// Bits per weight (stream advance per column).
    public let k: Int
    /// Weights per 12-bit state.
    public let g: Int
    /// Table entries.
    public let entries: Int
    /// `ht*` hash the state with 0x9E3779B1; `rtr*` index the table with it.
    public let hashed: Bool

    public var log2K: Int { entries.trailingZeroBitCount }

    public static let all: [String: InklingLamqTrellisFormat] = [
        "ht1": .init(name: "ht1", k: 1, g: 4, entries: 512, hashed: true),
        "ht2": .init(name: "ht2", k: 2, g: 4, entries: 512, hashed: true),
        "ht3": .init(name: "ht3", k: 3, g: 2, entries: 1024, hashed: true),
        "rtr1": .init(name: "rtr1", k: 1, g: 1, entries: 4096, hashed: false),
        "rtr2": .init(name: "rtr2", k: 2, g: 1, entries: 4096, hashed: false),
        "rtr3": .init(name: "rtr3", k: 3, g: 1, entries: 4096, hashed: false),
        "rtr4": .init(name: "rtr4", k: 4, g: 1, entries: 4096, hashed: false),
    ]

    /// Format ids of `{gu,dn}_rg_map[..., 0]` (manifest `fmt_ids`). 7 is NVFP4.
    public static let byID: [Int: String] = [
        0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4", 7: "nvfp4",
    ]
}

public enum InklingLamqError: LocalizedError {
    case file(String)
    case unsupported(String)
    case mismatch(String)

    public var errorDescription: String? {
        switch self {
        case .file(let s): return "lamq file: \(s)"
        case .unsupported(let s): return "lamq: unsupported \(s)"
        case .mismatch(let s): return "lamq self-check: \(s)"
        }
    }
}

// MARK: - Hadamard rotations (spec §5)

/// `H = kron(A, B)` for vector length `n`; the same construction as `lamq_ref.hadamard_factors`.
public enum InklingLamqHadamard {
    static func isPrime(_ q: Int) -> Bool {
        guard q > 1 else { return false }
        var d = 2
        while d * d <= q {
            if q % d == 0 { return false }
            d += 1
        }
        return true
    }

    /// Sylvester matrix of order 2^a, row-major.
    static func sylvester(_ a: Int) -> (order: Int, values: [Float]) {
        var h: [Float] = [1]
        var order = 1
        for _ in 0 ..< a {
            let o2 = order * 2
            var next = [Float](repeating: 0, count: o2 * o2)
            for i in 0 ..< order {
                for j in 0 ..< order {
                    let v = h[i * order + j]
                    next[i * o2 + j] = v
                    next[i * o2 + j + order] = v
                    next[(i + order) * o2 + j] = v
                    next[(i + order) * o2 + j + order] = -v
                }
            }
            h = next
            order = o2
        }
        return (order, h)
    }

    /// Paley construction I: order q + 1 for a prime q = 3 (mod 4), row-major.
    static func paley(_ q: Int) -> (order: Int, values: [Float]) {
        var squares = Set<Int>()
        for x in 1 ..< q { squares.insert((x * x) % q) }
        func chi(_ a: Int) -> Float { a == 0 ? 0 : (squares.contains(a) ? 1 : -1) }
        let o = q + 1
        var s = [Float](repeating: 0, count: o * o)
        for j in 1 ..< o { s[j] = 1 }               // s[0, 1:] = 1
        for i in 1 ..< o { s[i * o] = -1 }          // s[1:, 0] = -1
        for i in 0 ..< q {
            for j in 0 ..< q {
                s[(i + 1) * o + (j + 1)] = chi(((j - i) % q + q) % q)
            }
        }
        for i in 0 ..< o { s[i * o + i] += 1 }
        return (o, s)
    }

    /// (A, B) as float32 MLX arrays.
    public static func factors(_ n: Int) throws -> (a: MLXArray, b: MLXArray) {
        var a = 0
        while (n >> (a + 1)) << (a + 1) == n { a += 1 }   // largest power of two dividing n
        while a >= 0 {
            if n % (1 << a) == 0 {
                let p = n / (1 << a)
                if p == 1 || p == 2 || (isPrime(p - 1) && (p - 1) % 4 == 3) {
                    let left: (order: Int, values: [Float])
                    let right: (order: Int, values: [Float])
                    if p == 1 && a > 1 {
                        left = sylvester(a / 2)
                        right = sylvester(a - a / 2)
                    } else if p == 1 {
                        left = sylvester(a)
                        right = (1, [1])
                    } else if p == 2 {
                        left = sylvester(a)
                        right = (2, [1, 1, 1, -1])
                    } else {
                        left = sylvester(a)
                        right = paley(p - 1)
                    }
                    return (MLXArray(left.values, [left.order, left.order]),
                            MLXArray(right.values, [right.order, right.order]))
                }
            }
            a -= 1
        }
        throw InklingLamqError.unsupported("Hadamard order \(n)")
    }
}

/// Cached rotation of one vector length: `rht_in(x, s) = x @ R.T`, `rht_out(y, s) = y @ R`,
/// R = kron(A, B) diag(s) / sqrt(n). float32.
public final class InklingLamqRotation: @unchecked Sendable {
    public let n: Int
    let a: MLXArray
    let b: MLXArray
    let bT: MLXArray
    let aT: MLXArray
    let invSqrt: Float

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [Int: InklingLamqRotation] = [:]

    public static func of(_ n: Int) throws -> InklingLamqRotation {
        lock.lock()
        defer { lock.unlock() }
        if let r = cache[n] { return r }
        let r = try InklingLamqRotation(n: n)
        cache[n] = r
        return r
    }

    init(n: Int) throws {
        self.n = n
        let (a, b) = try InklingLamqHadamard.factors(n)
        self.a = a
        self.b = b
        self.aT = a.transposed()
        self.bT = b.transposed()
        self.invSqrt = 1 / Float(n).squareRoot()
    }

    /// `x @ R.T` for x `[..., n]` and signs `[n]` (or `[rows, n]` broadcastable).
    public func input(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let shape = x.shape
        let xs = (x.asType(.float32) * signs.asType(.float32)).reshaped([-1, a.dim(0), b.dim(0)])
        let y = matmul(matmul(a, xs), bT)
        return (y * invSqrt).reshaped(shape)
    }

    /// `y @ R` for y `[..., n]`.
    public func output(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let shape = y.shape
        let ys = y.asType(.float32).reshaped([-1, a.dim(0), b.dim(0)])
        let z = matmul(matmul(aT, ys), b).reshaped(shape)
        return z * (signs.asType(.float32) * invSqrt)
    }
}

// MARK: - npz / npy reader (spec §1: STORED zip members, npy v1-v3)

/// One array inside a `.npz`: where its raw little-endian payload sits in the file.
public struct InklingLamqNpyEntry: Sendable {
    public let name: String
    /// numpy `descr`, e.g. `<u4`, `<f2`, `|u1`, `<i2`, `|i1`, `<f4`, `<i4`.
    public let descr: String
    public let shape: [Int]
    /// File offset of the first payload byte, and the payload length.
    public let offset: UInt64
    public let byteCount: Int

    public var count: Int { shape.reduce(1, *) }
}

/// Reads arrays of a STORED `.npz` by `pread`, without unzipping (a lamq layer file is 145-300 MB).
public final class InklingLamqNpz: @unchecked Sendable {
    public let url: URL
    public let entries: [String: InklingLamqNpyEntry]
    private let handle: FileHandle

    public init(url: URL) throws {
        self.url = url
        self.handle = try FileHandle(forReadingFrom: url)
        self.entries = try Self.index(handle: handle, name: url.lastPathComponent)
    }

    deinit { try? handle.close() }

    public func has(_ name: String) -> Bool { entries[name] != nil }

    public func entry(_ name: String) throws -> InklingLamqNpyEntry {
        guard let e = entries[name] else { throw InklingLamqError.file("\(url.lastPathComponent) has no \(name)") }
        return e
    }

    /// The raw payload bytes of an array.
    public func bytes(_ name: String) throws -> Data {
        let e = try entry(name)
        return try read(offset: e.offset, count: e.byteCount)
    }

    /// Bytes `count` at `offset` of the file (for streaming a slab entry).
    public func read(offset: UInt64, count: Int) throws -> Data {
        try handle.seek(toOffset: offset)
        guard let data = try handle.read(upToCount: count), data.count == count else {
            throw InklingLamqError.file("\(url.lastPathComponent): short read at \(offset)")
        }
        return data
    }

    /// An array as an MLX array of its own dtype and shape (`shapeOverride` reshapes it).
    public func array(_ name: String, shape shapeOverride: [Int]? = nil) throws -> MLXArray {
        let e = try entry(name)
        let data = try bytes(name)
        let shape = shapeOverride ?? (e.shape.isEmpty ? [1] : e.shape)
        switch e.descr {
        case "<u4": return MLXArray(Self.values(data, UInt32.self), shape)
        case "<i4": return MLXArray(Self.values(data, Int32.self), shape)
        case "<f4": return MLXArray(Self.values(data, Float.self), shape)
        case "<f2": return MLXArray(Self.values(data, Float16.self), shape)
        case "<i2": return MLXArray(Self.values(data, Int16.self), shape)
        case "|u1": return MLXArray(Self.values(data, UInt8.self), shape)
        case "|i1": return MLXArray(Self.values(data, Int8.self), shape)
        default: throw InklingLamqError.unsupported("dtype \(e.descr) of \(name)")
        }
    }

    /// An array's elements on the host.
    public func host<T>(_ name: String, of _: T.Type) throws -> [T] {
        Self.values(try bytes(name), T.self)
    }

    static func values<T>(_ data: Data, _: T.Type) -> [T] {
        let n = data.count / MemoryLayout<T>.stride
        guard n > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.baseAddress!.assumingMemoryBound(to: T.self), count: n))
        }
    }

    // ZIP: end of central directory -> central directory -> local headers -> npy headers.
    static func index(handle: FileHandle, name: String) throws -> [String: InklingLamqNpyEntry] {
        let size = try handle.seekToEnd()
        let tailLength = Int(min(size, 65_557))
        try handle.seek(toOffset: size - UInt64(tailLength))
        guard let tail = try handle.read(upToCount: tailLength) else { throw InklingLamqError.file("\(name): empty") }
        let t = [UInt8](tail)
        func u16(_ b: [UInt8], _ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
        func u32(_ b: [UInt8], _ o: Int) -> Int { u16(b, o) | u16(b, o + 2) << 16 }
        var eocd = -1
        var i = t.count - 22
        while i >= 0 {
            if t[i] == 0x50, t[i + 1] == 0x4B, t[i + 2] == 0x05, t[i + 3] == 0x06 { eocd = i; break }
            i -= 1
        }
        guard eocd >= 0 else { throw InklingLamqError.file("\(name): no ZIP end record") }
        let count = u16(t, eocd + 10)
        let cdSize = u32(t, eocd + 12)
        let cdOffset = u32(t, eocd + 16)
        if count == 0xFFFF || cdOffset == 0xFFFF_FFFF {
            throw InklingLamqError.unsupported("\(name): zip64 archive")
        }
        try handle.seek(toOffset: UInt64(cdOffset))
        guard let cdData = try handle.read(upToCount: cdSize), cdData.count == cdSize else {
            throw InklingLamqError.file("\(name): short central directory")
        }
        let cd = [UInt8](cdData)
        var out: [String: InklingLamqNpyEntry] = [:]
        var p = 0
        for _ in 0 ..< count {
            guard u32(cd, p) == 0x0201_4B50 else { throw InklingLamqError.file("\(name): bad central directory") }
            let method = u16(cd, p + 10)
            let compressed = u32(cd, p + 20)
            let nameLength = u16(cd, p + 28)
            let extraLength = u16(cd, p + 30)
            let commentLength = u16(cd, p + 32)
            let localOffset = u32(cd, p + 42)
            let member = String(decoding: cd[(p + 46) ..< (p + 46 + nameLength)], as: UTF8.self)
            p += 46 + nameLength + extraLength + commentLength
            guard method == 0 else { throw InklingLamqError.unsupported("\(name):\(member) is compressed") }
            try handle.seek(toOffset: UInt64(localOffset))
            guard let local = try handle.read(upToCount: 30), local.count == 30 else {
                throw InklingLamqError.file("\(name): short local header")
            }
            let l = [UInt8](local)
            let dataStart = UInt64(localOffset + 30 + u16(l, 26) + u16(l, 28))
            // npy header
            try handle.seek(toOffset: dataStart)
            guard let pre = try handle.read(upToCount: 12), pre.count == 12 else { throw InklingLamqError.file("\(member): short") }
            let h = [UInt8](pre)
            guard h[0] == 0x93, h[1] == 0x4E else { throw InklingLamqError.file("\(member): not npy") }
            let major = Int(h[6])
            let headerLength = major == 1 ? u16(h, 8) : u32(h, 8)
            let headerStart = UInt64(major == 1 ? 10 : 12)
            try handle.seek(toOffset: dataStart + headerStart)
            guard let hd = try handle.read(upToCount: headerLength) else { throw InklingLamqError.file("\(member): header") }
            let header = String(decoding: hd, as: UTF8.self)
            let (descr, shape, fortran) = try parseHeader(header, member: member)
            guard !fortran else { throw InklingLamqError.unsupported("\(member): Fortran order") }
            let payload = dataStart + headerStart + UInt64(headerLength)
            let arrayName = member.hasSuffix(".npy") ? String(member.dropLast(4)) : member
            let byteCount = compressed - Int(payload - dataStart)
            out[arrayName] = InklingLamqNpyEntry(
                name: arrayName, descr: descr, shape: shape, offset: payload, byteCount: byteCount)
        }
        return out
    }

    /// `{'descr': '<u4', 'fortran_order': False, 'shape': (288, 128, 144), }`
    static func parseHeader(_ h: String, member: String) throws -> (String, [Int], Bool) {
        func value(after key: String) -> Substring? {
            guard let r = h.range(of: "'\(key)':") else { return nil }
            return h[r.upperBound...].drop(while: { $0 == " " })
        }
        guard let d = value(after: "descr"), d.first == "'",
            let end = d.dropFirst().firstIndex(of: "'")
        else { throw InklingLamqError.file("\(member): no descr") }
        let descr = String(d[d.index(after: d.startIndex) ..< end])
        let fortran = value(after: "fortran_order")?.hasPrefix("True") ?? false
        guard let s = value(after: "shape"), s.first == "(", let close = s.firstIndex(of: ")") else {
            throw InklingLamqError.file("\(member): no shape")
        }
        let shape = s[s.index(after: s.startIndex) ..< close].split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return (descr, shape, fortran)
    }
}

// MARK: - Metadata (spec §6-7)

public struct InklingLamqLinearMeta: Codable, Sendable {
    public let fmt: String
    public let m: Int
    public let n: Int
    public let k: Int
    public let g: Int?
    public let K: Int?
    public let hash_mult: UInt64?
    public let out_rot: Bool?
    public let safe: String
}

public struct InklingLamqLayerMeta: Codable, Sendable {
    public let layer: Int
    public let linears: [String: InklingLamqLinearMeta]
    public let experts: [String: [String: Int]]
    public let n_rg: Int?
    public let rg_rows: Int?

    public static func load(_ url: URL) throws -> InklingLamqLayerMeta {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

public struct InklingLamqBucketMeta: Codable, Sendable {
    public let kb: Int
    public let opt: String
    public let rows: Int
    public let k: Int
}

/// The six attention linears of a layer (spec §6).
public let inklingLamqLinearNames = [
    "self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear",
    "self_attn.relative_position_bias.linear", "self_attn.o_proj.gate_proj",
    "self_attn.o_proj.linear",
]
