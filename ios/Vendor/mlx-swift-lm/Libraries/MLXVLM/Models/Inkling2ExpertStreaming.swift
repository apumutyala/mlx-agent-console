// [agent-console patch C] SSD expert streaming for Inkling 2. Not part of PR #821.
//
// The routed experts of a streamed build live in `experts.bin` (see `tools/expert_repack.py` and
// `experts.json`), not in the safetensors. Each MoE layer keeps `S` expert slots inside its usual
// `[S, out, packed]` expert tensors; a call's expert ids are remapped to slot ids on the host and a
// miss is read with `preadv` straight into the slot's bytes. The quantized matmuls and fused
// kernels run unchanged against the slot tensors.
//
// Off by default: nothing changes unless `Inkling2ExpertStreaming.options.enabled` is set before
// the load and the model directory holds an `experts.json`.

import Foundation
import MLX

#if canImport(Darwin)
    import Darwin
#endif

// MARK: - Public surface

/// How a streamed Inkling 2 build is loaded. Read when the load starts.
public struct Inkling2ExpertStreamingOptions: Sendable, Equatable {
    /// Stream experts when the model directory holds `experts.json`.
    public var enabled: Bool
    /// Fraction of each layer's experts that stay resident, 0...1. 1 keeps every expert (each one
    /// is read once, on first use) and is the exactness gate: outputs must match a resident load.
    public var residentFraction: Double
    /// Lower bound on slots per layer; never below the experts per token.
    public var minimumSlots: Int
    /// Threads reading the misses of one call.
    public var readers: Int
    /// Writes one JSON line per MoE call (`forward`, `visit`, `layer`, `rows`, `ids`) to this file.
    /// Also works with `enabled == false` on a resident build (routing trace only).
    public var traceURL: URL?

    public init(
        enabled: Bool = false, residentFraction: Double = 0.5, minimumSlots: Int = 8,
        readers: Int = 4, traceURL: URL? = nil
    ) {
        self.enabled = enabled
        self.residentFraction = residentFraction
        self.minimumSlots = minimumSlots
        self.readers = readers
        self.traceURL = traceURL
    }
}

/// Counters of the model loaded last. Times are wall-clock seconds on the model's thread.
public struct Inkling2ExpertStreamingSnapshot: Sendable, Codable, Equatable {
    /// Whether experts stream (false: the trace runs on a resident model).
    public var streaming: Bool
    public var layers: Int
    public var slotsPerLayer: Int
    public var expertsPerLayer: Int
    /// Bytes of the slot tensors (resident expert memory).
    public var slotBytes: Int
    /// MoE layer calls (48 per decoded token for Inkling 2 Mini).
    public var calls: Int
    /// Distinct experts the calls needed, and how many of them had to be read.
    public var expertLookups: Int
    public var misses: Int
    /// Calls that needed more distinct experts than slots and ran in chunks (prefill).
    public var chunkedCalls: Int
    public var bytesRead: Int
    /// Time inside `preadv` (all reader threads of a call, wall clock).
    public var readSeconds: Double
    /// Time the host waited for the routing indices (the GPU work before each MoE call).
    public var routeWaitSeconds: Double
    /// Time in `Stream.gpu.synchronize()` before overwriting slots.
    public var gpuSyncSeconds: Double
    /// Forward passes (one per decoded token; one per prefill chunk).
    public var forwards: Int

    public var hitRate: Double {
        expertLookups == 0 ? 0 : 1 - Double(misses) / Double(expertLookups)
    }
    public var readGBPerSecond: Double {
        readSeconds == 0 ? 0 : Double(bytesRead) / readSeconds / 1e9
    }
}

/// Process-wide switch and readout for Inkling 2 expert streaming.
public enum Inkling2ExpertStreaming {
    /// Set before loading a streamed build; a load reads it once.
    public static var options: Inkling2ExpertStreamingOptions {
        get { streamingState.withLock { $0.options } }
        set { streamingState.withLock { $0.options = newValue } }
    }

    /// The counters of the most recently loaded Inkling 2 model that streams or traces, if it is
    /// still alive.
    public static func snapshot() -> Inkling2ExpertStreamingSnapshot? {
        streamingState.withLock { $0.current }?.snapshot()
    }

    /// Zeroes the counters (not the slot contents), e.g. between benchmark phases.
    public static func resetStatistics() {
        streamingState.withLock { $0.current }?.resetStatistics()
    }

    /// Writes buffered trace lines to the trace file.
    public static func flushTrace() {
        streamingState.withLock { $0.current }?.flushTrace()
    }

    /// Bytes of `experts.bin` and the per-expert record sizes, without loading anything.
    public static func manifestSummary(directory: URL) throws -> (
        fileBytes: Int, expertBytes: Int, layers: Int, experts: Int
    ) {
        let manifest = try Inkling2ExpertManifest.load(directory: directory)
        return (
            manifest.fileBytes, manifest.layers.reduce(0) { $0 + $1.experts * $1.recordBytes },
            manifest.layers.count, manifest.experts
        )
    }
}

private final class Inkling2ExpertStreamingState: @unchecked Sendable {
    struct Values {
        var options = Inkling2ExpertStreamingOptions()
        weak var current: Inkling2ExpertStreamRuntime?
    }
    private let lock = NSLock()
    private var values = Values()

    func withLock<R>(_ body: (inout Values) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&values)
    }
}

private let streamingState = Inkling2ExpertStreamingState()

// MARK: - Errors

enum Inkling2ExpertStreamingError: LocalizedError {
    case manifest(String)
    case io(String)
    case unsupported(String)
    case binding(String)

    var errorDescription: String? {
        switch self {
        case .manifest(let m): "experts.json: \(m)"
        case .io(let m): "experts.bin: \(m)"
        case .unsupported(let m): "Expert streaming does not support \(m)"
        case .binding(let m): "Expert slots: \(m)"
        }
    }
}

// MARK: - Manifest

/// `experts.json` as written by `tools/expert_repack.py` (format version 1).
struct Inkling2ExpertManifest: Decodable, Sendable {
    struct Projection: Decodable, Sendable {
        let mode: String
        let bits: Int
        let groupSize: Int
        let inputDims: Int
        let outputDims: Int

        enum CodingKeys: String, CodingKey {
            case mode, bits
            case groupSize = "group_size"
            case inputDims = "input_dims"
            case outputDims = "output_dims"
        }
    }

    struct Field: Decodable, Sendable {
        /// `gate_up_proj.weight`, `down_proj.scales`, ...
        let name: String
        let tensor: String
        let dtype: String
        /// Per expert, without the expert dimension.
        let shape: [Int]
        /// Inside the record.
        let offset: Int
        let nbytes: Int
    }

    struct Layer: Decodable, Sendable {
        let layer: Int
        let tensorPrefix: String
        let experts: Int
        let baseOffset: Int
        let recordBytes: Int
        let recordStride: Int
        let sha256: String
        let projections: [String: Projection]
        let fields: [Field]

        enum CodingKeys: String, CodingKey {
            case layer, experts, sha256, projections, fields
            case tensorPrefix = "tensor_prefix"
            case baseOffset = "base_offset"
            case recordBytes = "record_bytes"
            case recordStride = "record_stride"
        }
    }

    static let formatName = "inkling2-expert-stream"
    static let fileName = "experts.json"

    let format: String
    let version: Int
    let file: String
    let fileBytes: Int
    let experts: Int
    let expertsPerToken: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let layers: [Layer]

    enum CodingKeys: String, CodingKey {
        case format, version, file, experts, layers
        case fileBytes = "file_bytes"
        case expertsPerToken = "experts_per_token"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
    }

    static func exists(in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(fileName).path)
    }

    static func load(directory: URL) throws -> Inkling2ExpertManifest {
        let data = try Data(contentsOf: directory.appendingPathComponent(fileName))
        let manifest = try JSONDecoder().decode(Inkling2ExpertManifest.self, from: data)
        try manifest.validate()
        return manifest
    }

    func validate() throws {
        guard format == Self.formatName else {
            throw Inkling2ExpertStreamingError.manifest("format \(format)")
        }
        guard version == 1 else { throw Inkling2ExpertStreamingError.manifest("version \(version)") }
        guard !file.contains("/"), !file.hasPrefix(".") else {
            throw Inkling2ExpertStreamingError.manifest("file name \(file)")
        }
        for layer in layers {
            guard layer.experts == experts, layer.recordStride >= layer.recordBytes,
                layer.baseOffset >= 0,
                layer.baseOffset + layer.experts * layer.recordStride <= fileBytes
            else { throw Inkling2ExpertStreamingError.manifest("layer \(layer.layer) bounds") }
            for field in layer.fields
            where field.offset < 0 || field.offset + field.nbytes > layer.recordBytes {
                throw Inkling2ExpertStreamingError.manifest(
                    "layer \(layer.layer) field \(field.name) bounds")
            }
        }
    }
}

extension Inkling2ExpertManifest.Field {
    var dtypeValue: DType? {
        switch dtype {
        case "U32": .uint32
        case "BF16": .bfloat16
        case "F16": .float16
        case "F32": .float32
        case "U8": .uint8
        default: nil
        }
    }
}

// MARK: - File

/// `experts.bin`, opened without the page cache: a miss is a DMA into slot memory, and cached
/// pages would only add footprint the OS can't use for us (iOS does not keep them warm anyway).
final class Inkling2ExpertFile: @unchecked Sendable {
    let url: URL
    private let fd: Int32

    init(url: URL, expectedBytes: Int) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw Inkling2ExpertStreamingError.io("open \(url.lastPathComponent): errno \(errno)")
        }
        var info = stat()
        guard fstat(fd, &info) == 0, Int(info.st_size) == expectedBytes else {
            close(fd)
            throw Inkling2ExpertStreamingError.io(
                "\(url.lastPathComponent) is not the \(expectedBytes) bytes experts.json describes")
        }
        #if canImport(Darwin)
            _ = fcntl(fd, F_NOCACHE, 1)
            _ = fcntl(fd, F_RDAHEAD, 0)
        #endif
        self.fd = fd
    }

    deinit { close(fd) }

    /// Reads `total` bytes at `offset` into `vectors`, in order, retrying short reads and EINTR.
    func read(_ vectors: [iovec], offset: Int, total: Int) throws {
        var vectors = vectors
        var done = 0
        var attempts = 0
        while done < total {
            let n = vectors.withUnsafeBufferPointer { buffer in
                preadv(fd, buffer.baseAddress, Int32(buffer.count), off_t(offset + done))
            }
            if n < 0 {
                if errno == EINTR || errno == EAGAIN, attempts < 16 {
                    attempts += 1
                    continue
                }
                throw Inkling2ExpertStreamingError.io("preadv at \(offset + done): errno \(errno)")
            }
            if n == 0 { throw Inkling2ExpertStreamingError.io("end of file at \(offset + done)") }
            done += n
            if done < total { vectors = Self.advance(vectors, by: n) }
        }
    }

    /// `vectors` without their first `count` bytes.
    static func advance(_ vectors: [iovec], by count: Int) -> [iovec] {
        var skip = count
        var result = [iovec]()
        result.reserveCapacity(vectors.count)
        for v in vectors {
            if skip >= v.iov_len {
                skip -= v.iov_len
                continue
            }
            result.append(iovec(iov_base: v.iov_base.map { $0 + skip }, iov_len: v.iov_len - skip))
            skip = 0
        }
        return result
    }
}

// MARK: - Runtime

/// One per loaded model: the options it loaded with, the file, the counters and the trace.
final class Inkling2ExpertStreamRuntime: @unchecked Sendable {
    let options: Inkling2ExpertStreamingOptions
    let manifest: Inkling2ExpertManifest?
    let file: Inkling2ExpertFile?

    /// The forward pass and layer visit being built, for the trace.
    var forwardIndex = 0
    var visit = 0

    private let lock = NSLock()
    private var counters = Inkling2ExpertStreamingSnapshot(
        streaming: false, layers: 0, slotsPerLayer: 0, expertsPerLayer: 0, slotBytes: 0, calls: 0,
        expertLookups: 0, misses: 0, chunkedCalls: 0, bytesRead: 0, readSeconds: 0,
        routeWaitSeconds: 0, gpuSyncSeconds: 0, forwards: 0)
    private let trace: Inkling2ExpertTraceWriter?

    var streams: Bool { manifest != nil }

    /// The runtime for a load from `directory`, or nil when it neither streams nor traces.
    static func make(directory: URL) throws -> Inkling2ExpertStreamRuntime? {
        let options = Inkling2ExpertStreaming.options
        let streams = options.enabled && Inkling2ExpertManifest.exists(in: directory)
        guard streams || options.traceURL != nil else { return nil }
        var manifest: Inkling2ExpertManifest?
        var file: Inkling2ExpertFile?
        if streams {
            let loaded = try Inkling2ExpertManifest.load(directory: directory)
            file = try Inkling2ExpertFile(
                url: directory.appendingPathComponent(loaded.file), expectedBytes: loaded.fileBytes)
            manifest = loaded
        }
        let runtime = try Inkling2ExpertStreamRuntime(
            options: options, manifest: manifest, file: file)
        streamingState.withLock { $0.current = runtime }
        return runtime
    }

    private init(
        options: Inkling2ExpertStreamingOptions, manifest: Inkling2ExpertManifest?,
        file: Inkling2ExpertFile?
    ) throws {
        self.options = options
        self.manifest = manifest
        self.file = file
        self.trace = try options.traceURL.map { try Inkling2ExpertTraceWriter(url: $0) }
        counters.streaming = manifest != nil
        // One header line per load; the lines after it until the next header belong to it.
        trace?.appendLine(
            "{\"load\":\"\(ISO8601DateFormatter().string(from: Date()))\","
                + "\"streaming\":\(manifest != nil),\"resident_fraction\":\(options.residentFraction)}")
    }

    /// Slots per layer for `experts` experts and `topK` per token.
    func capacity(experts: Int, topK: Int) -> Int {
        let wanted = Int((options.residentFraction * Double(experts)).rounded())
        return min(experts, max(wanted, options.minimumSlots, topK))
    }

    func registerLayer(slots: Int, experts: Int, bytes: Int) {
        lock.withLock {
            counters.layers += 1
            counters.slotsPerLayer = slots
            counters.expertsPerLayer = experts
            counters.slotBytes += bytes
        }
    }

    func beginForward() {
        forwardIndex += 1
        visit = 0
        lock.withLock { counters.forwards += 1 }
    }

    func record(
        lookups: Int = 0, misses: Int = 0, bytes: Int = 0, read: Double = 0, routeWait: Double = 0,
        gpuSync: Double = 0, call: Bool = false, chunked: Bool = false
    ) {
        lock.withLock {
            counters.expertLookups += lookups
            counters.misses += misses
            counters.bytesRead += bytes
            counters.readSeconds += read
            counters.routeWaitSeconds += routeWait
            counters.gpuSyncSeconds += gpuSync
            if call { counters.calls += 1 }
            if chunked { counters.chunkedCalls += 1 }
        }
    }

    func traceCall(layer: Int, rows: Int, ids: [UInt32]) {
        trace?.append(forward: forwardIndex, visit: visit, layer: layer, rows: rows, ids: ids)
    }

    var tracing: Bool { trace != nil }

    func snapshot() -> Inkling2ExpertStreamingSnapshot { lock.withLock { counters } }

    func resetStatistics() {
        lock.withLock {
            let keep = counters
            counters = Inkling2ExpertStreamingSnapshot(
                streaming: keep.streaming, layers: keep.layers, slotsPerLayer: keep.slotsPerLayer,
                expertsPerLayer: keep.expertsPerLayer, slotBytes: keep.slotBytes, calls: 0,
                expertLookups: 0, misses: 0, chunkedCalls: 0, bytesRead: 0, readSeconds: 0,
                routeWaitSeconds: 0, gpuSyncSeconds: 0, forwards: 0)
        }
    }

    func flushTrace() { trace?.flush() }

    deinit { trace?.flush() }
}

@inline(__always) func inkling2Seconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
}

// MARK: - Trace

/// Buffered JSON lines: `{"f":forward,"v":visit,"l":layer,"r":rows,"ids":[...]}`. The ids are the
/// routed experts of every row, row-major (`rows * K` values).
final class Inkling2ExpertTraceWriter: @unchecked Sendable {
    private let handle: FileHandle
    private var buffer = Data()
    private let lock = NSLock()

    init(url: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
        handle = try FileHandle(forWritingTo: url)
        handle.seekToEndOfFile()
    }

    func append(forward: Int, visit: Int, layer: Int, rows: Int, ids: [UInt32]) {
        var line = "{\"f\":\(forward),\"v\":\(visit),\"l\":\(layer),\"r\":\(rows),\"ids\":["
        line.reserveCapacity(line.utf8.count + ids.count * 4 + 3)
        for (i, id) in ids.enumerated() {
            if i > 0 { line += "," }
            line += String(id)
        }
        line += "]}\n"
        lock.withLock {
            buffer.append(contentsOf: line.utf8)
            if buffer.count >= 1 << 20 { writeLocked() }
        }
    }

    func appendLine(_ line: String) {
        lock.withLock { buffer.append(contentsOf: (line + "\n").utf8) }
    }

    func flush() { lock.withLock { writeLocked() } }

    private func writeLocked() {
        guard !buffer.isEmpty else { return }
        handle.write(buffer)
        buffer.removeAll(keepingCapacity: true)
    }

    deinit {
        flush()
        try? handle.close()
    }
}

// MARK: - Slot cache

/// The expert slots of one MoE layer.
///
/// Storage is the layer's own expert tensors, `[S, out, ...]`, allocated once by MLX. A miss is
/// read straight into the slot's bytes of each tensor: no array is created per token, and MLX's
/// graph never sees a write (the arrays' contents change between evaluations, like a buffer the
/// app owns). That is safe because (1) every read happens after `eval` of the call's routing
/// indices, which depend on all earlier work of this model, plus a `Stream.gpu.synchronize()`,
/// so no queued kernel still reads a victim slot, and (2) a call never evicts an expert it uses.
///
/// Eviction is decayed-use (LFU with a half-life, as in oMLX #4951): each use adds 1 to the slot's
/// score and every 4 calls all scores decay by 0.7; ties go to the least recently used slot.
/// With `S == E` slot `e` holds expert `e`, so the ids need no remapping and the model computes
/// exactly what a resident load computes.
final class Inkling2ExpertSlotCache {
    /// One field of the record: where its bytes land for slot `s`.
    struct Destination {
        let name: String
        let array: MLXArray
        let bytesPerSlot: Int
        let offsetInRecord: Int
        var base: UnsafeMutableRawPointer?
    }

    let layer: Int
    let experts: Int
    let capacity: Int
    let identity: Bool
    private let spec: Inkling2ExpertManifest.Layer
    private let runtime: Inkling2ExpertStreamRuntime
    private var destinations: [Destination]
    /// Bytes of the record before its first field and between fields: read into scratch memory.
    private let gaps: [(offset: Int, length: Int)]
    private var scratch: UnsafeMutableRawPointer?
    private var scratchBytes = 0
    private let readSpan: Int
    private var bound = false

    private var slotOf: [Int16]
    private var expertOf: [Int16]
    private var score: [Float]
    private var lastUse: [Int]
    private var stamp: [Int]
    private var free: [Int16]
    private var loaded = 0
    private var call = 0

    init(
        spec: Inkling2ExpertManifest.Layer, capacity: Int, arrays: [String: MLXArray],
        runtime: Inkling2ExpertStreamRuntime
    ) throws {
        let experts = spec.experts
        let identity = capacity == experts
        guard experts <= Int(Int16.max), capacity > 0, capacity <= experts else {
            throw Inkling2ExpertStreamingError.manifest("layer \(spec.layer): \(capacity) slots")
        }

        var destinations = [Destination]()
        for field in spec.fields.sorted(by: { $0.offset < $1.offset }) {
            guard let array = arrays[field.name] else {
                throw Inkling2ExpertStreamingError.unsupported(
                    "field \(field.name) (layer \(spec.layer))")
            }
            guard array.shape == [capacity] + field.shape, array.dtype == field.dtypeValue,
                array.nbytes == capacity * field.nbytes
            else {
                throw Inkling2ExpertStreamingError.binding(
                    "layer \(spec.layer) \(field.name): slot tensor \(array.shape) \(array.dtype) "
                        + "does not hold \(capacity) x \(field.shape) \(field.dtype)")
            }
            destinations.append(
                Destination(
                    name: field.name, array: array, bytesPerSlot: field.nbytes,
                    offsetInRecord: field.offset))
        }
        var gaps = [(offset: Int, length: Int)]()
        var cursor = 0
        for d in destinations {
            if d.offsetInRecord > cursor {
                gaps.append((cursor, d.offsetInRecord - cursor))
            }
            cursor = d.offsetInRecord + d.bytesPerSlot
        }

        self.layer = spec.layer
        self.experts = experts
        self.capacity = capacity
        self.identity = identity
        self.spec = spec
        self.runtime = runtime
        self.destinations = destinations
        self.gaps = gaps
        self.readSpan = cursor
        self.slotOf = Array(repeating: -1, count: experts)
        self.expertOf = Array(repeating: -1, count: capacity)
        self.score = Array(repeating: 0, count: capacity)
        self.lastUse = Array(repeating: 0, count: capacity)
        self.stamp = Array(repeating: -1, count: capacity)
        self.free = identity ? [] : (0 ..< capacity).reversed().map { Int16($0) }
    }

    deinit { scratch?.deallocate() }

    var slotBytes: Int { destinations.reduce(0) { $0 + $1.array.nbytes } }

    /// Every expert is resident and the ids are the slot ids: the call needs no host sync.
    var fullyResident: Bool { identity && loaded == experts }

    func isResident(_ expert: UInt32) -> Bool { slotOf[Int(expert)] >= 0 }

    /// Finds each slot tensor's bytes once (allocating them), and checks that a write through that
    /// pointer is what the GPU reads. Called at load; `ensure` binds lazily if it was not.
    func bind() throws {
        guard !bound else { return }
        let workers = max(1, runtime.options.readers)
        let gapBytes = gaps.reduce(0) { $0 + $1.length }
        if gapBytes > 0 {
            scratchBytes = gapBytes
            scratch = UnsafeMutableRawPointer.allocate(
                byteCount: gapBytes * workers, alignment: 64)
        }
        for index in destinations.indices {
            let array = destinations[index].array
            eval(array)
            let view = array.asData(access: .noCopy)
            var expected = [Int](repeating: 1, count: array.ndim)
            for axis in stride(from: array.ndim - 2, through: 0, by: -1) {
                expected[axis] = expected[axis + 1] * array.shape[axis + 1]
            }
            guard view.data.count == array.nbytes, view.strides == expected else {
                throw Inkling2ExpertStreamingError.binding(
                    "\(destinations[index].name) of layer \(layer) is not one contiguous buffer")
            }
            let base = view.data.withUnsafeBytes { UnsafeMutableRawPointer(mutating: $0.baseAddress) }
            guard let base else {
                throw Inkling2ExpertStreamingError.binding("no storage for layer \(layer)")
            }
            try probe(array, base: base, name: destinations[index].name)
            destinations[index].base = base
        }
        bound = true
    }

    /// Writes a known value into the first element of slot 0, reads it back through a GPU op and
    /// restores zeros.
    private func probe(_ array: MLXArray, base: UnsafeMutableRawPointer, name: String) throws {
        let item = array.itemSize
        let pattern: [UInt8]
        switch array.dtype {
        case .bfloat16: pattern = [0x80, 0x3F]  // 1.0
        case .float16: pattern = [0x00, 0x3C]  // 1.0
        case .float32: pattern = [0x00, 0x00, 0x80, 0x3F]  // 1.0
        default:
            let bytes: [UInt8] = [0x5A, 0xC3, 0x0F, 0x96, 0x3C, 0xA5, 0x69, 0x11]
            pattern = Array(bytes.prefix(item))
        }
        pattern.withUnsafeBytes { base.copyMemory(from: $0.baseAddress!, byteCount: item) }
        let readBack = (array.reshaped(-1)[0 ..< 1] * 1).asData(access: .copy).data
        _ = memset(base, 0, item)
        guard Array(readBack) == pattern else {
            throw Inkling2ExpertStreamingError.binding(
                "a write into \(name) of layer \(layer) is not visible to MLX; "
                    + "streaming needs mlx-swift's shared-storage buffers")
        }
    }

    /// Makes every expert of `ids` resident and returns the slot of each id, in order. The distinct
    /// experts of `ids` must fit in `capacity`.
    func ensure(_ ids: [UInt32]) -> [UInt32] {
        if !bound {
            do { try bind() } catch { fatalError("Inkling expert streaming: \(error)") }
        }
        call += 1
        var needed = [Int16]()
        needed.reserveCapacity(min(ids.count, capacity))
        var assigned = [(expert: Int, slot: Int)]()
        // Hits first: they must be protected before any miss picks a victim.
        for id in ids {
            let e = Int(id)
            precondition(e < experts, "expert id \(e) out of range")
            let slot = slotOf[e]
            if slot >= 0, stamp[Int(slot)] != call {
                stamp[Int(slot)] = call
                needed.append(slot)
            }
        }
        for id in ids where slotOf[Int(id)] < 0 {
            let e = Int(id)
            let slot: Int
            if identity {
                slot = e
            } else if let f = free.popLast() {
                slot = Int(f)
            } else {
                slot = victim()  // never a slot stamped by this call
                slotOf[Int(expertOf[slot])] = -1
                loaded -= 1
            }
            expertOf[slot] = Int16(e)
            slotOf[e] = Int16(slot)
            stamp[slot] = call
            score[slot] = 0
            needed.append(Int16(slot))
            assigned.append((e, slot))
            loaded += 1
        }

        if !assigned.isEmpty {
            let syncStart = DispatchTime.now().uptimeNanoseconds
            MLX.Stream.gpu.synchronize()
            let sync = inkling2Seconds(since: syncStart)
            let readStart = DispatchTime.now().uptimeNanoseconds
            let failure = read(assigned)
            if let failure {
                fatalError("Inkling expert streaming, layer \(layer): \(failure)")
            }
            runtime.record(
                bytes: assigned.count * readSpan, read: inkling2Seconds(since: readStart),
                gpuSync: sync)
        }
        for slot in needed {
            score[Int(slot)] += 1
            lastUse[Int(slot)] = call
        }
        if call % 4 == 0 {
            for i in score.indices { score[i] *= 0.7 }
        }
        runtime.record(lookups: needed.count, misses: assigned.count)
        return ids.map { UInt32(slotOf[Int($0)]) }
    }

    /// The occupied, unprotected slot with the lowest score, then the least recently used.
    private func victim() -> Int {
        var best = -1
        for s in 0 ..< capacity where stamp[s] != call && expertOf[s] >= 0 {
            if best < 0 || score[s] < score[best]
                || (score[s] == score[best] && lastUse[s] < lastUse[best])
            {
                best = s
            }
        }
        precondition(best >= 0, "no evictable slot")
        return best
    }

    /// Reads each expert's record into its slot, `readers` at a time. Returns the first error.
    private func read(_ assigned: [(expert: Int, slot: Int)]) -> Error? {
        guard let file = runtime.file else { return Inkling2ExpertStreamingError.io("no file") }
        let workers = min(max(1, runtime.options.readers), assigned.count)
        // The vectors of every read, built here so the reader threads share only plain values.
        var reads = [Inkling2ExpertRead]()
        reads.reserveCapacity(assigned.count)
        for (i, (expert, slot)) in assigned.enumerated() {
            let worker = i % workers
            var pieces = [(offset: Int, iov: iovec)]()
            pieces.reserveCapacity(destinations.count + gaps.count)
            for d in destinations {
                pieces.append(
                    (d.offsetInRecord,
                     iovec(iov_base: d.base! + slot * d.bytesPerSlot, iov_len: d.bytesPerSlot)))
            }
            var gapCursor = 0
            for g in gaps {
                pieces.append(
                    (g.offset,
                     iovec(
                        iov_base: scratch! + worker * scratchBytes + gapCursor, iov_len: g.length)
                    ))
                gapCursor += g.length
            }
            pieces.sort { $0.offset < $1.offset }
            reads.append(
                Inkling2ExpertRead(
                    vectors: pieces.map { $0.iov }, offset: spec.baseOffset + expert * spec.recordStride,
                    total: readSpan))
        }
        let batch = Inkling2ExpertReadBatch(file: file, reads: reads, workers: workers)
        if workers == 1 {
            batch.run(worker: 0)
        } else {
            DispatchQueue.concurrentPerform(iterations: workers) { batch.run(worker: $0) }
        }
        return batch.errors.value
    }

    /// Positions of `ids` (flattened `[rows * K]`) in chunks of at most `capacity` distinct
    /// experts: resident experts first so a miss never evicts one this call still needs, then by
    /// expert so each expert's rows are contiguous.
    func chunks(_ ids: [UInt32]) -> [[Int]] {
        let order = ids.indices.sorted { a, b in
            let (ra, rb) = (isResident(ids[a]), isResident(ids[b]))
            if ra != rb { return ra }
            if ids[a] != ids[b] { return ids[a] < ids[b] }
            return a < b
        }
        var result = [[Int]]()
        var current = [Int]()
        var distinct = 0
        var previous: UInt32?
        for p in order {
            if ids[p] != previous {
                if distinct == capacity {
                    result.append(current)
                    current = []
                    distinct = 0
                }
                distinct += 1
                previous = ids[p]
            }
            current.append(p)
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

/// One record read: file range and where its bytes go (slot memory, or scratch for padding).
private struct Inkling2ExpertRead: @unchecked Sendable {
    let vectors: [iovec]
    let offset: Int
    let total: Int
}

/// The reads of one call; worker `w` does reads `w, w + workers, ...`. The destinations of
/// different reads never overlap (distinct slots; scratch per worker).
private final class Inkling2ExpertReadBatch: @unchecked Sendable {
    let file: Inkling2ExpertFile
    let reads: [Inkling2ExpertRead]
    let workers: Int
    let errors = Inkling2ErrorBox()

    init(file: Inkling2ExpertFile, reads: [Inkling2ExpertRead], workers: Int) {
        self.file = file
        self.reads = reads
        self.workers = workers
    }

    func run(worker: Int) {
        var i = worker
        while i < reads.count {
            let r = reads[i]
            do { try file.read(r.vectors, offset: r.offset, total: r.total) } catch {
                errors.set(error)
            }
            i += workers
        }
    }
}

private final class Inkling2ErrorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var first: Error?
    func set(_ error: Error) { lock.withLock { if first == nil { first = error } } }
    var value: Error? { lock.withLock { first } }
}

// MARK: - Per-layer hook

/// How an MoE call runs against the slots.
enum Inkling2RoutedExperts {
    /// Indices into the expert tensors (slot ids, or the original ids when they coincide).
    case direct(MLXArray)
    /// More distinct experts than slots: run in chunks over these host ids.
    case chunked([UInt32])
}

/// What an `Inkling2Experts` layer needs from the runtime: the trace and, when streaming, its
/// slots.
final class Inkling2ExpertStreamLayer {
    let layer: Int
    let runtime: Inkling2ExpertStreamRuntime
    let cache: Inkling2ExpertSlotCache?

    init(layer: Int, runtime: Inkling2ExpertStreamRuntime, cache: Inkling2ExpertSlotCache?) {
        self.layer = layer
        self.runtime = runtime
        self.cache = cache
    }

    /// Resolves the routing indices `[..., K]` of one call. Syncs with the GPU to read them unless
    /// every expert is already resident at its own id and nothing is traced.
    func route(_ indices: MLXArray) -> Inkling2RoutedExperts {
        if let cache, cache.fullyResident, !runtime.tracing {
            runtime.record(lookups: 0, call: true)
            return .direct(indices)
        }
        let start = DispatchTime.now().uptimeNanoseconds
        let host = indices.asType(.uint32).asArray(UInt32.self)
        runtime.record(routeWait: inkling2Seconds(since: start), call: true)
        let rows = indices.ndim > 0 && indices.dim(-1) > 0 ? host.count / indices.dim(-1) : 0
        runtime.traceCall(layer: layer, rows: rows, ids: host)
        guard let cache else { return .direct(indices) }
        if Set(host).count > cache.capacity {
            runtime.record(chunked: true)
            return .chunked(host)
        }
        let slots = cache.ensure(host)
        if cache.identity { return .direct(indices) }
        return .direct(MLXArray(slots, indices.shape))
    }
}
