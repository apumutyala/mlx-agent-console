import Foundation

/// A source of extra numbers for a benchmark receipt (e.g. Inkling expert-streaming hit rate and
/// IO). The on-device benchmark code (branch `bench`, Bench/*) can read every registered source
/// with `BenchMetricsRegistry.collect()` before and after a run, without either side importing the
/// other. Keys are flat and unit-suffixed (`hit_rate`, `bytes_read`, `read_s`).
protocol BenchMetricsSource: AnyObject {
    /// Prefix for this source's keys in `collect()`, e.g. `inkling_stream`.
    var benchMetricsName: String { get }
    /// Current cumulative values. Empty when there is nothing to report (e.g. no model loaded).
    func benchMetrics() -> [String: Double]
    /// Zero the cumulative counters, e.g. at the start of a measured phase.
    func resetBenchMetrics()
}

enum BenchMetricsRegistry {
    private final class Storage: @unchecked Sendable {
        let lock = NSLock()
        var sources: [BenchMetricsSource] = []
    }
    private static let storage = Storage()

    static func register(_ source: BenchMetricsSource) {
        storage.lock.withLock {
            if !storage.sources.contains(where: { $0 === source }) { storage.sources.append(source) }
        }
    }

    static var sources: [BenchMetricsSource] { storage.lock.withLock { storage.sources } }

    /// Every source's metrics as `"<name>.<key>"`.
    static func collect() -> [String: Double] {
        var out = [String: Double]()
        for source in sources {
            for (key, value) in source.benchMetrics() { out["\(source.benchMetricsName).\(key)"] = value }
        }
        return out
    }

    static func resetAll() { sources.forEach { $0.resetBenchMetrics() } }
}

/// Settings for the Inkling routing trace (design stage M1). No MLX import.
enum StreamingDiagnostics {
    /// When on, the next Inkling load (resident or streamed) writes one JSON line per MoE call.
    static let traceKey = "inkling.routingTrace"
    static var traceEnabled: Bool { UserDefaults.standard.bool(forKey: traceKey) }

    /// When on, new chats sample greedily (temperature 0), for the correctness gate.
    static let greedyKey = "inference.greedy"
    static var greedy: Bool { UserDefaults.standard.bool(forKey: greedyKey) }

    /// Application Support/Diagnostics/inkling-routing.jsonl (appended across loads).
    static var traceURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("inkling-routing.jsonl")
    }
}
