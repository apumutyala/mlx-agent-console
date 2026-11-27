import Foundation

/// Benchmark receipt, schema v1. One JSON document per (device, model) run.
/// Encoded with snake_case keys (`BenchReceipt.encoder`); the worker stores it
/// verbatim (worker/receipts.py) and Claude reads it through `list_receipts`.
///
/// Property names avoid acronyms (`decodeTps`, not `decodeTPS`) so the
/// snake_case conversion round-trips. Bump `schema` on any breaking change.
struct BenchReceipt: Codable, Sendable, Identifiable {
    var schema = 1
    var receiptId: String
    var createdAt: String                // ISO 8601, UTC
    var status: String                   // ok | skipped | failed | cancelled
    var skipReason: String?
    var error: String?
    var appGitSha: String
    var appVersion: String
    var library: Library
    var device: Device
    var model: Model
    var workloadSet: String
    var params: Params
    var load: Load?
    var environment: Environment
    var workloads: [WorkloadResult]
    var totalWallS: Double

    var id: String { receiptId }

    struct Library: Codable, Sendable {
        var mlxSwiftLm: String
        var mlxSwift: String
    }

    struct Device: Codable, Sendable {
        var machine: String              // utsname.machine, e.g. "iPad17,3"
        var model: String                // "iPad"
        var os: String                   // "iPadOS 27.2.1"
        var physicalMemoryBytes: UInt64
        var processorCount: Int
    }

    struct Model: Codable, Sendable {
        var id: String                   // ModelCatalog id
        var displayName: String
        var repo: String
        var catalogDownloadGb: Double
        var modelType: String?           // config.json model_type
        var quantization: Quantization?  // config.json "quantization"
        var weightsBytes: Int?           // ModelMemory.model (resident weights)
        var weightFilesBytes: Int64?     // *.safetensors on disk
    }

    struct Quantization: Codable, Sendable {
        var bits: Int?
        var groupSize: Int?
        var mode: String?
        var perLayerOverrides: Int       // mixed-bit checkpoints list per-layer entries
    }

    struct Params: Codable, Sendable {
        var temperature: Double
        var seed: UInt64
        var repetitions: Int
        var warmupTokens: Int
        var mlxCacheLimitBytes: Int
        var memorySampleIntervalMs: Int
        var promptMode: String           // raw token ids, no chat template
        var stopTokens: String           // ignored: always decodes max_tokens
        var timing: String
    }

    struct Load: Codable, Sendable {
        var firstLoadS: Double
        /// The first load reported partial download progress, so it likely includes a download.
        var downloadObserved: Bool
        /// Second load of the same weights right after unloading (OS file cache warm).
        var warmLoadS: Double
        var baselineFootprintBytes: UInt64
        var footprintAfterLoadBytes: UInt64
        var availableAfterLoadBytes: UInt64
        var warmupS: Double
    }

    struct Environment: Codable, Sendable {
        var thermalBefore: String
        var thermalAfter: String
        var thermalMaxDuring: String
        var batteryLevelBefore: Double?   // 0...1, nil when unknown
        var batteryLevelAfter: Double?
        var batteryStateBefore: String
        var batteryStateAfter: String
        var lowPowerMode: Bool
        var availableMemoryBeforeBytes: UInt64
        var availableMemoryAfterBytes: UInt64
    }

    struct WorkloadResult: Codable, Sendable {
        var id: String
        var version: Int
        var description: String
        var promptTokens: Int
        var maxTokens: Int
        var reps: [Rep]
        var median: Stats
        var min: Stats
        var max: Stats
        /// SHA-256 (hex) of the first 64 greedy token ids, comma-joined ("12,345,...").
        var outputHash: String
        var outputTokenIds: [Int]
        var outputPreview: String
        /// Every rep produced the same hash.
        var deterministic: Bool
        /// Index of the first stop token in the generated ids, if one appeared
        /// (decoding continues past it so every rep decodes `maxTokens`).
        var stopTokenIndex: Int?
    }

    struct Rep: Codable, Sendable {
        var index: Int
        /// Prompt submitted → first generated token available (prefill + first step).
        var ttftS: Double
        var prefillTps: Double           // prompt tokens / ttft
        /// Tokens after the first / time after the first token.
        var decodeTps: Double
        var decodeS: Double
        var generatedTokens: Int
        var wallS: Double
        /// mlx-swift-lm's own clocks, as GenerateCompletionInfo computes them:
        /// promptTime = TokenIterator.promptPrefillTime + first next(); tokensPerSecond = N / generateTime.
        var libPrefillS: Double
        var libPromptTimeS: Double
        var libTokensPerSecond: Double
        var peakFootprintBytes: UInt64   // max task_vm_info.phys_footprint while running
        var minAvailableMemoryBytes: UInt64
        var footprintSamples: Int
        var mlxPeakBytes: Int
        var mlxActiveBytes: Int
        var mlxCacheBytes: Int
        var thermalBefore: String
        var thermalAfter: String
        var outputHash: String
    }

    struct Stats: Codable, Sendable {
        var ttftS: Double
        var prefillTps: Double
        var decodeTps: Double
        var wallS: Double
        var peakFootprintBytes: UInt64
        var mlxPeakBytes: Int

        /// Element-wise median / min / max over reps.
        static func reduce(_ reps: [Rep], _ pick: @escaping ([Double]) -> Double) -> Stats {
            func p(_ f: (Rep) -> Double) -> Double { pick(reps.map(f)) }
            return Stats(ttftS: p(\.ttftS), prefillTps: p(\.prefillTps), decodeTps: p(\.decodeTps),
                         wallS: p(\.wallS), peakFootprintBytes: UInt64(p { Double($0.peakFootprintBytes) }),
                         mlxPeakBytes: Int(p { Double($0.mlxPeakBytes) }))
        }
    }

    static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    static var encoder: JSONEncoder {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }

    static var decoder: JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }

    /// Model and workload summary for list rows: "short-128 52.1 tok/s · 4k-256 41.0 tok/s".
    var headline: String {
        if status != "ok" { return status + (skipReason.map { ": \($0)" } ?? error.map { ": \($0)" } ?? "") }
        return workloads.map { "\($0.id) \(String(format: "%.1f", $0.median.decodeTps)) tok/s" }
            .joined(separator: " · ")
    }
}

/// Build-time and device facts shared by every receipt.
enum BuildInfo {
    /// Injected by CI: `APP_GIT_SHA` build setting → Info.plist `AppGitSHA`.
    static var gitSHA: String {
        let s = Bundle.main.object(forInfoDictionaryKey: "AppGitSHA") as? String ?? ""
        return s.isEmpty || s.hasPrefix("$(") ? "unknown" : s
    }

    static var appVersion: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let v = info["CFBundleShortVersionString"] as? String ?? "?"
        let b = info["CFBundleVersion"] as? String ?? "?"
        return "\(v) (\(b))"
    }

    /// Pinned in ios/project.yml; keep in sync when bumping packages.
    static let library = BenchReceipt.Library(mlxSwiftLm: "3.33.0", mlxSwift: "0.33.x")

    /// utsname machine, e.g. "iPad17,3" (the simulator reports the host's arch).
    static var machine: String {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
}
