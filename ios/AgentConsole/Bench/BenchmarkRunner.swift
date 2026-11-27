import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import UIKit

/// Runs the versioned workload set (BenchWorkloads) against one catalog entry
/// and produces a schema-v1 receipt.
///
/// The caller holds `GPUGate` for the whole run. The runner unloads the chat
/// model first so only the measured model is resident, loads the entry twice
/// (first load, then a warm reload), warms up, and runs every workload N times.
///
/// Timing: the token loop drives mlx-swift-lm's `TokenIterator` directly (the
/// same iterator `generate` uses) so each rep decodes exactly `maxTokens` even
/// if a stop token appears, and records both wall-clock points and the
/// library's own `promptPrefillTime`, computed the same way as
/// `GenerateCompletionInfo.promptTime` / `.tokensPerSecond`.
struct BenchmarkRunner: Sendable {
    let backend: MLXBackend

    static let seed: UInt64 = 1234
    static let warmupTokens = 8
    static let hashTokens = 64
    static let defaultRepetitions = 3

    /// Skip models whose weights plus this much KV/activation headroom don't fit
    /// in what the process may still allocate.
    static let headroomBytes: Double = 1.5 * 1_073_741_824

    struct Progress: Sendable {
        var message: String
        var fraction: Double
    }

    /// Run every workload in `workloadIDs` (nil = all) `repetitions` times.
    /// Never throws: failures, skips and cancellation are recorded in the receipt.
    func run(entry: ModelCatalog.Entry, workloadIDs: [String]? = nil,
             repetitions: Int = BenchmarkRunner.defaultRepetitions,
             progress: @escaping @Sendable (Progress) -> Void = { _ in },
             isCancelled: @escaping @Sendable () -> Bool = { false }) async -> BenchReceipt {
        let wallStart = Date()
        let reps = min(max(repetitions, 1), 10)
        let envBefore = await Self.environmentNow()
        var receipt = await Self.skeleton(entry: entry, reps: reps, env: envBefore)
        func finish(_ status: String, error: String? = nil, skip: String? = nil) async -> BenchReceipt {
            let after = await Self.environmentNow()
            receipt.status = status
            receipt.error = error
            receipt.skipReason = skip
            receipt.environment.thermalAfter = after.thermal
            receipt.environment.batteryLevelAfter = after.batteryLevel
            receipt.environment.batteryStateAfter = after.batteryState
            receipt.environment.availableMemoryAfterBytes = after.available
            receipt.totalWallS = Date().timeIntervalSince(wallStart)
            return receipt
        }

        let workloads: [BenchWorkloads.Workload]
        do {
            workloads = try BenchWorkloads.workloads(ids: workloadIDs)
        } catch {
            return await finish("failed", error: error.localizedDescription)
        }
        guard entry.isSupportedOnThisDevice else {
            return await finish("skipped", skip: String(
                format: "catalog minimum is %.0f GB RAM; device has %.1f GB",
                entry.minimumMemoryGB, ModelCatalog.deviceMemoryGB))
        }

        // Only the measured model may be resident.
        progress(.init(message: "Unloading chat model", fraction: 0))
        backend.unload()
        MLX.Memory.clearCache()
        let baseline = ResourceProbe.physFootprint()
        let available = ResourceProbe.availableMemory()
        let need = entry.downloadGB * 1_000_000_000 + Self.headroomBytes
        if need > Double(available) {
            return await finish("skipped", skip: String(
                format: "needs ~%.1f GB (weights %.2f GB + %.1f GB headroom); %.1f GB available to the app",
                need / 1e9, entry.downloadGB, Self.headroomBytes / 1e9, Double(available) / 1e9))
        }

        do {
            // 1. First load (may download), then a warm reload.
            progress(.init(message: "Loading \(entry.displayName)", fraction: 0.02))
            let sawPartial = PartialFlag()
            let t0 = Date()
            var container: ModelContainer? = try await MLXBackend.loadContainer(for: entry) { f in
                if f > 0 && f < 1 { sawPartial.set() }
                progress(.init(message: "Downloading \(entry.displayName) \(Int(f * 100))%", fraction: 0.02))
            }
            let firstLoad = Date().timeIntervalSince(t0)
            if let c = container { receipt.model = await Self.describe(entry: entry, container: c) }
            container = nil
            MLX.Memory.clearCache()
            if isCancelled() { return await finish("cancelled") }

            progress(.init(message: "Warm reload", fraction: 0.05))
            let t1 = Date()
            let loaded = try await MLXBackend.loadContainer(for: entry) { _ in }
            let warmLoad = Date().timeIntervalSince(t1)
            let footprintLoaded = ResourceProbe.physFootprint()
            let availableLoaded = ResourceProbe.availableMemory()

            // 2. Warm-up (kernel/pipeline compilation) — not recorded as a rep.
            progress(.init(message: "Warm-up", fraction: 0.07))
            let t2 = Date()
            let warm = await Self.promptTokens(BenchWorkloads.short, container: loaded)
            _ = try await Self.decode(container: loaded, prompt: warm, maxTokens: Self.warmupTokens)
            receipt.load = .init(
                firstLoadS: firstLoad, downloadObserved: sawPartial.value, warmLoadS: warmLoad,
                baselineFootprintBytes: baseline, footprintAfterLoadBytes: footprintLoaded,
                availableAfterLoadBytes: availableLoaded, warmupS: Date().timeIntervalSince(t2))

            // 3. Workloads × reps.
            var maxThermal = ResourceProbe.thermal()
            let total = Double(workloads.count * reps)
            var done = 0.0
            for w in workloads {
                let prompt = await Self.promptTokens(w, container: loaded)
                var repResults: [BenchReceipt.Rep] = []
                var firstIDs: [Int] = []
                var stopIndex: Int?
                for i in 0..<reps {
                    if isCancelled() || Task.isCancelled { return await finish("cancelled") }
                    progress(.init(message: "\(entry.displayName) · \(w.id) · rep \(i + 1)/\(reps)",
                                   fraction: 0.1 + 0.9 * done / total))
                    MLX.Memory.clearCache()
                    MLX.Memory.peakMemory = 0   // the setter resets MLX's peak counter
                    let thermalBefore = ResourceProbe.thermal()
                    let sampler = ResourceSampler()
                    sampler.start()
                    let raw = try await Self.decode(container: loaded, prompt: prompt, maxTokens: w.maxTokens)
                    let mem = sampler.stop()
                    let snap = MLX.Memory.snapshot()
                    let thermalAfter = ResourceProbe.thermal()
                    if mem.maxThermal.rawValue > maxThermal.rawValue { maxThermal = mem.maxThermal }

                    let ids = Array(raw.tokens.prefix(Self.hashTokens))
                    if i == 0 {
                        firstIDs = ids
                        stopIndex = raw.tokens.firstIndex { raw.stopTokens.contains($0) }
                    }
                    let n = raw.tokens.count
                    let ttft = raw.firstTokenAt - raw.start
                    let decodeS = raw.end - raw.firstTokenAt
                    repResults.append(.init(
                        index: i, ttftS: ttft,
                        prefillTps: ttft > 0 ? Double(prompt.count) / ttft : 0,
                        decodeTps: decodeS > 0 && n > 1 ? Double(n - 1) / decodeS : 0,
                        decodeS: decodeS, generatedTokens: n, wallS: raw.end - raw.start,
                        libPrefillS: raw.libPrefill,
                        libPromptTimeS: raw.libPrefill + (raw.firstTokenAt - raw.iteratorReady),
                        libTokensPerSecond: decodeS > 0 ? Double(n) / decodeS : 0,
                        peakFootprintBytes: mem.peakFootprint, minAvailableMemoryBytes: mem.minAvailable,
                        footprintSamples: mem.samples, mlxPeakBytes: snap.peakMemory,
                        mlxActiveBytes: snap.activeMemory, mlxCacheBytes: snap.cacheMemory,
                        thermalBefore: ResourceProbe.name(thermalBefore),
                        thermalAfter: ResourceProbe.name(thermalAfter),
                        outputHash: Self.hash(ids)))
                    done += 1
                }
                let preview = await loaded.decode(tokenIds: firstIDs)
                receipt.workloads.append(.init(
                    id: w.id, version: w.version, description: w.description, promptTokens: prompt.count,
                    maxTokens: w.maxTokens, reps: repResults,
                    median: .reduce(repResults, BenchReceipt.median),
                    min: .reduce(repResults) { $0.min() ?? 0 },
                    max: .reduce(repResults) { $0.max() ?? 0 },
                    outputHash: Self.hash(firstIDs), outputTokenIds: firstIDs,
                    outputPreview: String(preview.prefix(300)),
                    deterministic: Set(repResults.map(\.outputHash)).count == 1,
                    stopTokenIndex: stopIndex))
            }
            receipt.environment.thermalMaxDuring = ResourceProbe.name(maxThermal)
            progress(.init(message: "Done", fraction: 1))
            return await finish("ok")
        } catch {
            if isCancelled() || error is CancellationError { return await finish("cancelled") }
            return await finish("failed", error: error.localizedDescription)
        }
    }

    // MARK: Token loop

    private struct RawRun: Sendable {
        var tokens: [Int]
        var stopTokens: Set<Int>
        var start: TimeInterval          // before TokenIterator init (prefill starts)
        var iteratorReady: TimeInterval  // TokenIterator init returned
        var firstTokenAt: TimeInterval   // first next() returned
        var end: TimeInterval            // last token returned
        var libPrefill: TimeInterval     // TokenIterator.promptPrefillTime
    }

    /// Greedy decode of exactly `maxTokens` tokens from raw prompt ids.
    private static func decode(container: ModelContainer, prompt: [Int], maxTokens: Int) async throws -> RawRun {
        try await container.perform { (ctx: ModelContext) -> RawRun in
            let clock = { Date.timeIntervalSinceReferenceDate }
            let input = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
            var params = GenerateParameters()
            params.maxTokens = maxTokens
            params.temperature = 0          // greedy (argmax sampler)
            params.seed = BenchmarkRunner.seed
            let start = clock()
            var iterator = try TokenIterator(input: input, model: ctx.model, parameters: params)
            let ready = clock()
            var tokens: [Int] = []
            tokens.reserveCapacity(maxTokens)
            var first = ready
            while let t = iterator.next() {
                if tokens.isEmpty { first = clock() }
                tokens.append(t)
                if Task.isCancelled { break }
            }
            let end = clock()
            // next() queued one more step; let it drain outside the timed region.
            MLX.Stream.defaultStream.synchronize()

            var stops = ctx.configuration.eosTokenIds
            if let eos = ctx.tokenizer.eosTokenId { stops.insert(eos) }
            for s in ctx.configuration.extraEOSTokens {
                if let id = ctx.tokenizer.convertTokenToId(s) { stops.insert(id) }
            }
            return RawRun(tokens: tokens, stopTokens: stops, start: start, iteratorReady: ready,
                          firstTokenAt: first, end: end, libPrefill: iterator.promptPrefillTime)
        }
    }

    private static func promptTokens(_ w: BenchWorkloads.Workload, container: ModelContainer) async -> [Int] {
        await container.perform { (ctx: ModelContext) -> [Int] in
            BenchWorkloads.promptTokens(w) { text, special in
                ctx.tokenizer.encode(text: text, addSpecialTokens: special)
            }
        }
    }

    static func hash(_ ids: [Int]) -> String {
        let digest = SHA256.hash(data: Data(ids.map(String.init).joined(separator: ",").utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Receipt parts

    private struct EnvNow: Sendable {
        var thermal: String
        var batteryLevel: Double?
        var batteryState: String
        var lowPower: Bool
        var available: UInt64
    }

    @MainActor
    private static func environmentNow() -> EnvNow {
        let d = UIDevice.current
        d.isBatteryMonitoringEnabled = true
        let state: String = switch d.batteryState {
        case .charging: "charging"
        case .full: "full"
        case .unplugged: "unplugged"
        case .unknown: "unknown"
        @unknown default: "unknown"
        }
        return EnvNow(thermal: ResourceProbe.name(ResourceProbe.thermal()),
                      batteryLevel: d.batteryLevel < 0 ? nil : Double(d.batteryLevel),
                      batteryState: state, lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled,
                      available: ResourceProbe.availableMemory())
    }

    @MainActor
    private static func skeleton(entry: ModelCatalog.Entry, reps: Int, env: EnvNow) -> BenchReceipt {
        let d = UIDevice.current
        return BenchReceipt(
            receiptId: UUID().uuidString.lowercased(),
            createdAt: ISO8601DateFormatter().string(from: Date()),
            status: "running", appGitSha: BuildInfo.gitSHA, appVersion: BuildInfo.appVersion,
            library: BuildInfo.library,
            device: .init(machine: BuildInfo.machine, model: d.model, os: "\(d.systemName) \(d.systemVersion)",
                          physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                          processorCount: ProcessInfo.processInfo.processorCount),
            model: .init(id: entry.id, displayName: entry.displayName, repo: entry.repo,
                         catalogDownloadGb: entry.downloadGB),
            workloadSet: BenchWorkloads.setVersion,
            params: .init(temperature: 0, seed: seed, repetitions: reps, warmupTokens: warmupTokens,
                          mlxCacheLimitBytes: MLXBackend.mlxCacheLimit,
                          memorySampleIntervalMs: ResourceSampler.intervalMs,
                          promptMode: "raw token ids, no chat template; filled prompts cut to exact length",
                          stopTokens: "ignored: every rep decodes max_tokens (first stop index recorded)",
                          timing: "wall clock around mlx-swift-lm TokenIterator; ttft = init + first next(); "
                              + "decode_tps = (n-1)/(t_last - t_first)"),
            load: nil,
            environment: .init(thermalBefore: env.thermal, thermalAfter: env.thermal, thermalMaxDuring: env.thermal,
                               batteryLevelBefore: env.batteryLevel, batteryLevelAfter: env.batteryLevel,
                               batteryStateBefore: env.batteryState, batteryStateAfter: env.batteryState,
                               lowPowerMode: env.lowPower, availableMemoryBeforeBytes: env.available,
                               availableMemoryAfterBytes: env.available),
            workloads: [], totalWallS: 0)
    }

    /// Model facts from the loaded container and its config.json.
    private static func describe(entry: ModelCatalog.Entry, container: ModelContainer) async -> BenchReceipt.Model {
        var m = BenchReceipt.Model(id: entry.id, displayName: entry.displayName, repo: entry.repo,
                                   catalogDownloadGb: entry.downloadGB)
        m.weightsBytes = (try? await container.memory())?.model
        guard let dir = try? await container.modelDirectory else { return m }
        if let data = try? Data(contentsOf: dir.appending(path: "config.json")),
           let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            m.modelType = config["model_type"] as? String
                ?? (config["text_config"] as? [String: Any])?["model_type"] as? String
            if let q = (config["quantization"] ?? config["quantization_config"]) as? [String: Any] {
                m.quantization = .init(bits: q["bits"] as? Int, groupSize: q["group_size"] as? Int,
                                       mode: q["mode"] as? String,
                                       perLayerOverrides: q.values.filter { $0 is [String: Any] }.count)
            }
        }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let sizes = files.filter { $0.pathExtension == "safetensors" }
            .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
        if !sizes.isEmpty { m.weightFilesBytes = Int64(sizes.reduce(0, +)) }
        return m
    }
}

/// Set from the download progress callback (any thread), read after the load.
private final class PartialFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.withLock { flag = true } }
    var value: Bool { lock.withLock { flag } }
}
