import CryptoKit
import Foundation

/// On-device engine A/B, MLX vs Xybrid's llama.cpp (docs/ENGINES-RESEARCH-2026-11-27.md §2,
/// protocol A1–A7). Same text prompt (workload `ab-256`), greedy, 256 output tokens, 5 reps,
/// median. Every step writes a standard schema-1 receipt with `engine` and `ab_step`.
///
/// Step order is chosen so the likely kills come last: A6 (4.84 GB mmap canary) and then A5
/// (MLX 4-bit 8B, expected to be refused or killed). A7 needs residency disabled at launch,
/// so it runs in its own launch (see `XybridLLMBackend.applyLaunchEnvironment`).
enum EngineAB {
    struct Step: Identifiable, Hashable, Sendable {
        let id: String           // "A1-mlx", "A2", …
        let title: String
        let purpose: String
        let entry: ModelCatalog.Entry?   // nil = not runnable yet (A4)
        var force = false
        var noResidency = false

        var engine: ModelCatalog.Engine { entry?.engine ?? .mlx }
    }

    static let repetitions = 5
    static let workloadID = BenchWorkloads.engineAB.id

    static let steps: [Step] = [
        Step(id: "A1-mlx", title: "A1 · MLX LFM2.5 1.2B 4-bit", purpose: "engine speed, MLX side",
             entry: ModelCatalog.lfm25_1_2b),
        Step(id: "A1-xybrid", title: "A1 · Xybrid LFM2.5 1.2B Q4_K_M", purpose: "engine speed, Xybrid side",
             entry: ModelCatalog.xyLFM25_1_2b),
        Step(id: "A2", title: "A2 · Xybrid 8B-A1B UD-Q3_K_XL (4.00 GB)", purpose: "fit and speed",
             entry: ModelCatalog.xyLFM25_8b_Q3KXL),
        Step(id: "A3", title: "A3 · Xybrid 8B-A1B UD-IQ4_XS (4.26 GB)", purpose: "quality/speed point",
             entry: ModelCatalog.xyLFM25_8b_IQ4XS),
        Step(id: "A3-iq3xxs", title: "A3b · Xybrid 8B-A1B UD-IQ3_XXS (3.27 GB)", purpose: "smallest 8B point",
             entry: ModelCatalog.xyLFM25_8b_IQ3XXS),
        Step(id: "A4", title: "A4 · MLX 8B-A1B 3-bit", purpose: "like-for-like vs A2 (needs a conversion, T4)",
             entry: nil),
        Step(id: "A6", title: "A6 · Xybrid 8B-A1B Q4_0 (4.84 GB)", purpose: "mmap canary: is a mapped file charged?",
             entry: ModelCatalog.xyLFM25_8b_Q4_0),
        Step(id: "A7", title: "A7 · A6 with Metal residency off", purpose: "experimental: GGML_METAL_NO_RESIDENCY=1",
             entry: ModelCatalog.xyLFM25_8b_Q4_0, noResidency: true),
        Step(id: "A5", title: "A5 · MLX 8B-A1B 4-bit (4.85 GB), force try", purpose: "expected refused/killed: the limit",
             entry: ModelCatalog.lfm25_8b_a1b, force: true),
    ]

    static func hash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Xybrid step

    /// Runs one Xybrid step. Never throws: failures, skips and cancellation are receipts.
    /// The caller holds `GPUGate` and has made MLX drop its weights (`beginBenchmark`).
    static func runXybrid(_ step: Step, backend: XybridLLMBackend,
                          progress: @escaping @Sendable (BenchmarkRunner.Progress) -> Void,
                          isCancelled: @escaping @Sendable () -> Bool) async -> BenchReceipt {
        let entry = step.entry ?? ModelCatalog.xyLFM25_1_2b
        let wallStart = Date()
        let reps = repetitions
        let envBefore = await BenchmarkRunner.environmentNow()
        var receipt = await BenchmarkRunner.skeleton(entry: entry, reps: reps, env: envBefore)
        receipt.engine = "xybrid"
        receipt.abStep = step.id
        receipt.identity = .current
        receipt.params.mlxCacheLimitBytes = 0
        receipt.params.promptMode = "text as one user message through the GGUF chat template (Xybrid has no raw-token API)"
        receipt.params.stopTokens = "generation stops at max_tokens or end-of-turn (generated count per rep recorded)"
        receipt.params.timing = "Xybrid XybridInferenceMetrics (ttft_ms, prefill_tps, decode_tps); wall clock "
            + "around runAsync; one untimed 8-token warm-up"
        var notes: [String: String] = [
            "xybrid_version": "0.21.0",
            "llama_cpp": "Xybrid 0.21.0's pinned llama.cpp (research: b1b770d9, unverified for the release binary)",
            "load_mode": "llama.cpp default (mmap), n_gpu_layers 99, context 4096 (Xybrid defaults)",
            "metal_residency": noResidencyActive ? "off (GGML_METAL_NO_RESIDENCY=1 set at launch; experimental)" : "on (default)",
        ]
        switch entry.xybridSource {
        case .registry(let id): notes["source"] = "xybrid registry: \(id)"
        case .gguf(let repo, let file, let bytes): notes["source"] = "\(repo)/\(file) (\(bytes) bytes)"
        case nil: break
        }
        receipt.engineNotes = notes
        let crumbed = Flag()

        func finish(_ status: String, error: String? = nil, skip: String? = nil) async -> BenchReceipt {
            if crumbed.value { JetsamBreadcrumb.clear() }
            await backend.unload()
            let after = await BenchmarkRunner.environmentNow()
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

        guard entry.isSupportedOnThisDevice else {
            return await finish("skipped", skip: String(
                format: "catalog minimum is %.0f GB RAM; device has %.1f GB",
                entry.minimumMemoryGB, ModelCatalog.deviceMemoryGB))
        }
        if step.noResidency != noResidencyActive {
            return await finish("skipped", skip: step.noResidency
                ? "A7 needs Metal residency disabled at launch: turn on “Disable Metal residency at next launch”, "
                    + "quit the app from the app switcher, reopen it and run A7 first"
                : "this launch has Metal residency disabled (A7 mode); run this step after a normal relaunch")
        }

        // Idle settle, as in BenchmarkRunner, before the footprint baseline.
        progress(.init(message: "Settling (\(Int(BenchmarkRunner.settleSeconds)) s)", fraction: 0.01))
        await backend.unload()
        let settleEnd = Date().addingTimeInterval(BenchmarkRunner.settleSeconds)
        while Date() < settleEnd {
            if isCancelled() || Task.isCancelled { return await finish("cancelled") }
            try? await Task.sleep(for: .milliseconds(250))
        }

        let baseline = ResourceProbe.physFootprint()
        let available = ResourceProbe.availableMemory()
        let metalBefore = MetalProbe.currentAllocatedBytes
        let disk = XybridLLMBackend.onDiskBytes(entry)
        JetsamBreadcrumb.set(JetsamBreadcrumb.make(engine: "xybrid", entry: entry, abStep: step.id, phase: "first load",
                                                   noResidency: noResidencyActive, forced: false))
        crumbed.set()
        let loadSampler = ResourceSampler()
        loadSampler.start()
        defer { _ = loadSampler.stop() }

        do {
            // 1. First load (may download), then a reload with a warm page cache.
            progress(.init(message: "Loading \(entry.displayName)", fraction: 0.02))
            let t0 = Date()
            let first = try await backend.load(entry) { f in
                progress(.init(message: f < 0.9 ? "Downloading \(entry.displayName) \(Int(f / 0.9 * 100))%"
                                                : "Loading \(entry.displayName)", fraction: 0.02))
            }
            let firstLoad = Date().timeIntervalSince(t0)
            let diskAfter = XybridLLMBackend.onDiskBytes(entry)
            receipt.storage = .init(localSnapshotCompleteBefore: first.localCompleteBefore,
                                    bytesOnDiskBefore: disk.bytes, bytesOnDiskAfter: diskAfter.bytes,
                                    path: diskAfter.path ?? first.path,
                                    excludedFromBackup: entry.xybridSource.map { source -> Bool in
                                        if case .registry = source { return ModelStore.isExcludedFromBackup(ModelStore.xybridRoot) }
                                        return ModelStore.isExcludedFromBackup(ModelStore.modelsRoot)
                                    } ?? false)
            receipt.model.weightFilesBytes = diskAfter.bytes > 0 ? diskAfter.bytes : nil
            notes["first_load_only_s"] = String(format: "%.3f", first.loadS)
            receipt.engineNotes = notes
            if isCancelled() { return await finish("cancelled") }

            JetsamBreadcrumb.update(phase: "warm reload")
            progress(.init(message: "Warm reload", fraction: 0.05))
            await backend.unload()
            let t1 = Date()
            let warm = try await backend.load(entry) { _ in }
            let warmLoad = Date().timeIntervalSince(t1)
            let footprintLoaded = ResourceProbe.physFootprint()
            let availableLoaded = ResourceProbe.availableMemory()
            let loadMem = loadSampler.stop()
            var probe = BenchReceipt.MemoryProbeResult(
                availableBeforeLoadBytes: available, availableAfterLoadBytes: availableLoaded,
                physFootprintPeakBytes: max(loadMem.peakFootprint, footprintLoaded),
                physFootprintBeforeLoadBytes: baseline, physFootprintAfterLoadBytes: footprintLoaded,
                metalRecommendedMaxWorkingSetBytes: MetalProbe.recommendedMaxWorkingSetBytes,
                metalAllocatedBeforeLoadBytes: metalBefore,
                metalAllocatedAfterLoadBytes: MetalProbe.currentAllocatedBytes,
                metalAllocatedMaxBytes: MetalProbe.currentAllocatedBytes)
            receipt.memoryProbe = probe
            receipt.load = .init(
                firstLoadS: firstLoad, downloadObserved: first.downloaded, warmLoadS: warmLoad,
                baselineFootprintBytes: baseline, footprintAfterLoadBytes: footprintLoaded,
                availableAfterLoadBytes: availableLoaded, warmupS: 0)
            notes["warm_load_only_s"] = String(format: "%.3f", warm.loadS)
            receipt.engineNotes = notes

            // 2. Untimed warm-up, then the timed reps.
            let workload = BenchWorkloads.engineAB
            guard case .short(let prompt) = workload.prompt else { return await finish("failed", error: "bad workload") }
            JetsamBreadcrumb.update(phase: "warm-up")
            progress(.init(message: "\(entry.displayName) · warm-up", fraction: 0.1))
            let tw = Date()
            _ = try await backend.benchmarkRun(prompt: prompt, maxTokens: BenchmarkRunner.warmupTokens)
            let warmupS = Date().timeIntervalSince(tw)
            receipt.load?.warmupS = warmupS

            var maxThermal = ResourceProbe.thermal()
            var repResults: [BenchReceipt.Rep] = []
            var firstText = ""
            var promptTokens = 0
            var shortest = Int.max
            for i in 0..<reps {
                if isCancelled() || Task.isCancelled { return await finish("cancelled") }
                progress(.init(message: "\(entry.displayName) · rep \(i + 1)/\(reps)",
                               fraction: 0.1 + 0.9 * Double(i) / Double(reps)))
                let thermalBefore = ResourceProbe.thermal()
                let sampler = ResourceSampler()
                sampler.start()
                let m = try await backend.benchmarkRun(prompt: prompt, maxTokens: workload.maxTokens)
                let mem = sampler.stop()
                let thermalAfter = ResourceProbe.thermal()
                if mem.maxThermal.rawValue > maxThermal.rawValue { maxThermal = mem.maxThermal }
                probe.physFootprintPeakBytes = max(probe.physFootprintPeakBytes, mem.peakFootprint)
                if let now = MetalProbe.currentAllocatedBytes {
                    probe.metalAllocatedMaxBytes = max(probe.metalAllocatedMaxBytes ?? 0, now)
                }
                receipt.memoryProbe = probe
                JetsamBreadcrumb.update(phase: "rep \(i + 1)/\(reps) done")

                let n = m.tokensOut ?? 0
                shortest = min(shortest, n)
                if i == 0 { firstText = m.text }
                promptTokens = m.tokensIn ?? promptTokens
                let ttft = (m.ttftMs ?? 0) / 1000
                let decodeS = max(0, m.wallS - ttft)
                repResults.append(.init(
                    index: i, ttftS: ttft,
                    prefillTps: m.prefillTps ?? (ttft > 0 && promptTokens > 0 ? Double(promptTokens) / ttft : 0),
                    decodeTps: m.decodeTps ?? (decodeS > 0 && n > 1 ? Double(n - 1) / decodeS : 0),
                    decodeS: decodeS, generatedTokens: n, wallS: m.wallS,
                    libPrefillS: 0, libPromptTimeS: ttft, libTokensPerSecond: m.tokensPerSecond ?? 0,
                    peakFootprintBytes: mem.peakFootprint, minAvailableMemoryBytes: mem.minAvailable,
                    footprintSamples: mem.samples, mlxPeakBytes: 0, mlxActiveBytes: 0, mlxCacheBytes: 0,
                    thermalBefore: ResourceProbe.name(thermalBefore),
                    thermalAfter: ResourceProbe.name(thermalAfter),
                    outputHash: hash(m.text)))
            }
            receipt.workloads.append(.init(
                id: workload.id, version: workload.version, description: workload.description,
                promptTokens: promptTokens, maxTokens: workload.maxTokens, reps: repResults,
                median: .reduce(repResults, BenchReceipt.median),
                min: .reduce(repResults) { $0.min() ?? 0 },
                max: .reduce(repResults) { $0.max() ?? 0 },
                outputHash: hash(firstText), outputTokenIds: [],
                outputPreview: String(firstText.prefix(300)),
                deterministic: Set(repResults.map(\.outputHash)).count == 1,
                stopTokenIndex: shortest < workload.maxTokens ? shortest : nil, warmupS: warmupS))
            if shortest < workload.maxTokens {
                receipt.warnings = (receipt.warnings ?? []) + [
                    "a rep stopped at end-of-turn after \(shortest) of \(workload.maxTokens) tokens; "
                        + "decode tok/s is over fewer tokens than the MLX side"]
            }
            receipt.environment.thermalMaxDuring = ResourceProbe.name(maxThermal)
            progress(.init(message: "Done", fraction: 1))
            return await finish("ok")
        } catch {
            if isCancelled() || error is CancellationError { return await finish("cancelled") }
            return await finish("failed", error: error.localizedDescription)
        }
    }

    static var noResidencyActive: Bool { XybridLLMBackend.noResidencyThisLaunch }

    // MARK: Receipts that aren't runs

    /// A4 until an MLX 3-bit LFM2.5-8B-A1B conversion is in the catalog.
    @MainActor
    static func notRunnable(_ step: Step) async -> BenchReceipt {
        let placeholder = ModelCatalog.Entry(
            id: "lfm2.5-8b-a1b-mlx-3bit", displayName: "LFM2.5 8B-A1B MLX 3-bit (not converted yet)",
            repo: "(none)", downloadGB: 3.7, note: "", minimumMemoryGB: 14, thinks: true, temperature: 0.2)
        var r = BenchmarkRunner.skeleton(entry: placeholder, reps: repetitions, env: BenchmarkRunner.environmentNow())
        r.engine = "mlx"
        r.abStep = step.id
        r.identity = .current
        r.status = "skipped"
        r.skipReason = "No MLX 3-bit LFM2.5-8B-A1B build in the catalog yet (needs a conversion, backlog T4). "
            + "A2/A3 have no like-for-like MLX partner until then."
        return r
    }

    /// The run the previous launch died in (breadcrumb still set): saved as `killed`.
    @MainActor
    static func killedReceipt(_ crumb: JetsamBreadcrumb.Crumb) -> BenchReceipt {
        let entry = ModelCatalog.entry(id: crumb.entryID) ?? ModelCatalog.Entry(
            id: crumb.entryID, displayName: crumb.displayName, repo: crumb.repo, downloadGB: 0, note: "",
            minimumMemoryGB: 0, thinks: false, temperature: 0)
        var r = BenchmarkRunner.skeleton(entry: entry, reps: repetitions, env: BenchmarkRunner.environmentNow())
        let iso = ISO8601DateFormatter()
        r.engine = crumb.engine
        r.abStep = crumb.abStep
        r.identity = .current
        r.status = "killed"
        r.jetsamSuspected = true
        r.forced = crumb.forced ? true : nil
        r.error = "The app was terminated during “\(crumb.phase)” (no receipt was written). "
            + "Most likely iOS's memory limit (jetsam); the breadcrumb can't tell jetsam from other kills."
        r.memoryProbe = .init(
            availableBeforeLoadBytes: crumb.availableBeforeBytes, availableAfterLoadBytes: 0,
            physFootprintPeakBytes: crumb.footprintAtUpdateBytes, physFootprintBeforeLoadBytes: 0,
            physFootprintAfterLoadBytes: crumb.footprintAtUpdateBytes,
            metalRecommendedMaxWorkingSetBytes: MetalProbe.recommendedMaxWorkingSetBytes,
            metalAllocatedBeforeLoadBytes: nil, metalAllocatedAfterLoadBytes: crumb.metalAllocatedAtUpdateBytes,
            metalAllocatedMaxBytes: crumb.metalAllocatedAtUpdateBytes)
        r.engineNotes = [
            "breadcrumb_phase": crumb.phase,
            "breadcrumb_started_at": iso.string(from: crumb.startedAt),
            "breadcrumb_updated_at": iso.string(from: crumb.updatedAt),
            "found_at_launch": crumb.foundAtLaunch.map { iso.string(from: $0) } ?? "",
            "metal_residency": crumb.noResidency ? "off" : "on",
            "note": "footprint/metal figures are the last values recorded before the kill, not a true peak",
        ]
        return r
    }
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.withLock { flag = true } }
    var value: Bool { lock.withLock { flag } }
}
