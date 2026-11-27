import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The only file that touches MLX. Written against mlx-swift-lm 3.33 plus PR #821
/// (Inkling 2), vendored under ios/Vendor/mlx-swift-lm. The Hugging Face
/// downloader/tokenizer come from the MLXHuggingFace macros.
final class MLXBackend: ModelBackend, @unchecked Sendable {
    private var entry: ModelCatalog.Entry
    private var container: ModelContainer?
    /// Bumped on every switch so a load that finishes late can't install a stale model.
    private var epoch = 0
    /// The load in progress for `epoch`, shared by every caller that asks meanwhile.
    private var inflight: (epoch: Int, task: Task<ModelContainer, Error>)?
    private let lock = NSLock()

    init(entry: ModelCatalog.Entry = ModelCatalog.selected) {
        self.entry = entry
        MemoryBudget.noteLaunch()
        MemoryBudget.heldByModel = { MLX.Memory.activeMemory + MLX.Memory.cacheMemory }
        MemoryBudget.mlxSnapshot = {
            (MLX.Memory.activeMemory, MLX.Memory.peakMemory, MLX.Memory.cacheMemory)
        }
    }

    var displayName: String { lock.withLock { entry.displayName } }

    /// Hugging Face repo id of the loaded/selected model (federated rounds match on it).
    var modelID: String { lock.withLock { Self.configuration(for: entry).name } }

    /// The loaded container, for MLX-level work (LoRA training, token counts)
    /// that lives in MLX-only files. nil until `load` finishes.
    var modelContainer: ModelContainer? { lock.withLock { container } }

    var isLoaded: Bool {
        get async { lock.withLock { container != nil && entry.id == ModelCatalog.selected.id } }
    }

    /// mlx-swift-lm configuration for a catalog entry. Registered entries come
    /// from `LLMRegistry`; the rest are built here. When `toolCallFormat` is nil the
    /// factory resolves it from the model type / chat template at load time
    /// (Inkling 2 declares `.inkling` itself).
    static func configuration(for entry: ModelCatalog.Entry) -> ModelConfiguration {
        switch entry.id {
        case ModelCatalog.lfm3_1_7b.id: return LLMRegistry.lfm3_1_7b_4bit        // .xmlFunction calls, <think>
        case ModelCatalog.lfm25_8b_a1b.id: return LLMRegistry.lfm25_8b_a1b_4bit  // .lfm2 calls, always thinks
        case ModelCatalog.qwen3_1_7b.id: return LLMRegistry.qwen3_1_7b_4bit      // .json calls, <think>
        case ModelCatalog.lfm25_1_2b.id, ModelCatalog.lfm25_350m.id:
            return ModelConfiguration(id: entry.repo, toolCallFormat: .lfm2)
        default:
            return ModelConfiguration(id: entry.repo)
        }
    }

    func switchModel(to newEntry: ModelCatalog.Entry) async {
        ModelCatalog.selected = newEntry
        reconcileSelection()
    }

    /// Unload if the persisted selection differs from what's loaded.
    private func reconcileSelection() {
        let wanted = ModelCatalog.selected
        let changed: Bool = lock.withLock {
            guard wanted.id != entry.id else { return false }
            entry = wanted
            container = nil   // conversations still holding the old container keep it alive until they end
            inflight?.task.cancel()   // stops a download in progress; a weight read finishes and is dropped
            inflight = nil
            epoch += 1
            return true
        }
        if changed { MLX.Memory.clearCache() }
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        reconcileSelection()
        if await isLoaded { progress(1); return }
        // One load at a time: the chat screen and a first message can both ask, and two
        // concurrent loads of a 10 GB model would exceed the app's memory limit.
        let (task, startEpoch): (Task<ModelContainer, Error>, Int) = lock.withLock {
            if let inflight, inflight.epoch == epoch { return (inflight.task, epoch) }
            let target = entry
            let task = Task { try await self.performLoad(target, progress: progress) }
            inflight = (epoch, task)
            return (task, epoch)
        }
        defer { lock.withLock { if inflight?.epoch == startEpoch { inflight = nil } } }
        let loaded = try await task.value
        lock.withLock {
            if epoch == startEpoch { container = loaded }
        }
    }

    private func performLoad(_ target: ModelCatalog.Entry, progress: @escaping @Sendable (Double) -> Void)
        async throws -> ModelContainer
    {
        // Keep MLX's buffer cache modest on iPad; weights dominate memory.
        MLX.Memory.cacheLimit = 32 * 1024 * 1024

        // Big models: refuse up front rather than get the app killed mid-load.
        let fit = MemoryBudget.fit(target)
        if case .tooLarge(let needed, let budget) = fit, !fit.allowsLoad {
            throw MemoryBudget.LoadError.insufficientMemory(
                model: target.displayName, neededBytes: needed, budgetBytes: budget)
        }

        switch target.loader {
        case .llm:
            return try await #huggingFaceLoadModelContainer(
                configuration: Self.configuration(for: target),
                progress: .download { p in progress(p.fractionCompleted) }
            )
        case .vlm:
            return try await loadLarge(target, progress: progress)
        }
    }

    /// Inkling 2 and other big VLM-factory models: memory limits, a breadcrumb for loads the
    /// system kills, and a load report for Settings.
    private func loadLarge(_ target: ModelCatalog.Entry, progress: @escaping @Sendable (Double) -> Void)
        async throws -> ModelContainer
    {
        let defaults = UserDefaults.standard
        let availableBefore = MemoryBudget.availableBytes
        if availableBefore > 0 {
            // A soft cap: past it MLX waits for in-flight work before allocating more, which
            // keeps transient peaks (prefill scratch) under the jetsam limit. It is not a
            // hard limit, so it can't make a load fail on its own.
            let budget = availableBefore + MemoryBudget.heldByModel()
            MLX.Memory.memoryLimit = max(1 << 30, budget - 512 * 1024 * 1024)
        }
        MLX.Memory.peakMemory = 0
        defaults.set("\(target.id) at \(Date().formatted(date: .abbreviated, time: .standard))",
                     forKey: MemoryBudget.pendingLoadKey)
        let started = Date()
        Self.applyKernelOptions()
        if target.id.hasPrefix("inkling"), defaults.bool(forKey: Self.kernelSelfCheckKey) {
            Self.runKernelSelfCheck()
        }

        let handlers = LoadProgressHandlers(
            download: { p in progress(0.9 * p.fractionCompleted) },
            weights: { p in progress(0.9 + 0.1 * p.fractionCompleted) })
        let loaded = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: Self.configuration(for: target),
            progress: handlers)

        defaults.removeObject(forKey: MemoryBudget.pendingLoadKey)
        let snapshot = MemoryBudget.mlxSnapshot()
        let report = String(
            format: "%@: loaded in %.0f s · MLX active %@, peak %@ · app available before %@, after %@ · MLX limit %@",
            target.displayName, Date().timeIntervalSince(started),
            MemoryBudget.gb(snapshot.active), MemoryBudget.gb(snapshot.peak),
            MemoryBudget.gb(availableBefore), MemoryBudget.gb(MemoryBudget.availableBytes),
            MemoryBudget.gb(MLX.Memory.memoryLimit))
        defaults.set(report, forKey: MemoryBudget.lastLoadReportKey)
        return loaded
    }

    /// Settings toggle: before loading Inkling, compare its fused decode expert kernels with the
    /// unfused path on a random layer for each expert format (2/3/4-bit affine). Debug aid.
    static let kernelSelfCheckKey = "inkling.kernelSelfCheck"
    static let kernelSelfCheckReportKey = "inkling.kernelSelfCheckReport"
    /// Settings toggle, on by default: fused decode kernels for 2/3-bit Inkling experts. Off
    /// gives the unfused path in the same build, for the A/B. Applied only when a model loads
    /// (`loadLarge`), so one loaded model, and every generation on it, uses one path.
    static let fusedLowBitExpertsKey = "inkling.fusedLowBitExperts"

    private static func applyKernelOptions() {
        let defaults = UserDefaults.standard
        Inkling2FusedDecodeOptions.lowBitAffine =
            defaults.object(forKey: fusedLowBitExpertsKey) as? Bool ?? true
    }

    /// Runs before the weights load, while the memory it needs
    /// (`Inkling2KernelSelfCheck.peakBytes()`, 223 MB) is free; skips with a note when the
    /// app has less than that plus 256 MB left.
    private static func runKernelSelfCheck() {
        let started = Date()
        let needed = Inkling2KernelSelfCheck.peakBytes()
        let available = MemoryBudget.availableBytes
        if available > 0, available < needed + 256 * 1024 * 1024 {
            UserDefaults.standard.set(
                "\(started.formatted(date: .abbreviated, time: .standard)) · skipped: needs "
                    + "\(MemoryBudget.gb(needed)) + 256 MB, the app has \(MemoryBudget.gb(available))",
                forKey: kernelSelfCheckReportKey)
            return
        }
        let reports = Inkling2KernelSelfCheck.run()
        MLX.Memory.clearCache()
        let report = String(
            format: "%@ · %.0f s\n", Date().formatted(date: .abbreviated, time: .standard),
            Date().timeIntervalSince(started)) + Inkling2KernelSelfCheck.summary(reports)
        UserDefaults.standard.set(report, forKey: kernelSelfCheckReportKey)
    }

    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation {
        let loadedContainer: ModelContainer? = lock.withLock { self.container }
        let current: ModelCatalog.Entry = lock.withLock { self.entry }
        guard let loadedContainer else { throw BackendError.notLoaded }
        var params = GenerateParameters()
        params.maxTokens = current.thinks ? 2048 : 1024   // leave room for <think> before the answer
        params.temperature = current.temperature
        if let context = current.contextTokens { params.maxKVSize = context }
        if let step = current.prefillStepSize { params.prefill.stepSize = step }
        var context: [String: any Sendable]? = nil
        if let effort = current.reasoningEffort { context = ["reasoning_effort": effort] }
        var specs: [ToolSpec]? = nil
        if !tools.isEmpty {
            specs = tools.map { (tool: ToolSpecification) -> ToolSpec in
                var function: [String: any Sendable] = [:]
                function["name"] = tool.name
                function["description"] = tool.description
                function["parameters"] = tool.parameters
                var spec: ToolSpec = [:]
                spec["type"] = "function"
                spec["function"] = function
                return spec
            }
        }
        // No `toolDispatch`: calls come back as `.toolCall` items so the app can
        // validate, ask for confirmation and run them itself.
        let session = ChatSession(
            loadedContainer, instructions: instructions, generateParameters: params,
            additionalContext: context, tools: specs)
        return MLXConversation(session: session)
    }

    enum BackendError: LocalizedError {
        case notLoaded
        var errorDescription: String? { "Model not loaded yet." }
    }
}

final class MLXConversation: ModelConversation, @unchecked Sendable {
    let session: ChatSession
    init(session: ChatSession) { self.session = session }

    func send(_ message: String) -> AsyncThrowingStream<ModelEvent, Error> {
        Self.map(session.streamDetails(to: message))
    }

    func sendToolResults(_ results: [ToolResultMessage]) -> AsyncThrowingStream<ModelEvent, Error> {
        let messages = results.map { Chat.Message.tool($0.content, id: $0.callID, name: $0.name) }
        return Self.map(session.streamDetails(to: messages))
    }

    private static func map(_ upstream: AsyncThrowingStream<Generation, Error>) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await item in upstream {
                        switch item {
                        case .chunk(let text): continuation.yield(.text(text))
                        case .reasoning(let text): continuation.yield(.reasoning(text))
                        case .toolCall(let call):
                            let args = (try? JSONEncoder().encode(call.function.arguments)) ?? Data("{}".utf8)
                            continuation.yield(.toolCall(ToolCallProposal(
                                name: call.function.name, argumentsJSON: args, id: call.id)))
                        case .rejectedToolCall(let rejected):
                            let name = rejected.toolName.map { " (\($0))" } ?? ""
                            let detail = rejected.detail.map { ": \($0)" } ?? ""
                            continuation.yield(.rejectedToolCall(
                                reason: "\(rejected.reason.rawValue)\(name)\(detail)"))
                        default: break   // .info, .steering
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
