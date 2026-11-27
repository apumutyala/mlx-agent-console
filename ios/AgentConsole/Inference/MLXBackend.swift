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
    private var inflight: (epoch: Int, task: Task<ModelContainer, Error>, progress: LoadProgressFanout)?
    /// Loads a model switch superseded. A weight read can't be cancelled (it runs on a
    /// dispatch queue), so the next load waits for these before it measures or allocates.
    private var draining: [Task<ModelContainer, Error>] = []
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

    /// Generation settings that bound memory for big models. The KV cache is deliberately
    /// *not* capped with `maxKVSize`: on Inkling that turns the global-attention layers into
    /// rotating caches that keep only 4 prefix tokens, silently evicting the system prompt and
    /// tool specs. Context is bounded per conversation instead (`MLXConversation`).
    static func applyMemorySettings(of entry: ModelCatalog.Entry, to params: inout GenerateParameters) {
        if let step = entry.prefillStepSize { params.prefill.stepSize = step }
        if entry.residentGB != nil {
            // ChatSession takes full KV-cache copies as prompt checkpoints when chunking is
            // `.balanced` (up to ~2 extra copies). `.remainder` turns them off; a rewound turn
            // re-prefills instead of restoring a copy.
            params.prefill.chunking = .remainder
        }
    }

    func switchModel(to newEntry: ModelCatalog.Entry) async {
        ModelCatalog.selected = newEntry
        reconcileSelection()
    }

    /// Unload if the persisted selection differs from what's loaded.
    private func reconcileSelection() {
        let wanted = ModelCatalog.selected
        let superseded: Task<ModelContainer, Error>?? = lock.withLock {
            guard wanted.id != entry.id else { return nil }
            entry = wanted
            container = nil   // conversations still holding the old container keep it alive until they end
            let old = inflight?.task
            inflight = nil
            epoch += 1
            if let old {
                old.cancel()   // stops a download; a weight read runs to completion
                draining.append(old)
            }
            return .some(old)
        }
        guard let superseded else { return }
        MLX.Memory.clearCache()
        if let old = superseded {
            // Drop our reference once it finishes so its (discarded) weights can be freed.
            Task { [weak self] in
                _ = try? await old.value
                self?.lock.withLock { self?.draining.removeAll { $0 == old } }
                MLX.Memory.clearCache()
            }
        }
    }

    private enum LoadPlan {
        case loaded
        case wait(Task<ModelContainer, Error>, epoch: Int)
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        reconcileSelection()
        // One load at a time. The decision is made under the same lock that installs the
        // container, so a caller arriving as a load finishes can't start a duplicate.
        let plan: LoadPlan = lock.withLock {
            if container != nil { return .loaded }
            if let inflight, inflight.epoch == epoch {
                inflight.progress.add(progress)
                return .wait(inflight.task, epoch: epoch)
            }
            let fanout = LoadProgressFanout()
            fanout.add(progress)
            let previous = DrainBox(draining)
            let target = entry
            let loadEpoch = epoch
            let task = Task { () async throws -> ModelContainer in
                await previous.drain()   // superseded weight reads finish (and free) first
                MLX.Memory.clearCache()
                try Task.checkCancellation()
                return try await self.performLoad(target, epoch: loadEpoch) { fanout.send($0) }
            }
            inflight = (epoch, task, fanout)
            return .wait(task, epoch: epoch)
        }
        guard case .wait(let task, let startEpoch) = plan else { progress(1); return }
        do {
            let loaded = try await task.value
            lock.withLock {
                if epoch == startEpoch, container == nil { container = loaded }
                if inflight?.epoch == startEpoch { inflight = nil }
            }
            progress(1)
        } catch {
            lock.withLock { if inflight?.epoch == startEpoch { inflight = nil } }
            throw error
        }
    }

    private func isCurrent(_ loadEpoch: Int) -> Bool { lock.withLock { epoch == loadEpoch } }

    private func performLoad(_ target: ModelCatalog.Entry, epoch loadEpoch: Int,
                             progress: @escaping @Sendable (Double) -> Void) async throws -> ModelContainer
    {
        // Keep MLX's buffer cache modest on iPad; weights dominate memory.
        MLX.Memory.cacheLimit = 32 * 1024 * 1024

        // Big models: refuse up front rather than get the app killed mid-load. Only the
        // buffer cache counts as freeable here: weights another task or chat still holds
        // stay in MLX's active memory and are already missing from os_proc_available_memory.
        let fit = MemoryBudget.fit(target, includingHeld: false)
        if case .tooLarge(let needed, let budget) = fit, !fit.allowsLoad {
            throw MemoryBudget.LoadError.insufficientMemory(
                model: target.displayName, neededBytes: needed, budgetBytes: budget,
                stillHeldBytes: MLX.Memory.activeMemory)
        }

        switch target.loader {
        case .llm:
            return try await #huggingFaceLoadModelContainer(
                configuration: Self.configuration(for: target),
                progress: .download { p in progress(p.fractionCompleted) }
            )
        case .vlm:
            return try await loadLarge(target, epoch: loadEpoch, progress: progress)
        }
    }

    /// Inkling 2 and other big VLM-factory models: memory limits, a breadcrumb for loads the
    /// system kills, and a load report for Settings.
    private func loadLarge(_ target: ModelCatalog.Entry, epoch loadEpoch: Int,
                           progress: @escaping @Sendable (Double) -> Void) async throws -> ModelContainer
    {
        let defaults = UserDefaults.standard
        let availableBefore = MemoryBudget.availableBytes
        if availableBefore > 0 {
            // A soft cap: past it MLX waits for in-flight work before allocating more, which
            // keeps transient peaks (prefill scratch) under the jetsam limit. It is not a
            // hard limit, so it can't make a load fail on its own.
            let budget = availableBefore + MLX.Memory.cacheMemory
            MLX.Memory.memoryLimit = max(1 << 30, budget - 512 * 1024 * 1024)
        }
        MLX.Memory.peakMemory = 0
        // The breadcrumb survives only a process kill: cleared on success, error and
        // cancellation, and only by the load that wrote it.
        let token = "\(target.id) at \(Date().formatted(date: .abbreviated, time: .standard)) #\(UUID().uuidString.prefix(8))"
        defaults.set(token, forKey: MemoryBudget.pendingLoadKey)
        defer {
            if defaults.string(forKey: MemoryBudget.pendingLoadKey) == token {
                defaults.removeObject(forKey: MemoryBudget.pendingLoadKey)
            }
        }
        let started = Date()

        let handlers = LoadProgressHandlers(
            download: { p in progress(0.9 * p.fractionCompleted) },
            weights: { p in progress(0.9 + 0.1 * p.fractionCompleted) })
        let loaded = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: Self.configuration(for: target),
            progress: handlers)

        if isCurrent(loadEpoch) {
            let snapshot = MemoryBudget.mlxSnapshot()
            let report = String(
                format: "%@: loaded in %.0f s · MLX active %@, peak %@ · app available before %@, after %@ · MLX limit %@",
                target.displayName, Date().timeIntervalSince(started),
                MemoryBudget.gb(snapshot.active), MemoryBudget.gb(snapshot.peak),
                MemoryBudget.gb(availableBefore), MemoryBudget.gb(MemoryBudget.availableBytes),
                MemoryBudget.gb(MLX.Memory.memoryLimit))
            defaults.set(report, forKey: MemoryBudget.lastLoadReportKey)
        }
        return loaded
    }

    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation {
        let loadedContainer: ModelContainer? = lock.withLock { self.container }
        let current: ModelCatalog.Entry = lock.withLock { self.entry }
        guard let loadedContainer else { throw BackendError.notLoaded }
        var params = GenerateParameters()
        params.maxTokens = current.thinks ? 2048 : 1024   // leave room for <think> before the answer
        params.temperature = current.temperature
        Self.applyMemorySettings(of: current, to: &params)
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
        // Tool schemas are rendered into the prompt too; count them toward the context.
        let toolChars = specs.map { "\($0)".count } ?? 0
        return MLXConversation(session: session, contextTokens: current.contextTokens,
                               instructions: instructions, extraChars: toolChars)
    }

    enum BackendError: LocalizedError {
        case notLoaded
        var errorDescription: String? { "Model not loaded yet." }
    }
}

/// Fans one load's progress out to every caller waiting on it; late joiners get the last value.
private final class LoadProgressFanout: @unchecked Sendable {
    private let lock = NSLock()
    private var observers: [@Sendable (Double) -> Void] = []
    private var last: Double = 0

    func add(_ observer: @escaping @Sendable (Double) -> Void) {
        let current: Double = lock.withLock {
            observers.append(observer)
            return last
        }
        if current > 0 { observer(current) }
    }

    func send(_ value: Double) {
        let targets: [@Sendable (Double) -> Void] = lock.withLock {
            last = value
            return observers
        }
        for observer in targets { observer(value) }
    }
}

/// Superseded loads a new load must wait for. Emptied after waiting, so the new load
/// doesn't keep their (discarded) containers alive while it allocates its own.
private final class DrainBox: @unchecked Sendable {
    private let lock = NSLock()
    private var tasks: [Task<ModelContainer, Error>]
    init(_ tasks: [Task<ModelContainer, Error>]) { self.tasks = tasks }

    func drain() async {
        while let next: Task<ModelContainer, Error> = lock.withLock({ tasks.popLast() }) {
            _ = try? await next.value
        }
    }
}

/// A conversation that refuses to grow past the model's context budget instead of letting
/// the KV cache grow without bound (or a rotating cache silently drop the system prompt).
/// The size is estimated from characters (~3 per token, conservative for English and JSON).
final class MLXConversation: ModelConversation, @unchecked Sendable {
    let session: ChatSession
    private let limitChars: Int?
    private let lock = NSLock()
    private var usedChars: Int

    init(session: ChatSession, contextTokens: Int? = nil, instructions: String = "", extraChars: Int = 0) {
        self.session = session
        self.limitChars = contextTokens.map { $0 * 3 }
        self.usedChars = instructions.count + extraChars
    }

    struct ContextFull: LocalizedError {
        let tokens: Int
        var errorDescription: String? {
            "This chat has reached the model's ~\(tokens)-token context budget. Start a new chat to continue."
        }
    }

    /// Reserves room for `input`, or throws when the conversation is already full.
    private func admit(_ input: Int) throws {
        try lock.withLock {
            if let limitChars, usedChars + input > limitChars {
                throw ContextFull(tokens: limitChars / 3)
            }
            usedChars += input
        }
    }

    private func account(_ output: Int) { lock.withLock { usedChars += output } }

    func send(_ message: String) -> AsyncThrowingStream<ModelEvent, Error> {
        do { try admit(message.count) } catch { return Self.failed(error) }
        return map(session.streamDetails(to: message))
    }

    func sendToolResults(_ results: [ToolResultMessage]) -> AsyncThrowingStream<ModelEvent, Error> {
        do { try admit(results.reduce(0) { $0 + $1.content.count }) } catch { return Self.failed(error) }
        let messages = results.map { Chat.Message.tool($0.content, id: $0.callID, name: $0.name) }
        return map(session.streamDetails(to: messages))
    }

    private static func failed(_ error: Error) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: error) }
    }

    private func map(_ upstream: AsyncThrowingStream<Generation, Error>) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [self] in
                do {
                    for try await item in upstream {
                        switch item {
                        case .chunk(let text):
                            account(text.count)
                            continuation.yield(.text(text))
                        case .reasoning(let text):
                            account(text.count)
                            continuation.yield(.reasoning(text))
                        case .toolCall(let call):
                            let args = (try? JSONEncoder().encode(call.function.arguments)) ?? Data("{}".utf8)
                            account(args.count + call.function.name.count)
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
