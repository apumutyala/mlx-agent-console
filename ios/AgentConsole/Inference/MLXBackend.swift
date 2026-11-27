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
///
/// One load path (`load`) serves chat, fleet tasks, LoRA rounds and benchmarks:
/// - it runs on behalf of the caller's task, so cancelling the caller (pause, Serve
///   off, chat Stop) cancels the download, and the caller only returns once the load
///   has really finished (#6);
/// - a model switch cancels the superseded download, and the next load drains every
///   superseded weight read (which can't be cancelled) before it clears MLX's cache,
///   measures memory and allocates (#2);
/// - the `MemoryBudget` gate refuses models that don't fit, and VLM-factory entries
///   (Inkling 2) load through `loadLarge` (#2);
/// - in benchmark mode it serves the benchmark target instead of the selection (#7).
/// Every caller holds `GPUGate`, which serializes loads; the join/drain bookkeeping
/// below keeps the path safe even for a caller that doesn't.
final class MLXBackend: ModelBackend, @unchecked Sendable {
    private var entry: ModelCatalog.Entry
    private var container: ModelContainer?
    /// Bumped on every switch so a load that finishes late can't install a stale model.
    private var epoch = 0
    /// Set while a benchmark runs (see Benchmark mode below).
    private var benchmarkEntry: ModelCatalog.Entry?
    /// Loads currently running, for the current epoch or superseded ones still draining.
    private var active: [ActiveLoad] = []
    private let lock = NSLock()

    /// MLX's free-buffer cache limit on iPad; weights dominate memory.
    static let mlxCacheLimit = 32 * 1024 * 1024

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

    /// `configuration(for:)`, pointed at the local snapshot when ModelStore has a complete
    /// one for the repo at `refs/main` (then the downloader is never called for the
    /// weights, so the load works offline and never waits on the network). Only the model
    /// source changes, so registry settings such as `toolCallFormat` are kept.
    static func storeConfiguration(for entry: ModelCatalog.Entry) -> (ModelConfiguration, local: Bool) {
        var cfg = configuration(for: entry)
        if case .id(let repo, revision: let revision) = cfg.id, revision == "main",
           let snapshot = ModelStore.localSnapshot(repo: repo) {
            cfg.id = .directory(snapshot)
            return (cfg, true)
        }
        return (cfg, false)
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

    /// The entry the backend should hold: the benchmark target while a benchmark
    /// runs, otherwise the persisted selection.
    private var wantedEntry: ModelCatalog.Entry {
        lock.withLock { benchmarkEntry } ?? ModelCatalog.selected
    }

    /// Unload if the wanted entry differs from what's loaded (or always, with `force`).
    /// A load still running for the old entry is cancelled (that stops a download; a
    /// weight read runs to completion) and drained by the next load.
    private func reconcileSelection(force: Bool = false) {
        let wanted = wantedEntry
        let changed: Bool = lock.withLock {
            guard force || wanted.id != entry.id else { return false }
            entry = wanted
            container = nil   // conversations still holding the old container keep it alive until they end
            epoch += 1
            for load in active { load.cancel() }
            return true
        }
        if changed { MLX.Memory.clearCache() }
    }

    private enum LoadStep {
        case loaded
        /// Another task is loading the current entry: wait for it, then re-check.
        case join(LoadSignal)
        /// Superseded weight reads are still finishing: wait for them before measuring.
        case drain([LoadSignal])
        case start(ActiveLoad, target: ModelCatalog.Entry, epoch: Int)
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        var drained = false
        while true {
            reconcileSelection()
            // After reconciling, `entry` is the wanted one (the selection, or a benchmark
            // target). The decision is made under the lock that installs the container, so
            // a caller arriving as a load finishes can't start a duplicate.
            let step: LoadStep = lock.withLock {
                if container != nil { return .loaded }
                if let current = active.first(where: { $0.epoch == epoch }) {
                    current.progress.add(progress)
                    return .join(current.done)
                }
                let stale = active.map(\.done)
                if !stale.isEmpty { return .drain(stale) }
                let load = ActiveLoad(epoch: epoch)
                load.progress.add(progress)
                active.append(load)
                return .start(load, target: entry, epoch: epoch)
            }
            switch step {
            case .loaded:
                progress(1)
                return
            case .join(let signal):
                // The joined load's error is ours too, unless it was only cancelled
                // (its caller went away): then retry, starting our own.
                if let error = await signal.wait(), !(error is CancellationError) { throw error }
                try Task.checkCancellation()
            case .drain(let signals):
                for signal in signals { _ = await signal.wait() }
                drained = true
                try Task.checkCancellation()
            case .start(let load, target: let target, epoch: let startEpoch):
                if drained { MLX.Memory.clearCache() }   // the drained loads' discarded weights
                do {
                    try await run(load, target: target, epoch: startEpoch)
                } catch {
                    finish(load, error: error)
                    throw error
                }
                finish(load, error: nil)
                try Task.checkCancellation()
                progress(1)
                return
            }
        }
    }

    /// Runs one load to completion and installs it if it is still wanted. The load runs
    /// in a task this caller awaits (cancelling the caller cancels it), and the await only
    /// returns once the task has ended, so the caller (and its `GPUGate` hold) outlasts any
    /// weight read. A published container is never replaced.
    private func run(_ load: ActiveLoad, target: ModelCatalog.Entry, epoch startEpoch: Int) async throws {
        let task = Task { () async throws -> ModelContainer in
            try Task.checkCancellation()
            return try await self.performLoad(target, epoch: startEpoch) { load.progress.send($0) }
        }
        if !load.attach(task) { task.cancel() }   // superseded before it started
        let loaded = try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
        lock.withLock {
            if epoch == startEpoch, container == nil { container = loaded }
        }
    }

    /// Removes a finished load and wakes whoever joined or is draining it. Called after
    /// `run` returned, so this frame no longer references a discarded container.
    private func finish(_ load: ActiveLoad, error: Error?) {
        lock.withLock { active.removeAll { $0 === load } }
        load.detach()
        load.done.finish(error)
        if error != nil || !isCurrent(load.epoch) { MLX.Memory.clearCache() }
    }

    private func isCurrent(_ loadEpoch: Int) -> Bool { lock.withLock { epoch == loadEpoch } }

    private func performLoad(_ target: ModelCatalog.Entry, epoch loadEpoch: Int,
                             progress: @escaping @Sendable (Double) -> Void) async throws -> ModelContainer
    {
        // Keep MLX's buffer cache modest on iPad; weights dominate memory.
        MLX.Memory.cacheLimit = Self.mlxCacheLimit

        // Big models: refuse up front rather than get the app killed mid-load. Only the
        // buffer cache counts as freeable here: weights another task or chat still holds
        // stay in MLX's active memory and are already missing from os_proc_available_memory.
        let fit = MemoryBudget.fit(target, includingHeld: false)
        if case .tooLarge(let needed, let budget) = fit, !fit.allowsLoad {
            throw MemoryBudget.LoadError.insufficientMemory(
                model: target.displayName, neededBytes: needed, budgetBytes: budget,
                stillHeldBytes: MLX.Memory.activeMemory)
        }

        guard target.engine == .mlx else { throw BackendError.wrongEngine(target.displayName) }

        switch target.loader {
        case .llm:
            // Local-first: a complete snapshot in ModelStore loads with no network call.
            let (configuration, local) = Self.storeConfiguration(for: target)
            let loaded = try await loadModelContainer(
                from: #hubDownloader(ModelStore.hub),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progress: .download { p in progress(p.fractionCompleted) })
            if !local { ModelStore.noteDownloaded(repo: target.repo) }
            return loaded
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
        let (configuration, local) = Self.storeConfiguration(for: target)
        let loaded = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(ModelStore.hub),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration,
            progress: handlers)
        if !local { ModelStore.noteDownloaded(repo: target.repo) }

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

    // MARK: Benchmark mode
    //
    // Benchmarks load through `load` above (the one load path), so its draining, VLM
    // routing and memory budget apply to measured models too. The caller holds GPUGate
    // for the whole benchmark.

    /// Until `endBenchmark()`, serve `entry` instead of the persisted selection. The
    /// selection itself is not changed. Drops whatever is loaded (chat model, adapters).
    func beginBenchmark(_ entry: ModelCatalog.Entry) {
        lock.withLock { benchmarkEntry = entry }
        reconcileSelection(force: true)
    }

    /// Load the benchmark target through `load` and return its container.
    func loadBenchmarkModel(progress: @escaping @Sendable (Double) -> Void) async throws -> ModelContainer {
        try await load(progress: progress)
        guard let c = modelContainer else { throw BackendError.notLoaded }
        return c
    }

    /// Drop the benchmark target's weights (before a warm reload). The caller must
    /// also drop its own reference for the memory to be freed.
    func dropBenchmarkModel() { reconcileSelection(force: true) }

    /// Leave benchmark mode; the chat model reloads on its next `load`.
    func endBenchmark() {
        lock.withLock { benchmarkEntry = nil }
        reconcileSelection(force: true)
    }

    /// Generation parameters a benchmark decodes with: the same per-entry settings chat
    /// uses (prefill step and chunking from `applyMemorySettings`), greedy.
    static func benchmarkParameters(for entry: ModelCatalog.Entry, maxTokens: Int, seed: UInt64) -> GenerateParameters {
        var params = GenerateParameters()
        params.maxTokens = maxTokens
        params.temperature = 0          // greedy (argmax sampler)
        params.seed = seed
        applyMemorySettings(of: entry, to: &params)
        return params
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
        /// The entry runs on another engine (Xybrid); `EngineRouter` routes it there.
        case wrongEngine(String)
        var errorDescription: String? {
            switch self {
            case .notLoaded: "Model not loaded yet."
            case .wrongEngine(let name): "\(name) runs on the Xybrid engine, not MLX."
            }
        }
    }
}


/// One load in progress: its epoch, the progress fan-out for callers that join it, the
/// task doing the work (so a model switch can cancel its download) and the signal its
/// joiners and the next load's drain wait on.
private final class ActiveLoad: @unchecked Sendable {
    let epoch: Int
    let done = LoadSignal()
    let progress = LoadProgressFanout()
    private let lock = NSLock()
    private var task: Task<ModelContainer, Error>?
    private var cancelled = false

    init(epoch: Int) { self.epoch = epoch }

    /// Returns false when the load was already superseded (the caller cancels the task).
    func attach(_ task: Task<ModelContainer, Error>) -> Bool {
        lock.withLock {
            self.task = task
            return !cancelled
        }
    }

    func cancel() {
        let task: Task<ModelContainer, Error>? = lock.withLock {
            cancelled = true
            return self.task
        }
        task?.cancel()
    }

    /// Drops the finished task, and with it the container it returned, so a discarded
    /// model's weights are freed before anyone draining this load measures memory.
    func detach() { lock.withLock { task = nil } }
}

/// A one-shot completion with the load's error (nil on success). Waiting is not
/// cancellable: the load it waits for always ends (a weight read can't be stopped, but
/// it finishes), and the caller checks cancellation right after.
private final class LoadSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    private var error: Error?
    private var waiters: [CheckedContinuation<Error?, Never>] = []

    func finish(_ error: Error?) {
        let waiting: [CheckedContinuation<Error?, Never>] = lock.withLock {
            finished = true
            self.error = error
            defer { waiters = [] }
            return waiters
        }
        for waiter in waiting { waiter.resume(returning: error) }
    }

    func wait() async -> Error? {
        await withCheckedContinuation { (continuation: CheckedContinuation<Error?, Never>) in
            let result: (done: Bool, error: Error?) = lock.withLock {
                if finished { return (true, error) }
                waiters.append(continuation)
                return (false, nil)
            }
            if result.done { continuation.resume(returning: result.error) }
        }
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

/// A conversation that refuses to grow past the model's context budget instead of letting
/// the KV cache grow without bound (or a rotating cache silently drop the system prompt).
/// The size is estimated from characters (~3 per token, conservative for English and JSON).
final class MLXConversation: ModelConversation, @unchecked Sendable {
    let session: ChatSession
    private let limitChars: Int?
    private let lock = NSLock()
    private var usedChars: Int
    /// The task relaying the current reply; awaited by `synchronize`.
    private var relay: Task<Void, Never>?

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

    /// Cancelling the consumer only *requests* a stop; the session's runner may
    /// still be decoding. This waits for the relay and the session's runners.
    func synchronize() async {
        let task = lock.withLock { relay }
        await task?.value
        await session.synchronize()
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
            lock.withLock { relay = task }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
