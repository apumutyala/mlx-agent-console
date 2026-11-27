import Foundation
import Xybrid

// Written against Xybrid 0.21.0 (bindings/apple/Sources/Xybrid/{Xybrid,xybrid_bolt}.swift at
// tag v0.21.0, 491da33d). Xybrid's LLM path is llama.cpp with Metal on device; GGUF files are
// memory-mapped (load mode AUTO) and Metal residency sets are on unless
// GGML_METAL_NO_RESIDENCY is set before the first load (read with getenv in-process).
//
// Note: the Xybrid module exports typealiases named `Result`, `Model`, `ToolCall` and others.
// This file avoids those bare names, and no MLX module is imported here.

/// The opt-in second engine: Xybrid's llama.cpp. Holds at most one model. `EngineRouter`
/// picks it for `.xybrid` catalog entries and unloads it before MLX loads anything (and the
/// other way round), so the two engines never hold big weights at once. Every caller holds
/// `GPUGate` around loads and generation, as for MLX.
final class XybridLLMBackend: @unchecked Sendable {
    struct LoadInfo: Sendable {
        var entryID: String
        /// Wall time of `ModelLoader.load()` only (after any download).
        var loadS: Double
        /// Bytes were fetched for this load (GGUF or registry download).
        var downloaded: Bool
        /// The GGUF path, or the SDK cache path for a registry model.
        var path: String?
        /// The GGUF existed with its exact expected size before this load.
        var localCompleteBefore: Bool
        var noResidency: Bool
    }

    /// One run's numbers, as Xybrid reports them (`XybridInferenceMetrics`) plus our wall clock.
    struct RunMetrics: Sendable {
        var wallS: Double
        var totalMs: Double
        var ttftMs: Double?
        var prefillTps: Double?
        var decodeTps: Double?
        var tokensPerSecond: Double?
        var tokensOut: Int?
        var tokensIn: Int?
        var text: String
    }

    enum BackendError: LocalizedError {
        case notXybrid(String)
        case notLoaded
        case badURL(String)
        case insufficientStorage(needed: Int64, free: Int64)
        case downloadFailed(String)
        var errorDescription: String? {
            switch self {
            case .notXybrid(let name): "\(name) is not a Xybrid model."
            case .notLoaded: "Xybrid model not loaded yet."
            case .badURL(let s): "Invalid download URL: \(s)"
            case .insufficientStorage(let needed, let free):
                String(format: "Not enough storage: the download needs %.2f GB (plus 1 GB margin), %.2f GB free.",
                       Double(needed) / 1e9, Double(free) / 1e9)
            case .downloadFailed(let s): "Download failed: \(s)"
            }
        }
    }

    private let lock = NSLock()
    private var loadedEntry: ModelCatalog.Entry?
    private var model: XybridModel?
    /// Models `forget()` dropped while a conversation may still use them; `unload()` unloads
    /// them, so a model switch can't leave two native models resident.
    private var retired: [XybridModel] = []

    func isLoaded(_ entry: ModelCatalog.Entry) -> Bool {
        lock.withLock { model != nil && loadedEntry?.id == entry.id }
    }

    var loadedEntryID: String? { lock.withLock { model == nil ? nil : loadedEntry?.id } }

    static func prepareSDK() {
        // Same privacy default as SpeechService: opt out before the SDK starts.
        setenv("XYBRID_TELEMETRY_OPTOUT", "1", 1)
        Xybrid.initialize()   // no apiKey: fully on-device; idempotent
        ModelStore.excludeXybridFromBackup()
    }

    // MARK: A7 (experimental): Metal residency off
    //
    // ggml's Metal backend reads GGML_METAL_NO_RESIDENCY with getenv when it creates its
    // device, which may happen once per process (whisper.cpp/llama.cpp share ggml), so the
    // variable is only trustworthy when set at launch, before any Xybrid model exists. The
    // Bench tab arms it for the next launch only; `applyLaunchEnvironment()` sets it.

    static let noResidencyNextLaunchKey = "bench.ab.noResidencyNextLaunch"
    /// True when this process started with residency disabled (A7 runs only then; A6 refuses).
    nonisolated(unsafe) private(set) static var noResidencyThisLaunch = false

    /// Call first thing at launch (AppDelegate), before any Xybrid model loads.
    static func applyLaunchEnvironment() {
        let d = UserDefaults.standard
        guard d.bool(forKey: noResidencyNextLaunchKey) else { return }
        d.removeObject(forKey: noResidencyNextLaunchKey)   // one launch only
        setenv("GGML_METAL_NO_RESIDENCY", "1", 1)
        noResidencyThisLaunch = true
    }

    /// Bytes on disk for an entry's weights and their path (GGUF file or SDK cache entry).
    static func onDiskBytes(_ entry: ModelCatalog.Entry) -> (bytes: Int64, path: String?) {
        switch entry.xybridSource {
        case .gguf(_, let file, _):
            let url = ModelStore.ggufURL(entryID: entry.id, file: file)
            return (ModelStore.fileSize(url) ?? 0, url.path)
        case .registry(let id):
            prepareSDK()
            guard let path = try? Xybrid.cachedModelPath(id) else { return (0, nil) }
            let url = URL(fileURLWithPath: path)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else { return (0, path) }
            return (isDir.boolValue ? ModelStore.treeBytes(url).bytes : (ModelStore.fileSize(url) ?? 0), path)
        case nil:
            return (0, nil)
        }
    }

    /// Downloads (first use) and loads `entry`. Progress: download 0–0.9, load 0.9–1.
    /// Unloads whatever this backend held first.
    @discardableResult
    func load(_ entry: ModelCatalog.Entry, progress: @escaping @Sendable (Double) -> Void) async throws -> LoadInfo {
        guard entry.engine == .xybrid, let source = entry.xybridSource else {
            throw BackendError.notXybrid(entry.displayName)
        }
        if isLoaded(entry) {
            progress(1)
            return LoadInfo(entryID: entry.id, loadS: 0, downloaded: false, path: nil,
                            localCompleteBefore: true, noResidency: Self.noResidencyThisLaunch)
        }
        await unload()
        Self.prepareSDK()

        var downloaded = false
        var localComplete = false
        var path: String?
        let loader: ModelLoader
        switch source {
        case .registry(let id):
            loader = Xybrid.model(id)
            localComplete = (try? Xybrid.hasCachedModelData(id)) ?? false
            if !localComplete, let download = loader.download() {
                downloaded = true
                try await withTaskCancellationHandler {
                    for await status in download.progress() {
                        progress(0.9 * Double(status.progress))
                        if status.state == .failed { throw BackendError.downloadFailed(download.error() ?? id) }
                        if status.state == .cancelled { throw CancellationError() }
                        if status.state != .downloading { break }
                    }
                } onCancel: {
                    download.cancel()   // ending progress() only unsubscribes; this stops the transfer
                }
                try Task.checkCancellation()
            }
            if let cached = try? Xybrid.cachedModelPath(id) { path = cached }
        case .gguf(let repo, let file, let bytes):
            let url = ModelStore.ggufURL(entryID: entry.id, file: file)
            localComplete = ModelStore.ggufIsComplete(url, expectedBytes: bytes)
            if !localComplete {
                let free = ModelStore.freeBytes
                if free > 0, free < bytes + 1_000_000_000 {
                    throw BackendError.insufficientStorage(needed: bytes, free: free)
                }
                let remote = "https://huggingface.co/\(repo)/resolve/main/\(file)"
                guard let remoteURL = URL(string: remote) else { throw BackendError.badURL(remote) }
                downloaded = true
                try await GGUFDownload.run(from: remoteURL, to: url, expectedBytes: bytes) { f in progress(0.9 * f) }
                ModelStore.excludeFromBackup(ModelStore.modelsRoot)
            }
            path = url.path
            loader = Xybrid.model(.modelFile(url))   // no network; writes model_metadata.json beside it
        }
        progress(0.9)
        try Task.checkCancellation()

        let started = Date()
        let loaded = try await loader.load()
        let loadS = Date().timeIntervalSince(started)
        lock.withLock {
            model = loaded
            loadedEntry = entry
        }
        progress(1)
        return LoadInfo(entryID: entry.id, loadS: loadS, downloaded: downloaded, path: path,
                        localCompleteBefore: localComplete, noResidency: Self.noResidencyThisLaunch)
    }

    /// Drops this backend's reference without touching the native model (safe while a
    /// conversation still runs on it; its weights go when the last reference does).
    func forget() {
        lock.withLock {
            if let model { retired.append(model) }
            model = nil
            loadedEntry = nil
        }
    }

    /// Unloads the model and asks Xybrid to release idle native memory. Call with `GPUGate`
    /// held (no generation running).
    func unload() async {
        let held: [XybridModel] = lock.withLock {
            defer { model = nil; loadedEntry = nil; retired = [] }
            return retired + (model.map { [$0] } ?? [])
        }
        for m in held { try? await m.unloadAsync() }
        // Also frees a model a finished conversation dropped after `forget()` (idle models only).
        if Xybrid.isInitialized { Xybrid.releaseMemory() }
    }

    func makeConversation(instructions: String, tools: [ToolSpecification], maxTokens: Int,
                          temperature: Float) throws -> any ModelConversation {
        guard let model = lock.withLock({ self.model }) else { throw BackendError.notLoaded }
        return try XybridConversation(model: model, instructions: instructions, tools: tools,
                                      maxTokens: maxTokens, temperature: temperature)
    }

    /// One greedy, single-turn run for the benchmark: `prompt` goes through the GGUF's chat
    /// template as a user message; generation stops at `maxTokens` or at end-of-turn.
    func benchmarkRun(prompt: String, maxTokens: Int) async throws -> RunMetrics {
        guard let model = lock.withLock({ self.model }) else { throw BackendError.notLoaded }
        let options = XybridRunOptions.make(
            generationConfig: XybridGenerationConfig.greedy(maxTokens: UInt32(maxTokens)),
            suppressTrace: true)
        let envelope = XybridConversation.message(prompt, role: "user")
        let started = Date()
        let r = try await model.runAsync(envelope: envelope, options: options)
        let wall = Date().timeIntervalSince(started)
        let m = r.metrics
        return RunMetrics(
            wallS: wall, totalMs: Double(m.totalMs), ttftMs: m.ttftMs.map { Double($0) },
            prefillTps: m.prefillTps.map { Double($0) }, decodeTps: m.decodeTps.map { Double($0) },
            tokensPerSecond: m.tokensPerSecond.map { Double($0) },
            tokensOut: m.tokensOut.map { Int($0) } ?? r.tokensOut.map { Int($0) },
            tokensIn: r.tokensIn.map { Int($0) }, text: r.text ?? "")
    }
}

/// A multi-turn chat on a Xybrid model. History lives in a `XybridConversationContext`;
/// per Xybrid's executor contract a turn's input and reply are pushed *after* the run
/// (pushing first duplicates the message).
///
/// Tools: declared natively (`XybridGenerationConfig.tools`); a turn that ends in tool calls
/// yields `.toolCall` items, and `sendToolResults` continues it with
/// `XybridEnvelope.toolResults(userText, priorAssistantText:, results:)`. Limitation: a turn
/// with several tool rounds rebuilds only the latest round (Xybrid's continuation envelope
/// carries one), and tool exchanges are not kept in later turns' history.
final class XybridConversation: ModelConversation, @unchecked Sendable {
    private let model: XybridModel
    private let context: XybridConversationContext
    private let tools: [XybridToolDefinition]
    private let maxTokens: UInt32
    private let temperature: Float
    private let lock = NSLock()
    private var running: Task<Void, Never>?
    /// The user message of the turn in progress (kept across its tool rounds).
    private var turnUserText: String?
    /// Raw output (tool-call block included) of the last round that asked for tools.
    private var lastRawOutput = ""

    init(model: XybridModel, instructions: String, tools: [ToolSpecification], maxTokens: Int,
         temperature: Float) throws {
        self.model = model
        self.context = XybridConversationContext()
        self.maxTokens = UInt32(max(1, maxTokens))
        self.temperature = temperature
        let supportsTools = model.supportsToolCalling() ?? true
        self.tools = supportsTools ? tools.compactMap(XybridConversation.definition) : []
        if !instructions.isEmpty {
            try context.setSystem(envelope: Self.message(instructions, role: "system"))
        }
    }

    /// A text envelope tagged with a chat role (`xybrid.role`: system | user | assistant).
    static func message(_ text: String, role: String) -> XybridEnvelope {
        XybridEnvelope(kind: .text(text: text), metadata: [XybridMetadataEntry(key: "xybrid.role", value: role)])
    }

    private static func definition(_ tool: ToolSpecification) -> XybridToolDefinition? {
        let object = tool.parameters as [String: Any]
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return XybridToolDefinition(name: tool.name, description: tool.description, parametersJson: json)
    }

    /// Tool output must be JSON for Xybrid; plain text is sent as a JSON string.
    private static func jsonContent(_ content: String) -> String {
        if let data = content.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) != nil {
            return content
        }
        if let data = try? JSONEncoder().encode(content), let s = String(data: data, encoding: .utf8) { return s }
        return "\"\""
    }

    private var options: XybridRunOptions {
        XybridRunOptions.make(generationConfig: XybridGenerationConfig.make(
            maxTokens: maxTokens, temperature: temperature, tools: tools))
    }

    func send(_ message: String) -> AsyncThrowingStream<ModelEvent, Error> {
        lock.withLock {
            turnUserText = message
            lastRawOutput = ""
        }
        return run(Self.message(message, role: "user"))
    }

    func sendToolResults(_ results: [ToolResultMessage]) -> AsyncThrowingStream<ModelEvent, Error> {
        let (user, prior) = lock.withLock { (turnUserText ?? "", lastRawOutput) }
        do {
            let envelope = try XybridEnvelope.toolResults(
                user, priorAssistantText: prior,
                results: results.map { r in
                    XybridToolResult(callId: r.callID ?? "", name: r.name, contentJson: Self.jsonContent(r.content))
                })
            return run(envelope)
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
    }

    func synchronize() async {
        let task = lock.withLock { running }
        await task?.value
    }

    /// Streams one round through Xybrid's pull API on a detached task (`streamNext` blocks).
    private func run(_ envelope: XybridEnvelope) -> AsyncThrowingStream<ModelEvent, Error> {
        let model = self.model
        let context = self.context
        let options = self.options
        return AsyncThrowingStream { continuation in
            let cancel = XybridCancellationToken()
            let worker = Task.detached { [self] in
                do {
                    let streamID = try model.runStreamWithContext(
                        envelope: envelope, context: context, options: options, cancel: cancel)
                    defer { model.streamClose(streamId: streamID) }
                    var text = ""
                    var calls: [XybridToolCall] = []
                    var raw: String?
                    pull: while true {
                        if Task.isCancelled {
                            cancel.cancel()
                            throw CancellationError()
                        }
                        let event = try model.streamNext(streamId: streamID)
                        switch event.kind {
                        case .token:
                            guard let token = event.token else { continue }
                            if !token.token.isEmpty { continuation.yield(.text(token.token)) }
                            text = token.cumulativeText
                            if !token.toolCalls.isEmpty {
                                calls = token.toolCalls
                                raw = token.rawText
                            }
                        case .complete, .audio:
                            break pull
                        }
                    }
                    if Task.isCancelled { throw CancellationError() }
                    self.finishRound(text: text, calls: calls, raw: raw)
                    for call in calls {
                        continuation.yield(.toolCall(ToolCallProposal(
                            name: call.name, argumentsJSON: Data(call.argumentsJson.utf8),
                            id: call.id.isEmpty ? nil : call.id)))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            self.lock.withLock { self.running = worker }
            continuation.onTermination = { _ in
                cancel.cancel()   // native stop at the next token boundary
                worker.cancel()
            }
        }
    }

    /// A round that asked for tools keeps the turn open; a final answer closes it and records
    /// the user message and the answer in the history.
    private func finishRound(text: String, calls: [XybridToolCall], raw: String?) {
        if !calls.isEmpty {
            lock.withLock { lastRawOutput = raw ?? text }
            return
        }
        let user: String? = lock.withLock {
            defer { turnUserText = nil; lastRawOutput = "" }
            return turnUserText
        }
        if let user { try? context.push(envelope: Self.message(user, role: "user")) }
        try? context.push(envelope: Self.message(text, role: "assistant"))
    }
}

/// Foreground download of one GGUF into its own directory, with progress and an exact
/// size check (Hugging Face tree sizes are in the catalog). The file is written to the
/// system's temporary location and moved into place only when complete and verified, so a
/// partial file never exists at the destination.
private final class GGUFDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    enum DownloadError: LocalizedError {
        case http(Int)
        case sizeMismatch(expected: Int64, got: Int64)
        var errorDescription: String? {
            switch self {
            case .http(let code): "The server answered HTTP \(code)."
            case .sizeMismatch(let e, let g): "Downloaded \(g) bytes, expected \(e)."
            }
        }
    }

    private let destination: URL
    private let expectedBytes: Int64?
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var finishError: Error?

    private init(destination: URL, expectedBytes: Int64?, progress: @escaping @Sendable (Double) -> Void) {
        self.destination = destination
        self.expectedBytes = expectedBytes
        self.progress = progress
    }

    static func run(from source: URL, to destination: URL, expectedBytes: Int64?,
                    progress: @escaping @Sendable (Double) -> Void) async throws {
        try Task.checkCancellation()
        let delegate = GGUFDownload(destination: destination, expectedBytes: expectedBytes, progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }   // the session retains its delegate until then
        let task = session.downloadTask(with: source)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                delegate.lock.withLock { delegate.continuation = c }
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : (expectedBytes ?? 0)
        if total > 0 { progress(min(1, Double(totalBytesWritten) / Double(total))) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // `location` is deleted when this method returns: verify and move it now.
        do {
            let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
            guard (200..<300).contains(code) else { throw DownloadError.http(code) }
            let size = ModelStore.fileSize(location) ?? 0
            if let expectedBytes, size != expectedBytes {
                throw DownloadError.sizeMismatch(expected: expectedBytes, got: size)
            }
            let fm = FileManager.default
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }   // our own stale/partial copy
            try fm.moveItem(at: location, to: destination)
        } catch {
            lock.withLock { finishError = error }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let (c, moveError): (CheckedContinuation<Void, Error>?, Error?) = lock.withLock {
            defer { continuation = nil }
            return (continuation, finishError)
        }
        if let urlError = error as? URLError, urlError.code == .cancelled {
            c?.resume(throwing: CancellationError())   // a cancelled task, not a failed download
        } else if let failure = error ?? moveError {
            c?.resume(throwing: failure)
        } else {
            c?.resume()
        }
    }
}
