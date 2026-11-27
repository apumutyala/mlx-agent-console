import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// The only file that touches MLX. Written against mlx-swift-lm 3.33.x
/// (Hugging Face downloader/tokenizer come from the MLXHuggingFace macros).
final class MLXBackend: ModelBackend, @unchecked Sendable {
    private var entry: ModelCatalog.Entry
    private var container: ModelContainer?
    /// Bumped on every switch so a load that finishes late can't install a stale model.
    private var epoch = 0
    private let lock = NSLock()

    init(entry: ModelCatalog.Entry = ModelCatalog.selected) {
        self.entry = entry
    }

    var displayName: String { lock.withLock { entry.displayName } }

    var isLoaded: Bool {
        get async { lock.withLock { container != nil && entry.id == ModelCatalog.selected.id } }
    }

    /// mlx-swift-lm configuration for a catalog entry. Registered entries come
    /// from `LLMRegistry`; the rest are built here. When `toolCallFormat` is nil the
    /// factory resolves it from the model type / chat template at load time.
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
            epoch += 1
            return true
        }
        if changed { MLX.Memory.clearCache() }
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        reconcileSelection()
        if await isLoaded { progress(1); return }
        let (target, startEpoch) = lock.withLock { (entry, epoch) }
        // Keep MLX's buffer cache modest on iPad; weights dominate memory.
        MLX.Memory.cacheLimit = 32 * 1024 * 1024
        let loaded = try await #huggingFaceLoadModelContainer(
            configuration: Self.configuration(for: target),
            progress: .download { p in progress(p.fractionCompleted) }
        )
        lock.withLock {
            if epoch == startEpoch { container = loaded }
        }
    }

    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation {
        let loadedContainer: ModelContainer? = lock.withLock { self.container }
        let current: ModelCatalog.Entry = lock.withLock { self.entry }
        guard let loadedContainer else { throw BackendError.notLoaded }
        var params = GenerateParameters()
        params.maxTokens = current.thinks ? 2048 : 1024   // leave room for <think> before the answer
        params.temperature = current.temperature
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
        let session = ChatSession(loadedContainer, instructions: instructions, generateParameters: params, tools: specs)
        return MLXConversation(session: session)
    }

    enum BackendError: LocalizedError {
        case notLoaded
        var errorDescription: String? { "Model not loaded yet." }
    }
}

private final class MLXConversation: ModelConversation, @unchecked Sendable {
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
