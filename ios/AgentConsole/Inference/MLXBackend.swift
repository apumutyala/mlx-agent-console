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
    private let configuration: ModelConfiguration
    private var container: ModelContainer?
    private let lock = NSLock()

    init(configuration: ModelConfiguration = LLMRegistry.qwen3_1_7b_4bit) {
        self.configuration = configuration
    }

    var displayName: String { configuration.name }

    /// Hugging Face repo id of the configured model (federated rounds match on it).
    var modelID: String { configuration.name }

    /// The loaded container, for MLX-level work (LoRA training, token counts)
    /// that lives in MLX-only files. nil until `load` finishes.
    var modelContainer: ModelContainer? { lock.withLock { container } }

    var isLoaded: Bool { get async { lock.withLock { container != nil } } }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        if await isLoaded { progress(1); return }
        // Keep MLX's buffer cache modest on iPad; weights dominate memory.
        MLX.Memory.cacheLimit = 32 * 1024 * 1024
        let loaded = try await #huggingFaceLoadModelContainer(
            configuration: configuration,
            progress: .download { p in progress(p.fractionCompleted) }
        )
        lock.withLock { container = loaded }
    }

    func makeConversation(instructions: String) async throws -> any ModelConversation {
        guard let container = lock.withLock({ container }) else {
            throw BackendError.notLoaded
        }
        var params = GenerateParameters()
        params.maxTokens = 1024
        params.temperature = 0.4
        let session = ChatSession(container, instructions: instructions, generateParameters: params)
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

    func send(_ message: String) -> AsyncThrowingStream<String, Error> {
        session.streamResponse(to: message)
    }
}
