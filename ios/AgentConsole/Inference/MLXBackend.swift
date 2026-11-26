import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The only file that touches MLX. If mlx-swift-lm's API moves, fix it here.
/// Mirrors the loading path in Apple's LLMEval example.
final class MLXBackend: ModelBackend, @unchecked Sendable {
    private let configuration: ModelConfiguration
    private var container: ModelContainer?
    private let lock = NSLock()

    init(configuration: ModelConfiguration = LLMRegistry.qwen3_1_7b_4bit) {
        self.configuration = configuration
    }

    var displayName: String { configuration.name }

    var isLoaded: Bool { get async { lock.withLock { container != nil } } }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        if await isLoaded { progress(1); return }
        // Keep MLX's buffer cache modest on phones; weights dominate memory.
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
        let loaded = try await LLMModelFactory.shared.loadContainer(configuration: configuration) { p in
            progress(p.fractionCompleted)
        }
        lock.withLock { container = loaded }
    }

    func makeConversation(instructions: String) async throws -> any ModelConversation {
        guard let container = lock.withLock({ container }) else {
            throw BackendError.notLoaded
        }
        let session = ChatSession(
            container,
            instructions: instructions,
            generateParameters: GenerateParameters(maxTokens: 1024, temperature: 0.4)
        )
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

    func send(_ message: String) -> AsyncThrowingStream<String, Error> {
        session.streamResponse(to: message)
    }
}
