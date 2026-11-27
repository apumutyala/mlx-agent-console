import Foundation

/// Everything the app needs from a language model. The rest of the app never
/// imports MLX directly, so the backend can be swapped (another MLX model, a
/// stub for previews/tests, or a remote model) without touching the harness.
protocol ModelBackend: AnyObject, Sendable {
    var displayName: String { get }
    var isLoaded: Bool { get async }

    /// Download (first run) and load weights. `progress` is 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws

    /// Start a fresh conversation. Each conversation owns its own KV cache.
    func makeConversation(instructions: String) async throws -> any ModelConversation

    /// Same, with per-conversation sampling (fleet `generate` tasks set these).
    func makeConversation(instructions: String, maxTokens: Int, temperature: Float) async throws -> any ModelConversation
}

extension ModelBackend {
    // Backends without tunable sampling just ignore the knobs.
    func makeConversation(instructions: String, maxTokens: Int, temperature: Float) async throws -> any ModelConversation {
        try await makeConversation(instructions: instructions)
    }
}

protocol ModelConversation: AnyObject, Sendable {
    /// Stream the reply to one user/tool message, chunk by chunk.
    func send(_ message: String) -> AsyncThrowingStream<String, Error>
}

extension ModelConversation {
    func complete(_ message: String) async throws -> String {
        var out = ""
        for try await chunk in send(message) {
            try Task.checkCancellation()
            out += chunk
        }
        return out
    }
}

/// Deterministic backend for SwiftUI previews and logic tests on the simulator.
final class EchoBackend: ModelBackend, ModelConversation, @unchecked Sendable {
    let displayName = "Echo (no model)"
    var isLoaded: Bool { get async { true } }
    func load(progress: @escaping @Sendable (Double) -> Void) async throws { progress(1) }
    func makeConversation(instructions: String) async throws -> any ModelConversation { self }
    func send(_ message: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { c in
            c.yield("Echo: " + message.prefix(200))
            c.finish()
        }
    }
}
