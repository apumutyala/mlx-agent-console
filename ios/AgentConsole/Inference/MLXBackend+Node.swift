import Foundation
import MLXLMCommon

// Extras the compute node needs from MLX. Kept out of MLXBackend.swift so the
// core backend (and its default model) stays small and easy to swap.
extension MLXBackend {
    func makeConversation(instructions: String, maxTokens: Int, temperature: Float) async throws -> any ModelConversation {
        guard let container = modelContainer else { throw BackendError.notLoaded }
        var params = GenerateParameters()
        // Thinking models spend tokens on <think> first; leave room for the answer.
        params.maxTokens = maxTokens + (ModelCatalog.selected.thinks ? 1024 : 0)
        params.temperature = temperature
        // Same memory settings as chat (Inkling 2: 512-token prefill chunks, no KV checkpoints).
        let entry = ModelCatalog.selected
        Self.applyMemorySettings(of: entry, to: &params)
        var extra: [String: any Sendable]? = nil
        if let effort = entry.reasoningEffort { extra = ["reasoning_effort": effort] }
        return MLXConversation(
            session: ChatSession(
                container, instructions: instructions, generateParameters: params, additionalContext: extra),
            contextTokens: entry.contextTokens, instructions: instructions)
    }

    /// Exact token count (stream chunks are only roughly one token each).
    func tokenCount(of text: String) async -> Int? {
        guard let container = modelContainer else { return nil }
        return await container.encode(text).count
    }
}
