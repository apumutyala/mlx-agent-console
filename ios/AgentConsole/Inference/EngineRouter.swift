import Foundation

/// The app's single `ModelBackend`: routes to MLX (default) or to Xybrid's llama.cpp for
/// catalog entries with `engine == .xybrid` (opt-in in the model picker).
///
/// Memory rule: the two engines never hold big weights at once. Loading an MLX model
/// unloads Xybrid first, and selecting a Xybrid model makes MLX drop its container (its
/// entry no longer matches). Loads run with `GPUGate` held by the caller, as before.
final class EngineRouter: ModelBackend, @unchecked Sendable {
    let mlx: MLXBackend
    let xybrid: XybridLLMBackend

    init(mlx: MLXBackend = MLXBackend(), xybrid: XybridLLMBackend = XybridLLMBackend()) {
        self.mlx = mlx
        self.xybrid = xybrid
    }

    private var selected: ModelCatalog.Entry { ModelCatalog.selected }

    var displayName: String {
        let entry = selected
        return entry.engine == .xybrid ? "\(entry.displayName) (Xybrid)" : mlx.displayName
    }

    var isLoaded: Bool {
        get async {
            let entry = selected
            if entry.engine == .xybrid { return xybrid.isLoaded(entry) }
            return await mlx.isLoaded
        }
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        let entry = selected
        if entry.engine == .xybrid {
            await mlx.switchModel(to: entry)   // MLX drops its container: the selection changed engine
            try await xybrid.load(entry, progress: progress)
        } else {
            await xybrid.unload()   // no-op unless Xybrid holds (or recently dropped) a model
            try await mlx.load(progress: progress)
        }
    }

    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation {
        let entry = selected
        guard entry.engine == .xybrid else {
            return try await mlx.makeConversation(instructions: instructions, tools: tools)
        }
        guard xybrid.isLoaded(entry) else { throw XybridLLMBackend.BackendError.notLoaded }
        return try xybrid.makeConversation(instructions: instructions, tools: tools,
                                           maxTokens: entry.thinks ? 2048 : 1024,
                                           temperature: entry.temperature)
    }

    func makeConversation(instructions: String, maxTokens: Int, temperature: Float) async throws -> any ModelConversation {
        let entry = selected
        guard entry.engine == .xybrid else {
            return try await mlx.makeConversation(instructions: instructions, maxTokens: maxTokens,
                                                  temperature: temperature)
        }
        guard xybrid.isLoaded(entry) else { throw XybridLLMBackend.BackendError.notLoaded }
        return try xybrid.makeConversation(instructions: instructions, tools: [], maxTokens: maxTokens,
                                           temperature: temperature)
    }

    func switchModel(to entry: ModelCatalog.Entry) async {
        // MLX persists the selection and drops its container if it no longer matches.
        await mlx.switchModel(to: entry)
        // Xybrid: drop our reference now (a running chat may still use it); the native
        // unload happens on the next load, under the GPU gate.
        if xybrid.loadedEntryID != entry.id { xybrid.forget() }
    }

    /// Before an MLX benchmark or a LoRA round: free Xybrid's weights. Caller holds GPUGate.
    func releaseXybrid() async {
        await xybrid.unload()
    }
}

extension ModelBackend {
    /// The MLX engine behind this backend, for MLX-only work (LoRA, token counts, benchmarks).
    var mlxBackend: MLXBackend? { (self as? MLXBackend) ?? (self as? EngineRouter)?.mlx }
}
