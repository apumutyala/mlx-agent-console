import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Optional semantic half of memory retrieval: LFM2.5-Embedding-350M (4-bit,
/// ~230 MB) via mlx-swift-lm's MLXEmbedders. Loaded lazily and only when the
/// setting is on and the memory policy allows it next to the selected chat
/// model; retrieval falls back to BM25 whenever it isn't loaded.
///
/// Callers must not run it concurrently with LLM generation (one Metal queue,
/// shared MLX state): MemoryHarness only embeds between generations.
actor MemoryEmbedder {
    /// `EmbedderRegistry.lfm2_embedding_350m_4bit` in mlx-swift-lm 3.33.0.
    static let modelID = "mlx-community/LFM2.5-Embedding-350M-4bit"
    /// LFM2.5-Embedding's own prompt conventions (config.json "mlx.prompts").
    static let queryPrefix = "query: "
    static let documentPrefix = "document: "
    /// Memories are short; cap the encoder input anyway.
    static let maxTokens = 512

    enum Role { case query, document }

    private var container: EmbedderModelContainer?
    private var isLoading = false
    /// Bumped by `unload`, so a load that finishes afterwards is discarded.
    private var generation = 0

    var isLoaded: Bool { container != nil }

    /// Whether the embedder may stay resident next to `entry`. Weights of the
    /// chat model + embedder (+ slack for KV cache and activations) must fit in
    /// well under half the device's RAM. On a 16 GB iPad that admits the
    /// 1-5 GB models and excludes ~10 GB ones; on 8 GB only the ~1 GB models.
    static func allowed(alongside entry: ModelCatalog.Entry, deviceGB: Double = ModelCatalog.deviceMemoryGB) -> Bool {
        entry.downloadGB + 0.8 <= deviceGB * 0.45
    }

    /// Downloads (first time) and loads the weights. A concurrent second call
    /// returns immediately; callers check `isLoaded` afterwards.
    func load() async throws {
        guard container == nil, !isLoading else { return }
        isLoading = true
        defer { isLoading = false }
        let started = generation
        // Same store as the chat models (Application Support, local-first: no network
        // call when the snapshot is complete on disk).
        var configuration = EmbedderRegistry.lfm2_embedding_350m_4bit
        let local = ModelStore.localSnapshot(repo: Self.modelID)
        if let local { configuration.id = .directory(local) }
        let loaded = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(ModelStore.hub),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration)
        if local == nil { ModelStore.noteDownloaded(repo: Self.modelID) }
        try Task.checkCancellation()
        if generation == started { container = loaded }
    }

    func unload() {
        generation += 1
        guard container != nil else { return }
        container = nil
        MLX.Memory.clearCache()
    }

    /// L2-normalised CLS vectors, one per text. Throws if not loaded.
    func embed(_ texts: [String], role: Role) async throws -> [[Float]] {
        guard let container else { throw EmbedderError.notLoaded }
        let prefix = role == .query ? Self.queryPrefix : Self.documentPrefix
        let inputs = texts.map { prefix + $0 }
        let maxTokens = Self.maxTokens
        return try await container.perform { context -> [[Float]] in
            var out: [[Float]] = []
            for text in inputs {
                try Task.checkCancellation()
                // BOS is the CLS token for this model, so special tokens are required.
                let ids = Array(context.tokenizer.encode(text: text, addSpecialTokens: true).prefix(maxTokens))
                let output = context.model(MLXArray(ids), positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                guard let hidden = output.hiddenStates else { throw EmbedderError.noOutput }
                // CLS pooling done by hand (position 0), independent of pooling config.
                let cls = hidden[0, 0].asType(.float32)
                eval(cls)
                var v = cls.asArray(Float.self)
                let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
                if norm > 0 { v = v.map { $0 / norm } }
                out.append(v)
            }
            return out
        }
    }

    enum EmbedderError: LocalizedError {
        case notLoaded, noOutput
        var errorDescription: String? {
            switch self {
            case .notLoaded: "Embedding model not loaded."
            case .noOutput: "Embedding model returned no hidden states."
            }
        }
    }
}
