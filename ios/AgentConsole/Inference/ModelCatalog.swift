import Foundation

/// The on-device models the app offers. Plain data, no MLX import: MLXBackend
/// maps each entry's `id` to an mlx-swift-lm `ModelConfiguration`.
enum ModelCatalog {
    struct Entry: Identifiable, Hashable, Sendable {
        /// Which mlx-swift-lm factory loads the weights.
        enum Loader: Hashable, Sendable {
            /// `LLMModelFactory` (text-only architectures).
            case llm
            /// `VLMModelFactory` (Inkling 2: `inkling2_loop`, vision tower loads on first image).
            case vlm
        }

        /// Stable key persisted in UserDefaults (never rename a shipped one).
        let id: String
        let displayName: String
        /// Hugging Face repo the weights come from.
        let repo: String
        /// Approximate one-time download, in GB (weights + tokenizer).
        let downloadGB: Double
        /// Short hint shown under the picker.
        let note: String
        /// Devices with less RAM than this don't get the entry at all.
        let minimumMemoryGB: Double
        /// Emits reasoning before answering.
        let thinks: Bool
        let temperature: Float
        var loader: Loader = .llm
        /// Weights that stay resident once loaded, in GB (10^9 bytes). Only set for
        /// models big enough to need the runtime memory check (see `MemoryBudget`).
        var residentGB: Double? = nil
        /// KV-cache capacity in tokens (`maxKVSize`); nil = unbounded.
        var contextTokens: Int? = nil
        /// Prompt tokens per prefill chunk; smaller = lower peak memory, slower prefill.
        var prefillStepSize: Int? = nil
        /// Chat-template `reasoning_effort` (Inkling: none/minimal/low/medium/high/xhigh/max).
        var reasoningEffort: String? = nil
        /// Set for an SSD-streamed build: the routed experts stay in `experts.bin` and only a
        /// fraction of them is resident. Downloaded from `StreamedModelStore`, not Hugging Face.
        var streamed: StreamedBuild? = nil

        var sizeLabel: String {
            downloadGB < 1 ? "\(Int((downloadGB * 1000).rounded())) MB" : String(format: "%.1f GB", downloadGB)
        }
        var isSupportedOnThisDevice: Bool { ModelCatalog.deviceMemoryGB >= minimumMemoryGB }
    }

    static let lfm3_1_7b = Entry(
        id: "lfm3-1.7b", displayName: "LFM3 1.7B", repo: "LiquidAI/LFM3-1.7B-MLX-4bit",
        downloadGB: 0.99, note: "Liquid AI's newest small model. Best balance (default).",
        minimumMemoryGB: 0, thinks: true, temperature: 0.3)
    static let lfm25_1_2b = Entry(
        id: "lfm2.5-1.2b-instruct", displayName: "LFM2.5 1.2B Instruct", repo: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        downloadGB: 0.66, note: "Fast instruct model, no thinking.",
        minimumMemoryGB: 0, thinks: false, temperature: 0.3)
    static let lfm25_350m = Entry(
        id: "lfm2.5-350m", displayName: "LFM2.5 350M", repo: "LiquidAI/LFM2.5-350M-MLX-4bit",
        downloadGB: 0.23, note: "Tiny and very fast; weakest at tool use.",
        minimumMemoryGB: 0, thinks: false, temperature: 0.3)
    static let lfm25_8b_a1b = Entry(
        id: "lfm2.5-8b-a1b", displayName: "LFM2.5 8B-A1B (MoE)", repo: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
        downloadGB: 4.85, note: "Strongest; mixture-of-experts with ~1.5B active. 16 GB iPads only.",
        minimumMemoryGB: 14, thinks: true, temperature: 0.2)
    static let qwen3_1_7b = Entry(
        id: "qwen3-1.7b", displayName: "Qwen3 1.7B", repo: "mlx-community/Qwen3-1.7B-4bit",
        downloadGB: 0.98, note: "Fallback (the previous default).",
        minimumMemoryGB: 0, thinks: true, temperature: 0.4)

    // MARK: Inkling 2 Mini (Thinking Machines, 26.4B looped MoE, ~3.4B active/token)
    //
    // Only conversions in the mlx-lm #2086 tensor layout (`model.language_model.layers.N.mlp.experts.
    // {gate_up_proj,down_proj}`) load through mlx-swift-lm PR #821. mlx-vlm-layout repos
    // (`language_model.model...switch_mlp`, e.g. netprojects LayerFit, VisualInference) and oQ/JANG
    // formats do not. `residentGB` is the text-model tensor bytes from the safetensors headers
    // (vision tower excluded: it loads only on the first image; DSpark draft never loaded).
    // Adding a build = one `inkling(...)` line plus an entry in `all`.

    /// Shared settings for every Inkling 2 Mini build.
    static func inkling(id: String, name: String, repo: String, downloadGB: Double, residentGB: Double,
                        note: String, minimumMemoryGB: Double = 14) -> Entry {
        Entry(id: id, displayName: name, repo: repo, downloadGB: downloadGB, note: note,
              minimumMemoryGB: minimumMemoryGB, thinks: true, temperature: 0.6, loader: .vlm,
              residentGB: residentGB, contextTokens: 8192, prefillStepSize: 512, reasoningEffort: "low")
    }

    /// Default Inkling tier: 3.10 bpw mixed precision + DWQ (experts 2/3-bit, attention 6/8-bit).
    /// Card: KL 0.18 vs the native 19.6 GB release (tool calling 0.22), top-1 agreement 87.3%.
    static let inkling2MiniDWQ = inkling(
        id: "inkling2-mini-dwq3.1", name: "Inkling 2 Mini · 3.1-bit DWQ",
        repo: "LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ", downloadGB: 10.3, residentGB: 9.60,
        note: "Thinking Machines' 26B MoE (3.4B active). 3.1-bit distilled build sized for 16 GB devices.")
    /// GPTQ 8/2/3 (attention 8, experts gate/up 2, down 3). Card: +12.1% text / +9.3% code perplexity.
    static let inkling2MiniGPTQSmall = inkling(
        id: "inkling2-mini-gptq-small", name: "Inkling 2 Mini · GPTQ 2/3-bit",
        repo: "roman220220/Inkling-2-Mini-gptq-mlx-jang-small", downloadGB: 11.5, residentGB: 10.45,
        note: "GPTQ-calibrated alternative to the DWQ build; slightly larger.")
    /// GPTQ 8/3/3. Card: +3.4% text / +2.4% code perplexity. Needs ~14 GB free; on a 16 GB iPad
    /// this only fits if the per-app limit is unusually high.
    static let inkling2MiniGPTQ3 = inkling(
        id: "inkling2-mini-gptq3", name: "Inkling 2 Mini · GPTQ 3-bit",
        repo: "roman220220/Inkling-2-Mini-gptq-mlx-jang", downloadGB: 13.5, residentGB: 12.49,
        note: "Highest-quality Inkling build that might fit; needs a very high per-app memory limit.")

    // MARK: Inkling 2 Mini, SSD expert streaming
    //
    // The same DWQ weights, repacked offline by `tools/expert_repack.py` into core shards plus
    // `experts.bin` (layer-major 16 KiB-aligned expert records) and `experts.json`. The repack is
    // byte-exact, so quality equals the resident DWQ tier; only speed and memory change. Each layer
    // keeps `residentFraction` of its 128 experts in memory and reads misses from flash.
    // Sizes from the DWQ safetensors headers: text core 1.357 GB, routed experts 8.239 GB,
    // experts.bin 8.259 GB (record padding), vision 0.637 GB (loads on the first image).

    /// An SSD-streamed build: where it is hosted and how much of it stays resident.
    struct StreamedBuild: Hashable, Sendable {
        /// The directory name under the streamed-models URL (the repack output directory).
        let name: String
        /// Fraction of each layer's experts kept in memory (`S = round(f * 128)` slots).
        let residentFraction: Double
    }

    static let dwqStreamName = "Inkling-2-Mini-MLX-3.1bpw-DWQ-stream"
    static let dwqTextCoreGB = 1.357
    static let dwqExpertsGB = 8.239

    static func inklingStreamed(id: String, name: String, fraction: Double, minimumMemoryGB: Double,
                                note: String) -> Entry {
        var entry = inkling(
            id: id, name: name, repo: "LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ",
            downloadGB: 10.3, residentGB: dwqTextCoreGB + fraction * dwqExpertsGB, note: note,
            minimumMemoryGB: minimumMemoryGB)
        entry.streamed = StreamedBuild(name: dwqStreamName, residentFraction: fraction)
        return entry
    }

    // Budget on the free-sideload iPad (device receipt 2026-11-27): per-app limit ≈ 5.3–5.5 GB,
    // target footprint ≤ 4.6 GB. One slot across all 24 layers costs 64.36 MB. With the 1.36 GB core,
    // KV ≈ 0.2 GB at 4K tokens, prefill scratch ≈ 0.25 GB at 256-token steps and ≈ 0.2 GB runtime,
    // 32 slots (2.06 GB) come to ≈ 4.1 GB and 24 slots (1.54 GB) to ≈ 3.6 GB. 8K context adds
    // ≈ 0.2 GB, which still fits at 32 slots (≈ 4.3 GB).

    /// 32 of 128 experts per layer resident (3.42 GB weights): the largest tier that fits the
    /// ~5.3 GB free-sideload limit with headroom. 4K context keeps KV small; 512-token prefill steps
    /// halve TTFT vs 256 (a streamed prefill re-reads ~96 experts per visit per step regardless of
    /// the step length) for ~0.1 GB more attention scratch.
    static let inkling2MiniStream25: Entry = {
        var e = inklingStreamed(
            id: "inkling2-mini-dwq3.1-stream25", name: "Inkling 2 Mini · streamed 25%", fraction: 0.25,
            minimumMemoryGB: 7,
            note: "The 3.1-bit DWQ build with 25% of experts in memory, the rest read from storage. "
                + "Fits the default per-app memory limit; experimental.")
        e.contextTokens = 4096
        e.prefillStepSize = 512
        return e
    }()
    /// 24 of 128 (2.90 GB weights): fallback if the 25% tier is killed for memory.
    static let inkling2MiniStream19: Entry = {
        var e = inklingStreamed(
            id: "inkling2-mini-dwq3.1-stream19", name: "Inkling 2 Mini · streamed 19%", fraction: 0.1875,
            minimumMemoryGB: 7,
            note: "Like the 25% tier with fewer experts in memory: ~0.5 GB smaller, slower.")
        e.contextTokens = 4096
        e.prefillStepSize = 512
        return e
    }()
    /// 64 of 128 resident: ~5.5 GB weights. Needs the increased-memory-limit entitlement.
    static let inkling2MiniStream50 = inklingStreamed(
        id: "inkling2-mini-dwq3.1-stream50", name: "Inkling 2 Mini · streamed 50%", fraction: 0.5,
        minimumMemoryGB: 7,
        note: "The 3.1-bit DWQ build with half of its experts in memory. Experimental.")
    /// Every expert resident, through the streaming path: the exactness gate against the
    /// resident DWQ tier (must greedy-match it). Same memory as the resident build.
    static let inkling2MiniStream100 = inklingStreamed(
        id: "inkling2-mini-dwq3.1-stream100", name: "Inkling 2 Mini · streamed 100% (check)",
        fraction: 1.0, minimumMemoryGB: 14,
        note: "Diagnostic: the streamed build with every expert resident. Output must match the "
            + "resident DWQ build exactly.")

    static let inklingTiers: [Entry] = [inkling2MiniDWQ, inkling2MiniGPTQSmall, inkling2MiniGPTQ3,
                                        inkling2MiniStream25, inkling2MiniStream19, inkling2MiniStream50,
                                        inkling2MiniStream100]

    // MARK: LFM2.5-8B-A1B, SSD expert streaming
    //
    // `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` @ 2e92b640, repacked by tools/expert_repack.py (switch_glu
    // layout). From its header: 22 MoE layers × 32 experts of 6.19 MB (4-bit, group 64) = 4.360 GB of
    // experts, 0.474 GB everything else. One slot across the 22 layers costs 136.25 MB. Footprint ≈
    // core + S × 136 MB + KV (hybrid: 6 attention layers, ~0.1 GB at 8K) + ~0.3 GB prefill scratch at
    // 256-token steps + ~0.2 GB runtime: S = 20 ≈ 3.8 GB, S = 24 ≈ 4.3 GB (the 4.6 GB target).

    static let lfm8bStreamName = "LFM2.5-8B-A1B-MLX-4bit-stream"
    static let lfm8bCoreGB = 0.474
    static let lfm8bExpertsGB = 4.360

    static func lfm8bStreamed(id: String, name: String, fraction: Double, note: String) -> Entry {
        var e = Entry(
            id: id, displayName: name, repo: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", downloadGB: 4.85, note: note,
            minimumMemoryGB: 7, thinks: true, temperature: 0.2, loader: .llm,
            residentGB: lfm8bCoreGB + fraction * lfm8bExpertsGB, contextTokens: 8192, prefillStepSize: 256)
        e.streamed = StreamedBuild(name: lfm8bStreamName, residentFraction: fraction)
        return e
    }

    /// 24 of 32 experts per layer resident (3.74 GB weights).
    static let lfm25_8b_a1b_stream75 = lfm8bStreamed(
        id: "lfm2.5-8b-a1b-stream75", name: "LFM2.5 8B-A1B (MoE) · streamed 75%", fraction: 0.75,
        note: "The 8B MoE with three quarters of its experts in memory and the rest read from storage. "
            + "Fits the default per-app memory limit; experimental.")
    /// 20 of 32 (3.20 GB weights): more headroom, more misses.
    static let lfm25_8b_a1b_stream62 = lfm8bStreamed(
        id: "lfm2.5-8b-a1b-stream62", name: "LFM2.5 8B-A1B (MoE) · streamed 62%", fraction: 0.625,
        note: "Like the 75% tier with fewer experts in memory: ~0.5 GB smaller, slower.")

    static let lfmStreamTiers: [Entry] = [lfm25_8b_a1b_stream75, lfm25_8b_a1b_stream62]

    static let all: [Entry] = [lfm3_1_7b, lfm25_1_2b, lfm25_350m, lfm25_8b_a1b, qwen3_1_7b]
        + lfmStreamTiers + inklingTiers
    static let defaultEntry = lfm3_1_7b

    /// Entries this device can run, in display order.
    static var available: [Entry] { all.filter(\.isSupportedOnThisDevice) }

    /// Physical RAM in GB (a "16 GB" iPad reports slightly under 16).
    static var deviceMemoryGB: Double { Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 }

    static let storageKey = "model.selected"

    /// The persisted choice, falling back to the default when unset or not
    /// runnable here (e.g. a backup restored onto an 8 GB iPad).
    static var selected: Entry {
        get {
            let id = UserDefaults.standard.string(forKey: storageKey)
            return available.first { $0.id == id } ?? defaultEntry
        }
        set { UserDefaults.standard.set(newValue.id, forKey: storageKey) }
    }

    static func entry(id: String) -> Entry? { all.first { $0.id == id } }
}
