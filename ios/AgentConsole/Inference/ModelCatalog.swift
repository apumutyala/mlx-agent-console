import Foundation

/// The on-device models the app offers. Plain data, no MLX import: MLXBackend
/// maps each entry's `id` to an mlx-swift-lm `ModelConfiguration`.
enum ModelCatalog {
    struct Entry: Identifiable, Hashable, Sendable {
        /// Stable key persisted in UserDefaults (never rename a shipped one).
        let id: String
        let displayName: String
        /// Hugging Face repo the weights come from.
        let repo: String
        /// Approximate one-time download, in GB (4-bit weights + tokenizer).
        let downloadGB: Double
        /// Short hint shown under the picker.
        let note: String
        /// Devices with less RAM than this don't get the entry at all.
        let minimumMemoryGB: Double
        /// Emits `<think>` reasoning before answering.
        let thinks: Bool
        let temperature: Float

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

    static let all: [Entry] = [lfm3_1_7b, lfm25_1_2b, lfm25_350m, lfm25_8b_a1b, qwen3_1_7b]
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
