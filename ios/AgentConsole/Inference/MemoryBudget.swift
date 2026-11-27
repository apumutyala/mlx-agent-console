import Foundation
import Metal
import os

/// How much memory this process may still use before iOS terminates it, and whether a
/// catalog entry fits. No MLX import: the backend registers what MLX currently holds.
enum MemoryBudget {
    /// Kept free beyond a model's resident weights: KV cache (~0.4 GB at 8K tokens for
    /// Inkling 2), prefill scratch, Metal/runtime overhead and the app itself.
    static let headroomBytes = 1_610_612_736   // 1.5 GiB

    /// `os_proc_available_memory()`: bytes the app can still allocate before it hits its
    /// jetsam limit (which reflects the increased-memory-limit entitlement when granted).
    /// 0 when the system does not report one.
    static var availableBytes: Int { Int(os_proc_available_memory()) }

    static var physicalBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// Metal's recommended GPU working set for this device, if reported.
    static let gpuWorkingSetBytes: UInt64? = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize

    /// Bytes MLX holds right now (active + buffer cache): what unloading the current model
    /// would free *if nothing else still uses it*. Only the Settings estimate counts it; the
    /// load gate itself doesn't (see `fit(_:includingHeld:)`). Registered by the MLX backend.
    nonisolated(unsafe) static var heldByModel: @Sendable () -> Int = { 0 }

    /// MLX's active/peak/cache bytes, registered by the backend for the diagnostics readout.
    nonisolated(unsafe) static var mlxSnapshot: @Sendable () -> (active: Int, peak: Int, cache: Int) = { (0, 0, 0) }

    /// Settings toggle: load models even when the check says they don't fit.
    static let overrideKey = "memory.ignoreCheck"
    static var ignoreCheck: Bool { UserDefaults.standard.bool(forKey: overrideKey) }

    /// Set while a large model is loading and cleared when it finishes, so a load that
    /// was killed for memory can be reported on the next launch.
    static let pendingLoadKey = "memory.pendingLoad"
    /// A load that was still pending at launch, i.e. the previous run died during it.
    static let killedLoadKey = "memory.killedLoad"
    static let lastLoadReportKey = "memory.lastLoadReport"

    /// Call once per launch, before any load starts.
    static func noteLaunch() {
        let defaults = UserDefaults.standard
        if let pending = defaults.string(forKey: pendingLoadKey) {
            defaults.set(pending, forKey: killedLoadKey)
            defaults.removeObject(forKey: pendingLoadKey)
        }
    }

    enum Fit: Equatable {
        /// Small model, or the system reports no limit.
        case unchecked
        case fits(neededBytes: Int, budgetBytes: Int)
        case tooLarge(neededBytes: Int, budgetBytes: Int)

        var allowsLoad: Bool {
            if case .tooLarge = self { return MemoryBudget.ignoreCheck }
            return true
        }
    }

    /// Whether `entry` fits in what this process can use. `includingHeld` adds what the current
    /// model holds, as if it were unloaded: right for the picker's estimate, wrong for the gate
    /// before a load, where a running chat, fleet task or LoRA round may still hold it.
    static func fit(_ entry: ModelCatalog.Entry, includingHeld: Bool = true) -> Fit {
        guard let resident = entry.residentGB else { return .unchecked }
        let available = availableBytes
        guard available > 0 else { return .unchecked }
        let needed = Int(resident * 1e9) + headroom(entry)
        let budget = available + (includingHeld ? heldByModel() : 0)
        return needed <= budget ? .fits(neededBytes: needed, budgetBytes: budget)
                                : .tooLarge(neededBytes: needed, budgetBytes: budget)
    }

    /// Memory kept free beyond an entry's resident weights: its own estimate when it has one
    /// (LookAtMyQuant builds: no 0.9 GB+ dequantized copies, smaller KV budget), else 1.5 GiB.
    static func headroom(_ entry: ModelCatalog.Entry) -> Int {
        entry.headroomGB.map { Int($0 * 1e9) } ?? headroomBytes
    }

    static func gb(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
    static func gb(_ bytes: UInt64) -> String { gb(Int(clamping: bytes)) }

    enum LoadError: LocalizedError {
        case insufficientMemory(model: String, neededBytes: Int, budgetBytes: Int, stillHeldBytes: Int)
        var errorDescription: String? {
            switch self {
            case .insufficientMemory(let model, let needed, let budget, let held):
                // What iOS lets the whole app use: free memory now plus what MLX already holds.
                let appLimit = budget + held
                var text = "\(model) needs about \(MemoryBudget.gb(needed)) (weights + headroom); "
                    + "iOS allows this app about \(MemoryBudget.gb(appLimit)) and \(MemoryBudget.gb(budget)) is free now."
                if Double(appLimit) < 0.45 * Double(MemoryBudget.physicalBytes) {
                    // ~5.3 GB on a 16 GB iPad: the default jetsam limit, not the raised one.
                    text += " That is iOS's default limit: free Apple ID sideloads don't get the "
                        + "increased-memory-limit entitlement. Use the streamed tier or a smaller model."
                } else if held > 256 * 1024 * 1024 {
                    text += " \(MemoryBudget.gb(held)) is still held by the previous model (a running chat, "
                        + "fleet task or training round); try again when it finishes."
                } else {
                    text += " Pick a smaller model or close other apps."
                }
                return text + " (Settings › Memory has the readout; “Ignore memory check” overrides.)"
            }
        }
    }
}
