import BackgroundTasks
import Foundation
import Observation

/// Runs a summary batch so it can continue after the user switches apps
/// (iOS/iPadOS 26 `BGContinuedProcessingTask`). This is for finite,
/// user-started work only — the system may still expire it, in which case the
/// batch keeps every completed file and resumes at the next one.
@MainActor @Observable
final class BatchCoordinator {
    nonisolated static let identifierPrefix = (Bundle.main.bundleIdentifier ?? "AgentConsole") + ".summarize"

    private(set) var active: SummaryBatch?
    private(set) var mode: String = ""
    private(set) var lastMessage: String?
    private var foregroundTask: Task<Void, Never>?

    let runner: BatchRunner
    init(runner: BatchRunner) { self.runner = runner }

    /// Whether this device/OS will let MLX keep using the GPU in the background.
    nonisolated static var backgroundGPUAvailable: Bool {
        BGTaskScheduler.supportedResources.contains(.gpu)
    }

    // Must be called before the app finishes launching.
    nonisolated static func register(coordinator: @escaping @MainActor () -> BatchCoordinator?) {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: identifierPrefix + ".*", using: nil) { task in
            guard let task = task as? BGContinuedProcessingTask else {
                task.setTaskCompleted(success: false)
                return
            }
            let batchID = String(task.identifier.dropFirst(identifierPrefix.count + 1))
            Task { @MainActor in
                guard let c = coordinator() else { task.setTaskCompleted(success: false); return }
                await c.runInBackgroundTask(task, batchID: batchID)
            }
        }
    }

    func start(fileIDs: [String]) async {
        let batch = SummaryBatch(id: UUID().uuidString.prefix(8).lowercased(), fileIDs: fileIDs, createdAt: .now)
        try? await runner.store.save(batch)
        await resume(batchID: batch.id)
    }

    func resume(batchID: String) async {
        guard let batch = await runner.store.load(batchID), !batch.isDone else { return }
        active = batch
        // MLX needs Metal. Without background GPU access, running in the
        // background would fail, so stay foreground-only and checkpoint.
        guard Self.backgroundGPUAvailable else {
            mode = "Foreground only (no background GPU on this device)"
            runInForeground(batchID: batchID)
            return
        }
        let req = BGContinuedProcessingTaskRequest(
            identifier: "\(Self.identifierPrefix).\(batchID)",
            title: "Summarizing documents",
            subtitle: "\(batch.completedCount) of \(batch.fileIDs.count)")
        req.strategy = .fail          // start now or tell us, don't queue silently
        req.requiredResources = .gpu
        do {
            try BGTaskScheduler.shared.submit(req)
            mode = "Continued processing (background GPU)"
        } catch {
            lastMessage = "Background start refused (\(error.localizedDescription)); running in foreground."
            mode = "Foreground"
            runInForeground(batchID: batchID)
        }
    }

    /// Called by the app when it moves to the background without a BG task:
    /// stop at the next file boundary; completed files are already saved.
    func pauseForeground() {
        foregroundTask?.cancel()
    }

    private func runInForeground(batchID: String) {
        foregroundTask?.cancel()
        foregroundTask = Task {
            let final = await runner.run(batchID: batchID) { b in Task { @MainActor in self.active = b } }
            finish(final)
        }
    }

    private func runInBackgroundTask(_ task: BGContinuedProcessingTask, batchID: String) async {
        guard let initial = await runner.store.load(batchID) else { task.setTaskCompleted(success: false); return }
        task.progress.totalUnitCount = Int64(initial.fileIDs.count)
        task.progress.completedUnitCount = Int64(initial.completedCount)
        let work = Task {
            await runner.run(batchID: batchID) { b in
                task.progress.completedUnitCount = Int64(b.completedCount)
                task.updateTitle("Summarizing documents", subtitle: "\(b.completedCount) of \(b.fileIDs.count)")
                Task { @MainActor in self.active = b }
            }
        }
        task.expirationHandler = {
            // System is reclaiming time. Cancelling stops at the current file;
            // everything finished so far is already on disk.
            work.cancel()
        }
        let final = await work.value
        task.setTaskCompleted(success: final?.isDone == true)
        finish(final)
    }

    private func finish(_ batch: SummaryBatch?) {
        active = batch
        guard let batch else { return }
        lastMessage = batch.isDone
            ? "Done: \(batch.outputs.count) summarized, \(batch.failures.count) failed."
            : "Paused at \(batch.completedCount) of \(batch.fileIDs.count). Resume to continue from file \(batch.completedCount + 1)."
    }
}
