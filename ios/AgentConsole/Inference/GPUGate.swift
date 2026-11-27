import Foundation

/// One model, one GPU: chat, document summaries and fleet tasks all share the
/// same weights and Metal queue. Running two generations at once would double
/// KV memory and, worse, let a LoRA conversion swap layers under a live chat.
/// This FIFO gate serializes them. It is not reentrant, so hold it only around
/// model work (never across a tool call that may itself summarize).
///
/// `acquire` honours task cancellation: a cancelled waiter leaves the queue and
/// throws `CancellationError` (so Stop, a background pause or a BG-task expiry
/// works even while queued behind a fleet task). Call `release()` only after
/// `acquire` returned normally. Before releasing, wait for any generation you
/// cancelled to actually stop (`ModelConversation.synchronize()`), or the next
/// holder runs concurrently with it.
actor GPUGate {
    static let shared = GPUGate()

    private struct Waiter {
        let id: UUID
        let who: String
        let resume: CheckedContinuation<Void, Error>
    }

    private var busy = false
    private(set) var holder: String?
    private var waiters: [Waiter] = []

    func acquire(_ who: String) async throws {
        try Task.checkCancellation()
        if !busy {
            busy = true
            holder = who
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    c.resume(throwing: CancellationError())
                    return
                }
                waiters.append(Waiter(id: id, who: who, resume: c))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    /// Removes a still-queued waiter. If `release()` already handed it the gate,
    /// there is nothing to remove: its `acquire` returned normally and the
    /// caller owns (and must release) the gate.
    private func cancelWaiter(_ id: UUID) {
        guard let i = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: i).resume.resume(throwing: CancellationError())
    }

    func release() {
        if waiters.isEmpty {
            busy = false
            holder = nil
        } else {
            let next = waiters.removeFirst()
            holder = next.who          // ownership passes directly; `busy` stays true
            next.resume.resume()
        }
    }
}
