import Foundation

/// One model, one GPU: chat, document summaries and fleet tasks all share the
/// same weights and Metal queue. Running two generations at once would double
/// KV memory and, worse, let a LoRA conversion swap layers under a live chat.
/// This FIFO gate serializes them. It is not reentrant, so hold it only around
/// model work (never across a tool call that may itself summarize).
actor GPUGate {
    static let shared = GPUGate()

    private var busy = false
    private(set) var holder: String?
    private var waiters: [(who: String, resume: CheckedContinuation<Void, Never>)] = []

    func acquire(_ who: String) async {
        if !busy {
            busy = true
            holder = who
            return
        }
        await withCheckedContinuation { waiters.append((who, $0)) }
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
