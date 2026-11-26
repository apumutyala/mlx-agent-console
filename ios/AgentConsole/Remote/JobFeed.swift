import Foundation
import Observation

/// Live view of one remote job. The event log is cached on disk together with
/// its cursor, so after a force-quit the app shows what it already had and
/// fetches only `events?after=<cursor>` — the missed logs replay in order.
@MainActor @Observable
final class JobFeed {
    let jobID: String
    private(set) var job: RemoteJob?
    private(set) var events: [JobEvent] = []
    private(set) var cursor = 0
    private(set) var connectionError: String?
    private(set) var isFollowing = false

    private let cacheURL: URL

    init(jobID: String) {
        self.jobID = jobID
        let dir = URL.cachesDirectory.appending(path: "job-events")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        cacheURL = dir.appending(path: "\(jobID).json")
        if let d = try? Data(contentsOf: cacheURL),
           let cached = try? JSONDecoder().decode([JobEvent].self, from: d) {
            events = cached
            cursor = cached.last?.seq ?? 0
        }
    }

    var pendingApprovals: [RemoteJob.Approval] { job?.pendingApprovals ?? [] }

    /// Long-poll until the job finishes or the calling Task is cancelled (view
    /// disappears, app suspends). Safe to call again; it resumes from `cursor`.
    func follow() async {
        isFollowing = true
        defer { isFollowing = false }
        var backoff: UInt64 = 1
        while !Task.isCancelled {
            do {
                let client = try WorkerClient.fromSettings()
                job = try await client.job(jobID)
                let page = try await client.events(jobID, after: cursor, wait: 25)
                apply(page.events)
                connectionError = nil
                backoff = 1
                if page.events.contains(where: { $0.kind != "log" }) {
                    job = try await client.job(jobID)   // status/approvals changed
                }
                if let job, job.isTerminal, !page.more { return }
            } catch is CancellationError {
                return
            } catch let e as URLError where e.code == .cancelled {
                return
            } catch {
                connectionError = error.localizedDescription
                try? await Task.sleep(nanoseconds: backoff * 1_000_000_000)
                backoff = min(backoff * 2, 30)
            }
        }
    }

    private func apply(_ new: [JobEvent]) {
        // The worker guarantees contiguous seqs; drop anything we already hold.
        let fresh = new.filter { $0.seq > cursor }
        guard !fresh.isEmpty else { return }
        events += fresh
        cursor = fresh.last!.seq
        if events.count > 5_000 { events.removeFirst(events.count - 5_000) }
        if let d = try? JSONEncoder().encode(events) { try? d.write(to: cacheURL, options: .atomic) }
    }

    func cancel() async throws { job = try await WorkerClient.fromSettings().cancel(jobID) }
    func retry() async throws -> RemoteJob { try await WorkerClient.fromSettings().retry(jobID) }
    func decide(_ approval: String, approve: Bool) async throws {
        job = try await WorkerClient.fromSettings().decide(jobID, approval: approval, approve: approve)
    }
}
