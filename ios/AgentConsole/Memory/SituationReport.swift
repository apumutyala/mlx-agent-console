import Foundation
import UIKit

/// A compact "what's going on right now" block for the system prompt: time,
/// device, loaded model, imported documents, worker jobs and fleet nodes.
/// Network parts are best-effort with a short timeout and a cache, and are
/// simply omitted (or reported unreachable) when offline.
@MainActor
final class SituationReporter {
    struct RemoteSnapshot: Sendable {
        var worker: String?
        var fleet: String?
        let fetchedAt: Date
    }

    nonisolated static let networkTimeout: TimeInterval = 3
    nonisolated static let cacheLifetime: TimeInterval = 60
    private var cache: RemoteSnapshot?

    func report(modelName: String, documents: [ImportedDocument], budgetTokens: Int = 260) async -> String {
        var lines: [String] = []
        let now = Date.now
        let f = DateFormatter()
        f.dateFormat = "EEEE d MMMM yyyy, HH:mm zzz"
        lines.append("Now: \(f.string(from: now))")
        let device = UIDevice.current
        let ram = Int(ModelCatalog.deviceMemoryGB.rounded())
        lines.append("Device: \(device.model), \(device.systemName) \(device.systemVersion), \(ram) GB RAM")
        lines.append("Model: \(modelName) (on-device, MLX)")
        if documents.isEmpty {
            lines.append("Imported documents: none")
        } else {
            let names = documents.suffix(12).map { "\($0.id) \($0.name)" }.joined(separator: "; ")
            lines.append("Imported documents (\(documents.count)): \(names)")
        }
        let remote = await remoteSnapshot()
        if let w = remote.worker { lines.append(w) }
        if let n = remote.fleet { lines.append(n) }
        return TokenEstimate.truncate(lines.joined(separator: "\n"), toTokens: budgetTokens)
    }

    func invalidate() { cache = nil }

    private func remoteSnapshot() async -> RemoteSnapshot {
        if let cache, Date.now.timeIntervalSince(cache.fetchedAt) < Self.cacheLifetime { return cache }
        guard let config = WorkerConfig.load() else {
            let s = RemoteSnapshot(worker: "Worker: not configured", fleet: nil, fetchedAt: .now)
            cache = s
            return s
        }
        async let worker = Self.workerLine(config)
        async let fleet = Self.fleetLine(config)
        let s = RemoteSnapshot(worker: await worker, fleet: await fleet, fetchedAt: .now)
        cache = s
        return s
    }

    /// Remote strings are untrusted: only short, cleaned fields go in.
    nonisolated static func field(_ s: String, max: Int = 32) -> String {
        let clean = MemoryText.clean(s, maxLength: max) ?? "?"
        return MemoryText.looksLikeControlMarkup(clean) ? "?" : clean.replacingOccurrences(of: "\n", with: " ")
    }

    nonisolated private static func workerLine(_ config: WorkerConfig) async -> String {
        do {
            let jobs = try await withTimeout(networkTimeout) { try await WorkerClient(config: config).jobs() }
            guard !jobs.isEmpty else { return "Worker: online, no jobs yet" }
            let now = Date.now.timeIntervalSince1970
            let recent = jobs.prefix(5).map { j -> String in
                let mins = max(Int((now - j.updatedAt) / 60), 0)
                let approvals = j.pendingApprovals.isEmpty ? "" : ", awaiting approval"
                return "\(field(j.id, max: 12)) \(field(j.type)) \(field(j.status, max: 12))\(approvals), \(mins)m ago"
            }
            let running = jobs.filter { !$0.isTerminal }.count
            return "Worker: online, \(running) active; recent jobs: " + recent.joined(separator: "; ")
        } catch {
            return "Worker: unreachable"
        }
    }

    /// `GET /nodes` on the worker (compute fleet). Decoded loosely so this
    /// doesn't depend on the fleet client's types.
    nonisolated private static func fleetLine(_ config: WorkerConfig) async -> String? {
        var req = URLRequest(url: config.baseURL.appending(path: "nodes"), timeoutInterval: networkTimeout)
        req.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        guard let reply = try? await URLSession.shared.data(for: req),
              (reply.1 as? HTTPURLResponse)?.statusCode == 200,
              let obj = try? JSONSerialization.jsonObject(with: reply.0) as? [String: Any],
              let nodes = obj["nodes"] as? [[String: Any]] else { return nil }
        guard !nodes.isEmpty else { return "Fleet: no nodes registered" }
        let online = nodes.filter { ($0["online"] as? Bool) == true }
        let names = online.prefix(6).map { n in field((n["name"] as? String) ?? (n["id"] as? String) ?? "?", max: 24) }
        return "Fleet: \(online.count) of \(nodes.count) nodes online" + (names.isEmpty ? "" : " (\(names.joined(separator: ", ")))")
    }
}

struct TimeoutError: Error {}

/// Runs `body`, throwing `TimeoutError` if it takes longer than `seconds`.
func withTimeout<T: Sendable>(_ seconds: TimeInterval, _ body: @escaping @Sendable () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw TimeoutError()
        }
        defer { group.cancelAll() }
        guard let first = try await group.next() else { throw TimeoutError() }
        return first
    }
}
