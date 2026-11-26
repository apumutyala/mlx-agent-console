import Foundation
import Security

// MARK: - Wire models (match worker/jobworker.py)

struct RemoteJob: Codable, Identifiable, Hashable, Sendable {
    struct Approval: Codable, Hashable, Sendable { let id: String; let prompt: String }
    let id: String
    let type: String
    let params: [String: String]
    let status: String
    let createdAt: Double
    let updatedAt: Double
    let exitCode: Int?
    let retryOf: String?
    let pendingApprovals: [Approval]
    let lastSeq: Int?

    var isTerminal: Bool { ["succeeded", "failed", "cancelled", "lost", "denied"].contains(status) }
}

struct JobEvent: Codable, Identifiable, Hashable, Sendable {
    let seq: Int
    let ts: Double
    let kind: String          // log | status | created | approval_requested | approval_decided | cancel_requested
    let stream: String?
    let line: String?
    let status: String?
    let approvalId: String?
    let prompt: String?
    let decision: String?
    let reason: String?
    var id: Int { seq }

    var displayText: String {
        switch kind {
        case "log": line ?? ""
        case "status": "● \(status ?? "")" + (reason.map { " — \($0)" } ?? "")
        case "approval_requested": "? approval “\(approvalId ?? "")”: \(prompt ?? "")"
        case "approval_decided": "✓ \(approvalId ?? "") → \(decision ?? "")"
        default: "· \(kind)"
        }
    }
}

struct EventPage: Codable, Sendable {
    let events: [JobEvent]
    let cursor: Int
    let status: String
    let more: Bool
}

struct JobTypeInfo: Codable, Hashable, Sendable {
    struct Param: Codable, Hashable, Sendable { let pattern: String; let `default`: String? }
    let name: String
    let description: String
    let params: [String: Param]
    let requiresApproval: Bool
}

// MARK: - Client

struct WorkerConfig: Sendable {
    var baseURL: URL
    var token: String

    static func load() -> WorkerConfig? {
        guard let s = UserDefaults.standard.string(forKey: "worker.baseURL"),
              let url = URL(string: s),
              let token = Keychain.read("worker.token") else { return nil }
        return WorkerConfig(baseURL: url, token: token)
    }

    func save() {
        UserDefaults.standard.set(baseURL.absoluteString, forKey: "worker.baseURL")
        Keychain.write("worker.token", token)
    }
}

enum WorkerError: LocalizedError {
    case notConfigured
    case http(Int, String)
    var errorDescription: String? {
        switch self {
        case .notConfigured: "Set the worker URL and token in Settings."
        case .http(let code, let msg): "Worker returned \(code): \(msg)"
        }
    }
}

/// Stateless HTTP client. Constructed per use from the saved config, so App
/// Intents (which may run without the UI) can use it too.
struct WorkerClient: Sendable {
    let config: WorkerConfig
    private let session: URLSession

    init(config: WorkerConfig) {
        self.config = config
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 40   // > the 25 s long-poll
        c.waitsForConnectivity = false
        session = URLSession(configuration: c)
    }

    static func fromSettings() throws -> WorkerClient {
        guard let cfg = WorkerConfig.load() else { throw WorkerError.notConfigured }
        return WorkerClient(config: cfg)
    }

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private func request<T: Decodable>(_ method: String, _ path: String, query: [URLQueryItem] = [],
                                       body: [String: Any]? = nil) async throws -> T {
        var comps = URLComponents(url: config.baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query }
        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        if let body {
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, resp) = try await session.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let msg = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
            throw WorkerError.http(code, msg ?? "")
        }
        return try Self.decoder.decode(T.self, from: data)
    }

    func jobTypes() async throws -> [JobTypeInfo] {
        struct R: Decodable { let jobTypes: [JobTypeInfo] }
        return try await (request("GET", "job-types") as R).jobTypes
    }

    func jobs() async throws -> [RemoteJob] {
        struct R: Decodable { let jobs: [RemoteJob] }
        return try await (request("GET", "jobs") as R).jobs
    }

    func job(_ id: String) async throws -> RemoteJob { try await request("GET", "jobs/\(id)") }

    func submit(type: String, params: [String: String]) async throws -> RemoteJob {
        try await request("POST", "jobs", body: ["type": type, "params": params])
    }

    func events(_ id: String, after: Int, wait: Int) async throws -> EventPage {
        try await request("GET", "jobs/\(id)/events",
                          query: [.init(name: "after", value: "\(after)"), .init(name: "wait", value: "\(wait)")])
    }

    func cancel(_ id: String) async throws -> RemoteJob { try await request("POST", "jobs/\(id)/cancel") }
    func retry(_ id: String) async throws -> RemoteJob { try await request("POST", "jobs/\(id)/retry") }

    func decide(_ id: String, approval: String, approve: Bool) async throws -> RemoteJob {
        try await request("POST", "jobs/\(id)/approvals/\(approval)", body: ["decision": approve ? "approve" : "deny"])
    }
}

// MARK: - Keychain

enum Keychain {
    private static func query(_ key: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: Bundle.main.bundleIdentifier ?? "AgentConsole",
         kSecAttrAccount as String: key]
    }

    static func write(_ key: String, _ value: String) {
        SecItemDelete(query(key) as CFDictionary)
        var q = query(key)
        q[kSecValueData as String] = Data(value.utf8)
        // Readable after first unlock so Shortcuts/App Intents work with the phone locked.
        q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(q as CFDictionary, nil)
    }

    static func read(_ key: String) -> String? {
        var q = query(key)
        q[kSecReturnData as String] = true
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }
}
