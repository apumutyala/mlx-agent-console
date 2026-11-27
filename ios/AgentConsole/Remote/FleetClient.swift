import Foundation

// MARK: - Wire models (match worker/fleet.py and worker/federated.py)

struct NodeInfo: Encodable, Sendable {
    struct Capabilities: Encodable, Sendable {
        let kinds: [String]
        let model: String
        let memoryGB: Double
        let os: String
        /// Whether train_lora tasks may use this device's imported documents
        /// (`use_local_data`). Off unless the user opted in on the device.
        var trainOnDocuments = false
        enum CodingKeys: String, CodingKey {
            case kinds, model, memoryGB = "memory_gb", os, trainOnDocuments = "train_on_documents"
        }
    }
    let name: String
    let kind: String
    let capabilities: Capabilities
}

/// A task as handed out by `GET nodes/{id}/next`. The payload shape depends on
/// `kind`, so the same response is decoded twice: once with `Empty` to route,
/// then with the kind's payload type.
struct FleetTask<Payload: Decodable>: Decodable {
    let id: String
    let kind: String
    let payload: Payload
}

struct EmptyPayload: Decodable {}

/// `{"task": {...} | null}` (top level: local types can't nest in generic functions).
struct NextTaskResponse<Payload: Decodable>: Decodable {
    let task: FleetTask<Payload>?
}

struct FleetEvent: Encodable, Sendable {
    let kind: String              // token | progress | log
    var text: String?
    var fraction: Double?
    var message: String?

    static func token(_ t: String) -> FleetEvent { FleetEvent(kind: "token", text: t) }
    static func progress(_ f: Double, _ m: String) -> FleetEvent { FleetEvent(kind: "progress", fraction: f, message: m) }
    static func log(_ t: String) -> FleetEvent { FleetEvent(kind: "log", text: t) }
}

struct TaskResult: Codable, Sendable {
    var status: String            // succeeded | failed
    var output: String?
    var error: String?
    var metrics: [String: Double] = [:]

    static func failure(_ message: String) -> TaskResult { TaskResult(status: "failed", output: nil, error: message) }
}

struct FedRound: Decodable, Sendable {
    struct Hyperparameters: Decodable, Sendable {
        let rank: Int
        let layers: Int
    }
    let id: String
    let status: String
    let baseModel: String
    let hyperparameters: Hyperparameters
    let seed: Int?
}

// MARK: - Client

/// Node-side half of the fleet protocol. The node only ever makes outbound
/// requests (heartbeat, long-poll, events, artifacts), so it works behind NAT
/// and within iOS's foreground networking rules.
struct FleetClient: Sendable {
    let config: WorkerConfig
    private var session: URLSession { Self.session }

    /// One session for every FleetClient: the timeouts don't depend on the
    /// worker config, and a session per client leaked one per loop pass.
    private static let session: URLSession = {
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 45      // > the 25 s long-poll
        c.timeoutIntervalForResource = 600    // adapter uploads/downloads
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()

    init(config: WorkerConfig) {
        self.config = config
    }

    static func fromSettings() throws -> FleetClient {
        guard let cfg = WorkerConfig.load() else { throw WorkerError.notConfigured }
        return FleetClient(config: cfg)
    }

    /// True when `other` points at the same worker with the same token.
    func matches(_ other: WorkerConfig) -> Bool {
        config.baseURL == other.baseURL && config.token == other.token
    }

    /// The worker already settled this task (404 unknown, 409 no longer ours):
    /// resending a completion can't help.
    static func isSettled(_ error: Error) -> Bool {
        if let w = error as? WorkerError, case .http(let code, _) = w { return code == 404 || code == 409 }
        return false
    }

    /// A rejection that retrying won't change (4xx other than timeout/rate limit),
    /// or a body that can't be encoded.
    static func isPermanent(_ error: Error) -> Bool {
        if error is EncodingError { return true }
        if let w = error as? WorkerError, case .http(let code, _) = w {
            return (400..<500).contains(code) && code != 408 && code != 429
        }
        return false
    }

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    @discardableResult
    private func send(_ method: String, _ path: String, query: [URLQueryItem] = [],
                      json: Data? = nil, raw: Data? = nil) async throws -> Data {
        var comps = URLComponents(url: config.baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query }
        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        if let json {
            req.httpBody = json
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        } else if let raw {
            req.httpBody = raw
            req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        }
        let (data, resp) = try await session.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let msg = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
            throw WorkerError.http(code, msg ?? "")
        }
        return data
    }

    func heartbeat(nodeID: String, info: NodeInfo) async throws {
        try await send("POST", "nodes/\(nodeID)/heartbeat", json: try JSONEncoder().encode(info))
    }

    /// Long-polls for work. Returns the raw response (decode with `decodeTask`)
    /// or nil when the wait elapsed with nothing queued for this node.
    func nextTask(nodeID: String, wait: Int = 25) async throws -> Data? {
        let data = try await send("GET", "nodes/\(nodeID)/next", query: [.init(name: "wait", value: "\(wait)")])
        return try Self.decoder.decode(NextTaskResponse<EmptyPayload>.self, from: data).task == nil ? nil : data
    }

    static func decodeTask<P: Decodable>(_ data: Data, payload: P.Type) throws -> FleetTask<P> {
        guard let task = try decoder.decode(NextTaskResponse<P>.self, from: data).task else {
            throw WorkerError.http(0, "response has no task")
        }
        return task
    }

    /// Returns true when the requester asked to cancel this task.
    func postEvents(taskID: String, nodeID: String, events: [FleetEvent]) async throws -> Bool {
        struct Body: Encodable {
            let nodeID: String
            let events: [FleetEvent]
            enum CodingKeys: String, CodingKey { case nodeID = "node_id", events }
        }
        struct R: Decodable { let cancel: Bool }
        let data = try await send("POST", "tasks/\(taskID)/events",
                                  json: try JSONEncoder().encode(Body(nodeID: nodeID, events: events)))
        return try Self.decoder.decode(R.self, from: data).cancel
    }

    func putArtifact(taskID: String, nodeID: String, name: String, data: Data) async throws {
        try await send("PUT", "tasks/\(taskID)/artifacts/\(name)",
                       query: [.init(name: "node_id", value: nodeID)], raw: data)
    }

    func complete(taskID: String, nodeID: String, result: TaskResult) async throws {
        struct Body: Encodable {
            let nodeID: String
            let status: String
            let output: String?
            let error: String?
            let metrics: [String: Double]
            enum CodingKeys: String, CodingKey { case nodeID = "node_id", status, output, error, metrics }
        }
        // JSONEncoder throws on NaN/Inf, which would leave the task unreported.
        let body = Body(nodeID: nodeID, status: result.status, output: result.output,
                        error: result.error, metrics: result.metrics.filter { $0.value.isFinite })
        try await send("POST", "tasks/\(taskID)/complete", json: try JSONEncoder().encode(body))
    }

    /// Latest aggregated federated round, if any.
    func latestRound() async throws -> FedRound? {
        struct R: Decodable { let round: FedRound? }
        return try Self.decoder.decode(R.self, from: try await send("GET", "fed/latest")).round
    }

    /// Downloads a worker-relative path, e.g. `fed/rounds/<id>/adapter`.
    func download(path: String) async throws -> Data {
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return try await send("GET", trimmed)
    }
}

/// Thread-safe outbox between the work (which may be a synchronous MLX
/// training callback) and the async pump that posts events. Also carries the
/// cancel flag the worker returns, so the work can poll it cheaply.
final class TaskEventBuffer: @unchecked Sendable {
    /// The worker reads at most 500 events per POST.
    static let maxPerPost = 500
    /// Cap on buffered events during a worker outage (oldest are dropped).
    static let maxBuffered = 5_000

    private let lock = NSLock()
    private var pending: [FleetEvent] = []
    private var cancelled = false
    private var finished = false
    private var dropped = 0
    var droppedEvents: Int { lock.withLock { dropped } }

    func append(_ e: FleetEvent) {
        lock.withLock {
            pending.append(e)
            trimLocked()
        }
    }

    /// Takes up to `maxPerPost` events off the front.
    func drain() -> [FleetEvent] {
        lock.withLock {
            let n = min(Self.maxPerPost, pending.count)
            let out = Array(pending.prefix(n))
            pending.removeFirst(n)
            return out
        }
    }

    /// Puts events whose POST failed back at the front, in order.
    func requeue(_ events: [FleetEvent]) {
        guard !events.isEmpty else { return }
        lock.withLock {
            pending.insert(contentsOf: events, at: 0)
            trimLocked()
        }
    }

    private func trimLocked() {
        let excess = pending.count - Self.maxBuffered
        if excess > 0 {
            pending.removeFirst(excess)
            dropped += excess
        }
    }

    var count: Int { lock.withLock { pending.count } }

    var cancelRequested: Bool { lock.withLock { cancelled } }
    func requestCancel() { lock.withLock { cancelled = true } }

    var isFinished: Bool { lock.withLock { finished } }
    func finish() { lock.withLock { finished = true } }
}
