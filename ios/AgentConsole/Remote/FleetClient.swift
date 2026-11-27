import Foundation

// MARK: - Wire models (match worker/fleet.py and worker/federated.py)

struct NodeInfo: Encodable, Sendable {
    struct Capabilities: Encodable, Sendable {
        let kinds: [String]
        let model: String
        let memoryGB: Double
        let os: String
        enum CodingKeys: String, CodingKey { case kinds, model, memoryGB = "memory_gb", os }
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

struct TaskResult: Sendable {
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
    private let session: URLSession

    init(config: WorkerConfig) {
        self.config = config
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 45      // > the 25 s long-poll
        c.timeoutIntervalForResource = 600    // adapter uploads/downloads
        c.waitsForConnectivity = false
        session = URLSession(configuration: c)
    }

    static func fromSettings() throws -> FleetClient {
        guard let cfg = WorkerConfig.load() else { throw WorkerError.notConfigured }
        return FleetClient(config: cfg)
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
        let body = Body(nodeID: nodeID, status: result.status, output: result.output,
                        error: result.error, metrics: result.metrics)
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
    private let lock = NSLock()
    private var pending: [FleetEvent] = []
    private var cancelled = false
    private var finished = false

    func append(_ e: FleetEvent) { lock.withLock { pending.append(e) } }
    func drain() -> [FleetEvent] { lock.withLock { defer { pending.removeAll() }; return pending } }
    var count: Int { lock.withLock { pending.count } }

    var cancelRequested: Bool { lock.withLock { cancelled } }
    func requestCancel() { lock.withLock { cancelled = true } }

    var isFinished: Bool { lock.withLock { finished } }
    func finish() { lock.withLock { finished = true } }
}
