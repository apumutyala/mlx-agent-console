import Foundation

/// The model *proposes* tool calls as text. This file decides whether a
/// proposal is well-formed and allowed, and only then runs Swift code.
///
/// Wire format (Qwen-style, which the default model already emits):
///   <tool_call>{"name": "summarizeImportedFile", "arguments": {"fileID": "…"}}</tool_call>

struct ToolCallProposal: Equatable {
    let name: String
    let argumentsJSON: Data
}

enum ToolCallParser {
    /// Returns the first proposal in `text`, or nil if the model just answered.
    static func firstCall(in text: String) -> ToolCallProposal? {
        guard let open = text.range(of: "<tool_call>"),
              let close = text.range(of: "</tool_call>", range: open.upperBound..<text.endIndex)
        else { return nil }
        let body = text[open.upperBound..<close.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String,
              let args = obj["arguments"] as? [String: Any],
              let argData = try? JSONSerialization.data(withJSONObject: args)
        else { return ToolCallProposal(name: "<malformed>", argumentsJSON: Data()) }
        return ToolCallProposal(name: name, argumentsJSON: argData)
    }
}

enum ToolError: LocalizedError {
    case unknownTool(String)
    case invalidArguments(String)
    case denied(String)
    var errorDescription: String? {
        switch self {
        case .unknownTool(let n): "Unknown tool \(n). Available tools are listed in your instructions."
        case .invalidArguments(let m): "Invalid arguments: \(m)"
        case .denied(let m): "The user declined: \(m)"
        }
    }
}

/// A tool with typed, strictly-decoded arguments.
protocol AgentTool: Sendable {
    associatedtype Arguments: Decodable & Sendable
    static var name: String { get }
    static var schemaDescription: String { get }
    /// Semantic validation beyond decoding (e.g. "file must exist").
    func validate(_ args: Arguments) async throws
    /// Non-nil = ask the user before running, with this explanation.
    func confirmationPrompt(_ args: Arguments) -> String?
    func run(_ args: Arguments) async throws -> String
}

extension AgentTool {
    func confirmationPrompt(_ args: Arguments) -> String? { nil }
}

/// Decodes into `T` and rejects keys `T` doesn't declare, so the model can't
/// smuggle extra fields past the type.
func strictDecode<T: Decodable>(_ type: T.Type, from data: Data, allowedKeys: Set<String>) throws -> T {
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw ToolError.invalidArguments("arguments must be an object")
    }
    let extra = Set(obj.keys).subtracting(allowedKeys)
    guard extra.isEmpty else { throw ToolError.invalidArguments("unexpected keys \(extra.sorted())") }
    do { return try JSONDecoder().decode(T.self, from: data) }
    catch { throw ToolError.invalidArguments(String(describing: error)) }
}

// MARK: - Tools

struct SummarizeImportedFileTool: AgentTool {
    struct Arguments: Decodable, Sendable { let fileID: String }
    static let name = "summarizeImportedFile"
    static let schemaDescription = #"summarizeImportedFile(fileID: string) — summarize one imported document. fileID must be one of the IDs listed under "Imported files"."#

    let documents: DocumentStore
    let summarizer: DocumentSummarizer

    func validate(_ args: Arguments) async throws {
        guard await documents.document(id: args.fileID) != nil else {
            throw ToolError.invalidArguments("no imported file with id \(args.fileID)")
        }
    }

    func run(_ args: Arguments) async throws -> String {
        guard let doc = await documents.document(id: args.fileID) else { throw ToolError.invalidArguments("file vanished") }
        return try await summarizer.summarize(doc)
    }
}

/// Lets the local model hand work to the worker — only named job types, and
/// only after the user taps Approve.
struct SubmitWorkerJobTool: AgentTool {
    struct Arguments: Decodable, Sendable {
        let type: String
        let params: [String: String]?
    }
    static let name = "submitWorkerJob"
    static let schemaDescription = #"submitWorkerJob(type: string, params?: {string: string}) — queue a named job on the user's remote worker. Requires user approval."#

    let client: WorkerClient

    func validate(_ args: Arguments) async throws {
        let types = try await client.jobTypes()
        guard let jt = types.first(where: { $0.name == args.type }) else {
            throw ToolError.invalidArguments("unknown job type \(args.type); known: \(types.map(\.name))")
        }
        let unknown = Set((args.params ?? [:]).keys).subtracting(jt.params.keys)
        guard unknown.isEmpty else { throw ToolError.invalidArguments("unknown params \(unknown.sorted())") }
        // The worker re-validates every param against its regex; this is just early feedback.
    }

    func confirmationPrompt(_ args: Arguments) -> String? {
        let p = (args.params ?? [:]).map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ", ")
        return "Start worker job “\(args.type)”\(p.isEmpty ? "" : " (\(p))")?"
    }

    func run(_ args: Arguments) async throws -> String {
        let job = try await client.submit(type: args.type, params: args.params ?? [:])
        return "Submitted job \(job.id) (status: \(job.status)). The user can follow it in the Jobs tab."
    }
}

// MARK: - Registry

typealias ConfirmHandler = @Sendable (String) async -> Bool

struct ToolRegistry: Sendable {
    private let runners: [String: @Sendable (Data, ConfirmHandler) async throws -> String]
    let descriptions: [String]

    init(summarize: SummarizeImportedFileTool, submitJob: SubmitWorkerJobTool?) {
        var r: [String: @Sendable (Data, ConfirmHandler) async throws -> String] = [:]
        var d: [String] = []
        func add<T: AgentTool>(_ tool: T, keys: Set<String>) {
            r[T.name] = { data, confirm in
                let args = try strictDecode(T.Arguments.self, from: data, allowedKeys: keys)
                try await tool.validate(args)
                if let prompt = tool.confirmationPrompt(args), !(await confirm(prompt)) {
                    throw ToolError.denied(prompt)
                }
                return try await tool.run(args)
            }
            d.append(T.schemaDescription)
        }
        add(summarize, keys: ["fileID"])
        if let submitJob { add(submitJob, keys: ["type", "params"]) }
        runners = r
        descriptions = d
    }

    func execute(_ call: ToolCallProposal, confirm: ConfirmHandler) async throws -> String {
        guard let run = runners[call.name] else { throw ToolError.unknownTool(call.name) }
        return try await run(call.argumentsJSON, confirm)
    }
}
