import Foundation

/// The model *proposes* tool calls. This file decides whether a proposal is
/// well-formed and allowed, and only then runs Swift code.
///
/// Normally proposals arrive structured: tools are declared natively to
/// mlx-swift-lm, which renders them with the model's own chat template and
/// parses the model's call dialect back into `ToolCall`s (see MLXBackend). The
/// text parser below is only a fallback for a call that slipped through as text.

struct ToolCallProposal: Equatable, Sendable {
    let name: String
    let argumentsJSON: Data
    /// Correlates the tool result with the call, when the model/template uses ids.
    let id: String?

    init(name: String, argumentsJSON: Data, id: String? = nil) {
        self.name = name
        self.argumentsJSON = argumentsJSON
        self.id = id
    }

    static let malformed = ToolCallProposal(name: "<malformed>", argumentsJSON: Data())
}

/// Fallback text parser. Understands the three dialects our models emit:
///   Qwen/Hermes JSON: <tool_call>{"name": "f", "arguments": {...}}</tool_call>
///   LFM3 XML:         <tool_call><function=f><parameter=k>v</parameter></function></tool_call>
///   LFM2/2.5 Python:  <|tool_call_start|>[f(k='v', p={'a': 'b'})]<|tool_call_end|>
enum ToolCallParser {
    /// Returns the first proposal in `text`, or nil if the model just answered.
    static func firstCall(in text: String) -> ToolCallProposal? {
        if let body = between(text, "<|tool_call_start|>", "<|tool_call_end|>") {
            return pythonic(body) ?? .malformed
        }
        guard let body = between(text, "<tool_call>", "</tool_call>") else { return nil }
        if body.contains("<function=") { return xmlFunction(body) ?? .malformed }
        return json(body) ?? .malformed
    }

    private static func between(_ text: String, _ open: String, _ close: String) -> String? {
        guard let o = text.range(of: open),
              let c = text.range(of: close, range: o.upperBound..<text.endIndex) else { return nil }
        return text[o.upperBound..<c.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func proposal(_ name: String, _ args: [String: Any]) -> ToolCallProposal? {
        guard !name.isEmpty, let data = try? JSONSerialization.data(withJSONObject: args) else { return nil }
        return ToolCallProposal(name: name, argumentsJSON: data)
    }

    static func json(_ body: String) -> ToolCallProposal? {
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String
        else { return nil }
        if obj["arguments"] == nil { return proposal(name, [:]) }
        guard let args = obj["arguments"] as? [String: Any] else { return nil }
        return proposal(name, args)
    }

    static func xmlFunction(_ body: String) -> ToolCallProposal? {
        guard let f = body.range(of: "<function="),
              let gt = body.range(of: ">", range: f.upperBound..<body.endIndex) else { return nil }
        let name = body[f.upperBound..<gt.lowerBound].trimmingCharacters(in: .whitespaces)
        var args: [String: Any] = [:]
        var rest = body[gt.upperBound...]
        while let p = rest.range(of: "<parameter=") {
            guard let pgt = rest.range(of: ">", range: p.upperBound..<rest.endIndex),
                  let end = rest.range(of: "</parameter>", range: pgt.upperBound..<rest.endIndex) else { return nil }
            let key = rest[p.upperBound..<pgt.lowerBound].trimmingCharacters(in: .whitespaces)
            let raw = rest[pgt.upperBound..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            // Objects/arrays/numbers/bools are written as JSON; anything else is a string.
            if let d = raw.data(using: .utf8),
               let v = try? JSONSerialization.jsonObject(with: d, options: [.fragmentsAllowed]),
               !(v is String) {
                args[key] = v
            } else {
                args[key] = raw
            }
            rest = rest[end.upperBound...]
        }
        return proposal(name, args)
    }

    /// `[name(k=v, ...)]` with Python literal values (str, int/float, True/False/None, dict, list).
    static func pythonic(_ body: String) -> ToolCallProposal? {
        var p = PyLiteralParser(body)
        p.skipSpace()
        let bracketed = p.eat("[")
        p.skipSpace()
        let name = p.identifier()
        guard !name.isEmpty, p.eat("(") else { return nil }
        var args: [String: Any] = [:]
        p.skipSpace()
        if !p.eat(")") {
            repeat {
                p.skipSpace()
                let key = p.identifier()
                p.skipSpace()
                guard !key.isEmpty, p.eat("="), let value = p.value() else { return nil }
                args[key] = value
                p.skipSpace()
            } while p.eat(",")
            guard p.eat(")") else { return nil }
        }
        if bracketed { p.skipSpace(); _ = p.eat("]") }
        return proposal(name, args)
    }
}

/// Minimal parser for the Python literals LFM2 writes in tool calls.
private struct PyLiteralParser {
    private let s: [Character]
    private var i = 0
    init(_ text: String) { s = Array(text) }

    mutating func skipSpace() { while i < s.count, s[i].isWhitespace { i += 1 } }

    mutating func eat(_ c: Character) -> Bool {
        guard i < s.count, s[i] == c else { return false }
        i += 1
        return true
    }

    private func at(_ c: Character) -> Bool { i < s.count && s[i] == c }

    mutating func identifier() -> String {
        var out = ""
        while i < s.count, s[i].isLetter || s[i].isNumber || s[i] == "_" || s[i] == "." {
            out.append(s[i])
            i += 1
        }
        return out
    }

    mutating func value() -> Any? {
        skipSpace()
        guard i < s.count else { return nil }
        switch s[i] {
        case "'", "\"":
            return string()
        case "{":
            i += 1
            var dict: [String: Any] = [:]
            while true {
                skipSpace()
                if eat("}") { return dict }
                guard let key = value() as? String else { return nil }
                skipSpace()
                guard eat(":"), let v = value() else { return nil }
                dict[key] = v
                skipSpace()
                if !eat(",") && !at("}") { return nil }
            }
        case "[":
            i += 1
            var list: [Any] = []
            while true {
                skipSpace()
                if eat("]") { return list }
                guard let v = value() else { return nil }
                list.append(v)
                skipSpace()
                if !eat(",") && !at("]") { return nil }
            }
        default:
            var word = ""
            if eat("-") { word = "-" }
            word += identifier()
            switch word {
            case "True", "true": return true
            case "False", "false": return false
            case "None", "null": return NSNull()
            default:
                if let n = Int(word) { return n }
                return Double(word)
            }
        }
    }

    private mutating func string() -> String? {
        let quote = s[i]
        i += 1
        var out = ""
        while i < s.count {
            let c = s[i]
            i += 1
            if c == quote { return out }
            if c == "\\", i < s.count {
                let e = s[i]
                i += 1
                switch e {
                case "n": out.append("\n")
                case "t": out.append("\t")
                default: out.append(e)
                }
            } else {
                out.append(c)
            }
        }
        return nil
    }
}

enum ToolError: LocalizedError {
    case unknownTool(String)
    case invalidArguments(String)
    case denied(String)
    var errorDescription: String? {
        switch self {
        case .unknownTool(let n): "Unknown tool \(n). Use only the tools you were given."
        case .invalidArguments(let m): "Invalid arguments: \(m)"
        case .denied(let m): "The user declined: \(m)"
        }
    }
}

/// A tool with typed, strictly-decoded arguments.
protocol AgentTool: Sendable {
    associatedtype Arguments: Decodable & Sendable
    static var name: String { get }
    /// One-line description given to the model (natively, via the chat template).
    static var summary: String { get }
    /// JSON-Schema for `Arguments`, given to the model natively.
    static var parametersSchema: [String: any Sendable] { get }
    /// Semantic validation beyond decoding (e.g. "file must exist").
    func validate(_ args: Arguments) async throws
    /// Non-nil = ask the user before running, with this explanation.
    func confirmationPrompt(_ args: Arguments) -> String?
    func run(_ args: Arguments) async throws -> String
}

extension AgentTool {
    func confirmationPrompt(_ args: Arguments) -> String? { nil }
    static var specification: ToolSpecification {
        ToolSpecification(name: name, description: summary, parameters: parametersSchema)
    }
}

/// Decodes into `T` and rejects keys `T` doesn't declare, so the model can't
/// smuggle extra fields past the type.
func strictDecode<T: Decodable>(_ type: T.Type, from data: Data, allowedKeys: Set<String>) throws -> T {
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
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
    static let summary = "Summarize one document the user imported into Agent Console. Use the file's id from the Imported files list."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "fileID": ["type": "string", "description": "id of an imported file"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["fileID"],
    ]

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
    static let summary = "Queue a named job on the user's remote worker computer. The user must approve it first."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "type": ["type": "string", "description": "a job type name known to the worker"] as [String: any Sendable],
            "params": [
                "type": "object",
                "description": "string parameters for the job type",
                "additionalProperties": ["type": "string"] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["type"],
    ]

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
    /// Native declarations handed to the model.
    let specifications: [ToolSpecification]
    /// "name: summary" lines for the system prompt.
    var descriptions: [String] { specifications.map { "\($0.name): \($0.description)" } }

    init(summarize: SummarizeImportedFileTool, submitJob: SubmitWorkerJobTool?) {
        var r: [String: @Sendable (Data, ConfirmHandler) async throws -> String] = [:]
        var s: [ToolSpecification] = []
        func add<T: AgentTool>(_ tool: T, keys: Set<String>) {
            r[T.name] = { data, confirm in
                let args = try strictDecode(T.Arguments.self, from: data, allowedKeys: keys)
                try await tool.validate(args)
                if let prompt = tool.confirmationPrompt(args), !(await confirm(prompt)) {
                    throw ToolError.denied(prompt)
                }
                return try await tool.run(args)
            }
            s.append(T.specification)
        }
        add(summarize, keys: ["fileID"])
        if let submitJob { add(submitJob, keys: ["type", "params"]) }
        runners = r
        specifications = s
    }

    func execute(_ call: ToolCallProposal, confirm: ConfirmHandler) async throws -> String {
        guard let run = runners[call.name] else { throw ToolError.unknownTool(call.name) }
        return try await run(call.argumentsJSON, confirm)
    }
}
