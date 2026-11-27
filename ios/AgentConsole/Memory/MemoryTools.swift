import Foundation

// Model-facing memory tools. Same rules as every other tool: strictly decoded
// arguments, Swift-side validation, and `forget` needs the user's approval.

struct RememberTool: AgentTool {
    struct Arguments: Decodable, Sendable {
        let text: String
        let kind: String?
        let importance: Int?
    }
    static let name = "remember"
    static let summary = "Save a durable memory for future chats: a fact about the user or their preferences, a lesson about what worked, or the current state of something. One short standalone sentence."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "text": ["type": "string", "description": "the memory, one short sentence (max 300 characters)"] as [String: any Sendable],
            "kind": ["type": "string", "enum": ["fact", "lesson", "world"],
                     "description": "fact (about the user), lesson (what worked/failed), world (current state)"] as [String: any Sendable],
            "importance": ["type": "integer", "minimum": 1, "maximum": 5,
                           "description": "1 trivia ... 5 always relevant (default 3)"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["text"],
    ]
    static let keys: Set<String> = ["text", "kind", "importance"]
    static let maxLength = 300

    let harness: MemoryHarness
    let conversationID: UUID

    func validate(_ args: Arguments) async throws {
        let t = args.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { throw ToolError.invalidArguments("text is empty") }
        guard t.count <= Self.maxLength else { throw ToolError.invalidArguments("text longer than \(Self.maxLength) characters") }
        guard !MemoryText.looksLikeControlMarkup(t) else { throw ToolError.invalidArguments("text contains markup") }
        if let k = args.kind {
            guard let kind = MemoryKind(loose: k), kind != .episode else {
                throw ToolError.invalidArguments("kind must be fact, lesson or world")
            }
        }
        if let i = args.importance, !(1...5).contains(i) {
            throw ToolError.invalidArguments("importance must be 1...5")
        }
    }

    func run(_ args: Arguments) async throws -> String {
        let kind = args.kind.flatMap(MemoryKind.init(loose:)) ?? .fact
        // A chat that has seen remote output can't write trusted memories.
        let trust: MemoryTrust = await harness.isTainted(conversationID) ? .untrusted : .model
        let stored = try await harness.memories.upsert(MemoryRecord(
            kind: kind, text: args.text, sourceConversation: conversationID,
            importance: args.importance ?? 3, trust: trust))
        await harness.embedIfPossible(stored)
        await harness.noteChanged()
        return "Saved \(stored.id) (\(stored.kind.rawValue))\(trust == .untrusted ? ", marked untrusted" : "")."
    }
}

struct RecallTool: AgentTool {
    struct Arguments: Decodable, Sendable {
        let query: String
        let limit: Int?
    }
    static let name = "recall"
    static let summary = "Search your long-term memory (facts, past chats, lessons) for anything relevant to a query."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "query": ["type": "string", "description": "what to look for"] as [String: any Sendable],
            "limit": ["type": "integer", "minimum": 1, "maximum": 10, "description": "max results (default 6)"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["query"],
    ]
    static let keys: Set<String> = ["query", "limit"]

    let harness: MemoryHarness

    func validate(_ args: Arguments) async throws {
        let q = args.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, q.count <= 500 else { throw ToolError.invalidArguments("query must be 1...500 characters") }
        if let l = args.limit, !(1...10).contains(l) { throw ToolError.invalidArguments("limit must be 1...10") }
    }

    func run(_ args: Arguments) async throws -> String {
        let qv = await harness.queryVectorOutsideGate(args.query)
        let hits = await harness.memories.search(args.query, queryVector: qv, embeddingModel: MemoryEmbedder.modelID,
                                                 limit: args.limit ?? 6)
            .filter { $0.score > 0.05 }.map(\.record)
        let chosen = ContextAssembler.packMemories(hits, budget: 500, maxItems: args.limit ?? 6)
        guard !chosen.isEmpty else { return "No matching memories." }
        await harness.memories.markAccessed(chosen.map(\.id))
        return chosen.map { "- " + ContextAssembler.memoryLine($0) }.joined(separator: "\n")
    }
}

/// Removes one memory, after the user approves in the confirmation alert.
struct ForgetTool: AgentTool {
    struct Arguments: Decodable, Sendable { let id: String }
    static let name = "forget"
    static let summary = "Delete one memory by its id (e.g. m-1a2b3c4d) when the user asks you to forget it. The user must confirm."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "id": ["type": "string", "description": "memory id, like m-1a2b3c4d"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["id"],
    ]
    static let keys: Set<String> = ["id"]

    let harness: MemoryHarness
    /// Text of the memory found in `validate`, shown in the confirmation
    /// (which is synchronous and can't query the store itself).
    private let preview = PreviewBox()

    init(harness: MemoryHarness) { self.harness = harness }

    func validate(_ args: Arguments) async throws {
        guard args.id.range(of: #"^m-[0-9a-f]{8}$"#, options: .regularExpression) != nil else {
            throw ToolError.invalidArguments("id must look like m-1a2b3c4d")
        }
        guard let r = await harness.memories.record(id: args.id) else {
            throw ToolError.invalidArguments("no memory with id \(args.id)")
        }
        preview.set(r.text)
    }

    func confirmationPrompt(_ args: Arguments) -> String? {
        "Forget this memory? “\(TokenEstimate.truncate(preview.get() ?? args.id, toTokens: 40))”"
    }

    func run(_ args: Arguments) async throws -> String {
        try await harness.memories.delete(ids: [args.id])
        await harness.noteChanged()
        return "Forgot \(args.id)."
    }
}

final class PreviewBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: String?
    func set(_ v: String) { lock.withLock { value = v } }
    func get() -> String? { lock.withLock { value } }
}

extension ToolRegistry {
    /// Adds remember/recall/forget, bound to one conversation.
    mutating func registerMemoryTools(_ harness: MemoryHarness, conversationID: UUID) {
        register(RememberTool(harness: harness, conversationID: conversationID), keys: RememberTool.keys)
        register(RecallTool(harness: harness), keys: RecallTool.keys)
        register(ForgetTool(harness: harness), keys: ForgetTool.keys)
    }
}
