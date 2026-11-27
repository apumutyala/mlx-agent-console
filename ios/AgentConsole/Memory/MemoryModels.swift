import Foundation

// Plain data for the memory harness. No UI, no MLX: everything here is Codable
// and Sendable so the stores (actors) and the views can share it freely.

/// What a memory is about. Raw values are persisted; never rename a shipped one.
enum MemoryKind: String, Codable, CaseIterable, Sendable, Identifiable {
    /// Something about the user or their preferences ("prefers metric units").
    case fact
    /// Summary of one past conversation.
    case episode
    /// How to do something / what worked or failed (tool tips, workflows).
    case lesson
    /// Snapshot of the current state of the world (projects, devices, jobs).
    case world

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fact: "Fact"
        case .episode: "Episode"
        case .lesson: "Lesson"
        case .world: "World"
        }
    }

    /// Accepts the synonyms the model tends to use (`procedure` for `lesson`).
    init?(loose raw: String) {
        switch raw.lowercased().trimmingCharacters(in: .whitespaces) {
        case "fact", "preference", "user": self = .fact
        case "episode", "summary": self = .episode
        case "lesson", "procedure", "tip": self = .lesson
        case "world", "state", "status": self = .world
        default: return nil
        }
    }
}

/// Where a memory's text came from. Prompt-injection defense: anything that
/// came from outside the device (worker job logs, fleet output) is `untrusted`
/// and is never promoted into the playbook without the user saying so.
enum MemoryTrust: String, Codable, Sendable {
    /// Typed or edited by the user in the Memory screen.
    case user
    /// Written by the on-device model (remember tool or reflection) from a chat
    /// that contained only the user's words and on-device tool output.
    case model
    /// Derived from a chat that also contained remote (worker/fleet) output.
    case untrusted
}

struct MemoryRecord: Codable, Identifiable, Hashable, Sendable {
    /// Short, stable id the model can cite (`m-1a2b3c4d`).
    let id: String
    var kind: MemoryKind
    var text: String
    /// Conversation it came from, if any.
    var sourceConversation: UUID?
    let createdAt: Date
    var updatedAt: Date
    /// 1 (trivia) ... 5 (always relevant).
    var importance: Int
    /// Times the memory was retrieved into a prompt.
    var accessCount: Int
    var lastAccessedAt: Date?
    /// Pinned memories never decay and are always considered for the prompt.
    var pinned: Bool
    var trust: MemoryTrust
    /// L2-normalised Float32 vector (little endian), JSON-encoded as base64.
    var embedding: Data?
    /// Which embedder produced `embedding` (vectors from another model are ignored).
    var embeddingModel: String?

    static func newID() -> String { "m-" + String(UUID().uuidString.prefix(8)).lowercased() }

    init(id: String = MemoryRecord.newID(), kind: MemoryKind, text: String, sourceConversation: UUID? = nil,
         createdAt: Date = .now, importance: Int = 3, pinned: Bool = false, trust: MemoryTrust = .model) {
        self.id = id
        self.kind = kind
        self.text = text
        self.sourceConversation = sourceConversation
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.importance = min(max(importance, 1), 5)
        self.accessCount = 0
        self.lastAccessedAt = nil
        self.pinned = pinned
        self.trust = trust
        self.embedding = nil
        self.embeddingModel = nil
    }

    var vector: [Float]? {
        guard let embedding, !embedding.isEmpty, embedding.count % 4 == 0 else { return nil }
        // Copy rather than rebind: decoded Data isn't guaranteed Float-aligned.
        var out = [Float](repeating: 0, count: embedding.count / 4)
        _ = out.withUnsafeMutableBytes { embedding.copyBytes(to: $0) }
        return out
    }

    static func pack(_ vector: [Float]) -> Data { vector.withUnsafeBufferPointer { Data(buffer: $0) } }
}

// MARK: - Playbook

/// One self-maintained note in the agent's playbook ("Before submitting a
/// worker job, list job types first").
struct PlaybookEntry: Codable, Identifiable, Hashable, Sendable {
    /// `p-1a2b3c`; the model revises/removes entries by id.
    let id: String
    var text: String
    var updatedAt: Date

    static func newID() -> String { "p-" + String(UUID().uuidString.prefix(6)).lowercased() }
}

/// An immutable snapshot of the whole playbook. Every edit appends a version,
/// so any earlier state can be restored from Settings.
struct PlaybookVersion: Codable, Identifiable, Hashable, Sendable {
    let id: Int
    let createdAt: Date
    /// Why this version exists ("reflection on chat …", "rollback to v3", "edited").
    let reason: String
    let entries: [PlaybookEntry]
}

/// A playbook edit the reflection pass proposed from a chat that contained
/// untrusted (remote) text. Waits in the Memory screen for the user's approval.
struct PendingPlaybookEdit: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    let createdAt: Date
    let sourceConversation: UUID?
    let edit: ReflectionResult.LessonEdit
}

// MARK: - Conversations

struct StoredTurn: Codable, Hashable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant, tool, error }
    var role: Role
    var text: String
    var thinking: String
    /// Tool result that came from a remote machine (worker/fleet): untrusted.
    var remote: Bool

    init(role: Role, text: String, thinking: String = "", remote: Bool = false) {
        self.role = role
        self.text = text
        self.thinking = thinking
        self.remote = remote
    }
}

struct StoredConversation: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var title: String
    /// True once the user renamed it (auto-titling stops).
    var titleIsCustom: Bool
    let createdAt: Date
    var updatedAt: Date
    var modelName: String
    var turns: [StoredTurn]
    /// Turns [0, reflectedTurnCount) have already been through reflection.
    var reflectedTurnCount: Int

    var containsRemoteText: Bool { turns.contains(where: \.remote) }
}

/// Lightweight row for the conversations list (the index file holds only these).
struct ConversationSummary: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var title: String
    var updatedAt: Date
    var turnCount: Int
    var preview: String
}
