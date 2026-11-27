import Foundation
import Observation

/// Every chat, persisted: one atomic JSON file per conversation plus a small
/// index for the list. Main-actor so the conversations list can observe it;
/// the files are small (text only), so synchronous atomic writes are fine.
@MainActor @Observable
final class ConversationStore {
    static let maxConversations = 300
    /// Longest stored tool/assistant text per turn (keeps files bounded).
    static let maxTurnLength = 8_000

    private(set) var summaries: [ConversationSummary] = []
    private let dir: URL
    private let indexURL: URL

    init(dir: URL = AtomicJSON.directory("conversations")) {
        self.dir = dir
        indexURL = dir.appending(path: "index.json")
        summaries = (AtomicJSON.readArray(ConversationSummary.self, from: indexURL) ?? [])
            .sorted { $0.updatedAt > $1.updatedAt }
        rebuildMissingSummaries()
    }

    /// Index missing, set aside, or behind the files on disk: re-list every
    /// transcript so none is orphaned.
    private func rebuildMissingSummaries() {
        let known = Set(summaries.map(\.id))
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        var added = false
        for f in files where f.pathExtension == "json" {
            guard let id = UUID(uuidString: f.deletingPathExtension().lastPathComponent), !known.contains(id),
                  let c = load(id) else { continue }
            summaries.append(Self.summary(c))
            added = true
        }
        if added {
            summaries.sort { $0.updatedAt > $1.updatedAt }
            saveIndex()
        }
    }

    private func fileURL(_ id: UUID) -> URL { dir.appending(path: "\(id.uuidString).json") }

    func load(_ id: UUID) -> StoredConversation? { AtomicJSON.read(StoredConversation.self, from: fileURL(id)) }

    /// Saves the transcript (creating the conversation on first save) and
    /// returns the stored version. Empty transcripts aren't persisted.
    @discardableResult
    func save(id: UUID, turns: [StoredTurn], modelName: String, tainted: Bool = false) -> StoredConversation? {
        guard turns.contains(where: { $0.role == .user }) else { return nil }
        let capped = turns.map { t -> StoredTurn in
            var t = t
            if t.text.count > Self.maxTurnLength { t.text = String(t.text.prefix(Self.maxTurnLength)) + "…" }
            if t.thinking.count > Self.maxTurnLength { t.thinking = String(t.thinking.prefix(Self.maxTurnLength)) + "…" }
            return t
        }
        var conv = load(id) ?? StoredConversation(
            id: id, title: "", titleIsCustom: false, createdAt: .now, updatedAt: .now,
            modelName: modelName, turns: [], reflectedTurnCount: 0)
        conv.turns = capped
        conv.updatedAt = .now
        conv.modelName = modelName
        conv.reflectedTurnCount = min(conv.reflectedTurnCount, capped.count)
        if tainted || conv.tainted == true { conv.tainted = true }   // sticky
        if !conv.titleIsCustom { conv.title = Self.autoTitle(capped) }
        write(conv)
        return conv
    }

    func rename(_ id: UUID, to title: String) {
        guard var conv = load(id), let clean = MemoryText.clean(title, maxLength: 80) else { return }
        conv.title = clean
        conv.titleIsCustom = true
        write(conv)
    }

    func setReflected(_ id: UUID, upTo count: Int) {
        guard var conv = load(id) else { return }
        conv.reflectedTurnCount = min(max(count, conv.reflectedTurnCount), conv.turns.count)
        // Not a user-visible change: keep updatedAt (list order) as it was.
        try? AtomicJSON.write(conv, to: fileURL(id))
    }

    /// Removes the transcript file. Memories distilled from it stay (they're
    /// managed separately in the Memory screen).
    func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: fileURL(id))
        summaries.removeAll { $0.id == id }
        saveIndex()
    }

    /// Conversations with turns not yet reflected on, oldest first.
    func needingReflection(excluding active: UUID?) -> [UUID] {
        summaries.reversed().compactMap { s in
            guard s.id != active, let c = load(s.id), c.turns.count > c.reflectedTurnCount,
                  c.turns[c.reflectedTurnCount...].contains(where: { $0.role == .user }) else { return nil }
            return s.id
        }
    }

    // MARK: Private

    private func write(_ conv: StoredConversation) {
        try? AtomicJSON.write(conv, to: fileURL(conv.id))
        let summary = Self.summary(conv)
        summaries.removeAll { $0.id == conv.id }
        summaries.insert(summary, at: 0)
        if summaries.count > Self.maxConversations {
            // Oldest transcripts beyond the cap are dropped; their episodes live on in memory.
            for old in summaries[Self.maxConversations...] { try? FileManager.default.removeItem(at: fileURL(old.id)) }
            summaries = Array(summaries.prefix(Self.maxConversations))
        }
        saveIndex()
    }

    static func summary(_ conv: StoredConversation) -> ConversationSummary {
        let preview = conv.turns.last(where: { $0.role == .assistant && !$0.text.isEmpty })?.text
            ?? conv.turns.last?.text ?? ""
        return ConversationSummary(id: conv.id, title: conv.title, updatedAt: conv.updatedAt,
                                   turnCount: conv.turns.count, preview: String(preview.prefix(140)))
    }

    private func saveIndex() { try? AtomicJSON.write(summaries, to: indexURL) }

    /// First user message, trimmed to a short title.
    static func autoTitle(_ turns: [StoredTurn]) -> String {
        let first = turns.first { $0.role == .user }?.text ?? "Chat"
        return MemoryText.clean(first, maxLength: 48) ?? "Chat"
    }
}
