import Foundation

/// The agent's self-maintained notes ("playbook"), with full version history.
/// Every change appends an immutable version; rollback appends a copy of an old
/// version (history is never rewritten, so a rollback can itself be undone).
actor PlaybookStore {
    static let maxEntries = 20
    static let maxVersions = 60
    static let maxPending = 30

    private struct State: Codable {
        var versions: [PlaybookVersion]
        var pending: [PendingPlaybookEdit]
    }

    private let url: URL
    private var state: State

    init(url: URL = AtomicJSON.directory("store").appending(path: "playbook.json")) {
        self.url = url
        state = AtomicJSON.read(State.self, from: url) ?? State(versions: [], pending: [])
        if state.versions.isEmpty {
            state.versions = [PlaybookVersion(id: 1, createdAt: .now, reason: "initial", entries: Self.seed)]
        }
    }

    /// Starting notes, so the playbook shows its shape before the first reflection.
    private static let seed: [PlaybookEntry] = [
        PlaybookEntry(id: "p-seed01", text: "Answer directly when no tool is needed; keep replies short.", updatedAt: .now),
        PlaybookEntry(id: "p-seed02", text: "Use recall before saying you don't know something the user may have told you earlier.", updatedAt: .now),
    ]

    var current: PlaybookVersion { state.versions.last! }
    var entries: [PlaybookEntry] { current.entries }
    func versions() -> [PlaybookVersion] { state.versions.reversed() }
    func pending() -> [PendingPlaybookEdit] { state.pending }

    /// Applies validated edits as one new version. Unknown ids are ignored
    /// (the validator already rejects them; this guards against races).
    /// Returns the new version, or nil if nothing changed.
    @discardableResult
    func apply(_ edits: [ReflectionResult.LessonEdit], reason: String) throws -> PlaybookVersion? {
        var list = current.entries
        var changed = false
        for e in edits {
            switch e.op {
            case .add:
                guard let t = e.text.flatMap({ MemoryText.clean($0, maxLength: ReflectionValidator.maxLessonLength) }),
                      !list.contains(where: { MemoryMaintenance.similarity($0.text, t) >= MemoryMaintenance.mergeThreshold })
                else { continue }
                list.append(PlaybookEntry(id: PlaybookEntry.newID(), text: t, updatedAt: .now))
                changed = true
            case .revise:
                guard let id = e.id, let i = list.firstIndex(where: { $0.id == id }),
                      let t = e.text.flatMap({ MemoryText.clean($0, maxLength: ReflectionValidator.maxLessonLength) }),
                      t != list[i].text else { continue }
                list[i].text = t
                list[i].updatedAt = .now
                changed = true
            case .remove:
                guard let id = e.id, list.contains(where: { $0.id == id }) else { continue }
                list.removeAll { $0.id == id }
                changed = true
            }
        }
        guard changed else { return nil }
        // Over the cap, the oldest-updated notes fall off (they're still in history).
        if list.count > Self.maxEntries {
            list = Array(list.sorted { $0.updatedAt > $1.updatedAt }.prefix(Self.maxEntries))
        }
        return try append(entries: list, reason: reason)
    }

    /// User edit from Settings: replaces the whole list.
    func replace(with entries: [PlaybookEntry], reason: String = "edited by you") throws {
        let cleaned = entries.compactMap { e -> PlaybookEntry? in
            guard let t = MemoryText.clean(e.text, maxLength: ReflectionValidator.maxLessonLength) else { return nil }
            return PlaybookEntry(id: e.id, text: t, updatedAt: e.text == t ? e.updatedAt : .now)
        }
        try append(entries: Array(cleaned.prefix(Self.maxEntries)), reason: reason)
    }

    func rollback(to versionID: Int) throws {
        guard let v = state.versions.first(where: { $0.id == versionID }) else { return }
        try append(entries: v.entries, reason: "rollback to v\(versionID)")
    }

    // MARK: Pending (untrusted) edits

    func enqueuePending(_ edits: [ReflectionResult.LessonEdit], source: UUID?) throws {
        guard !edits.isEmpty else { return }
        state.pending += edits.map { PendingPlaybookEdit(id: UUID(), createdAt: .now, sourceConversation: source, edit: $0) }
        if state.pending.count > Self.maxPending { state.pending.removeFirst(state.pending.count - Self.maxPending) }
        try AtomicJSON.write(state, to: url)
    }

    /// Approve (apply) or reject a pending edit.
    func resolvePending(_ id: UUID, approve: Bool) throws {
        guard let i = state.pending.firstIndex(where: { $0.id == id }) else { return }
        let p = state.pending.remove(at: i)
        if approve, try apply([p.edit], reason: "approved suggestion") != nil { return }   // apply() saved
        try AtomicJSON.write(state, to: url)
    }

    @discardableResult
    private func append(entries: [PlaybookEntry], reason: String) throws -> PlaybookVersion {
        let v = PlaybookVersion(id: (state.versions.last?.id ?? 0) + 1, createdAt: .now, reason: reason, entries: entries)
        state.versions.append(v)
        if state.versions.count > Self.maxVersions { state.versions.removeFirst(state.versions.count - Self.maxVersions) }
        try AtomicJSON.write(state, to: url)
        return v
    }
}
