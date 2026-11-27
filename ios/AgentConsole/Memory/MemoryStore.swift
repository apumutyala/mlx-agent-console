import Foundation

/// Durable store of typed memories (facts, episodes, lessons, world state).
/// One JSON file, written atomically after every change. Size is bounded:
/// inserts merge near-duplicates, and over the cap the lowest-retention
/// unpinned records are evicted (see `MemoryMaintenance`).
actor MemoryStore {
    static let cap = 400
    static let perKindCap: [MemoryKind: Int] = [.episode: 150, .world: 60]

    private let url: URL
    private var records: [MemoryRecord]
    /// Bumped on every change, so views can cheaply notice updates.
    private(set) var revision = 0

    init(url: URL = AtomicJSON.directory("store").appending(path: "memories.json")) {
        self.url = url
        records = AtomicJSON.read([MemoryRecord].self, from: url) ?? []
    }

    // MARK: Reading

    func all() -> [MemoryRecord] { records.sorted { $0.updatedAt > $1.updatedAt } }
    func record(id: String) -> MemoryRecord? { records.first { $0.id == id } }
    var count: Int { records.count }

    /// Hybrid search (BM25 + optional embeddings). Does not touch access counts;
    /// call `markAccessed` for what actually went into a prompt.
    func search(_ query: String, queryVector: [Float]? = nil, embeddingModel: String? = nil,
                kinds: Set<MemoryKind>? = nil, limit: Int = 20) -> [MemoryRanker.Scored] {
        let pool = kinds.map { k in records.filter { k.contains($0.kind) } } ?? records
        return Array(MemoryRanker.rank(pool, query: query, queryVector: queryVector,
                                       embeddingModel: embeddingModel).prefix(limit))
    }

    /// Records that still need a vector from `model`.
    func missingEmbeddings(model: String, limit: Int = 32) -> [MemoryRecord] {
        Array(records.filter { $0.embeddingModel != model || $0.embedding == nil }.prefix(limit))
    }

    // MARK: Writing

    /// Inserts (or merges into a near-duplicate) and returns the stored record.
    /// Rejects empty text and anything that looks like tool/role markup.
    @discardableResult
    func upsert(_ candidate: MemoryRecord) throws -> MemoryRecord {
        guard let text = MemoryText.clean(candidate.text) else { throw MemoryError.empty }
        guard !MemoryText.looksLikeControlMarkup(text) else { throw MemoryError.markup }
        var c = candidate
        c.text = text
        let stored: MemoryRecord
        if let i = MemoryMaintenance.mergeTarget(for: c, in: records) {
            stored = MemoryMaintenance.merge(records[i], with: c)
            records[i] = stored
        } else {
            stored = c
            records.append(c)
        }
        enforceCap()
        try save()
        return records.first { $0.id == stored.id } ?? stored
    }

    /// User edit from the Memory screen. User-edited text becomes `.user` trust.
    func edit(id: String, text: String? = nil, kind: MemoryKind? = nil, importance: Int? = nil,
              pinned: Bool? = nil) throws {
        guard let i = records.firstIndex(where: { $0.id == id }) else { throw MemoryError.notFound(id) }
        if let text {
            guard let clean = MemoryText.clean(text) else { throw MemoryError.empty }
            if clean != records[i].text {
                records[i].text = clean
                records[i].trust = .user
                records[i].embedding = nil          // re-embedded lazily
                records[i].embeddingModel = nil
            }
        }
        if let kind { records[i].kind = kind }
        if let importance { records[i].importance = min(max(importance, 1), 5) }
        if let pinned { records[i].pinned = pinned }
        records[i].updatedAt = .now
        try save()
    }

    func delete(ids: [String]) throws {
        let set = Set(ids)
        records.removeAll { set.contains($0.id) }
        try save()
    }

    func deleteAll() throws {
        records = []
        try save()
    }

    func markAccessed(_ ids: [String]) {
        guard !ids.isEmpty else { return }
        let set = Set(ids)
        let now = Date.now
        for i in records.indices where set.contains(records[i].id) {
            records[i].accessCount += 1
            records[i].lastAccessedAt = now
        }
        try? save()
    }

    func setEmbedding(id: String, vector: [Float], model: String) {
        guard let i = records.firstIndex(where: { $0.id == id }) else { return }
        records[i].embedding = MemoryRecord.pack(vector)
        records[i].embeddingModel = model
        try? save()
    }

    // MARK: Export / import

    /// Merges imported records: same id replaces, otherwise normal upsert
    /// (dedupe). Imported text keeps its trust, except `.user` is downgraded to
    /// `.model` — a file isn't the user typing. Returns how many were taken.
    func importRecords(_ incoming: [MemoryRecord]) throws -> Int {
        var taken = 0
        for var r in incoming.prefix(Self.cap * 2) {
            guard let text = MemoryText.clean(r.text), !MemoryText.looksLikeControlMarkup(text) else { continue }
            r.text = text
            r.importance = min(max(r.importance, 1), 5)
            if r.trust == .user { r.trust = .model }
            if let i = records.firstIndex(where: { $0.id == r.id }) {
                records[i] = r
            } else if let i = MemoryMaintenance.mergeTarget(for: r, in: records) {
                records[i] = MemoryMaintenance.merge(records[i], with: r)
            } else {
                records.append(r)
            }
            taken += 1
        }
        enforceCap()
        try save()
        return taken
    }

    // MARK: Private

    private func enforceCap() {
        let (kept, _) = MemoryMaintenance.evict(records, cap: Self.cap, perKindCap: Self.perKindCap)
        records = kept
    }

    private func save() throws {
        revision += 1
        try AtomicJSON.write(records, to: url)
    }
}

enum MemoryError: LocalizedError {
    case empty
    case markup
    case notFound(String)
    case invalidKind(String)

    var errorDescription: String? {
        switch self {
        case .empty: "Memory text is empty."
        case .markup: "Memory text contains tool or role markup and was rejected."
        case .notFound(let id): "No memory with id \(id)."
        case .invalidKind(let k): "Unknown memory kind \(k); use fact, episode, lesson or world."
        }
    }
}
