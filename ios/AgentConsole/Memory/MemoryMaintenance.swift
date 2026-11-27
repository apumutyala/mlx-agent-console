import Foundation

// Pure policies that keep the memory store small and non-redundant:
// sanitising text, merging near-duplicates, and evicting by retention score.

enum MemoryText {
    static let maxLength = 500

    /// Collapses whitespace, strips control characters and caps length.
    /// Returns nil for text that's empty after cleaning.
    static func clean(_ raw: String, maxLength: Int = MemoryText.maxLength) -> String? {
        let noControl = String(raw.unicodeScalars.filter { !CharacterSet.controlCharacters.contains($0) || $0 == "\n" }
            .map(Character.init))
        let collapsed = noControl.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard !collapsed.isEmpty else { return nil }
        return collapsed.count > maxLength ? String(collapsed.prefix(maxLength - 1)) + "…" : collapsed
    }

    /// Text that looks like an attempt to make the model call tools or change
    /// roles. Never stored, never injected.
    static func looksLikeControlMarkup(_ text: String) -> Bool {
        let t = text.lowercased()
        return ["<tool_call", "<|tool_call", "</tool_call", "<function=", "<|im_start", "<|im_end",
                "<|startoftext", "<|endoftext", "<think>", "</think>", "<|system", "[inst]"]
            .contains(where: t.contains)
    }
}

enum MemoryMaintenance {
    /// Jaccard similarity of the two texts' term sets (0...1).
    static func similarity(_ a: String, _ b: String) -> Double {
        let sa = Set(TextTokenizer.terms(a)), sb = Set(TextTokenizer.terms(b))
        guard !sa.isEmpty || !sb.isEmpty else { return 1 }
        return Double(sa.intersection(sb).count) / Double(sa.union(sb).count)
    }

    /// Above this, a new memory is treated as a restatement of an old one.
    static let mergeThreshold = 0.8

    /// Index of an existing record the candidate should merge into, if any.
    /// Episodes merge only with the same conversation's episode.
    static func mergeTarget(for candidate: MemoryRecord, in records: [MemoryRecord]) -> Int? {
        if candidate.kind == .episode, let conv = candidate.sourceConversation {
            return records.firstIndex { $0.kind == .episode && $0.sourceConversation == conv }
        }
        var best: (Int, Double)?
        for (i, r) in records.enumerated() where r.kind == candidate.kind {
            let s = similarity(r.text, candidate.text)
            if s >= mergeThreshold, s > (best?.1 ?? 0) { best = (i, s) }
        }
        return best?.0
    }

    /// Folds `incoming` into `existing`: newer text wins, importance and usage
    /// accumulate, pinning is sticky. Text the user wrote is never overwritten
    /// by the model; otherwise the merged record takes the *least* trusted origin,
    /// so untrusted text can't launder itself by restating a trusted memory.
    static func merge(_ existing: MemoryRecord, with incoming: MemoryRecord, now: Date = .now) -> MemoryRecord {
        var m = existing
        if existing.trust != .user || incoming.trust == .user {
            m.text = incoming.text
            m.embedding = incoming.embedding
            m.embeddingModel = incoming.embeddingModel
        }
        m.importance = max(existing.importance, incoming.importance)
        m.pinned = existing.pinned || incoming.pinned
        m.updatedAt = now
        m.sourceConversation = incoming.sourceConversation ?? existing.sourceConversation
        switch (existing.trust, incoming.trust) {
        case (.user, _): m.trust = .user
        case (_, .untrusted), (.untrusted, _): m.trust = .untrusted
        default: m.trust = incoming.trust
        }
        return m
    }

    /// How much a record deserves to stay. Importance dominates; use and
    /// recency (half-life ~45 days since last touch) break ties.
    static func retention(_ r: MemoryRecord, now: Date = .now) -> Double {
        if r.pinned { return .infinity }
        let last = max(r.updatedAt, r.lastAccessedAt ?? r.updatedAt)
        let days = max(now.timeIntervalSince(last), 0) / 86_400
        let decay = pow(0.5, days / 45)
        let trustFactor = r.trust == .untrusted ? 0.7 : (r.trust == .user ? 1.3 : 1)
        return (Double(r.importance) + log2(1 + Double(r.accessCount))) * (0.3 + 0.7 * decay) * trustFactor
    }

    /// Drops the lowest-retention unpinned records until at most `cap` remain
    /// (per kind caps keep episodes from crowding out facts). Returns the
    /// survivors in their original order and the evicted ids.
    static func evict(_ records: [MemoryRecord], cap: Int, perKindCap: [MemoryKind: Int] = [:],
                      now: Date = .now) -> (kept: [MemoryRecord], evicted: [String]) {
        var evicted = Set<String>()
        func dropLowest(from pool: [MemoryRecord], keeping n: Int) {
            let candidates = pool.filter { !$0.pinned && !evicted.contains($0.id) }
            let excess = pool.filter { !evicted.contains($0.id) }.count - n
            guard excess > 0 else { return }
            for r in candidates.sorted(by: { retention($0, now: now) < retention($1, now: now) }).prefix(excess) {
                evicted.insert(r.id)
            }
        }
        for (kind, n) in perKindCap { dropLowest(from: records.filter { $0.kind == kind }, keeping: n) }
        dropLowest(from: records, keeping: cap)
        return (records.filter { !evicted.contains($0.id) }, evicted.sorted())
    }
}
