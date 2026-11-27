import Foundation

// Pure prompt-budget helpers. The small on-device models have short useful
// context, so everything the harness adds to the prompt is measured here.

enum TokenEstimate {
    /// ~4 characters per token for English with these tokenizers. Rounds up,
    /// so a budget computed from it is conservative.
    static let charsPerToken = 4

    static func tokens(_ text: String) -> Int { (text.count + charsPerToken - 1) / charsPerToken }

    /// Longest prefix of `text` that fits in `tokens`, cut at a word boundary,
    /// with an ellipsis when something was dropped.
    static func truncate(_ text: String, toTokens tokens: Int) -> String {
        let limit = max(tokens, 0) * charsPerToken
        guard text.count > limit else { return text }
        guard limit > 1 else { return "" }
        var cut = String(text.prefix(limit - 1))
        if let space = cut.lastIndex(where: \.isWhitespace), cut.distance(from: cut.startIndex, to: space) > limit / 2 {
            cut = String(cut[..<space])
        }
        return cut + "…"
    }
}

enum BudgetPacker {
    struct Item<ID: Hashable> {
        let id: ID
        let text: String
    }

    /// Greedy packing in the given (priority) order: take each item that still
    /// fits, skip ones that don't, and stop after `maxItems`. Skipping keeps a
    /// long low-value item from crowding out several short useful ones.
    /// `separatorTokens` is charged per item for its bullet/newline.
    static func pack<ID: Hashable>(_ items: [Item<ID>], budgetTokens: Int, maxItems: Int = .max,
                                   separatorTokens: Int = 1) -> [Item<ID>] {
        var used = 0
        var out: [Item<ID>] = []
        var seen = Set<ID>()
        for item in items where out.count < maxItems {
            guard !seen.contains(item.id) else { continue }
            let cost = TokenEstimate.tokens(item.text) + separatorTokens
            guard used + cost <= budgetTokens else { continue }
            used += cost
            out.append(item)
            seen.insert(item.id)
        }
        return out
    }

    /// Keeps the most recent turns of a transcript that fit the budget
    /// (oldest dropped first), returned in chronological order.
    static func tail(_ lines: [String], budgetTokens: Int) -> [String] {
        var used = 0
        var out: [String] = []
        for line in lines.reversed() {
            let cost = TokenEstimate.tokens(line) + 1
            guard used + cost <= budgetTokens else { break }
            used += cost
            out.append(line)
        }
        return out.reversed()
    }
}
