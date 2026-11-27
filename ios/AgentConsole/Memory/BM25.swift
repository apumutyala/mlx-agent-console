import Foundation

// Pure keyword retrieval: no dependencies, no I/O, deterministic. Always on,
// so recall works before (or without) the optional embedding model.

enum TextTokenizer {
    /// Words too common to say anything about relevance.
    static let stopwords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
        "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its",
        "me", "my", "no", "not", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "to", "too", "us", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "you", "your", "about", "would",
        "should", "could", "just", "also", "than", "very", "some", "any", "all", "been", "being",
    ]

    /// Lowercased alphanumeric terms with stopwords removed and a light
    /// suffix-stripping stem ("jobs"/"job", "running"/"run"). Numbers are kept.
    static func terms(_ text: String) -> [String] {
        var out: [String] = []
        var word = ""
        func flush() {
            defer { word = "" }
            guard !word.isEmpty, !stopwords.contains(word) else { return }
            out.append(stem(word))
        }
        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                word.unicodeScalars.append(scalar)
            } else {
                flush()
            }
        }
        flush()
        return out
    }

    /// Deliberately tiny stemmer: strips a few English suffixes when enough
    /// of the word remains. Good enough to match plurals and verb forms.
    static func stem(_ w: String) -> String {
        guard w.count > 3, w.allSatisfy(\.isLetter) else { return w }
        func strip(_ n: Int) -> String { String(w.dropLast(n)) }
        if w.hasSuffix("ies"), w.count > 4 { return strip(3) + "y" }                    // queries -> query
        if w.hasSuffix("sses") || w.hasSuffix("xes") || w.hasSuffix("ches") || w.hasSuffix("shes") {
            return strip(2)                                                              // classes -> class
        }
        if w.hasSuffix("s"), !w.hasSuffix("ss"), !w.hasSuffix("us"), !w.hasSuffix("is") {
            return strip(1)                                                              // jobs -> job
        }
        for suffix in ["ing", "ed"] where w.hasSuffix(suffix) && w.count - suffix.count >= 3 {
            let base = strip(suffix.count)
            // running -> runn -> run (but keep "fill", "pass", "buzz")
            if let last = base.last, base.count >= 4, base.dropLast().last == last, !"lsz".contains(last) {
                return String(base.dropLast())
            }
            return base
        }
        return w
    }
}

/// Okapi BM25 over a small in-memory corpus. Built per query; at the memory
/// store's cap (a few hundred short records) that takes well under a millisecond.
struct BM25Index {
    struct Document {
        let id: String
        let termFrequencies: [String: Int]
        let length: Int
    }

    let k1: Double
    let b: Double
    private(set) var documents: [Document] = []
    private var documentFrequency: [String: Int] = [:]
    private var totalLength = 0

    init(k1: Double = 1.2, b: Double = 0.75) {
        self.k1 = k1
        self.b = b
    }

    init(_ corpus: [(id: String, text: String)], k1: Double = 1.2, b: Double = 0.75) {
        self.init(k1: k1, b: b)
        for item in corpus { add(id: item.id, text: item.text) }
    }

    mutating func add(id: String, text: String) {
        let terms = TextTokenizer.terms(text)
        var tf: [String: Int] = [:]
        for t in terms { tf[t, default: 0] += 1 }
        for t in tf.keys { documentFrequency[t, default: 0] += 1 }
        documents.append(Document(id: id, termFrequencies: tf, length: terms.count))
        totalLength += terms.count
    }

    var averageLength: Double { documents.isEmpty ? 0 : Double(totalLength) / Double(documents.count) }

    /// BM25 "plus one" IDF: always positive, so a term in every document still
    /// counts a little instead of going negative.
    func idf(_ term: String) -> Double {
        let n = Double(documents.count)
        let df = Double(documentFrequency[term] ?? 0)
        return log(1 + (n - df + 0.5) / (df + 0.5))
    }

    /// Scores of every document with at least one query term, highest first.
    func search(_ query: String) -> [(id: String, score: Double)] {
        let qTerms = Array(Set(TextTokenizer.terms(query)))
        guard !qTerms.isEmpty, !documents.isEmpty else { return [] }
        let avg = max(averageLength, 1)
        var results: [(id: String, score: Double)] = []
        for doc in documents {
            var score = 0.0
            for t in qTerms {
                guard let f = doc.termFrequencies[t] else { continue }
                let tf = Double(f)
                let norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * Double(doc.length) / avg))
                score += idf(t) * norm
            }
            if score > 0 { results.append((doc.id, score)) }
        }
        return results.sorted { $0.score > $1.score }
    }
}

enum VectorMath {
    /// Cosine similarity; 0 for mismatched or empty vectors.
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return 0 }
        return Double(dot / (na.squareRoot() * nb.squareRoot()))
    }
}

/// Hybrid ranking: keyword relevance always, semantic relevance when vectors
/// exist, then mild boosts for importance, recency and pinning. Pure.
enum MemoryRanker {
    struct Scored: Sendable {
        let record: MemoryRecord
        /// Ranking score (relative: keyword part normalised to the best hit).
        let score: Double
        /// Absolute relevance in 0...1 for thresholds: the share of the
        /// query's IDF mass this record matches, or the rescaled cosine when
        /// that is higher. Independent of the other records.
        let relevance: Double
    }

    /// Weight of the embedding similarity when both signals exist.
    static let semanticWeight = 0.55
    /// Below this cosine, a semantic match is treated as noise.
    static let semanticFloor = 0.25

    static func rank(_ records: [MemoryRecord], query: String, queryVector: [Float]?,
                     embeddingModel: String?, now: Date = .now) -> [Scored] {
        let index = BM25Index(records.map { ($0.id, $0.text) })
        let keyword = Dictionary(index.search(query).map { ($0.id, $0.score) }, uniquingKeysWith: { max($0, $1) })
        let maxKeyword = keyword.values.max() ?? 0
        let qTerms = Set(TextTokenizer.terms(query))
        let qMass = qTerms.reduce(0.0) { $0 + index.idf($1) }

        var out: [Scored] = []
        for r in records {
            let kw = maxKeyword > 0 ? (keyword[r.id] ?? 0) / maxKeyword : 0   // 0...1
            var sem = 0.0
            if let q = queryVector, let v = r.vector, r.embeddingModel == embeddingModel {
                let c = VectorMath.cosine(q, v)
                sem = c > semanticFloor ? (c - semanticFloor) / (1 - semanticFloor) : 0
            }
            let relevance = queryVector == nil ? kw : (1 - semanticWeight) * kw + semanticWeight * sem
            var coverage = 0.0
            if qMass > 0, keyword[r.id] != nil {
                let docTerms = Set(TextTokenizer.terms(r.text))
                coverage = qTerms.intersection(docTerms).reduce(0.0) { $0 + index.idf($1) } / qMass
            }
            guard relevance > 0 || r.pinned else { continue }
            let days = now.timeIntervalSince(r.updatedAt) / 86_400
            let recency = exp(-max(days, 0) / 60)                     // ~2-month memory
            let boost = 0.75 + 0.05 * Double(r.importance) + 0.1 * recency + (r.pinned ? 0.2 : 0)
            out.append(Scored(record: r, score: max(relevance, r.pinned ? 0.15 : 0) * boost,
                              relevance: max(coverage, sem)))
        }
        return out.sorted { $0.score == $1.score ? $0.record.updatedAt > $1.record.updatedAt : $0.score > $1.score }
    }
}
