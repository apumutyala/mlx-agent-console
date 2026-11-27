import Foundation

// The reflection pass asks the local model for STRICT JSON. Everything it
// returns is validated here, in Swift, before anything is stored: unknown keys,
// wrong types, oversize text, too many items or control markup reject the
// whole reply (or the single offending item, where noted). Reflection output
// is data only: nothing in it is ever executed as a tool call.

struct ReflectionResult: Equatable, Sendable {
    struct Fact: Equatable, Sendable {
        let kind: MemoryKind      // .fact, .lesson or .world (never .episode)
        let text: String
        let importance: Int
    }

    struct LessonEdit: Codable, Hashable, Sendable {
        enum Op: String, Codable, Sendable { case add, revise, remove }
        let op: Op
        /// Required for revise/remove: an existing playbook entry id.
        let id: String?
        /// Required for add/revise.
        let text: String?
    }

    let facts: [Fact]
    /// One-paragraph summary of the conversation (nil = nothing worth keeping).
    let episode: String?
    let playbook: [LessonEdit]
}

enum ReflectionValidator {
    static let maxFacts = 6
    static let maxEdits = 3
    static let maxFactLength = 240
    static let maxEpisodeLength = 600
    static let maxLessonLength = 240
    static let maxReplyLength = 12_000

    enum Failure: Error, Equatable, CustomStringConvertible {
        case noJSONObject
        case notAnObject
        case unexpectedKeys([String])
        case wrongType(String)
        case tooMany(String)
        case tooLong(String)
        case invalid(String)

        var description: String {
            switch self {
            case .noJSONObject: "no JSON object in reply"
            case .notAnObject: "top level is not an object"
            case .unexpectedKeys(let k): "unexpected keys \(k)"
            case .wrongType(let f): "wrong type for \(f)"
            case .tooMany(let f): "too many items in \(f)"
            case .tooLong(let f): "\(f) too long"
            case .invalid(let m): m
            }
        }
    }

    /// The first balanced `{…}` in `reply` (models like to wrap JSON in
    /// ```json fences or add a sentence). String contents are respected, so
    /// braces inside strings don't confuse the scan. Only the extraction is
    /// lenient; the object itself must validate strictly.
    static func extractObject(_ reply: String) -> String? {
        let chars = Array(reply.prefix(maxReplyLength))
        guard let start = chars.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        for i in start..<chars.count {
            let c = chars[i]
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"": inString = true
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 { return String(chars[start...i]) }
            default: break
            }
        }
        return nil
    }

    /// Validates a reply against the schema:
    /// ```
    /// { "facts":    [ { "text": String, "kind": "fact"|"lesson"|"world"?, "importance": 1...5? } ],
    ///   "episode":  String | null,
    ///   "playbook": [ { "op": "add"|"revise"|"remove", "id": String?, "text": String? } ] }
    /// ```
    /// All three keys are optional; no others are allowed. `knownEntryIDs` are
    /// the current playbook ids (revise/remove of anything else is rejected).
    static func validate(_ reply: String, knownEntryIDs: Set<String>) -> Result<ReflectionResult, Failure> {
        guard let json = extractObject(reply), let data = json.data(using: .utf8) else { return .failure(.noJSONObject) }
        guard let any = try? JSONSerialization.jsonObject(with: data), let obj = any as? [String: Any] else {
            return .failure(.notAnObject)
        }
        let extra = Set(obj.keys).subtracting(["facts", "episode", "playbook"])
        guard extra.isEmpty else { return .failure(.unexpectedKeys(extra.sorted())) }

        // facts
        var facts: [ReflectionResult.Fact] = []
        if let raw = obj["facts"], !(raw is NSNull) {
            guard let list = raw as? [Any] else { return .failure(.wrongType("facts")) }
            guard list.count <= maxFacts else { return .failure(.tooMany("facts")) }
            for item in list {
                guard let f = item as? [String: Any] else { return .failure(.wrongType("facts[]")) }
                let extra = Set(f.keys).subtracting(["text", "kind", "importance"])
                guard extra.isEmpty else { return .failure(.unexpectedKeys(extra.sorted())) }
                guard let text = f["text"] as? String else { return .failure(.wrongType("facts[].text")) }
                guard text.count <= maxFactLength else { return .failure(.tooLong("facts[].text")) }
                var kind = MemoryKind.fact
                if let k = f["kind"] {
                    guard let ks = k as? String, let parsed = MemoryKind(loose: ks), parsed != .episode else {
                        return .failure(.wrongType("facts[].kind"))
                    }
                    kind = parsed
                }
                var importance = 3
                if let i = f["importance"] {
                    // JSONSerialization yields NSNumber; reject booleans and fractions.
                    guard let n = i as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                          n.doubleValue == n.doubleValue.rounded(), (1...5).contains(n.intValue) else {
                        return .failure(.wrongType("facts[].importance"))
                    }
                    importance = n.intValue
                }
                // Per-item drops (not whole-reply failures): blank or markup-laden text.
                guard let clean = MemoryText.clean(text, maxLength: maxFactLength),
                      !MemoryText.looksLikeControlMarkup(clean) else { continue }
                facts.append(.init(kind: kind, text: clean, importance: importance))
            }
        }

        // episode
        var episode: String?
        if let raw = obj["episode"], !(raw is NSNull) {
            guard let s = raw as? String else { return .failure(.wrongType("episode")) }
            guard s.count <= maxEpisodeLength else { return .failure(.tooLong("episode")) }
            if let clean = MemoryText.clean(s, maxLength: maxEpisodeLength), !MemoryText.looksLikeControlMarkup(clean) {
                episode = clean
            }
        }

        // playbook
        var edits: [ReflectionResult.LessonEdit] = []
        if let raw = obj["playbook"], !(raw is NSNull) {
            guard let list = raw as? [Any] else { return .failure(.wrongType("playbook")) }
            guard list.count <= maxEdits else { return .failure(.tooMany("playbook")) }
            for item in list {
                guard let e = item as? [String: Any] else { return .failure(.wrongType("playbook[]")) }
                let extra = Set(e.keys).subtracting(["op", "id", "text"])
                guard extra.isEmpty else { return .failure(.unexpectedKeys(extra.sorted())) }
                guard let opRaw = e["op"] as? String, let op = ReflectionResult.LessonEdit.Op(rawValue: opRaw) else {
                    return .failure(.wrongType("playbook[].op"))
                }
                guard let idField = optionalString(e["id"]) else { return .failure(.wrongType("playbook[].id")) }
                let id = idField.value
                guard let textField = optionalString(e["text"]) else { return .failure(.wrongType("playbook[].text")) }
                var text: String?
                if let s = textField.value {
                    guard s.count <= maxLessonLength else { return .failure(.tooLong("playbook[].text")) }
                    text = MemoryText.clean(s, maxLength: maxLessonLength)
                }
                if let text, MemoryText.looksLikeControlMarkup(text) { continue }
                switch op {
                case .add:
                    guard let text else { return .failure(.invalid("add needs text")) }
                    edits.append(.init(op: .add, id: nil, text: text))
                case .revise:
                    guard let id, knownEntryIDs.contains(id) else { return .failure(.invalid("revise of unknown id")) }
                    guard let text else { return .failure(.invalid("revise needs text")) }
                    edits.append(.init(op: .revise, id: id, text: text))
                case .remove:
                    guard let id, knownEntryIDs.contains(id) else { return .failure(.invalid("remove of unknown id")) }
                    edits.append(.init(op: .remove, id: id, text: nil))
                }
            }
        }
        return .success(ReflectionResult(facts: facts, episode: episode, playbook: edits))
    }

    /// Absent / null / string are fine (wrapped); any other JSON type is nil.
    private struct OptionalField { let value: String? }
    private static func optionalString(_ v: Any?) -> OptionalField? {
        guard let v else { return OptionalField(value: nil) }
        if v is NSNull { return OptionalField(value: nil) }
        if let s = v as? String { return OptionalField(value: s) }
        return nil
    }
}
