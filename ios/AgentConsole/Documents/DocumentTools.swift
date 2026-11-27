import Foundation

// Model tools that write plain-text documents into the app's DocumentStore
// (they show up in the Documents tab). Strict arguments, Swift validation,
// sanitised names, size caps; only .txt and .md are created or appended to.

enum DocumentNames {
    static let maxNameLength = 60
    static let writableExtensions: Set<String> = ["txt", "md"]

    /// A safe file name: no paths, no control or reserved characters, no
    /// leading dots, at most 60 characters, and a .txt/.md extension (added
    /// when missing or not allowed). Nil if nothing usable remains.
    static func sanitize(_ raw: String) -> String? {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_.()"))
        var base = String(String.UnicodeScalarView(raw.unicodeScalars.map { allowed.contains($0) ? $0 : "-" }))
        base = base.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        while base.contains("--") { base = base.replacingOccurrences(of: "--", with: "-") }
        while base.contains("..") { base = base.replacingOccurrences(of: "..", with: ".") }
        base = base.trimmingCharacters(in: CharacterSet(charactersIn: " .-"))
        let ext = (base as NSString).pathExtension.lowercased()
        var stem = writableExtensions.contains(ext) ? (base as NSString).deletingPathExtension : base
        stem = stem.trimmingCharacters(in: CharacterSet(charactersIn: " .-"))
        guard !stem.isEmpty else { return nil }
        let finalExt = writableExtensions.contains(ext) ? ext : "txt"
        stem = String(stem.prefix(maxNameLength - finalExt.count - 1))
        return "\(stem).\(finalExt)"
    }

    static func isWritable(_ name: String) -> Bool {
        writableExtensions.contains((name as NSString).pathExtension.lowercased())
    }

    /// `notes.txt`, `notes (2).txt`, ... avoiding `existing` (lowercased names).
    static func unique(_ name: String, existing: Set<String>) -> String {
        guard existing.contains(name.lowercased()) else { return name }
        let stem = (name as NSString).deletingPathExtension, ext = (name as NSString).pathExtension
        for n in 2...999 {
            let candidate = "\(stem) (\(n)).\(ext)"
            if !existing.contains(candidate.lowercased()) { return candidate }
        }
        return "\(stem)-\(UUID().uuidString.prefix(6)).\(ext)"
    }
}

enum DocumentLimits {
    static let maxCreateCharacters = 100_000
    static let maxDocumentCharacters = 200_000
    static let maxAppendCharacters = 50_000
}

struct CreateDocumentTool: AgentTool {
    struct Arguments: Decodable, Sendable {
        let name: String
        let text: String
    }
    static let name = "create_document"
    static let summary = "Create a new plain-text document (.txt or .md) in the user's Documents tab with the given text."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "name": ["type": "string", "description": "file name, e.g. notes.txt or plan.md"] as [String: any Sendable],
            "text": ["type": "string", "description": "full text content"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["name", "text"],
    ]
    static let keys: Set<String> = ["name", "text"]

    let documents: DocumentStore

    func validate(_ args: Arguments) async throws {
        guard DocumentNames.sanitize(args.name) != nil else { throw ToolError.invalidArguments("name has no usable characters") }
        guard !args.text.isEmpty else { throw ToolError.invalidArguments("text is empty") }
        guard args.text.count <= DocumentLimits.maxCreateCharacters else {
            throw ToolError.invalidArguments("text longer than \(DocumentLimits.maxCreateCharacters) characters")
        }
    }

    func run(_ args: Arguments) async throws -> String {
        guard let name = DocumentNames.sanitize(args.name) else { throw ToolError.invalidArguments("bad name") }
        let doc = try await documents.createTextDocument(name: name, text: args.text)
        return "Created document \(doc.id) “\(doc.name)” (\(args.text.count) characters). It's in the Documents tab."
    }
}

struct AppendToDocumentTool: AgentTool {
    struct Arguments: Decodable, Sendable {
        let id_or_name: String
        let text: String
    }
    static let name = "append_to_document"
    static let summary = "Append text to the end of an existing .txt or .md document, found by id or file name."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "id_or_name": ["type": "string", "description": "document id or file name"] as [String: any Sendable],
            "text": ["type": "string", "description": "text to append"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["id_or_name", "text"],
    ]
    static let keys: Set<String> = ["id_or_name", "text"]

    let documents: DocumentStore
    /// Set in `validate` for documents the user imported: appending to those
    /// needs the user's approval (confirmation is synchronous).
    private let importedTarget = PreviewBox()

    init(documents: DocumentStore) { self.documents = documents }

    func validate(_ args: Arguments) async throws {
        guard let doc = await documents.resolve(args.id_or_name) else {
            throw ToolError.invalidArguments("no document \(args.id_or_name); use list_documents")
        }
        guard DocumentNames.isWritable(doc.name) else {
            throw ToolError.invalidArguments("\(doc.name) is not a .txt or .md document")
        }
        guard await documents.isUTF8Text(doc) else {
            throw ToolError.invalidArguments("\(doc.name) is not readable UTF-8 text; create a new document instead")
        }
        guard !args.text.isEmpty, args.text.count <= DocumentLimits.maxAppendCharacters else {
            throw ToolError.invalidArguments("text must be 1...\(DocumentLimits.maxAppendCharacters) characters")
        }
        importedTarget.set(doc.agentCreated == true ? "" : doc.name)
    }

    func confirmationPrompt(_ args: Arguments) -> String? {
        guard let name = importedTarget.get(), !name.isEmpty else { return nil }
        return "Append \(args.text.count) characters to your imported file “\(name)”?"
    }

    func run(_ args: Arguments) async throws -> String {
        guard let doc = await documents.resolve(args.id_or_name) else { throw ToolError.invalidArguments("document vanished") }
        do {
            let length = try await documents.appendText(id: doc.id, text: args.text,
                                                        maxLength: DocumentLimits.maxDocumentCharacters)
            return "Appended to \(doc.id) “\(doc.name)” (now \(length) characters)."
        } catch let e as CocoaError where e.code == .fileWriteOutOfSpace {
            throw ToolError.invalidArguments("document would exceed \(DocumentLimits.maxDocumentCharacters) characters")
        }
    }
}

struct ListDocumentsTool: AgentTool {
    struct Arguments: Decodable, Sendable {}
    static let name = "list_documents"
    static let summary = "List the documents in the user's Documents tab (id, name, size)."
    static let parametersSchema: [String: any Sendable] = [
        "type": "object",
        "properties": [:] as [String: any Sendable],
    ]
    static let keys: Set<String> = []

    let documents: DocumentStore

    func validate(_ args: Arguments) async throws {}

    func run(_ args: Arguments) async throws -> String {
        let docs = await documents.all()
        guard !docs.isEmpty else { return "No documents yet." }
        var lines: [String] = []
        for d in docs.suffix(40) {
            let bytes = await documents.size(of: d)
            lines.append("- \(d.id): \(d.name) (\(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)))")
        }
        return lines.joined(separator: "\n")
    }
}

extension ToolRegistry {
    mutating func registerDocumentTools(_ documents: DocumentStore) {
        register(CreateDocumentTool(documents: documents), keys: CreateDocumentTool.keys)
        register(AppendToDocumentTool(documents: documents), keys: AppendToDocumentTool.keys)
        register(ListDocumentsTool(documents: documents), keys: ListDocumentsTool.keys)
    }
}
