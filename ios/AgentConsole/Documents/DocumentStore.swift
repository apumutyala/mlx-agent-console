import Foundation
import PDFKit

struct ImportedDocument: Codable, Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let storedFilename: String
    let importedAt: Date
}

/// Imported files are copied into the app container, so summarization works
/// offline and doesn't depend on security-scoped access later.
actor DocumentStore {
    private let dir: URL
    private let indexURL: URL
    private var docs: [ImportedDocument] = []

    init(root: URL = .applicationSupportDirectory.appending(path: "Documents")) {
        dir = root
        indexURL = root.appending(path: "index.json")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let data = try? Data(contentsOf: indexURL),
           let saved = try? JSONDecoder().decode([ImportedDocument].self, from: data) {
            docs = saved
        }
    }

    func all() -> [ImportedDocument] { docs }
    func document(id: String) -> ImportedDocument? { docs.first { $0.id == id } }

    @discardableResult
    func importFile(from url: URL) throws -> ImportedDocument {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let id = String(UUID().uuidString.prefix(8)).lowercased()
        let stored = "\(id)-\(url.lastPathComponent)"
        try FileManager.default.copyItem(at: url, to: dir.appending(path: stored))
        let doc = ImportedDocument(id: id, name: url.lastPathComponent, storedFilename: stored, importedAt: .now)
        docs.append(doc)
        try save()
        return doc
    }

    func remove(_ id: String) throws {
        guard let doc = document(id: id) else { return }
        try? FileManager.default.removeItem(at: dir.appending(path: doc.storedFilename))
        docs.removeAll { $0.id == id }
        try save()
    }

    func text(of doc: ImportedDocument) throws -> String {
        let url = dir.appending(path: doc.storedFilename)
        if url.pathExtension.lowercased() == "pdf" {
            guard let pdf = PDFDocument(url: url) else { throw CocoaError(.fileReadCorruptFile) }
            return pdf.string ?? ""
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func save() throws {
        try JSONEncoder().encode(docs).write(to: indexURL, options: .atomic)
        Task { @MainActor in NotificationCenter.default.post(name: .documentsChanged, object: nil) }
    }

    // MARK: Agent-written text documents (see DocumentTools.swift)

    /// Creates a UTF-8 text document. `name` must already be sanitized
    /// (`DocumentNames.sanitize`); a numeric suffix keeps names unique.
    func createTextDocument(name: String, text: String) throws -> ImportedDocument {
        let unique = DocumentNames.unique(name, existing: Set(docs.map { $0.name.lowercased() }))
        let id = String(UUID().uuidString.prefix(8)).lowercased()
        let stored = "\(id)-\(unique)"
        try Data(text.utf8).write(to: dir.appending(path: stored), options: .atomic)
        let doc = ImportedDocument(id: id, name: unique, storedFilename: stored, importedAt: .now)
        docs.append(doc)
        try save()
        return doc
    }

    /// Appends to a text document and returns its new length in characters.
    func appendText(id: String, text: String, maxLength: Int) throws -> Int {
        guard let doc = document(id: id), DocumentNames.isWritable(doc.name) else {
            throw CocoaError(.fileWriteNoPermission)
        }
        let url = dir.appending(path: doc.storedFilename)
        let current = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        let joined = current.isEmpty || current.hasSuffix("\n") ? current + text : current + "\n" + text
        guard joined.count <= maxLength else { throw CocoaError(.fileWriteOutOfSpace) }
        try Data(joined.utf8).write(to: url, options: .atomic)
        try save()
        return joined.count
    }

    /// Looks a document up by id, or by name (case-insensitive, extension optional).
    func resolve(_ idOrName: String) -> ImportedDocument? {
        let key = idOrName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let d = docs.first(where: { $0.id == key }) { return d }
        if let d = docs.last(where: { $0.name.lowercased() == key }) { return d }
        return docs.last { ($0.name as NSString).deletingPathExtension.lowercased() == key }
    }

    func size(of doc: ImportedDocument) -> Int {
        let url = dir.appending(path: doc.storedFilename)
        return ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int) ?? 0
    }
}

extension Notification.Name {
    /// Posted after the document index changes (import, removal, agent writes).
    static let documentsChanged = Notification.Name("AgentConsole.documentsChanged")
}
