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
    }
}
