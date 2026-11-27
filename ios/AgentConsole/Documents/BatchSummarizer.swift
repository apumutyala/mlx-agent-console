import Foundation

/// Map-reduce summary of one document. Each chunk uses a fresh conversation so
/// the KV cache never grows past one chunk.
struct DocumentSummarizer: Sendable {
    let backend: any ModelBackend
    let documents: DocumentStore
    var chunkChars = 6_000

    /// Holds the shared GPU gate for the whole document, so a fleet task can't
    /// slip in between chunks (covers both batches and the chat's summarize tool).
    func summarize(_ doc: ImportedDocument) async throws -> String {
        try await GPUGate.shared.acquire("summary")
        do {
            let summary = try await summarizeHoldingGate(doc)
            await GPUGate.shared.release()
            return summary
        } catch {
            await GPUGate.shared.release()
            throw error
        }
    }

    private func summarizeHoldingGate(_ doc: ImportedDocument) async throws -> String {
        if !(await backend.isLoaded) { try await backend.load { _ in } }
        let text = try await documents.text(of: doc)
        guard !text.isEmpty else { return "(no extractable text)" }
        let chunks = stride(from: 0, to: text.count, by: chunkChars).map { start -> String in
            let s = text.index(text.startIndex, offsetBy: start)
            let e = text.index(s, offsetBy: chunkChars, limitedBy: text.endIndex) ?? text.endIndex
            return String(text[s..<e])
        }
        var partials: [String] = []
        for chunk in chunks {
            try Task.checkCancellation()
            let convo = try await backend.makeConversation(instructions: "Summarize faithfully and concisely. No preamble.")
            partials.append(try await convo.complete("Summarize this excerpt of \(doc.name) in under 120 words:\n\n\(chunk)"))
        }
        if partials.count == 1 { return partials[0] }
        let convo = try await backend.makeConversation(instructions: "Combine partial summaries into one. No preamble.")
        return try await convo.complete("Merge these partial summaries of \(doc.name) into one summary under 200 words:\n\n" +
                                        partials.enumerated().map { "[\($0.offset + 1)] \($0.element)" }.joined(separator: "\n\n"))
    }
}

/// A checkpointed batch: finished files are saved as soon as they complete,
/// so expiry/cancel/crash loses at most the file in progress.
struct SummaryBatch: Codable, Identifiable, Sendable {
    let id: String
    let fileIDs: [String]
    var outputs: [String: String] = [:]   // fileID -> summary
    var failures: [String: String] = [:]  // fileID -> error
    let createdAt: Date

    var remaining: [String] { fileIDs.filter { outputs[$0] == nil && failures[$0] == nil } }
    var isDone: Bool { remaining.isEmpty }
    var completedCount: Int { fileIDs.count - remaining.count }
}

actor BatchStore {
    private let dir: URL
    init(root: URL = .applicationSupportDirectory.appending(path: "Batches")) {
        dir = root
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    func save(_ b: SummaryBatch) throws {
        try JSONEncoder().encode(b).write(to: dir.appending(path: "\(b.id).json"), options: .atomic)
    }
    func load(_ id: String) -> SummaryBatch? {
        guard let d = try? Data(contentsOf: dir.appending(path: "\(id).json")) else { return nil }
        return try? JSONDecoder().decode(SummaryBatch.self, from: d)
    }
    func all() -> [SummaryBatch] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.compactMap { try? JSONDecoder().decode(SummaryBatch.self, from: Data(contentsOf: $0)) }
            .sorted { $0.createdAt > $1.createdAt }
    }
}

struct BatchRunner: Sendable {
    let store: BatchStore
    let summarizer: DocumentSummarizer
    let documents: DocumentStore

    /// Resumes from the first unfinished file. Returns the final batch state.
    /// Stops cleanly (keeping progress) when the surrounding Task is cancelled.
    func run(batchID: String, onProgress: @escaping @Sendable (SummaryBatch) -> Void) async -> SummaryBatch? {
        guard var batch = await store.load(batchID) else { return nil }
        onProgress(batch)
        for fileID in batch.remaining {
            if Task.isCancelled { break }
            guard let doc = await documents.document(id: fileID) else {
                batch.failures[fileID] = "file was removed"
                try? await store.save(batch)
                continue
            }
            do {
                batch.outputs[fileID] = try await summarizer.summarize(doc)
            } catch is CancellationError {
                break  // expired mid-file: this file restarts on resume
            } catch {
                if Task.isCancelled { break }  // e.g. a cancellation surfaced as another error type
                batch.failures[fileID] = error.localizedDescription
            }
            try? await store.save(batch)   // checkpoint at every file boundary
            onProgress(batch)
        }
        return batch
    }
}
