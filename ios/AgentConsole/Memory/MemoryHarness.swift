import Foundation
import Observation
import SwiftUI

/// Coordinator for the agent's persistent memory: chat history, typed
/// memories, the self-maintained playbook, context assembly and reflection.
///
/// Hooks (kept small so other branches merge easily):
/// - LocalAgent asks for `systemContext` / `turnContext`, calls `persist`,
///   `conversationEnded`, and `preemptBackgroundWork` before generating.
/// - AppModel registers `memoryTools` and forwards scene-phase changes.
///
/// GPU discipline: reflection and embedding run only while the agent is idle
/// and are cancelled (and awaited) before a chat turn generates, so two model
/// workloads never share the GPU. In the background (no Metal), nothing runs;
/// pending reflections drain on the next foreground.
@MainActor @Observable
final class MemoryHarness {
    // MARK: Settings (UserDefaults; read by the Memory screen via @AppStorage)

    enum Key {
        static let learn = "memory.learnFromChats"
        static let useMemory = "memory.useInChats"
        static let embeddings = "memory.embeddings"
        static let budget = "memory.budgetTokens"
    }

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            Key.learn: true,
            Key.useMemory: true,
            // 230 MB extra download + ~300 MB resident: on by default only on big iPads.
            Key.embeddings: ModelCatalog.deviceMemoryGB >= 11,
            Key.budget: 1200,
        ])
    }

    var learnFromChats: Bool { UserDefaults.standard.bool(forKey: Key.learn) }
    var useMemory: Bool { UserDefaults.standard.bool(forKey: Key.useMemory) }
    var embeddingsEnabled: Bool { UserDefaults.standard.bool(forKey: Key.embeddings) }
    var budgetTokens: Int { min(max(UserDefaults.standard.integer(forKey: Key.budget), 200), 4000) }

    /// Tool results from these tools came from another machine: untrusted.
    nonisolated static func isRemoteTool(_ name: String) -> Bool {
        let n = name.lowercased()
        return ["worker", "fleet", "job", "node", "remote"].contains(where: n.contains)
    }

    // MARK: State

    nonisolated let memories = MemoryStore()
    nonisolated let playbook = PlaybookStore()
    let conversations = ConversationStore()
    nonisolated let embedder = MemoryEmbedder()
    @ObservationIgnored private let reporter = SituationReporter()
    private let backend: any ModelBackend
    private let documents: DocumentStore
    @ObservationIgnored weak var agent: LocalAgent?

    /// Human-readable status of the last background pass (Memory screen).
    private(set) var status = "Idle"
    private(set) var isWorking = false
    private(set) var embedderLoaded = false
    /// Bumped whenever memories/playbook change, so views reload.
    private(set) var revision = 0

    /// Memory ids already shown to the model, per conversation (no repeats).
    @ObservationIgnored private var injected: [UUID: Set<String>] = [:]
    @ObservationIgnored private var backgroundTask: Task<Void, Never>?
    @ObservationIgnored private var defaultsObserver: NSObjectProtocol?
    /// Reflect mid-conversation after this many new user turns.
    static let reflectEveryUserTurns = 6

    init(backend: any ModelBackend, documents: DocumentStore) {
        self.backend = backend
        self.documents = documents
        Self.registerDefaults()
        // Model picker / settings changes: drop the embedder at once if the
        // newly selected model needs the memory (or embeddings were turned off).
        defaultsObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.enforceEmbedderPolicy() }
        }
    }

    var embedderAllowed: Bool {
        embeddingsEnabled && MemoryEmbedder.allowed(alongside: ModelCatalog.selected)
    }

    func enforceEmbedderPolicy() {
        guard !embedderAllowed else { return }
        Task {
            await embedder.unload()
            embedderLoaded = false
        }
    }

    private func changed() { revision += 1 }

    // MARK: Context assembly

    /// Appended to the system prompt when a conversation starts (or resumes).
    func systemContext(firstMessage: String, conversationID: UUID, resumed: [StoredTurn]) async -> String {
        let docs = await documents.all()
        let situation = await reporter.report(modelName: backend.displayName, documents: docs)
        let entries = await playbook.entries
        var chosen: [MemoryRecord] = []
        if useMemory {
            let query = ([firstMessage] + resumed.suffix(4).filter { $0.role == .user }.map(\.text)).joined(separator: " ")
            chosen = await retrieve(query, budget: budgetTokens, excluding: [])
            injected[conversationID, default: []].formUnion(chosen.map(\.id))
        }
        let budget = ContextAssembler.Budget(memories: budgetTokens)
        return ContextAssembler.systemBlock(situation: situation, playbook: useMemory ? entries : [],
                                            memories: chosen, resumed: resumed, budget: budget)
    }

    /// Note prepended to a later user message with newly relevant memories.
    func turnContext(_ message: String, conversationID: UUID) async -> String? {
        guard useMemory else { return nil }
        let seen = injected[conversationID] ?? []
        let chosen = await retrieve(message, budget: max(budgetTokens / 3, 150), excluding: seen, maxItems: 4,
                                    minScore: 0.25)
        injected[conversationID, default: []].formUnion(chosen.map(\.id))
        return ContextAssembler.turnNote(chosen)
    }

    private func retrieve(_ query: String, budget: Int, excluding: Set<String>, maxItems: Int = 12,
                          minScore: Double = 0) async -> [MemoryRecord] {
        var qv: [Float]?
        if embedderAllowed, await embedder.isLoaded {
            qv = try? await embedder.embed([query], role: .query).first
        }
        let ranked = await memories.search(query, queryVector: qv, embeddingModel: MemoryEmbedder.modelID, limit: 40)
            .filter { $0.score >= minScore }
            .map(\.record)
        let chosen = ContextAssembler.packMemories(ranked, budget: budget, maxItems: maxItems, excluding: excluding)
        await memories.markAccessed(chosen.map(\.id))
        return chosen
    }

    // MARK: Persistence

    func persist(conversationID: UUID, turns: [StoredTurn]) {
        conversations.save(id: conversationID, turns: turns, modelName: backend.displayName)
        if let c = conversations.load(conversationID),
           c.turns[c.reflectedTurnCount...].filter({ $0.role == .user }).count >= Self.reflectEveryUserTurns {
            scheduleBackgroundWork(includeActive: true)
        }
    }

    func conversationEnded(_ id: UUID) {
        injected[id] = nil
        reporter.invalidate()
        scheduleBackgroundWork(includeActive: false)
    }

    func storedTurns(_ id: UUID) -> [StoredTurn]? { conversations.load(id)?.turns }

    // MARK: Scene phase

    func scenePhaseChanged(_ phase: ScenePhase) {
        switch phase {
        case .background:
            // No Metal in the background: stop now, resume on next foreground.
            backgroundTask?.cancel()
        case .active:
            scheduleBackgroundWork(includeActive: false)
        default: break
        }
    }

    // MARK: Background work (reflection + embeddings)

    /// Cancels reflection/embedding and waits (bounded) for it to stop, so a
    /// chat turn has the GPU to itself. Call before every generation.
    func preemptBackgroundWork() async {
        guard let task = backgroundTask else { return }
        task.cancel()
        _ = try? await withTimeout(5) { await task.value }
    }

    func scheduleBackgroundWork(includeActive: Bool) {
        guard backgroundTask == nil else { return }
        backgroundTask = Task { [weak self] in
            // Let the UI settle (and a follow-up message start) before using the GPU.
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            await self?.runBackgroundWork(includeActive: includeActive)
            self?.backgroundTask = nil
            self?.isWorking = false
        }
    }

    /// Manual "Reflect now" from the Memory screen (includes the open chat).
    func reflectNow() { scheduleBackgroundWork(includeActive: true) }

    private var agentBusy: Bool { agent?.isRunning ?? false }

    private func runBackgroundWork(includeActive: Bool) async {
        guard !Task.isCancelled, !agentBusy else { return }
        isWorking = true
        if learnFromChats, await backend.isLoaded {
            let active = includeActive ? nil : agent?.conversationID
            for id in conversations.needingReflection(excluding: active).suffix(5) {
                guard !Task.isCancelled, !agentBusy else { return }
                await reflect(conversationID: id)
            }
        }
        guard !Task.isCancelled, !agentBusy else { return }
        await refreshEmbeddings()
        let n = await memories.count
        if !Task.isCancelled { status = "Up to date · \(n) memories" }
        changed()
    }

    private func reflect(conversationID id: UUID) async {
        guard let conv = conversations.load(id) else { return }
        let start = conv.reflectedTurnCount
        let fresh = Array(conv.turns[start...])
        guard fresh.contains(where: { $0.role == .user }) else { return }
        status = "Reflecting on “\(conv.title)”…"
        let entries = await playbook.entries
        let previous = await memories.all().first { $0.kind == .episode && $0.sourceConversation == id }
        // Give the model a little context from before the watermark.
        let context = Array(conv.turns[max(0, start - 4)...])
        do {
            let result = try await Reflector(backend: backend)
                .reflect(turns: context, playbook: entries, previousEpisode: previous?.text)
            try Task.checkCancellation()
            switch result {
            case .success(let r):
                try await apply(r, from: conv)
                status = "Learned from “\(conv.title)”"
            case .failure(let why):
                status = "Reflection skipped (\(why))"
            }
            // Advance even on invalid output: one bad reply shouldn't loop forever.
            conversations.setReflected(id, upTo: conv.turns.count)
        } catch is CancellationError {
            status = "Paused"
        } catch {
            status = "Reflection failed: \(error.localizedDescription)"
        }
    }

    /// Stores a validated reflection. Chats containing remote text produce
    /// untrusted memories and only *pending* playbook edits (user approves).
    private func apply(_ r: ReflectionResult, from conv: StoredConversation) async throws {
        let tainted = conv.containsRemoteText
        let trust: MemoryTrust = tainted ? .untrusted : .model
        for f in r.facts {
            _ = try? await memories.upsert(MemoryRecord(kind: f.kind, text: f.text, sourceConversation: conv.id,
                                                        importance: f.importance, trust: trust))
        }
        if let e = r.episode {
            let text = "\(conv.title): \(e)"
            _ = try? await memories.upsert(MemoryRecord(kind: .episode, text: text, sourceConversation: conv.id,
                                                        importance: 2, trust: trust))
        }
        if !r.playbook.isEmpty {
            if tainted {
                try await playbook.enqueuePending(r.playbook, source: conv.id)
            } else {
                try await playbook.apply(r.playbook, reason: "reflection on “\(conv.title)”")
            }
        }
        changed()
    }

    /// Loads the embedder if allowed and fills in missing vectors.
    private func refreshEmbeddings() async {
        guard embedderAllowed else { return }
        do {
            if !(await embedder.isLoaded) {
                status = "Loading embedding model…"
                try await embedder.load()
            }
            embedderLoaded = true
            while !Task.isCancelled, !agentBusy {
                let batch = await memories.missingEmbeddings(model: MemoryEmbedder.modelID, limit: 16)
                guard !batch.isEmpty else { break }
                status = "Indexing memories…"
                let vectors = try await embedder.embed(batch.map(\.text), role: .document)
                for (r, v) in zip(batch, vectors) {
                    await memories.setEmbedding(id: r.id, vector: v, model: MemoryEmbedder.modelID)
                }
            }
        } catch is CancellationError {
        } catch {
            status = "Embeddings unavailable: \(error.localizedDescription)"
        }
    }

    // MARK: Tool support

    /// Embeds one fresh record right away when the embedder is resident
    /// (called from tools, between generations).
    func embedIfPossible(_ record: MemoryRecord) async {
        guard embedderAllowed, await embedder.isLoaded,
              let v = try? await embedder.embed([record.text], role: .document).first else { return }
        await memories.setEmbedding(id: record.id, vector: v, model: MemoryEmbedder.modelID)
    }

    func isTainted(_ conversationID: UUID) -> Bool {
        guard let agent, agent.conversationID == conversationID else {
            return conversations.load(conversationID)?.containsRemoteText ?? false
        }
        return agent.turns.contains(where: \.remote)
    }

    func noteChanged() { changed() }

    // MARK: User actions (Memory screen)

    func rollbackPlaybook(to version: Int) async {
        try? await playbook.rollback(to: version)
        changed()
    }

    func resolvePending(_ id: UUID, approve: Bool) async {
        try? await playbook.resolvePending(id, approve: approve)
        changed()
    }

    // MARK: Export / import

    struct ExportFile: Codable {
        var format = "agent-console-memory"
        var version = 1
        var exportedAt = Date.now
        var memories: [MemoryRecord]
        var playbook: [PlaybookEntry]
    }

    /// Writes a JSON export to a temporary file for the share sheet.
    func exportFile() async throws -> URL {
        var mems = await memories.all()
        for i in mems.indices { mems[i].embedding = nil; mems[i].embeddingModel = nil }   // re-derivable, large
        let file = ExportFile(memories: mems, playbook: await playbook.entries)
        let url = URL.temporaryDirectory.appending(path: "agent-memory-\(Int(Date.now.timeIntervalSince1970)).json")
        try AtomicJSON.write(file, to: url)
        return url
    }

    /// Imports memories (deduplicated) and offers the file's playbook entries
    /// as pending suggestions: a file is not the user typing, so it can't
    /// rewrite the playbook without approval. Returns a summary line.
    func importFile(_ url: URL) async throws -> String {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url)
        guard data.count < 20_000_000 else { throw CocoaError(.fileReadTooLarge) }
        let file = try AtomicJSON.decoder.decode(ExportFile.self, from: data)
        guard file.format == "agent-console-memory" else { throw CocoaError(.fileReadCorruptFile) }
        let n = try await memories.importRecords(file.memories)
        let edits = file.playbook.prefix(PlaybookStore.maxEntries).map {
            ReflectionResult.LessonEdit(op: .add, id: nil, text: $0.text)
        }
        try await playbook.enqueuePending(Array(edits), source: nil)
        changed()
        return "Imported \(n) memories; \(edits.count) playbook notes await your approval."
    }
}
