import SwiftUI
import UniformTypeIdentifiers

/// Drop-in section for SettingsView's Form.
struct MemorySettingsSection: View {
    let memory: MemoryHarness
    @AppStorage(MemoryHarness.Key.learn) private var learn = true

    var body: some View {
        Section {
            NavigationLink {
                MemoryScreen(memory: memory)
            } label: {
                LabeledContent("Memory", value: learn ? "learning" : "not learning")
            }
        } header: { Text("Memory") } footer: {
            Text("Chats, memories and the agent's playbook stay on this device.")
        }
    }
}

struct MemoryScreen: View {
    let memory: MemoryHarness
    @AppStorage(MemoryHarness.Key.learn) private var learn = true
    @AppStorage(MemoryHarness.Key.useMemory) private var useMemory = true
    @AppStorage(MemoryHarness.Key.embeddings) private var embeddings = false
    @AppStorage(MemoryHarness.Key.budget) private var budget = 1200
    @AppStorage(ModelCatalog.storageKey) private var selectedModelID = ModelCatalog.defaultEntry.id
    @State private var count = 0
    @State private var pendingCount = 0
    @State private var exportURL: URL?
    @State private var importing = false
    @State private var message: String?
    @State private var confirmErase = false

    var body: some View {
        Form {
            Section {
                Toggle("Learn from chats", isOn: $learn)
                Toggle("Use memory in chats", isOn: $useMemory)
                Toggle("Semantic search (embeddings)", isOn: $embeddings)
                Stepper("Memory budget: \(budget) tokens", value: $budget, in: 200...4000, step: 200)
            } footer: {
                Text(embeddingFooter)
            }
            Section {
                NavigationLink("Memories (\(count))") { MemoryListView(memory: memory) }
                NavigationLink {
                    PlaybookView(memory: memory)
                } label: {
                    LabeledContent("Playbook", value: pendingCount > 0 ? "\(pendingCount) to review" : "")
                }
                LabeledContent("Status", value: memory.status)
                Button(memory.isWorking ? "Working…" : "Reflect on chats now") { memory.reflectNow() }
                    .disabled(memory.isWorking || !learn)
            } footer: {
                Text("After each chat the on-device model writes a summary, extracts facts and may revise its playbook. Text from your worker or fleet is marked untrusted, and playbook changes based on it wait for your approval.")
            }
            Section("Backup") {
                if let exportURL {
                    ShareLink("Share memory export", item: exportURL)
                } else {
                    Button("Export memory (JSON)") {
                        Task {
                            do { exportURL = try await memory.exportFile() } catch { message = error.localizedDescription }
                        }
                    }
                }
                Button("Import memory…") { importing = true }
                Button("Erase all memories", role: .destructive) { confirmErase = true }
            }
            if let message { Text(message).font(.footnote) }
        }
        .navigationTitle("Memory")
        .task(id: memory.revision) {
            count = await memory.memories.count
            pendingCount = await memory.playbook.pending().count
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.json]) { result in
            Task {
                do { message = try await memory.importFile(result.get()) } catch { message = error.localizedDescription }
            }
        }
        .confirmationDialog("Erase all memories?", isPresented: $confirmErase, titleVisibility: .visible) {
            Button("Erase memories", role: .destructive) {
                Task {
                    try? await memory.memories.deleteAll()
                    memory.noteChanged()
                }
            }
        } message: {
            Text("Facts, episodes, lessons and world notes are deleted. Saved chats and the playbook are kept. This can't be undone unless you exported a backup.")
        }
    }

    private var embeddingFooter: String {
        let entry = ModelCatalog.entry(id: selectedModelID) ?? ModelCatalog.selected
        guard embeddings else { return "Keyword search only. Turn on semantic search to also match by meaning (one-time ~230 MB download of LFM2.5-Embedding-350M)." }
        if !MemoryEmbedder.allowed(alongside: entry) {
            return "Semantic search is paused while \(entry.displayName) is selected: it needs the memory. Keyword search is used instead."
        }
        return memory.embedderLoaded ? "Semantic search ready (LFM2.5-Embedding-350M)." : "The embedding model loads when the agent is idle."
    }
}

// MARK: - Memories

struct MemoryListView: View {
    let memory: MemoryHarness
    @State private var records: [MemoryRecord] = []
    @State private var query = ""
    @State private var kind: MemoryKind?
    @State private var editing: MemoryRecord?
    @State private var adding = false

    private var shown: [MemoryRecord] {
        var list = records
        if let kind { list = list.filter { $0.kind == kind } }
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return list.sorted { ($0.pinned ? 1 : 0, $0.updatedAt) > ($1.pinned ? 1 : 0, $1.updatedAt) } }
        let ranked = MemoryRanker.rank(list, query: q, queryVector: nil, embeddingModel: nil)
        let substring = list.filter { $0.text.localizedCaseInsensitiveContains(q) }
        var seen = Set<String>()
        return (ranked.map(\.record) + substring).filter { seen.insert($0.id).inserted }
    }

    var body: some View {
        List {
            Picker("Kind", selection: $kind) {
                Text("All").tag(MemoryKind?.none)
                ForEach(MemoryKind.allCases) { Text($0.label).tag(Optional($0)) }
            }
            .pickerStyle(.segmented)
            ForEach(shown) { r in
                Button { editing = r } label: { MemoryRow(record: r) }
                    .buttonStyle(.plain)
                    .swipeActions(edge: .trailing) {
                        Button("Delete", systemImage: "trash", role: .destructive) {
                            Task { try? await memory.memories.delete(ids: [r.id]); await reload() }
                        }
                    }
                    .swipeActions(edge: .leading) {
                        Button(r.pinned ? "Unpin" : "Pin", systemImage: r.pinned ? "pin.slash" : "pin") {
                            Task { try? await memory.memories.edit(id: r.id, pinned: !r.pinned); await reload() }
                        }
                        .tint(.orange)
                    }
            }
        }
        .searchable(text: $query, prompt: "Search memories")
        .navigationTitle("Memories")
        .toolbar { Button("Add", systemImage: "plus") { adding = true } }
        .task(id: memory.revision) { await reload() }
        .sheet(item: $editing) { r in
            MemoryEditor(title: "Edit memory", initial: r) { text, kind, importance, pinned in
                try await memory.memories.edit(id: r.id, text: text, kind: kind, importance: importance, pinned: pinned)
                await reload()
            }
        }
        .sheet(isPresented: $adding) {
            MemoryEditor(title: "New memory", initial: nil) { text, kind, importance, pinned in
                let stored = try await memory.memories.upsert(MemoryRecord(
                    kind: kind, text: text, importance: importance, pinned: pinned, trust: .user))
                await memory.embedIfPossible(stored)
                await reload()
            }
        }
    }

    private func reload() async {
        records = await memory.memories.all()
        memory.noteChanged()
    }
}

private struct MemoryRow: View {
    let record: MemoryRecord
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(record.text).lineLimit(4)
            HStack(spacing: 6) {
                if record.pinned { Image(systemName: "pin.fill").foregroundStyle(.orange) }
                Text(record.kind.label)
                Text("·")
                Text(String(repeating: "★", count: record.importance))
                if record.trust == .untrusted {
                    Label("untrusted", systemImage: "exclamationmark.shield").foregroundStyle(.red)
                } else if record.trust == .user {
                    Label("yours", systemImage: "person")
                }
                Spacer()
                Text(record.updatedAt.formatted(date: .abbreviated, time: .omitted))
            }
            .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

private struct MemoryEditor: View {
    let title: String
    let initial: MemoryRecord?
    let save: (String, MemoryKind, Int, Bool) async throws -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var kind = MemoryKind.fact
    @State private var importance = 3
    @State private var pinned = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                TextField("Memory", text: $text, axis: .vertical).lineLimit(3...10)
                Picker("Kind", selection: $kind) { ForEach(MemoryKind.allCases) { Text($0.label).tag($0) } }
                Stepper("Importance: \(importance)", value: $importance, in: 1...5)
                Toggle("Pinned", isOn: $pinned)
                if let initial {
                    LabeledContent("Id", value: initial.id)
                    LabeledContent("Used", value: "\(initial.accessCount)×")
                    LabeledContent("Source", value: initial.trust.rawValue)
                }
                if let error { Text(error).foregroundStyle(.red) }
            }
            .navigationTitle(title)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Task {
                            do { try await save(text, kind, importance, pinned); dismiss() }
                            catch { self.error = error.localizedDescription }
                        }
                    }
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .onAppear {
                if let initial {
                    text = initial.text; kind = initial.kind; importance = initial.importance; pinned = initial.pinned
                }
            }
        }
    }
}

// MARK: - Playbook

struct PlaybookView: View {
    let memory: MemoryHarness
    @State private var current: PlaybookVersion?
    @State private var versions: [PlaybookVersion] = []
    @State private var pending: [PendingPlaybookEdit] = []
    @State private var rollbackTarget: PlaybookVersion?
    @State private var newEntry = ""

    var body: some View {
        List {
            if !pending.isEmpty {
                Section {
                    ForEach(pending) { p in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(describe(p.edit))
                            HStack {
                                Button("Approve") { Task { await memory.resolvePending(p.id, approve: true); await reload() } }
                                    .buttonStyle(.borderedProminent)
                                Button("Reject", role: .destructive) {
                                    Task { await memory.resolvePending(p.id, approve: false); await reload() }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                } header: { Text("Suggestions to review") } footer: {
                    Text("Proposed from chats that included text from another machine (or from an imported file). Approve only what you'd want the agent to follow.")
                }
            }
            Section("Current playbook\(current.map { " · v\($0.id)" } ?? "")") {
                ForEach(current?.entries ?? []) { e in Text(e.text) }
                    .onDelete { idx in
                        guard var list = current?.entries else { return }
                        list.remove(atOffsets: idx)
                        Task { try? await memory.playbook.replace(with: list); memory.noteChanged(); await reload() }
                    }
                HStack {
                    TextField("Add a note", text: $newEntry)
                    Button("Add") {
                        let entry = PlaybookEntry(id: PlaybookEntry.newID(), text: newEntry, updatedAt: .now)
                        let list = (current?.entries ?? []) + [entry]
                        newEntry = ""
                        Task { try? await memory.playbook.replace(with: list); memory.noteChanged(); await reload() }
                    }
                    .disabled(newEntry.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            Section("History") {
                ForEach(versions) { v in
                    NavigationLink {
                        List(v.entries) { Text($0.text) }.navigationTitle("Version \(v.id)")
                    } label: {
                        VStack(alignment: .leading) {
                            Text("v\(v.id) · \(v.entries.count) notes").font(.subheadline)
                            Text("\(v.reason) · \(v.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        if v.id != current?.id {
                            Button("Restore", systemImage: "arrow.uturn.backward") { rollbackTarget = v }.tint(.blue)
                        }
                    }
                    .contextMenu {
                        if v.id != current?.id {
                            Button("Restore this version", systemImage: "arrow.uturn.backward") { rollbackTarget = v }
                        }
                    }
                }
            }
        }
        .navigationTitle("Playbook")
        .task(id: memory.revision) { await reload() }
        .confirmationDialog("Restore this playbook version?",
                            isPresented: Binding(get: { rollbackTarget != nil }, set: { if !$0 { rollbackTarget = nil } }),
                            titleVisibility: .visible) {
            Button("Restore v\(rollbackTarget?.id ?? 0)") {
                if let v = rollbackTarget { Task { await memory.rollbackPlaybook(to: v.id); await reload() } }
                rollbackTarget = nil
            }
        } message: {
            Text("The current notes are replaced by version \(rollbackTarget?.id ?? 0). The current version stays in history, so you can restore it later.")
        }
    }

    private func describe(_ e: ReflectionResult.LessonEdit) -> String {
        let old = current?.entries.first { $0.id == e.id }?.text ?? e.id ?? ""
        switch e.op {
        case .add: return "Add: “\(e.text ?? "")”"
        case .revise: return "Change “\(old)” to “\(e.text ?? "")”"
        case .remove: return "Remove: “\(old)”"
        }
    }

    private func reload() async {
        current = await memory.playbook.current
        versions = await memory.playbook.versions()
        pending = await memory.playbook.pending()
    }
}
