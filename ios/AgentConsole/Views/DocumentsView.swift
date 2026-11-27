import SwiftUI
import UniformTypeIdentifiers

struct DocumentsView: View {
    let documents: DocumentStore
    let batches: BatchStore
    @Bindable var coordinator: BatchCoordinator

    @State private var docs: [ImportedDocument] = []
    @State private var history: [SummaryBatch] = []
    @State private var selection = Set<String>()
    @State private var importing = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List(selection: $selection) {
                Section {
                    Label(BatchCoordinator.backgroundGPUAvailable
                          ? "Background GPU available: batches continue after you switch apps."
                          : "No background GPU on this device: batches pause when you leave the app and resume later.",
                          systemImage: BatchCoordinator.backgroundGPUAvailable ? "checkmark.seal" : "pause.circle")
                        .font(.footnote)
                }
                Section("Imported files") {
                    ForEach(docs) { d in Text(d.name).tag(d.id) }
                        .onDelete { idx in
                            Task {
                                for i in idx { try? await documents.remove(docs[i].id) }
                                await refresh()
                            }
                        }
                }
                if let b = coordinator.active {
                    Section("Current batch · \(coordinator.mode)") {
                        ProgressView(value: Double(b.completedCount), total: Double(max(b.fileIDs.count, 1))) {
                            Text("\(b.completedCount) of \(b.fileIDs.count)")
                        }
                        if let m = coordinator.lastMessage { Text(m).font(.footnote) }
                    }
                }
                Section("Batches") {
                    ForEach(history) { b in
                        NavigationLink {
                            BatchDetail(batch: b, docs: docs, coordinator: coordinator)
                        } label: {
                            LabeledContent(b.createdAt.formatted(date: .abbreviated, time: .shortened),
                                           value: b.isDone ? "done" : "\(b.completedCount)/\(b.fileIDs.count)")
                        }
                    }
                }
            }
            .environment(\.editMode, .constant(.active))
            .navigationTitle("Documents")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Import", systemImage: "square.and.arrow.down") { importing = true }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Summarize \(selection.count)") {
                        let ids = docs.map(\.id).filter(selection.contains)
                        Task { await coordinator.start(fileIDs: ids); await refresh() }
                    }
                    .disabled(selection.isEmpty || coordinator.active.map { !$0.isDone } == true)
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.plainText, .pdf, .text, .json],
                          allowsMultipleSelection: true) { result in
                Task {
                    do {
                        for url in try result.get() { try await documents.importFile(from: url) }
                    } catch { self.error = error.localizedDescription }
                    await refresh()
                }
            }
            .task { await refresh() }
            .onReceive(NotificationCenter.default.publisher(for: .documentsChanged)) { _ in Task { await refresh() } }
            .onChange(of: coordinator.active?.completedCount) { _, _ in Task { await refresh() } }
            .alert("Import failed", isPresented: .constant(error != nil)) {
                Button("OK") { error = nil }
            } message: { Text(error ?? "") }
        }
    }

    private func refresh() async {
        docs = await documents.all()
        history = await batches.all()
    }
}

private struct BatchDetail: View {
    let batch: SummaryBatch
    let docs: [ImportedDocument]
    let coordinator: BatchCoordinator

    var body: some View {
        List {
            if !batch.isDone {
                Button("Resume from file \(batch.completedCount + 1)") {
                    Task { await coordinator.resume(batchID: batch.id) }
                }
            }
            ForEach(batch.fileIDs, id: \.self) { id in
                Section(docs.first { $0.id == id }?.name ?? id) {
                    if let s = batch.outputs[id] { Text(s).textSelection(.enabled) }
                    else if let f = batch.failures[id] { Text(f).foregroundStyle(.red) }
                    else { Text("Not yet summarized").foregroundStyle(.secondary) }
                }
            }
        }
        .navigationTitle("Batch \(batch.id)")
    }
}
