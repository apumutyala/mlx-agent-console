import SwiftUI
import Xybrid

/// Settings › Downloaded models: every model on disk (MLX snapshots, Xybrid GGUFs, Xybrid
/// SDK cache), its files and real on-disk size, and a confirmed delete. The selected model
/// can't be deleted (switch first), and nothing is deleted while a benchmark runs.
struct DownloadedModelsView: View {
    @State private var models: [ModelStore.StoredModel] = []
    @State private var xybridEntries: [XybridCacheEntry] = []
    @State private var pendingDelete: ModelStore.StoredModel?
    @State private var pendingXybridDelete: String?
    @State private var message: String?

    var body: some View {
        Form {
            Section {
                LabeledContent("Models folder", value: "Application Support/Models")
                LabeledContent("Total on disk", value: Self.gb(totalBytes))
                LabeledContent("Free on device", value: Self.gb(ModelStore.freeBytes))
                LabeledContent("Excluded from iCloud backup",
                               value: ModelStore.isExcludedFromBackup(ModelStore.modelsRoot) ? "yes" : "no")
                if let report = UserDefaults.standard.string(forKey: ModelStore.migrationReportKey) {
                    Text("Migration: \(report)").font(.footnote).foregroundStyle(.secondary)
                }
                if let message { Text(message).font(.footnote) }
            } footer: {
                Text("Weights live in Application Support, which iOS doesn't purge, so a reinstall over the app "
                     + "(same Apple ID and bundle id) keeps them. Deleting the app deletes them.")
            }

            Section("MLX (Hugging Face)") {
                let hf = models.filter { $0.kind == .huggingFace }
                if hf.isEmpty { Text("None").foregroundStyle(.secondary) }
                ForEach(hf) { m in row(m) }
            }
            Section("Xybrid GGUF") {
                let gguf = models.filter { $0.kind == .gguf }
                if gguf.isEmpty { Text("None").foregroundStyle(.secondary) }
                ForEach(gguf) { m in row(m) }
            }
            Section {
                if xybridEntries.isEmpty { Text("None").foregroundStyle(.secondary) }
                ForEach(xybridEntries, id: \.modelId) { e in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(e.modelId).font(.subheadline)
                            Text(e.path).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                        }
                        Spacer()
                        Text(Self.gb(Int64(clamping: e.sizeBytes))).font(.caption).monospacedDigit()
                        Button(role: .destructive) { pendingXybridDelete = e.modelId } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            } header: { Text("Xybrid SDK cache (speech, registry LLMs)") } footer: {
                Text("Application Support/Xybrid/Models. Speech models download again on next use.")
            }
        }
        .navigationTitle("Downloaded models")
        .onAppear(perform: refresh)
        .refreshable { refresh() }
        .confirmationDialog(
            "Delete \(pendingDelete?.name ?? "")?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete \(Self.gb(pendingDelete?.bytes ?? 0))", role: .destructive) {
                if let m = pendingDelete { Task { await delete(m) } }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: {
            Text("The files are removed from this iPad. Using the model again downloads it again.")
        }
        .confirmationDialog(
            "Delete \(pendingXybridDelete ?? "") from the Xybrid cache?",
            isPresented: Binding(get: { pendingXybridDelete != nil }, set: { if !$0 { pendingXybridDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let id = pendingXybridDelete { deleteXybrid(id) }
                pendingXybridDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingXybridDelete = nil }
        } message: {
            Text("Using the model again downloads it again.")
        }
    }

    private var totalBytes: Int64 {
        models.reduce(0) { $0 + $1.bytes } + xybridEntries.reduce(0) { $0 + Int64(clamping: $1.sizeBytes) }
    }

    @ViewBuilder private func row(_ m: ModelStore.StoredModel) -> some View {
        DisclosureGroup {
            if let commit = m.commit { LabeledContent("Commit", value: String(commit.prefix(12))) }
            ForEach(m.files, id: \.name) { f in
                LabeledContent(f.name, value: Self.gb(f.bytes)).font(.caption)
            }
            Button("Delete…", role: .destructive) { pendingDelete = m }
                .disabled(inUse(m) != nil)
            if let why = inUse(m) { Text(why).font(.caption).foregroundStyle(.secondary) }
        } label: {
            HStack {
                Text(m.name).font(.subheadline).lineLimit(2)
                Spacer()
                Text(Self.gb(m.bytes)).font(.caption).monospacedDigit()
            }
        }
    }

    /// Why `m` can't be deleted right now, or nil.
    private func inUse(_ m: ModelStore.StoredModel) -> String? {
        if AppModel.shared.bench.isRunning { return "A benchmark is running." }
        let selected = ModelCatalog.selected
        switch m.kind {
        case .huggingFace:
            if selected.engine == .mlx, MLXBackend.configuration(for: selected).name == m.name {
                return "This is the selected chat model; pick another model first."
            }
        case .gguf:
            if selected.id == m.name { return "This is the selected chat model; pick another model first." }
        case .xybrid:
            break
        }
        return nil
    }

    private func refresh() {
        models = ModelStore.storedModels()
        XybridLLMBackend.prepareSDK()
        xybridEntries = ((try? Xybrid.modelCacheEntries()) ?? []).sorted { $0.sizeBytes > $1.sizeBytes }
    }

    private func delete(_ m: ModelStore.StoredModel) async {
        if let why = inUse(m) { message = why; return }
        if m.kind == .huggingFace, m.name == MemoryEmbedder.modelID {
            await AppModel.shared.memory.releaseForBenchmark()   // unloads the embedder first
        }
        do {
            try ModelStore.delete(m)
            message = "Deleted \(m.name) (\(Self.gb(m.bytes)))."
        } catch {
            message = "Couldn't delete \(m.name): \(error.localizedDescription)"
        }
        refresh()
    }

    private func deleteXybrid(_ id: String) {
        let selected = ModelCatalog.selected
        if case .registry(let selectedID) = selected.xybridSource, selectedID == id {
            message = "\(id) is the selected chat model; pick another model first."
            return
        }
        SpeechService.shared.releaseModels()   // Whisper/Kokoro reload from disk (or download) on next use
        do {
            _ = try Xybrid.removeCachedModel(id)
            message = "Deleted \(id) from the Xybrid cache."
        } catch {
            message = "Couldn't delete \(id): \(error.localizedDescription)"
        }
        refresh()
    }

    static func gb(_ bytes: Int64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
}
