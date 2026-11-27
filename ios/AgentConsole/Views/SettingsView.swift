import SwiftUI

struct SettingsView: View {
    @State private var url = UserDefaults.standard.string(forKey: "worker.baseURL") ?? "https://<host>.ts.net:8443"
    @State private var token = Keychain.read("worker.token") ?? ""
    @State private var status: String?
    @State private var scanning = false
    @AppStorage(ModelCatalog.storageKey) private var selectedModelID = ModelCatalog.defaultEntry.id
    @AppStorage(MemoryBudget.overrideKey) private var ignoreMemoryCheck = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Worker URL", text: $url)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Token", text: $token)
                } header: { Text("Remote worker") } footer: {
                    Text("Reach the worker over Tailscale/VPN or HTTPS. The token is stored in the Keychain.")
                }
                Button("Scan pairing code", systemImage: "qrcode.viewfinder") { scanning = true }
                    .disabled(!PairingScannerView.isAvailable)
                Button("Save and test") {
                    guard let u = URL(string: url) else { status = "Invalid URL"; return }
                    saveAndTest(WorkerConfig(baseURL: u, token: token))
                }
                if let status { Text(status).font(.footnote) }
                NodeSettingsSection(node: AppModel.shared.node)
                Section {
                    Picker("Model", selection: $selectedModelID) {
                        ForEach(ModelCatalog.available) { entry in
                            Text(Self.pickerLabel(entry)).tag(entry.id)
                        }
                    }
                    .onChange(of: selectedModelID) { _, id in
                        guard let entry = ModelCatalog.entry(id: id) else { return }
                        // New model = new chat; the old weights are released.
                        AppModel.shared.agent.reset()
                        SpeechService.shared.modelSelected(entry)   // big LLM: free Whisper/Kokoro first
                        Task { await AppModel.shared.backend.switchModel(to: entry) }
                    }
                    LabeledContent("Background GPU", value: BatchCoordinator.backgroundGPUAvailable ? "supported" : "not supported")
                } header: { Text("On-device") } footer: {
                    let entry = ModelCatalog.entry(id: selectedModelID) ?? ModelCatalog.selected
                    let hidden = ModelCatalog.all.count - ModelCatalog.available.count
                    let suffix: String = hidden > 0 ? " Larger models need a 16 GB iPad." : ""
                    let message: String = "\(entry.note) One-time download of about \(entry.sizeLabel); "
                        + "switching resets the chat." + Self.fitNote(entry) + suffix
                    Text(message)
                }
                SpeechSettingsSection()
                MemorySettingsSection(memory: AppModel.shared.memory)
                MemorySection(ignoreMemoryCheck: $ignoreMemoryCheck)
            }
            .navigationTitle("Settings")
            .sheet(isPresented: $scanning) {
                NavigationStack {
                    PairingScannerView { payload in
                        url = payload.url.absoluteString
                        token = payload.token
                        scanning = false
                        saveAndTest(WorkerConfig(baseURL: payload.url, token: payload.token))
                    }
                    .ignoresSafeArea()
                    .navigationTitle("Scan the code from pair.ps1")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { scanning = false } } }
                }
            }
        }
    }

    private func saveAndTest(_ cfg: WorkerConfig) {
        cfg.save()
        status = "Testing…"
        Task {
            do {
                let types = try await WorkerClient(config: cfg).jobTypes()
                status = "Connected · \(types.count) job types"
            } catch { status = error.localizedDescription }
        }
    }

    private static func pickerLabel(_ entry: ModelCatalog.Entry) -> String {
        let base = "\(entry.displayName) · \(entry.sizeLabel)"
        if case .tooLarge = MemoryBudget.fit(entry) { return base + " · too large" }
        return base
    }

    private static func fitNote(_ entry: ModelCatalog.Entry) -> String {
        switch MemoryBudget.fit(entry) {
        case .unchecked: return ""
        case .fits(let needed, let budget):
            return " Needs ~\(MemoryBudget.gb(needed)) of the \(MemoryBudget.gb(budget)) the app may use."
        case .tooLarge(let needed, let budget):
            return " ⚠️ Needs ~\(MemoryBudget.gb(needed)) but the app may only use \(MemoryBudget.gb(budget)); "
                + (MemoryBudget.ignoreCheck ? "loading anyway (check ignored)." : "it won't load.")
        }
    }
}

/// Live memory readout: what iOS lets this app use (reflects whether the
/// increased-memory-limit entitlement took effect) and what MLX holds.
private struct MemorySection: View {
    @Binding var ignoreMemoryCheck: Bool

    static func rows() -> [(String, String)] {
        let available = MemoryBudget.availableBytes
        let mlx = MemoryBudget.mlxSnapshot()
        var rows: [(String, String)] = [
            ("Physical RAM", MemoryBudget.gb(MemoryBudget.physicalBytes)),
            ("Available to app", available > 0 ? MemoryBudget.gb(available) : "not reported"),
            ("App limit (est.)", available > 0 ? MemoryBudget.gb(available + MemoryBudget.heldByModel()) : "–"),
        ]
        if let gpu = MemoryBudget.gpuWorkingSetBytes {
            rows.append(("GPU working set", MemoryBudget.gb(gpu)))
        }
        rows.append(("MLX active / peak", MemoryBudget.gb(mlx.active) + " / " + MemoryBudget.gb(mlx.peak)))
        rows.append(("MLX cache", MemoryBudget.gb(mlx.cache)))
        return rows
    }
    @State private var killedLoad = UserDefaults.standard.string(forKey: MemoryBudget.killedLoadKey)

    var body: some View {
        Section {
            TimelineView(.periodic(from: .now, by: 2)) { _ in
                MemoryReadout(rows: Self.rows())
            }
            Toggle("Ignore memory check", isOn: $ignoreMemoryCheck)
            if let report = UserDefaults.standard.string(forKey: MemoryBudget.lastLoadReportKey) {
                Text("Last large load: \(report)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if let killedLoad {
                Label("The load of \(killedLoad) never finished — iOS probably stopped the app for using too much memory.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
                Button("Dismiss") {
                    UserDefaults.standard.removeObject(forKey: MemoryBudget.killedLoadKey)
                    self.killedLoad = nil
                }
                .font(.caption)
            }
        } header: { Text("Memory") } footer: {
            Text("“Available to app” is os_proc_available_memory(): what iOS still lets this app allocate. "
                 + "Large models need their weights plus ~1.5 GB of it. “App limit” adds back what the "
                 + "loaded model holds. If it stays near half the RAM, the increased-memory-limit "
                 + "entitlement was not granted by the signing profile.")
        }
    }
}

private struct MemoryReadout: View {
    let rows: [(String, String)]
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(rows.indices, id: \.self) { i in
                LabeledContent(rows[i].0, value: rows[i].1)
            }
        }
        .font(.footnote.monospacedDigit())
    }
}
