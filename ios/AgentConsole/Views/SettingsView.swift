import SwiftUI

struct SettingsView: View {
    @State private var url = UserDefaults.standard.string(forKey: "worker.baseURL") ?? "https://<host>.ts.net:8443"
    @State private var token = Keychain.read("worker.token") ?? ""
    @State private var status: String?
    @State private var scanning = false
    @AppStorage(ModelCatalog.storageKey) private var selectedModelID = ModelCatalog.defaultEntry.id

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
                            Text("\(entry.displayName) · \(entry.sizeLabel)").tag(entry.id)
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
                    Text("\(entry.note) One-time download of about \(entry.sizeLabel); switching resets the chat."
                         + (hidden > 0 ? " Larger models need a 16 GB iPad." : ""))
                }
                SpeechSettingsSection()
                MemorySettingsSection(memory: AppModel.shared.memory)
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
}
