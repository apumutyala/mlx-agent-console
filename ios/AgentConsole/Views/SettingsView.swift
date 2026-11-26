import SwiftUI

struct SettingsView: View {
    @State private var url = UserDefaults.standard.string(forKey: "worker.baseURL") ?? "http://100.x.y.z:8787"
    @State private var token = Keychain.read("worker.token") ?? ""
    @State private var status: String?

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
                Button("Save and test") {
                    guard let u = URL(string: url) else { status = "Invalid URL"; return }
                    let cfg = WorkerConfig(baseURL: u, token: token)
                    cfg.save()
                    Task {
                        do {
                            let types = try await WorkerClient(config: cfg).jobTypes()
                            status = "Connected · \(types.count) job types"
                        } catch { status = error.localizedDescription }
                    }
                }
                if let status { Text(status).font(.footnote) }
                Section("On-device") {
                    LabeledContent("Model", value: AppModel.shared.backend.displayName)
                    LabeledContent("Background GPU", value: BatchCoordinator.backgroundGPUAvailable ? "supported" : "not supported")
                }
            }
            .navigationTitle("Settings")
        }
    }
}
