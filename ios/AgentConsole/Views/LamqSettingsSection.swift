import SwiftUI

/// Settings › Inkling 2 Mini (LookAtMyQuant): the memory plan per build and the kernel self-check.
/// The builds are not chat-loadable yet (docs/LAMQ-FORMAT-SPEC.md, "What is left").
struct LamqSettingsSection: View {
    @State private var running = false
    @State private var status: String?
    @State private var lines: [String] = UserDefaults.standard.string(forKey: LamqSelfCheck.reportKey)?
        .components(separatedBy: "\n") ?? []

    var body: some View {
        Section {
            ForEach(ModelCatalog.lamqBuilds) { build in
                VStack(alignment: .leading, spacing: 2) {
                    Text(build.name).font(.subheadline)
                    Text(Self.plan(build)).font(.caption).foregroundStyle(.secondary)
                    Text(build.card).font(.caption2).foregroundStyle(.secondary)
                }
            }
            Button(running ? "Running…" : "Run kernel self-check (downloads ~190 MB once)") {
                running = true
                status = nil
                Task {
                    do {
                        lines = try await LamqSelfCheck.run { msg in Task { @MainActor in status = msg } }
                        status = nil
                    } catch {
                        status = "Self-check failed to run: \(error.localizedDescription)"
                    }
                    running = false
                }
            }
            .disabled(running)
            if let status { Text(status).font(.caption) }
            if !lines.isEmpty {
                Text(lines.joined(separator: "\n"))
                    .font(.caption2.monospaced()).textSelection(.enabled)
                Button("Delete self-check download") { LamqSelfCheck.removeDownloads() }.font(.caption)
            }
        } header: { Text("Inkling 2 Mini · LookAtMyQuant (experimental)") } footer: {
            Text("Trellis-coded 1.2–1.65 bit builds. Their Metal kernels are new and the model is not yet "
                 + "wired into chat, so they can't be picked above. The self-check compares this iPad's "
                 + "kernels with the reference decoder, bit for bit; share its report.")
        }
    }

    private static func plan(_ b: ModelCatalog.LamqBuild) -> String {
        let need = String(format: "%.2f", b.residentNeedGB)
        let available = MemoryBudget.availableBytes
        let budget = available > 0 ? " · app may use \(MemoryBudget.gb(available + MemoryBudget.heldByModel()))" : ""
        switch b.plan {
        case .resident(let ctx):
            return String(format: "%.2f GB weights · resident ≈ %@ GB at %dK context (est.)", b.totalGB, need, ctx / 1024) + budget
        case .streamed(let gb):
            return String(format: "%.2f GB weights · resident would need ≈ %@ GB; plan: stream ≈ %.1f GB of experts (not built)",
                          b.totalGB, need, gb) + budget
        }
    }
}
