import SwiftUI

/// Settings › Inkling 2 Mini (LookAtMyQuant): the memory plan per build and the kernel self-check.
/// The 3.9 GB build is also a picker entry (`ModelCatalog.inkling2MiniLamq39`); the streamed
/// 4.7 / 5.4 GB builds are listed here only (docs/LAMQ-FORMAT-SPEC.md, "What is left").
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
            Text("Trellis-coded 1.2–1.65 bit builds with new Metal kernels. The 3.9 GB build is in the model "
                 + "picker above (fully resident); the 4.7 / 5.4 GB builds need expert streaming, not built yet. "
                 + "Run the self-check before the first chat: it compares this iPad's kernels with the "
                 + "reference decoder, bit for bit. Share its report.")
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
