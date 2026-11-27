import SwiftUI
import UIKit

/// Bench tab: run the versioned workload set on one model or the whole
/// catalog, list saved receipts, inspect and share them.
struct BenchView: View {
    let coordinator: BenchCoordinator
    @State private var choice = BenchView.allModels
    @State private var repetitions = BenchmarkRunner.defaultRepetitions
    @State private var deviceRefresh = Date()

    static let allModels = "__all__"

    private var entries: [ModelCatalog.Entry] {
        // Every catalog entry, so unsupported ones get a receipt with the skip reason.
        choice == Self.allModels ? ModelCatalog.all : ModelCatalog.all.filter { $0.id == choice }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    let _ = deviceRefresh
                    LabeledContent("Device", value: "\(BuildInfo.machine) · \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
                    LabeledContent("RAM", value: String(format: "%.1f GB", ModelCatalog.deviceMemoryGB))
                    LabeledContent("Available to app", value: Self.gb(ResourceProbe.availableMemory()))
                    LabeledContent("Footprint now", value: Self.gb(ResourceProbe.physFootprint()))
                    LabeledContent("Thermal", value: ResourceProbe.name(ResourceProbe.thermal()))
                    LabeledContent("Build", value: String(BuildInfo.gitSHA.prefix(12)))
                } header: { Text("This device") }

                Section {
                    Picker("Model", selection: $choice) {
                        Text("All models").tag(Self.allModels)
                        ForEach(ModelCatalog.all) { e in
                            Text("\(e.displayName) · \(e.sizeLabel)").tag(e.id)
                        }
                    }
                    Stepper("Repetitions: \(repetitions)", value: $repetitions, in: 1...5)
                    if coordinator.isRunning {
                        ProgressView(value: coordinator.fraction)
                        Text(coordinator.status).font(.footnote)
                        Button("Stop", role: .destructive) { coordinator.cancel() }
                    } else {
                        Button(choice == Self.allModels ? "Run all models" : "Run benchmark",
                               systemImage: "speedometer") {
                            coordinator.run(entries, repetitions: repetitions)
                        }
                        .disabled(!coordinator.canRun)
                        if coordinator.status != "Idle" { Text(coordinator.status).font(.footnote) }
                    }
                } header: { Text("Run") } footer: {
                    Text("Workload set \(BenchWorkloads.setVersion): "
                         + BenchWorkloads.all.map(\.description).joined(separator: "; ")
                         + ". Greedy, seed \(BenchmarkRunner.seed). Unloads the chat model while running; "
                         + "models that don't fit in memory are skipped with the reason recorded. "
                         + "Receipts upload to the worker when one is configured.")
                }

                Section {
                    if coordinator.saved.isEmpty {
                        Text("No receipts yet.").foregroundStyle(.secondary)
                    }
                    ForEach(coordinator.saved) { s in
                        NavigationLink {
                            ReceiptDetailView(saved: s, coordinator: coordinator)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(s.receipt.model.displayName) · \(s.receipt.createdAt)").font(.subheadline)
                                Text(s.receipt.headline).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: { Text("Receipts") }
            }
            .navigationTitle("Benchmarks")
            .toolbar {
                if !coordinator.saved.isEmpty {
                    ShareLink(items: coordinator.saved.map(\.url)) { Label("Export all", systemImage: "square.and.arrow.up") }
                }
            }
            .refreshable { coordinator.refresh(); deviceRefresh = Date() }
            .onAppear { coordinator.refresh(); deviceRefresh = Date() }
        }
    }

    static func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
    static func mb(_ bytes: UInt64) -> String { String(format: "%.0f MB", Double(bytes) / 1e6) }
}

struct ReceiptDetailView: View {
    let saved: ReceiptStore.Saved
    let coordinator: BenchCoordinator
    private var r: BenchReceipt { saved.receipt }

    var body: some View {
        Form {
            Section {
                LabeledContent("Status", value: r.status)
                if let s = r.skipReason { Text(s).font(.footnote) }
                if let e = r.error { Text(e).font(.footnote).foregroundStyle(.red) }
                LabeledContent("Model", value: r.model.repo)
                if let q = r.model.quantization {
                    let bits = q.bits.map { "\($0)" } ?? "?"
                    let group = q.groupSize.map { "\($0)" } ?? "?"
                    let extra = q.perLayerOverrides > 0 ? ", \(q.perLayerOverrides) per-layer overrides" : ""
                    LabeledContent("Quantization", value: "\(bits)-bit, group \(group)\(extra)")
                }
                if let w = r.model.weightsBytes { LabeledContent("Weights", value: BenchView.gb(UInt64(w))) }
                LabeledContent("Device", value: "\(r.device.machine) · \(r.device.os)")
                LabeledContent("Build", value: String(r.appGitSha.prefix(12)))
                LabeledContent("Workload set", value: "\(r.workloadSet) · \(r.params.repetitions) reps")
            }
            if let l = r.load {
                Section("Load") {
                    LabeledContent("First load", value: String(format: "%.2f s", l.firstLoadS)
                                   + (l.downloadObserved ? " (included download)" : ""))
                    LabeledContent("Warm load", value: String(format: "%.2f s", l.warmLoadS))
                    LabeledContent("Footprint after load", value: BenchView.gb(l.footprintAfterLoadBytes))
                    LabeledContent("Available after load", value: BenchView.gb(l.availableAfterLoadBytes))
                }
            }
            if !r.workloads.isEmpty {
                Section {
                    ScrollView(.horizontal) {
                        Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                            GridRow {
                                ForEach(["Workload", "Prompt", "TTFT", "Prefill tok/s", "Decode tok/s",
                                         "Decode min–max", "Peak mem", "Output", "Same"], id: \.self) {
                                    Text($0).font(.caption.bold())
                                }
                            }
                            ForEach(r.workloads, id: \.id) { w in
                                GridRow {
                                    Text(w.id).gridColumnAlignment(.leading)
                                    Text("\(w.promptTokens)")
                                    Text(String(format: "%.0f ms", w.median.ttftS * 1000))
                                    Text(String(format: "%.0f", w.median.prefillTps))
                                    Text(String(format: "%.1f", w.median.decodeTps)).bold()
                                    Text(String(format: "%.1f–%.1f", w.min.decodeTps, w.max.decodeTps))
                                    Text(BenchView.mb(w.median.peakFootprintBytes))
                                    Text(String(w.outputHash.prefix(8))).monospaced()
                                    Image(systemName: w.deterministic ? "checkmark" : "xmark")
                                }
                                .font(.caption)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                } header: { Text("Results (median of \(r.params.repetitions))") } footer: {
                    Text("TTFT includes prefill. Decode excludes the first token. Peak mem is the process "
                         + "phys_footprint sampled every \(r.params.memorySampleIntervalMs) ms. Output is the hash "
                         + "of the first 64 greedy tokens; Same = identical across reps.")
                }
            }
            Section("Environment") {
                LabeledContent("Thermal", value: "\(r.environment.thermalBefore) → \(r.environment.thermalAfter) (max \(r.environment.thermalMaxDuring))")
                LabeledContent("Battery", value: "\(r.environment.batteryStateBefore)"
                               + (r.environment.batteryLevelBefore.map { String(format: " %.0f%%", $0 * 100) } ?? ""))
                LabeledContent("Low Power Mode", value: r.environment.lowPowerMode ? "on" : "off")
                LabeledContent("Total time", value: String(format: "%.1f s", r.totalWallS))
            }
            Section {
                ShareLink(item: saved.url) { Label("Share receipt JSON", systemImage: "square.and.arrow.up") }
                Button("Upload to worker", systemImage: "icloud.and.arrow.up") {
                    Task { await coordinator.upload(saved.url) }
                }
                if let s = coordinator.uploadStatus[saved.id] { Text(s).font(.footnote) }
            }
        }
        .navigationTitle(r.model.displayName)
    }
}
