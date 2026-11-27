import SwiftUI
import UIKit

/// Bench tab: run the versioned workload set on one model or the whole
/// catalog, list saved receipts, inspect and share them.
struct BenchView: View {
    let coordinator: BenchCoordinator
    @State private var choice = BenchView.allModels
    @State private var repetitions = BenchmarkRunner.defaultRepetitions
    @State private var deviceRefresh = Date()
    @State private var abSteps: Set<String> = Set(EngineAB.steps.map(\.id)).subtracting(["A7"])
    @AppStorage(XybridLLMBackend.noResidencyNextLaunchKey) private var noResidencyNextLaunch = false

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

                engineABSection

                Section {
                    if coordinator.saved.isEmpty {
                        Text("No receipts yet.").foregroundStyle(.secondary)
                    }
                    ForEach(coordinator.saved) { s in
                        NavigationLink {
                            ReceiptDetailView(saved: s, coordinator: coordinator)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(Self.rowTitle(s.receipt)).font(.subheadline)
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
            .task { await coordinator.uploadPending() }   // receipts a failed upload left behind
        }
    }

    /// Engine A/B (A1–A7): what runs, in which order, and the A7 launch switch.
    @ViewBuilder private var engineABSection: some View {
        Section {
            if let notice = coordinator.killedNotice {
                Label(notice, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.orange)
            }
            ForEach(EngineAB.steps) { step in
                Toggle(isOn: Binding(
                    get: { abSteps.contains(step.id) },
                    set: { on in if on { abSteps.insert(step.id) } else { abSteps.remove(step.id) } })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(step.title).font(.subheadline)
                        Text(step.purpose + (step.entry.map { " · \($0.sizeLabel)" } ?? "")).font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .disabled(coordinator.isRunning)
            }
            Toggle("Disable Metal residency at next launch (A7, experimental)", isOn: $noResidencyNextLaunch)
            if EngineAB.noResidencyActive {
                Text("This launch has Metal residency OFF (GGML_METAL_NO_RESIDENCY=1): only A7 runs; "
                     + "relaunch normally for the other steps.").font(.footnote).foregroundStyle(.orange)
            }
            if coordinator.isRunning {
                ProgressView(value: coordinator.fraction)
                Text(coordinator.status).font(.footnote)
                Button("Stop", role: .destructive) { coordinator.cancel() }
            } else {
                Button("Run engine A/B (\(abSteps.count) steps)", systemImage: "arrow.left.arrow.right") {
                    coordinator.runEngineAB(EngineAB.steps.filter { abSteps.contains($0.id) })
                }
                .disabled(!coordinator.canRunEngineAB || abSteps.isEmpty)
            }
        } header: { Text("Engine A/B: MLX vs Xybrid") } footer: {
            Text(Self.engineABFooter)
        }
    }

    private static let engineABFooter: String = "Same prompt (\(EngineAB.workloadID)), greedy, 256 tokens, \(EngineAB.repetitions) reps, median. "
                 + "The first run downloads what is missing (up to ~23 GB for all steps: 5 GGUFs + 2 MLX models) into Application Support; "
                 + "keep the app in the foreground and plugged in. A6 and A5 may be killed by iOS: that is the "
                 + "data point, and the next launch saves a “killed” receipt. A7 needs its own launch: turn on "
                 + "the residency switch, quit the app from the app switcher, reopen, select only A7 and run."

    /// "A2 · Xybrid · LFM2.5 8B-A1B · UD-Q3_K_XL · <date>" (engine/step only when recorded).
    static func rowTitle(_ r: BenchReceipt) -> String {
        var parts: [String] = []
        if let step = r.abStep { parts.append(step) }
        if let engine = r.engine { parts.append(engine == "xybrid" ? "Xybrid" : "MLX") }
        parts.append(r.model.displayName)
        parts.append(r.createdAt)
        return parts.joined(separator: " · ")
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
                if let e = r.engine { LabeledContent("Engine", value: e + (r.abStep.map { " · \($0)" } ?? "")) }
                if r.jetsamSuspected == true { Text("Killed during the run: jetsam suspected.").font(.footnote).foregroundStyle(.red) }
                if let w = r.warnings, !w.isEmpty { Text(w.joined(separator: "\n")).font(.footnote).foregroundStyle(.orange) }
            }
            if let m = r.memoryProbe {
                Section("Memory") {
                    LabeledContent("Peak footprint", value: BenchView.gb(m.physFootprintPeakBytes))
                    LabeledContent("Available before → after load",
                                   value: "\(BenchView.gb(m.availableBeforeLoadBytes)) → \(BenchView.gb(m.availableAfterLoadBytes))")
                    if let w = m.metalRecommendedMaxWorkingSetBytes {
                        LabeledContent("Metal working set (recommended)", value: BenchView.gb(w))
                    }
                    if let a = m.metalAllocatedAfterLoadBytes {
                        LabeledContent("Metal allocated after load", value: BenchView.gb(a))
                    }
                }
            }
            if let s = r.storage {
                Section("Storage") {
                    LabeledContent("Local copy before load", value: s.localSnapshotCompleteBefore ? "complete" : "no")
                    LabeledContent("On disk before → after",
                                   value: "\(BenchView.gb(UInt64(max(0, s.bytesOnDiskBefore)))) → \(BenchView.gb(UInt64(max(0, s.bytesOnDiskAfter))))")
                    LabeledContent("Excluded from backup", value: s.excludedFromBackup ? "yes" : "no")
                }
            }
            if let id = r.identity {
                Section("Container") {
                    LabeledContent("Bundle id", value: id.bundleIdentifier)
                    LabeledContent("Container marker", value: String(id.containerMarkerId.prefix(8)))
                    LabeledContent("Path hash", value: id.containerPathHash)
                }
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
