import Foundation
import Observation
import SwiftUI
import UIKit

/// Turns this iPad into a fleet compute node (worker/fleet.py): heartbeat,
/// long-poll for a task, run it on the local model, stream events back,
/// report the result. One task at a time, and never at the same time as the
/// chat or a summary batch (all model work goes through `GPUGate`).
///
/// iOS only lets the app use the network and GPU in the foreground, so the
/// node pauses when the app goes to the background and resumes when it's
/// active again. (.inactive alone, e.g. Control Center or a system alert,
/// doesn't pause: Metal is still allowed there, and every release costs the
/// task a server attempt.) A task in flight at that moment is *released*, not
/// reported: the node stops working on it and never calls /complete, so the
/// worker requeues it (on the node's next poll, or by the offline sweep). A finished task's result is kept in a small persisted
/// outbox and resent with backoff until the worker settles it (2xx/404/409),
/// always before the node asks for more work.
///
/// Training never changes what the chat runs on: the chat's adapter is
/// snapshotted before a train_lora task and restored afterwards. Only "Use
/// federated adapter" applies a round's adapter to the chat.
@MainActor @Observable
final class NodeService {
    // MARK: Identity

    /// Stable per install: "ipad-" + 8 hex chars.
    nonisolated static let nodeID: String = {
        let key = "node.id"
        if let id = UserDefaults.standard.string(forKey: key) { return id }
        let id = "ipad-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(8)).lowercased()
        UserDefaults.standard.set(id, forKey: key)
        return id
    }()

    nonisolated static let defaultSystem = "You are a helpful assistant."

    // MARK: Observable state

    private(set) var isServing = UserDefaults.standard.bool(forKey: "node.serve")
    var keepAwake = UserDefaults.standard.object(forKey: "node.keepAwake") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(keepAwake, forKey: "node.keepAwake")
            applyIdleTimer()
        }
    }
    /// Take train_lora tasks at all (off by default; generate tasks don't need it).
    var acceptTraining = UserDefaults.standard.bool(forKey: "node.acceptTraining") {
        didSet {
            UserDefaults.standard.set(acceptTraining, forKey: "node.acceptTraining")
            heartbeatDue = true   // re-advertise capabilities now
        }
    }
    /// Let train_lora tasks with `use_local_data` read this iPad's imported
    /// documents (off by default). Without it such tasks fail, and training uses
    /// only the examples the round itself carries.
    var allowDocumentTraining = UserDefaults.standard.bool(forKey: "node.trainOnDocuments") {
        didSet {
            UserDefaults.standard.set(allowDocumentTraining, forKey: "node.trainOnDocuments")
            heartbeatDue = true
        }
    }
    private(set) var status = "Stopped"
    private(set) var currentTask: String?
    private(set) var tasksServed = UserDefaults.standard.integer(forKey: "node.tasksServed")
    private(set) var lastTokensPerSecond: Double?
    private(set) var lastError: String?
    private(set) var adapterStatus: String?
    private(set) var isApplyingAdapter = false

    // MARK: Dependencies

    @ObservationIgnored private let backend: any ModelBackend
    @ObservationIgnored private let documents: DocumentStore
    @ObservationIgnored private let trainer: FederatedTrainer?
    @ObservationIgnored private let onModelChanged: @MainActor () -> Void
    @ObservationIgnored private var loop: Task<Void, Never>?
    /// The last cancelled loop, still finishing (events, result report).
    @ObservationIgnored private var winding: Task<Void, Never>?
    @ObservationIgnored private var currentGrace: BackgroundGrace?
    @ObservationIgnored private var heartbeatDue = false
    /// False from .background until .active: no new MLX work in the background.
    @ObservationIgnored private var sceneActive = true
    /// The chat's adapter, when a training task ended while the app was in the
    /// background. While it is set the node keeps holding `GPUGate` (training
    /// weights are still loaded), and `applyPendingRestore` restores the adapter
    /// and only then releases the gate.
    @ObservationIgnored private var pendingRestore: PendingRestore?

    private struct PendingRestore {
        let snapshot: Data
        let converted: Bool
    }

    init(backend: any ModelBackend, documents: DocumentStore, trainer: FederatedTrainer?,
         onModelChanged: @escaping @MainActor () -> Void) {
        self.backend = backend
        self.documents = documents
        self.trainer = trainer
        self.onModelChanged = onModelChanged
    }

    // MARK: Control

    func setServing(_ on: Bool) {
        isServing = on
        UserDefaults.standard.set(on, forKey: "node.serve")
        if on { start() } else { pause(status: "Stopped") }
    }

    /// Called at launch and when the app becomes active again.
    func resumeIfEnabled() {
        if isServing { start() }
    }

    func scenePhaseChanged(_ phase: ScenePhase) {
        switch phase {
        case .active:
            sceneActive = true
            if pendingRestore != nil { Task { await self.applyPendingRestore() } }
            resumeIfEnabled()
        case .inactive:
            // Control Center, alerts, the app switcher: Metal is still allowed and the
            // app usually comes straight back, so keep working. Releasing here would
            // burn a server attempt per interruption.
            break
        case .background:
            sceneActive = false
            pause(status: "Paused (app in background)")
        @unknown default:
            break
        }
    }

    /// Cancels the loop (the task in flight is released, not reported) and
    /// keeps a background grace until the loop has flushed its events and any
    /// pending result report.
    private func pause(status: String) {
        guard let l = loop else {
            if !isServing { self.status = "Stopped" }
            return
        }
        // The new grace covers the old loop too (a new loop waits for the old
        // one), so end the previous grace instead of leaking it.
        currentGrace?.end()
        let grace = BackgroundGrace(name: "node-pause")
        currentGrace = grace
        stop(status: status)
        Task { @MainActor in
            await l.value
            grace.end()
            if self.currentGrace === grace { self.currentGrace = nil }
        }
    }

    private func start() {
        guard loop == nil else { return }
        let previous = winding
        loop = Task { [weak self] in
            await previous?.value   // never two loops working at once
            await self?.runLoop()
        }
        applyIdleTimer()
    }

    private func stop(status: String) {
        if let l = loop { winding = l }
        loop?.cancel()
        loop = nil
        self.status = status
        applyIdleTimer()
    }

    private func applyIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = keepAwake && loop != nil
    }

    /// LookAtMyQuant packed models have no LoRA-convertible linears (packed trellis layers).
    private var canTrain: Bool {
        trainer != nil && acceptTraining && ModelCatalog.selected.lamqRevision == nil
    }

    private func nodeInfo() -> NodeInfo {
        let device = UIDevice.current
        var kinds = ["generate"]
        if canTrain { kinds.append("train_lora") }
        if backend.mlxBackend != nil { kinds.append("bench") }
        return NodeInfo(
            name: device.name, kind: "ipad",
            capabilities: .init(kinds: kinds, model: backend.displayName,
                                memoryGB: (Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 * 10).rounded() / 10,
                                os: "\(device.systemName) \(device.systemVersion)",
                                trainOnDocuments: canTrain && allowDocumentTraining))
    }

    // MARK: Loop

    private func runLoop() async {
        var backoff: Double = 1
        var lastBeat = Date.distantPast
        var client: FleetClient?
        while !Task.isCancelled {
            do {
                guard let cfg = WorkerConfig.load() else { throw WorkerError.notConfigured }
                let c: FleetClient
                if let existing = client, existing.matches(cfg) {
                    c = existing
                } else {
                    c = FleetClient(config: cfg)
                    client = c
                    lastBeat = .distantPast
                }
                if heartbeatDue || Date().timeIntervalSince(lastBeat) > 30 {
                    try await c.heartbeat(nodeID: Self.nodeID, info: nodeInfo())
                    lastBeat = Date()
                    heartbeatDue = false
                }
                // Report finished work before asking for more: the worker treats a
                // poll from a node that still owns a task as abandonment of it.
                if CompletionOutbox.hasPending {
                    status = "Reporting results"
                    let flush = await Task.detached { await NodeService.flushOutbox(client: c, attempts: 1) }.value
                    noteFlush(flush)
                    if let error = flush.error { throw error }
                }
                if pendingRestore != nil { await applyPendingRestore() }
                // Don't claim work while the chat/batch has the model; a claimed
                // task would just sit, and the worker would requeue it.
                if let holder = await GPUGate.shared.holder {
                    status = "Waiting: model busy (\(holder))"
                    try await Task.sleep(for: .seconds(2))
                    continue
                }
                status = "Waiting for tasks"
                if let data = try await c.nextTask(nodeID: Self.nodeID, wait: 25) {
                    await execute(data, client: c)
                }
                backoff = 1
            } catch {
                if Task.isCancelled { break }
                if let w = error as? WorkerError, case .http(404, _) = w {
                    lastBeat = .distantPast   // worker forgot us (restart): re-register
                }
                lastError = error.localizedDescription
                status = "Retrying in \(Int(backoff)) s"
                try? await Task.sleep(for: .seconds(backoff))
                backoff = min(backoff * 2, 60)
            }
        }
    }

    private func execute(_ data: Data, client: FleetClient) async {
        let head: FleetTask<EmptyPayload>
        do {
            head = try FleetClient.decodeTask(data, payload: EmptyPayload.self)
        } catch {
            lastError = "Undecodable task: \(error.localizedDescription)"
            return
        }
        let nodeID = Self.nodeID
        let taskID = head.id
        currentTask = "\(head.kind) · \(taskID)"
        status = "Running \(head.kind)"
        let buffer = TaskEventBuffer()
        let info = nodeInfo()
        // Detached: posting events must keep working even if the loop is cancelled.
        let pump = Task.detached {
            await NodeService.pump(buffer, client: client, taskID: taskID, nodeID: nodeID, info: info)
        }

        // nil = released (paused): the worker requeues it, nothing to report.
        var result: TaskResult?
        do {
            try await GPUGate.shared.acquire("fleet task")
            do {
                try Task.checkCancellation()
                switch head.kind {
                case "generate":
                    result = try await runGenerate(data, buffer: buffer)
                case "train_lora":
                    result = try await runTrain(data, taskID: taskID, buffer: buffer, client: client)
                case "bench":
                    result = try await runBench(data, buffer: buffer)
                default:
                    result = .failure("This node can't run \(head.kind) tasks.")
                }
            } catch {
                result = taskResult(after: error, buffer: buffer)
            }
            // A deferred adapter restore keeps the gate: the training weights are still
            // loaded, and nothing else may run on them until applyPendingRestore.
            if pendingRestore == nil { await GPUGate.shared.release() }
        } catch {
            // Cancelled while queued behind the chat or a summary.
            result = taskResult(after: error, buffer: buffer)
        }

        buffer.finish()
        await pump.value
        currentTask = nil

        guard let report = result else {
            lastError = "Paused during \(head.kind) \(taskID); released it for the worker to requeue."
            return
        }
        CompletionOutbox.add(PendingCompletion(taskID: taskID, nodeID: nodeID, result: report))
        // Fresh task: a cancelled loop must still be able to report.
        let flush = await Task.detached { await NodeService.flushOutbox(client: client, attempts: 5) }.value

        tasksServed += 1
        UserDefaults.standard.set(tasksServed, forKey: "node.tasksServed")
        if let err = report.error { lastError = "\(head.kind): \(err)" }
        noteFlush(flush)
    }

    /// What to tell the worker after the work threw.
    private func taskResult(after error: Error, buffer: TaskEventBuffer) -> TaskResult? {
        if buffer.cancelRequested { return .failure("cancelled by requester") }
        if Task.isCancelled { return nil }   // paused: release, don't report a terminal failure
        return .failure(error.localizedDescription)
    }

    private func noteFlush(_ flush: FlushResult) {
        if !flush.dropped.isEmpty {
            lastError = "Worker rejected the result of \(flush.dropped.joined(separator: ", ")); dropped it."
        }
        if let error = flush.error {
            lastError = "Couldn't report results yet (\(error.localizedDescription)); will retry before taking new work."
        }
    }

    struct FlushResult: Sendable {
        var dropped: [String] = []
        var error: Error?
    }

    /// Sends every stored completion, retrying each up to `attempts` times with
    /// exponential backoff. Settled (2xx/404/409) and permanently rejected ones
    /// leave the outbox; the first one still failing stops the flush.
    nonisolated private static func flushOutbox(client: FleetClient, attempts: Int) async -> FlushResult {
        var out = FlushResult()
        for pending in CompletionOutbox.all() {
            var delay: Double = 1
            for attempt in 1...max(attempts, 1) {
                do {
                    try await client.complete(taskID: pending.taskID, nodeID: pending.nodeID, result: pending.result)
                    CompletionOutbox.remove(pending.taskID)
                    break
                } catch {
                    if FleetClient.isSettled(error) {
                        CompletionOutbox.remove(pending.taskID)
                        break
                    }
                    if FleetClient.isPermanent(error) {
                        CompletionOutbox.remove(pending.taskID)
                        out.dropped.append(pending.taskID)
                        break
                    }
                    if attempt == max(attempts, 1) {
                        out.error = error
                        return out
                    }
                    try? await Task.sleep(for: .seconds(delay))
                    delay = min(delay * 2, 16)
                }
            }
        }
        return out
    }

    /// Posts buffered events every ~150 ms (or at 32 queued), keeps the node
    /// fresh during long silent stretches (model load, LoRA conversion), and
    /// relays the worker's cancel flag into the buffer. A failed POST puts its
    /// events back (bounded); once the work is finished it gives up after a few
    /// consecutive failures so the result report isn't held hostage.
    nonisolated private static func pump(_ buffer: TaskEventBuffer, client: FleetClient, taskID: String,
                                         nodeID: String, info: NodeInfo) async {
        var lastPost = Date()
        var lastBeat = Date()
        var failures = 0
        while true {
            let finished = buffer.isFinished
            let queued = buffer.count
            let due = Date().timeIntervalSince(lastPost)
            let wait = failures == 0 ? 0 : min(pow(2, Double(failures - 1)), 8)
            if due >= wait, (queued > 0 && (queued >= 32 || due >= 0.15 || finished)) || due >= 5 {
                let events = buffer.drain()
                do {
                    if try await client.postEvents(taskID: taskID, nodeID: nodeID, events: events) {
                        buffer.requestCancel()
                    }
                    failures = 0
                } catch {
                    // 409: the task is no longer ours (requeued/cancelled). Stop the work.
                    if let w = error as? WorkerError, case .http(409, _) = w {
                        buffer.requestCancel()
                        if finished { return }
                    } else {
                        buffer.requeue(events)
                        failures += 1
                        if finished && failures >= 4 { return }
                    }
                }
                lastPost = Date()
            }
            if finished && buffer.count == 0 { return }
            if Date().timeIntervalSince(lastBeat) > 20 {
                try? await client.heartbeat(nodeID: nodeID, info: info)
                lastBeat = Date()
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    private func ensureLoaded() async throws {
        if !(await backend.isLoaded) {
            status = "Loading model"
            try await backend.load { _ in }
        }
    }

    // MARK: generate

    private struct GeneratePayload: Decodable {
        let prompt: String
        let system: String?
        let maxTokens: Int?
        let temperature: Double?
    }

    private func runGenerate(_ data: Data, buffer: TaskEventBuffer) async throws -> TaskResult {
        let p = try FleetClient.decodeTask(data, payload: GeneratePayload.self).payload
        try await ensureLoaded()
        try Task.checkCancellation()
        status = "Generating"
        let convo = try await backend.makeConversation(
            instructions: p.system ?? Self.defaultSystem,
            maxTokens: min(max(p.maxTokens ?? 512, 1), 8192),
            temperature: Float(min(max(p.temperature ?? 0.4, 0), 2)))
        let start = Date()
        var output = ""
        var reasoning = ""
        var chunks = 0
        do {
            for try await event in convo.send(p.prompt) {
                if buffer.cancelRequested || Task.isCancelled { break }
                switch event {
                case .text(let chunk):
                    output += chunk
                    chunks += 1
                    buffer.append(.token(chunk))
                case .reasoning(let chunk):
                    // Thinking tokens cost time but aren't part of the answer.
                    reasoning += chunk
                    chunks += 1
                case .toolCall, .rejectedToolCall:
                    break   // generate tasks declare no tools
                }
            }
        } catch {
            await convo.synchronize()   // the gate is released right after we return
            throw error
        }
        await convo.synchronize()
        output = ThinkSplitter.split(output).answer
        try Task.checkCancellation()
        if buffer.cancelRequested { throw CancellationError() }
        let seconds = Date().timeIntervalSince(start)
        var tokens = chunks
        if let mlx = backend.mlxBackend, let n = await mlx.tokenCount(of: reasoning + output) { tokens = n }
        let tps = seconds > 0 ? Double(tokens) / seconds : 0
        lastTokensPerSecond = tps
        return TaskResult(status: "succeeded", output: output, error: nil,
                          metrics: ["tokens": Double(tokens), "seconds": seconds, "tokens_per_second": tps])
    }

    // MARK: bench

    /// A benchmark replaced the loaded model, so an applied federated adapter is gone
    /// (FederatedTrainer notices by itself: it holds the converted container weakly).
    func modelWasUnloadedByBenchmark() {
        if adapterStatus?.hasPrefix("Using round") == true {
            adapterStatus = "A benchmark reloaded the base model, so the federated adapter is no longer applied. "
                + "Tap Use federated adapter to apply it again."
        }
    }

    private struct BenchPayload: Decodable {
        let modelId: String?
        let workloadIds: [String]?
        let repetitions: Int?
    }

    /// Runs the Bench screen's workload set (BenchmarkRunner) on one catalog
    /// model and completes with the schema-v1 receipt JSON as output. The
    /// receipt is also saved on the iPad and posted to the worker's /receipts.
    private func runBench(_ data: Data, buffer: TaskEventBuffer) async throws -> TaskResult {
        guard let mlx = backend.mlxBackend else { throw NodeError.cannotBench }
        let p = try FleetClient.decodeTask(data, payload: BenchPayload.self).payload
        let entry: ModelCatalog.Entry
        if let id = p.modelId {
            guard let e = ModelCatalog.entry(id: id), e.engine == .mlx else {   // fleet bench = MLX runner
                throw BenchError.unknownModel(id, known: ModelCatalog.all.map(\.id))
            }
            entry = e
        } else {
            // The chat selection may be a Xybrid entry; the fleet bench runs the MLX runner only.
            let selected = ModelCatalog.selected
            guard selected.engine == .mlx else {
                throw BenchError.unknownModel(selected.id, known: ModelCatalog.all.map(\.id))
            }
            entry = selected
        }
        _ = try BenchWorkloads.workloads(ids: p.workloadIds)   // fail fast on an unknown id
        status = "Benchmarking \(entry.displayName)"
        onModelChanged()   // the runner unloads the chat model
        modelWasUnloadedByBenchmark()
        await (backend as? EngineRouter)?.releaseXybrid()   // only one engine holds weights
        let receipt = await BenchmarkRunner(backend: mlx).run(
            entry: entry, workloadIDs: p.workloadIds,
            repetitions: p.repetitions ?? BenchmarkRunner.defaultRepetitions,
            progress: { buffer.append(.progress($0.fraction, $0.message)) },
            isCancelled: { buffer.cancelRequested })
        onModelChanged()
        if receipt.status == "cancelled" { throw CancellationError() }

        if let url = try? ReceiptStore.save(receipt) {
            do {
                if let name = try await ReceiptStore.upload(url) { buffer.append(.log("receipt stored as \(name)")) }
            } catch {
                buffer.append(.log("receipt upload failed: \(error.localizedDescription)"))
            }
        }
        var metrics: [String: Double] = ["total_s": receipt.totalWallS]
        if let l = receipt.load {
            metrics["first_load_s"] = l.firstLoadS
            metrics["warm_load_s"] = l.warmLoadS
        }
        for w in receipt.workloads {
            metrics["\(w.id).ttft_s"] = w.median.ttftS
            metrics["\(w.id).prefill_tps"] = w.median.prefillTps
            metrics["\(w.id).decode_tps"] = w.median.decodeTps
            metrics["\(w.id).peak_footprint_mb"] = Double(w.median.peakFootprintBytes) / 1e6
        }
        if let tps = receipt.workloads.first?.median.decodeTps { lastTokensPerSecond = tps }
        let json = String(decoding: try BenchReceipt.encoder.encode(receipt), as: UTF8.self)
        let ok = receipt.status == "ok" || receipt.status == "skipped"
        return TaskResult(status: ok ? "succeeded" : "failed", output: json,
                          error: ok ? nil : (receipt.error ?? receipt.status), metrics: metrics)
    }

    // MARK: train_lora

    /// Per-field cap, matching the MCP `start_fed_round` schema.
    nonisolated static let maxExampleChars = 8_000
    nonisolated static let maxExamples = 2_000

    private struct TrainPayload: Decodable {
        struct Example: Decodable {
            let text: String?
            let prompt: String?
            let completion: String?

            /// nil when the example is empty, malformed or over the length cap.
            var sample: FederatedTrainer.Sample? {
                func usable(_ s: String?) -> String? {
                    guard let s, s.count <= NodeService.maxExampleChars,
                          !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                    return s
                }
                if let text = usable(text) { return .text(text) }
                guard let prompt = usable(prompt), let completion = usable(completion) else { return nil }
                return .chat(prompt: prompt, completion: completion)
            }
        }
        let roundId: String?
        let baseModel: String
        let iterations: Int?
        let learningRate: Double?
        let rank: Int?
        let layers: Int?
        let batchSize: Int?
        let examples: [Example]?
        let useLocalData: Bool?
        let initAdapter: String?
        let seed: Int?
    }

    enum NodeError: LocalizedError {
        case cannotTrain
        case trainingOff
        case documentsOff
        case badInitAdapter(String)
        case cannotBench
        case wrongModel(want: String, have: String)
        var errorDescription: String? {
            switch self {
            case .cannotTrain: "This node's backend can't train LoRA adapters."
            case .trainingOff:
                "Training tasks are turned off on this iPad (Settings → Compute node → Accept training tasks)."
            case .documentsOff:
                "This round wants to train on the iPad's imported documents, which the user hasn't allowed (Settings → Compute node → Allow federated training on my documents)."
            case .badInitAdapter(let path): "Refusing init_adapter \(path): not a federated round adapter path."
            case .cannotBench: "This node's backend can't run benchmarks."
            case .wrongModel(let want, let have):
                "Round is for \(want) but this iPad has \(have) loaded."
            }
        }
    }

    /// The only worker paths a train task may load as its starting adapter:
    /// `^fed/rounds/r[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}/adapter$` (federated.py round ids).
    nonisolated static func isRoundAdapterPath(_ path: String) -> Bool {
        let prefix = "fed/rounds/", suffix = "/adapter"
        guard path.hasPrefix(prefix), path.hasSuffix(suffix) else { return false }
        let id = Array(path.utf8.dropFirst(prefix.utf8.count).dropLast(suffix.utf8.count))
        guard id.count == 26 else { return false }
        func digits(_ r: Range<Int>) -> Bool { id[r].allSatisfy { $0 >= 0x30 && $0 <= 0x39 } }
        let hex = id[18..<26].allSatisfy { ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x61 && $0 <= 0x66) }
        return id[0] == UInt8(ascii: "r") && digits(1..<9) && id[9] == UInt8(ascii: "T")
            && digits(10..<16) && id[16] == UInt8(ascii: "Z") && id[17] == UInt8(ascii: "-") && hex
    }

    private func runTrain(_ data: Data, taskID: String, buffer: TaskEventBuffer,
                          client: FleetClient) async throws -> TaskResult {
        guard let trainer else { throw NodeError.cannotTrain }
        guard acceptTraining else { throw NodeError.trainingOff }
        guard ModelCatalog.selected.lamqRevision == nil else { throw NodeError.cannotTrain }
        let p = try FleetClient.decodeTask(data, payload: TrainPayload.self).payload
        let useDocuments = p.useLocalData ?? false
        if useDocuments && !allowDocumentTraining { throw NodeError.documentsOff }
        guard p.baseModel.caseInsensitiveCompare(trainer.modelID) == .orderedSame else {
            throw NodeError.wrongModel(want: p.baseModel, have: trainer.modelID)
        }
        if let path = p.initAdapter, !Self.isRoundAdapterPath(path) { throw NodeError.badInitAdapter(path) }
        let start = Date()
        let seed = p.seed ?? 1234
        // Same bounds the worker enforces; don't trust the payload blindly.
        let iterations = min(max(p.iterations ?? 50, 1), 5_000)
        let learningRate = min(max(p.learningRate ?? 1e-4, 1e-7), 1e-1)
        let rank = min(max(p.rank ?? 8, 1), 64)
        let layers = min(max(p.layers ?? 8, 1), 64)
        let batchSize = min(max(p.batchSize ?? 1, 1), 16)

        // Validate the data before touching the model.
        let examples = (p.examples ?? []).prefix(Self.maxExamples)
        var raw = examples.compactMap(\.sample)
        let invalid = examples.count - raw.count
        if useDocuments { raw += (await localSamples()).map { FederatedTrainer.Sample.text($0) } }
        if invalid > 0 { buffer.append(.log("skipped \(invalid) empty, malformed or over-long examples")) }
        guard raw.count >= 2 else { throw FederatedTrainer.TrainerError.notEnoughData(raw.count) }

        buffer.append(.progress(0, "preparing"))
        try await ensureLoaded()
        try Task.checkCancellation()
        let prepared = try await trainer.prepareSamples(raw)
        if prepared.dropped > 0 { buffer.append(.log("dropped \(prepared.dropped) samples under 2 tokens")) }
        guard prepared.texts.count >= 2 else { throw FederatedTrainer.TrainerError.notEnoughData(prepared.texts.count) }
        // Download before changing any weights.
        var initBytes: Data?
        if let path = p.initAdapter { initBytes = try await client.download(path: path) }
        try Task.checkCancellation()

        status = "Preparing LoRA"
        let converted = try await trainer.ensureAdapter(rank: rank, layers: layers, seed: seed)
        // What the chat runs on now. (No restore can be pending here: a pending
        // restore holds the gate, and we hold it.)
        let snapshot = try await trainer.snapshot()

        let outcome: FederatedTrainer.Outcome
        do {
            if let initBytes {
                buffer.append(.log("loading global adapter \(p.initAdapter ?? "")"))
                try await trainer.loadAdapter(initBytes)
            } else {
                try await trainer.resetToInitial(seed: seed)
            }
            buffer.append(.log("training on \(prepared.texts.count) samples, \(iterations) iterations"))
            status = "Training LoRA"
            outcome = try await trainer.train(
                samples: prepared.texts, iterations: iterations, learningRate: learningRate,
                batchSize: batchSize, roundID: p.roundId ?? "", events: buffer)
        } catch {
            await restoreChatAdapter(snapshot, trainer: trainer, converted: converted)
            throw error
        }
        // The trained adapter is already serialised; give the chat its weights back
        // before the (slow) upload.
        await restoreChatAdapter(snapshot, trainer: trainer, converted: converted)

        status = "Uploading adapter"
        try await client.putArtifact(taskID: taskID, nodeID: Self.nodeID, name: "adapter.safetensors",
                                     data: outcome.adapter)
        var metrics: [String: Double] = [
            "num_examples": Double(outcome.trainSamples),
            "seconds": Date().timeIntervalSince(start),
            "iterations": Double(outcome.iterations),
            "adapter_bytes": Double(outcome.adapter.count),
        ]
        if let l = outcome.trainLoss, l.isFinite { metrics["final_loss"] = l }
        if let l = outcome.validationLoss, l.isFinite { metrics["val_loss"] = l }
        return TaskResult(status: "succeeded", output: "adapter.safetensors", error: nil, metrics: metrics)
    }

    /// Puts the chat's adapter back after a training task (caller holds the gate).
    /// In the background no Metal work is started: the restore waits for the next
    /// .active, and the gate stays held until then (see `execute`).
    private func restoreChatAdapter(_ snapshot: Data, trainer: FederatedTrainer, converted: Bool) async {
        guard sceneActive else {
            pendingRestore = PendingRestore(snapshot: snapshot, converted: converted)
            return
        }
        do {
            try await trainer.loadAdapter(snapshot)
            // Same weights as before the task, so live chats stay valid, unless this
            // task was the one that first converted the model to LoRA.
            if converted { onModelChanged() }
        } catch {
            try? await trainer.resetToBase()
            adapterStatus = "Couldn't restore the chat's adapter after training (\(error.localizedDescription)); chat is back on the base model."
            onModelChanged()
        }
    }

    /// Finishes a deferred restore. The gate is still held from the training task,
    /// so this restores first and then releases it.
    private func applyPendingRestore() async {
        guard sceneActive, let pending = pendingRestore else { return }
        pendingRestore = nil
        if let trainer {
            do {
                try await trainer.loadAdapter(pending.snapshot)
                if pending.converted { onModelChanged() }
            } catch {
                try? await trainer.resetToBase()
                adapterStatus = "Couldn't restore the chat's adapter after training (\(error.localizedDescription)); chat is back on the base model."
                onModelChanged()
            }
        }
        await GPUGate.shared.release()
    }

    /// Imported documents, cut into ~1500-character training samples. Only read
    /// when the user allowed training on documents.
    private func localSamples(chunkChars: Int = 1_500, limit: Int = 400) async -> [String] {
        guard allowDocumentTraining else { return [] }
        var out: [String] = []
        for doc in await documents.all() {
            guard let text = try? await documents.text(of: doc) else { continue }
            var rest = Substring(text)
            while !rest.isEmpty, out.count < limit {
                let piece = rest.prefix(chunkChars)
                rest = rest.dropFirst(piece.count)
                let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.count >= 20 { out.append(trimmed) }
            }
        }
        return out
    }

    // MARK: Federated adapter (Settings button)

    /// Download the latest aggregated round's adapter and apply it to the chat
    /// model. The only path that changes the chat's adapter.
    func useFederatedAdapter() async {
        guard let trainer else { adapterStatus = NodeError.cannotTrain.localizedDescription; return }
        guard !isApplyingAdapter else { return }
        isApplyingAdapter = true
        defer { isApplyingAdapter = false }
        adapterStatus = "Checking latest round…"
        do {
            let client = try FleetClient.fromSettings()
            guard let round = try await client.latestRound() else {
                adapterStatus = "No aggregated round yet."
                return
            }
            guard round.baseModel.caseInsensitiveCompare(trainer.modelID) == .orderedSame else {
                throw NodeError.wrongModel(want: round.baseModel, have: trainer.modelID)
            }
            let path = "fed/rounds/\(round.id)/adapter"
            guard Self.isRoundAdapterPath(path) else { throw NodeError.badInitAdapter(path) }
            adapterStatus = "Downloading \(round.id)…"
            let bytes = try await client.download(path: path)
            adapterStatus = "Applying…"
            try await GPUGate.shared.acquire("federated adapter")
            do {
                try await ensureLoaded()
                try await trainer.ensureAdapter(rank: round.hyperparameters.rank,
                                                layers: round.hyperparameters.layers, seed: round.seed ?? 1234)
                try await trainer.loadAdapter(bytes)
                onModelChanged()       // inside the gate: no chat step can slip in on stale KV
                await GPUGate.shared.release()
            } catch {
                await GPUGate.shared.release()
                throw error
            }
            adapterStatus = "Using round \(round.id) (\(bytes.count / 1024) KB)."
        } catch is CancellationError {
            adapterStatus = "Cancelled."
        } catch {
            adapterStatus = error.localizedDescription
        }
    }
}

// MARK: - Background grace

/// One `beginBackgroundTask` window, ended exactly once: by its owner once the
/// paused loop has finished, or synchronously by the expiration handler.
@MainActor
private final class BackgroundGrace {
    private var id: UIBackgroundTaskIdentifier = .invalid

    init(name: String) {
        id = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in
            MainActor.assumeIsolated { self?.end() }
        }
    }

    func end() {
        guard id != .invalid else { return }
        UIApplication.shared.endBackgroundTask(id)
        id = .invalid
    }
}

// MARK: - Result outbox

struct PendingCompletion: Codable, Sendable {
    let taskID: String
    let nodeID: String
    let result: TaskResult
}

/// Results not yet acknowledged by the worker, persisted so they survive a
/// pause or relaunch. Small and bounded (oldest dropped past `limit`).
enum CompletionOutbox {
    private static let key = "node.pendingCompletions"
    private static let limit = 20
    private static let lock = NSLock()

    static var hasPending: Bool { !all().isEmpty }

    static func all() -> [PendingCompletion] {
        lock.withLock { read() }
    }

    static func add(_ p: PendingCompletion) {
        lock.withLock {
            var items = read().filter { $0.taskID != p.taskID }
            items.append(p)
            write(Array(items.suffix(limit)))
        }
    }

    static func remove(_ taskID: String) {
        lock.withLock { write(read().filter { $0.taskID != taskID }) }
    }

    private static func read() -> [PendingCompletion] {
        guard let data = UserDefaults.standard.data(forKey: key) else { return [] }
        return (try? JSONDecoder().decode([PendingCompletion].self, from: data)) ?? []
    }

    private static func write(_ items: [PendingCompletion]) {
        if items.isEmpty {
            UserDefaults.standard.removeObject(forKey: key)
        } else if let data = try? JSONEncoder().encode(items.map(sanitized)) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    /// JSONEncoder throws on NaN/Inf; never let a metric block persistence.
    private static func sanitized(_ p: PendingCompletion) -> PendingCompletion {
        var r = p.result
        r.metrics = r.metrics.filter { $0.value.isFinite }
        return PendingCompletion(taskID: p.taskID, nodeID: p.nodeID, result: r)
    }
}
