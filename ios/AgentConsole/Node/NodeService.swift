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
/// node pauses when the app is backgrounded and resumes when it's active
/// again. A task in flight at that moment is reported as failed (the worker
/// would otherwise requeue it after 75 s anyway).
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
    @ObservationIgnored private var backgroundGrace: UIBackgroundTaskIdentifier = .invalid

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
        if on { start() } else { stop(status: "Stopped") }
    }

    /// Called at launch and when the app becomes active again.
    func resumeIfEnabled() {
        if isServing { start() }
    }

    func scenePhaseChanged(_ phase: ScenePhase) {
        switch phase {
        case .active:
            resumeIfEnabled()
        case .background:
            guard loop != nil else { return }
            // A few seconds of grace so a task in flight can report its failure.
            backgroundGrace = UIApplication.shared.beginBackgroundTask(withName: "node-pause") { [weak self] in
                Task { @MainActor in self?.endBackgroundGrace() }
            }
            let l = loop
            stop(status: "Paused (app in background)")
            Task { @MainActor in
                await l?.value
                self.endBackgroundGrace()
            }
        default:
            break
        }
    }

    private func endBackgroundGrace() {
        guard backgroundGrace != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundGrace)
        backgroundGrace = .invalid
    }

    private func start() {
        guard loop == nil else { return }
        applyIdleTimer()
        loop = Task { [weak self] in await self?.runLoop() }
    }

    private func stop(status: String) {
        loop?.cancel()
        loop = nil
        self.status = status
        applyIdleTimer()
    }

    private func applyIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = keepAwake && loop != nil
    }

    private func nodeInfo() -> NodeInfo {
        let device = UIDevice.current
        var kinds = ["generate"]
        if trainer != nil { kinds.append("train_lora") }
        return NodeInfo(
            name: device.name, kind: "ipad",
            capabilities: .init(kinds: kinds, model: backend.displayName,
                                memoryGB: (Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 * 10).rounded() / 10,
                                os: "\(device.systemName) \(device.systemVersion)"))
    }

    // MARK: Loop

    private func runLoop() async {
        var backoff: Double = 1
        var lastBeat = Date.distantPast
        while !Task.isCancelled {
            do {
                let client = try FleetClient.fromSettings()
                if Date().timeIntervalSince(lastBeat) > 30 {
                    try await client.heartbeat(nodeID: Self.nodeID, info: nodeInfo())
                    lastBeat = Date()
                }
                // Don't claim work while the chat/batch has the model; a claimed
                // task would just sit, and the worker would requeue it.
                if let holder = await GPUGate.shared.holder {
                    status = "Waiting: model busy (\(holder))"
                    try await Task.sleep(for: .seconds(2))
                    continue
                }
                status = "Waiting for tasks"
                if let data = try await client.nextTask(nodeID: Self.nodeID, wait: 25) {
                    await execute(data, client: client)
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

        let result: TaskResult
        await GPUGate.shared.acquire("fleet task")
        do {
            switch head.kind {
            case "generate":
                result = try await runGenerate(data, buffer: buffer)
            case "train_lora":
                result = try await runTrain(data, taskID: taskID, buffer: buffer, client: client)
            default:
                result = .failure("This node can't run \(head.kind) tasks.")
            }
        } catch {
            if buffer.cancelRequested {
                result = .failure("cancelled by requester")
            } else if Task.isCancelled {
                result = .failure("node paused (app backgrounded or serving turned off)")
            } else {
                result = .failure(error.localizedDescription)
            }
        }
        await GPUGate.shared.release()

        buffer.finish()
        await pump.value
        let report = result
        // Fresh task: a cancelled loop must still be able to report.
        let reported = await Task.detached { () -> String? in
            do {
                try await client.complete(taskID: taskID, nodeID: nodeID, result: report)
                return nil
            } catch {
                return error.localizedDescription
            }
        }.value

        tasksServed += 1
        UserDefaults.standard.set(tasksServed, forKey: "node.tasksServed")
        if let err = result.error { lastError = "\(head.kind): \(err)" }
        if let reported { lastError = "Couldn't report \(taskID): \(reported)" }
        currentTask = nil
    }

    /// Posts buffered events every ~150 ms (or at 32 queued), keeps the node
    /// fresh during long silent stretches (model load, LoRA conversion), and
    /// relays the worker's cancel flag into the buffer.
    nonisolated private static func pump(_ buffer: TaskEventBuffer, client: FleetClient, taskID: String,
                                         nodeID: String, info: NodeInfo) async {
        var lastPost = Date()
        var lastBeat = Date()
        while true {
            let finished = buffer.isFinished
            let queued = buffer.count
            let due = Date().timeIntervalSince(lastPost)
            if (queued > 0 && (queued >= 32 || due >= 0.15 || finished)) || due >= 5 {
                let events = buffer.drain()
                do {
                    if try await client.postEvents(taskID: taskID, nodeID: nodeID, events: events) {
                        buffer.requestCancel()
                    }
                } catch {
                    // 409: the task is no longer ours (requeued/cancelled). Stop the work.
                    if let w = error as? WorkerError, case .http(409, _) = w { buffer.requestCancel() }
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
        status = "Generating"
        let convo = try await backend.makeConversation(
            instructions: p.system ?? Self.defaultSystem,
            maxTokens: min(max(p.maxTokens ?? 512, 1), 8192),
            temperature: Float(min(max(p.temperature ?? 0.4, 0), 2)))
        let start = Date()
        var output = ""
        var chunks = 0
        for try await chunk in convo.send(p.prompt) {
            if buffer.cancelRequested { break }
            output += chunk
            chunks += 1
            buffer.append(.token(chunk))
        }
        try Task.checkCancellation()
        if buffer.cancelRequested { throw CancellationError() }
        let seconds = Date().timeIntervalSince(start)
        var tokens = chunks
        if let mlx = backend as? MLXBackend, let n = await mlx.tokenCount(of: output) { tokens = n }
        let tps = seconds > 0 ? Double(tokens) / seconds : 0
        lastTokensPerSecond = tps
        return TaskResult(status: "succeeded", output: output, error: nil,
                          metrics: ["tokens": Double(tokens), "seconds": seconds, "tokens_per_second": tps])
    }

    // MARK: train_lora

    private struct TrainPayload: Decodable {
        struct Example: Decodable {
            let text: String?
            let prompt: String?
            let completion: String?
            var sample: String? {
                if let text, !text.isEmpty { return text }
                guard let prompt, let completion else { return nil }
                return prompt + "\n" + completion
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
        case wrongModel(want: String, have: String)
        var errorDescription: String? {
            switch self {
            case .cannotTrain: "This node's backend can't train LoRA adapters."
            case .wrongModel(let want, let have):
                "Round is for \(want) but this iPad has \(have) loaded."
            }
        }
    }

    private func runTrain(_ data: Data, taskID: String, buffer: TaskEventBuffer,
                          client: FleetClient) async throws -> TaskResult {
        guard let trainer else { throw NodeError.cannotTrain }
        let p = try FleetClient.decodeTask(data, payload: TrainPayload.self).payload
        guard p.baseModel.caseInsensitiveCompare(trainer.modelID) == .orderedSame else {
            throw NodeError.wrongModel(want: p.baseModel, have: trainer.modelID)
        }
        let start = Date()
        let seed = p.seed ?? 1234
        let iterations = max(p.iterations ?? 50, 1)
        buffer.append(.progress(0, "preparing"))
        try await ensureLoaded()

        status = "Preparing LoRA"
        try await trainer.ensureAdapter(rank: p.rank ?? 8, layers: p.layers ?? 8, seed: seed)
        if let path = p.initAdapter {
            buffer.append(.log("loading global adapter \(path)"))
            try await trainer.loadAdapter(try await client.download(path: path))
        } else {
            try await trainer.resetToInitial(seed: seed)
        }
        onModelChanged()

        var samples = (p.examples ?? []).compactMap(\.sample)
        if p.useLocalData ?? false { samples += await localSamples() }
        buffer.append(.log("training on \(samples.count) samples, \(iterations) iterations"))

        status = "Training LoRA"
        let outcome = try await trainer.train(
            samples: samples, iterations: iterations, learningRate: p.learningRate ?? 1e-4,
            batchSize: max(p.batchSize ?? 1, 1), roundID: p.roundId ?? "", events: buffer)
        onModelChanged()   // chat KV caches were built with the old weights

        status = "Uploading adapter"
        try await client.putArtifact(taskID: taskID, nodeID: Self.nodeID, name: "adapter.safetensors",
                                     data: outcome.adapter)
        var metrics: [String: Double] = [
            "num_examples": Double(outcome.trainSamples),
            "seconds": Date().timeIntervalSince(start),
            "iterations": Double(outcome.iterations),
            "adapter_bytes": Double(outcome.adapter.count),
        ]
        if let l = outcome.trainLoss { metrics["final_loss"] = l }
        if let l = outcome.validationLoss { metrics["val_loss"] = l }
        return TaskResult(status: "succeeded", output: "adapter.safetensors", error: nil, metrics: metrics)
    }

    /// Imported documents, cut into ~1500-character training samples.
    private func localSamples(chunkChars: Int = 1_500, limit: Int = 400) async -> [String] {
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

    /// Download the latest aggregated round's adapter and apply it to the chat model.
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
            adapterStatus = "Downloading \(round.id)…"
            let bytes = try await client.download(path: "fed/rounds/\(round.id)/adapter")
            adapterStatus = "Applying…"
            await GPUGate.shared.acquire("federated adapter")
            do {
                try await ensureLoaded()
                try await trainer.ensureAdapter(rank: round.hyperparameters.rank,
                                                layers: round.hyperparameters.layers, seed: round.seed ?? 1234)
                try await trainer.loadAdapter(bytes)
                await GPUGate.shared.release()
            } catch {
                await GPUGate.shared.release()
                throw error
            }
            onModelChanged()
            adapterStatus = "Using round \(round.id) (\(bytes.count / 1024) KB)."
        } catch {
            adapterStatus = error.localizedDescription
        }
    }
}
