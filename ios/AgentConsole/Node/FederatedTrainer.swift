import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers

/// On-device LoRA fine-tuning for federated rounds (worker/federated.py).
///
/// FedAvg averages lora_a and lora_b element-wise, which only makes sense if
/// every device starts from the same adapter. So the LoRA layers are created
/// once per loaded model right after seeding MLX's RNG with the round seed,
/// and a snapshot of that initial state is kept: a round without a parent
/// adapter resets to it instead of continuing from whatever was trained last.
///
/// Callers must hold `GPUGate` around every method here (conversion swaps
/// layers inside the model the chat uses).
actor FederatedTrainer {
    struct Outcome: Sendable {
        let adapter: Data              // safetensors of the trainable (LoRA) parameters
        let trainLoss: Double?         // mean of the last few training steps
        let validationLoss: Double?
        let iterations: Int
        let trainSamples: Int
    }

    enum TrainerError: LocalizedError {
        case notLoaded
        case adapterShape(have: String, want: String)
        case seedMismatch(have: Int, want: Int)
        case noAdapter
        case notEnoughData(Int)

        var errorDescription: String? {
            switch self {
            case .notLoaded: "Model not loaded yet."
            case .adapterShape(let have, let want):
                "This model already has a LoRA adapter (\(have)); the task wants \(want). Relaunch the app to change rank/layers."
            case .seedMismatch(let have, let want):
                "LoRA was initialized with seed \(have), the round uses \(want). Relaunch the app to start from the round's seed."
            case .noAdapter: "No LoRA adapter applied yet."
            case .notEnoughData(let n): "Need at least 2 training samples, have \(n)."
            }
        }
    }

    private struct Applied {
        let container: ObjectIdentifier
        let rank: Int
        let layers: Int
        let seed: Int
        let initial: Data
    }

    private let backend: MLXBackend
    private var applied: Applied?

    init(backend: MLXBackend) { self.backend = backend }

    nonisolated var modelID: String { backend.modelID }

    /// "rank 8 · 8 layers", or nil before any conversion.
    var adapterSummary: String? { applied.map { "rank \($0.rank) · \($0.layers) layers" } }

    private func container() throws -> ModelContainer {
        guard let c = backend.modelContainer else { throw TrainerError.notLoaded }
        return c
    }

    /// Convert the model to LoRA once (seeded), or check the existing adapter matches.
    func ensureAdapter(rank: Int, layers: Int, seed: Int) async throws {
        let c = try container()
        let id = ObjectIdentifier(c)
        if let a = applied, a.container == id {
            guard a.rank == rank, a.layers == layers else {
                throw TrainerError.adapterShape(have: "rank \(a.rank), \(a.layers) layers",
                                                want: "rank \(rank), \(layers) layers")
            }
            return
        }
        let initial: Data = try await c.perform { ctx in
            // Same seed on every device → identical initial lora_a everywhere.
            // A scoped RandomState (rather than seeding the global one) means
            // nothing else drawing random keys can shift the sequence.
            try withRandomState(MLXRandom.RandomState(seed: UInt64(truncatingIfNeeded: seed))) {
                _ = try LoRAContainer.from(
                    model: ctx.model,
                    configuration: LoRAConfiguration(numLayers: layers, loraParameters: .init(rank: rank)))
            }
            eval(ctx.model)
            // Some models cache fused views of their weights; rebuild after any
            // weight change (mlx-swift-lm does the same after loading adapters).
            try ctx.model.prepare()
            return try saveToData(arrays: Dictionary(uniqueKeysWithValues: ctx.model.trainableParameters().flattened()))
        }
        applied = Applied(container: id, rank: rank, layers: layers, seed: seed, initial: initial)
    }

    /// Back to the seeded initial adapter (first round of a federation).
    func resetToInitial(seed: Int) async throws {
        guard let a = applied else { throw TrainerError.noAdapter }
        guard a.seed == seed else { throw TrainerError.seedMismatch(have: a.seed, want: seed) }
        try await loadAdapter(a.initial)
    }

    /// Install adapter weights (safetensors bytes) into the already-converted model.
    func loadAdapter(_ data: Data) async throws {
        guard applied != nil else { throw TrainerError.noAdapter }
        try await container().perform { ctx in
            let weights = try loadArrays(data: data)
            _ = try ctx.model.update(parameters: ModuleParameters.unflattened(weights),
                                     verify: [.noUnusedKeys, .shapeMismatch])
            eval(ctx.model)   // loadArrays is lazy
            try ctx.model.prepare()
        }
    }

    /// Fine-tune the current adapter. `events` receives progress and its cancel
    /// flag stops training at the next step.
    func train(samples: [String], iterations: Int, learningRate: Double, batchSize: Int,
               roundID: String, events: TaskEventBuffer) async throws -> Outcome {
        guard applied != nil else { throw TrainerError.noAdapter }
        guard samples.count >= 2 else { throw TrainerError.notEnoughData(samples.count) }
        // Last 10% (at least one) validates; the rest trains.
        let holdout = max(1, samples.count / 10)
        let validate = Array(samples.suffix(holdout))
        let train = Array(samples.dropLast(holdout))
        let modelID = backend.modelID

        return try await container().perform { ctx in
            let optimizer = Adam(learningRate: Float(learningRate))
            let parameters = LoRATrain.Parameters(
                batchSize: batchSize, iterations: iterations,
                stepsPerReport: 1,                 // check cancel every step
                stepsPerEval: iterations, validationBatches: 1,
                saveEvery: iterations + 1)         // no adapterURL: never saves to disk
            var losses: [Float] = []
            var validationLoss: Float?
            var stopped = false
            try LoRATrain.train(
                model: ctx.model, train: train, validate: validate, optimizer: optimizer,
                tokenizer: ctx.tokenizer, parameters: parameters
            ) { progress in
                switch progress {
                case .train(let i, let loss, _, let tps):
                    losses.append(loss)
                    let done = i + 1
                    if done % 5 == 0 || done == iterations {
                        events.append(.progress(Double(done) / Double(iterations),
                                                String(format: "iter %d/%d loss %.4f · %.0f tok/s", done, iterations, loss, tps)))
                    }
                case .validation(let i, let loss, _):
                    validationLoss = loss
                    events.append(.log(String(format: "iter %d validation loss %.4f", i + 1, loss)))
                case .save:
                    break
                }
                if events.cancelRequested || Task.isCancelled {
                    stopped = true
                    return .stop
                }
                return .more
            }
            eval(ctx.model)
            try ctx.model.prepare()
            if stopped { throw CancellationError() }
            let adapter = try saveToData(
                arrays: Dictionary(uniqueKeysWithValues: ctx.model.trainableParameters().flattened()),
                metadata: ["round_id": roundID, "base_model": modelID])
            let tail = losses.suffix(5)
            return Outcome(adapter: adapter,
                           trainLoss: tail.isEmpty ? nil : Double(tail.reduce(0, +) / Float(tail.count)),
                           validationLoss: validationLoss.map { Double($0) },
                           iterations: losses.count,
                           trainSamples: train.count)
        }
    }
}
