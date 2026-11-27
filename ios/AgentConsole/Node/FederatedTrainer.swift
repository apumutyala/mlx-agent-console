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
/// The initial adapter has lora_b = 0, so it behaves exactly like the base model.
///
/// Training runs on the same weights the chat uses, so callers take a
/// `snapshot()` before a task and `loadAdapter(snapshot)` afterwards: only an
/// explicit "Use federated adapter" changes what the chat runs on.
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

    /// One training example before tokenization.
    enum Sample: Sendable {
        case text(String)
        case chat(prompt: String, completion: String)
    }

    enum TrainerError: LocalizedError {
        case notLoaded
        case adapterShape(have: String, want: String)
        case seedMismatch(have: Int, want: Int)
        case noAdapter
        case notEnoughData(Int)
        case diverged

        var errorDescription: String? {
            switch self {
            case .notLoaded: "Model not loaded yet."
            case .adapterShape(let have, let want):
                "This model already has a LoRA adapter (\(have)); the task wants \(want). Relaunch the app to change rank/layers."
            case .seedMismatch(let have, let want):
                "LoRA was initialized with seed \(have), the round uses \(want). Relaunch the app to start from the round's seed."
            case .noAdapter: "No LoRA adapter applied to the loaded model yet."
            case .notEnoughData(let n):
                "Need at least 2 usable training samples (non-empty, at least 2 tokens); have \(n)."
            case .diverged: "Training diverged (non-finite loss)."
            }
        }
    }

    private struct Applied {
        /// Weak, not ObjectIdentifier: once the converted container is freed (model
        /// switch, benchmark), a new container at a reused address must not match.
        weak var container: ModelContainer?
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
    var adapterSummary: String? { applied.flatMap { $0.container == nil ? nil : $0 }.map { "rank \($0.rank) · \($0.layers) layers" } }

    private func container() throws -> ModelContainer {
        guard let c = backend.modelContainer else { throw TrainerError.notLoaded }
        return c
    }

    /// The loaded container, provided it is the one that was converted to LoRA
    /// (a model switch or reload replaces it with an unconverted one).
    private func convertedContainer() throws -> ModelContainer {
        let c = try container()
        guard let a = applied, a.container === c else { throw TrainerError.noAdapter }
        return c
    }

    /// Convert the model to LoRA once (seeded), or check the existing adapter
    /// matches. Returns true when this call converted the model.
    @discardableResult
    func ensureAdapter(rank: Int, layers: Int, seed: Int) async throws -> Bool {
        let c = try container()
        if let a = applied, a.container === c {
            guard a.rank == rank, a.layers == layers else {
                throw TrainerError.adapterShape(have: "rank \(a.rank), \(a.layers) layers",
                                                want: "rank \(rank), \(layers) layers")
            }
            return false
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
        applied = Applied(container: c, rank: rank, layers: layers, seed: seed, initial: initial)
        return true
    }

    /// The adapter the chat currently runs on (safetensors bytes), for restoring
    /// after a training task.
    func snapshot() async throws -> Data {
        try await convertedContainer().perform { ctx in
            try saveToData(arrays: Dictionary(uniqueKeysWithValues: ctx.model.trainableParameters().flattened()))
        }
    }

    /// Back to the seeded initial adapter (first round of a federation).
    func resetToInitial(seed: Int) async throws {
        guard let a = applied, let held = a.container, held === backend.modelContainer else {
            throw TrainerError.noAdapter
        }
        guard a.seed == seed else { throw TrainerError.seedMismatch(have: a.seed, want: seed) }
        try await loadAdapter(a.initial)
    }

    /// Back to base-model behaviour (the seeded initial adapter, lora_b = 0).
    func resetToBase() async throws {
        guard let a = applied else { throw TrainerError.noAdapter }
        try await loadAdapter(a.initial)
    }

    /// Install adapter weights (safetensors bytes) into the already-converted model.
    func loadAdapter(_ data: Data) async throws {
        try await convertedContainer().perform { ctx in
            let weights = try loadArrays(data: data)
            _ = try ctx.model.update(parameters: ModuleParameters.unflattened(weights),
                                     verify: [.noUnusedKeys, .shapeMismatch])
            eval(ctx.model)   // loadArrays is lazy
            try ctx.model.prepare()
        }
    }

    /// Tokenizer-aware cleanup of training examples: prompt/completion pairs are
    /// rendered with the model's chat template (falling back to "prompt\ncompletion"),
    /// samples under 2 tokens are dropped and long ones truncated to `maxTokens`.
    /// Needs the model loaded, but doesn't touch the weights.
    func prepareSamples(_ samples: [Sample], maxTokens: Int = 2_048) async throws -> (texts: [String], dropped: Int) {
        let c = try container()
        let tokenizer = await c.tokenizer
        var out: [String] = []
        var dropped = 0
        for sample in samples {
            var text: String
            switch sample {
            case .text(let t):
                text = t
            case .chat(let prompt, let completion):
                let messages: [[String: any Sendable]] = [
                    ["role": "user", "content": prompt],
                    ["role": "assistant", "content": completion],
                ]
                if let ids = try? tokenizer.applyChatTemplate(messages: messages), !ids.isEmpty {
                    text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                    // LoRATrain re-encodes with special tokens; don't double the BOS.
                    if let bos = tokenizer.bosToken, !bos.isEmpty, text.hasPrefix(bos) {
                        text = String(text.dropFirst(bos.count))
                    }
                } else {
                    text = prompt + "\n" + completion
                }
            }
            var ids = tokenizer.encode(text: text)
            if ids.count < 2 { dropped += 1; continue }
            if ids.count > maxTokens {
                ids = Array(ids.prefix(maxTokens))
                text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
                if let bos = tokenizer.bosToken, !bos.isEmpty, text.hasPrefix(bos) {
                    text = String(text.dropFirst(bos.count))
                }
                if tokenizer.encode(text: text).count < 2 { dropped += 1; continue }
            }
            out.append(text)
        }
        return (out, dropped)
    }

    /// Fine-tune the current adapter. `events` receives progress and its cancel
    /// flag stops training at the next step. A non-finite loss stops training
    /// and throws `.diverged`. On any throw the model holds partial weights:
    /// the caller restores its snapshot.
    func train(samples: [String], iterations: Int, learningRate: Double, batchSize: Int,
               roundID: String, events: TaskEventBuffer) async throws -> Outcome {
        let c = try convertedContainer()
        guard samples.count >= 2 else { throw TrainerError.notEnoughData(samples.count) }
        // Last 10% (at least one) validates; the rest trains.
        let holdout = max(1, samples.count / 10)
        let validate = Array(samples.suffix(holdout))
        let train = Array(samples.dropLast(holdout))
        let modelID = backend.modelID

        return try await c.perform { ctx in
            let optimizer = Adam(learningRate: Float(learningRate))
            let parameters = LoRATrain.Parameters(
                batchSize: batchSize, iterations: iterations,
                stepsPerReport: 1,                 // check cancel every step
                stepsPerEval: iterations, validationBatches: 1,
                saveEvery: iterations + 1)         // no adapterURL: never saves to disk
            var losses: [Float] = []
            var validationLoss: Float?
            var stopped = false
            var diverged = false
            try LoRATrain.train(
                model: ctx.model, train: train, validate: validate, optimizer: optimizer,
                tokenizer: ctx.tokenizer, parameters: parameters
            ) { progress in
                switch progress {
                case .train(let i, let loss, _, let tps):
                    losses.append(loss)
                    if !loss.isFinite { diverged = true; return .stop }
                    let done = i + 1
                    if done % 5 == 0 || done == iterations {
                        events.append(.progress(Double(done) / Double(iterations),
                                                String(format: "iter %d/%d loss %.4f · %.0f tok/s", done, iterations, loss, tps)))
                    }
                case .validation(let i, let loss, _):
                    validationLoss = loss
                    if !loss.isFinite { diverged = true; return .stop }
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
            // A stopped or diverged run is thrown away (the caller restores its
            // snapshot), so don't spend more GPU work on it here: the app may be
            // on its way to the background.
            if diverged { throw TrainerError.diverged }
            if stopped { throw CancellationError() }
            eval(ctx.model)
            try ctx.model.prepare()
            let adapter = try saveToData(
                arrays: Dictionary(uniqueKeysWithValues: ctx.model.trainableParameters().flattened()),
                metadata: ["round_id": roundID, "base_model": modelID])
            let tail = losses.suffix(5)
            let trainLoss = tail.isEmpty ? nil : Double(tail.reduce(0, +) / Float(tail.count))
            if let trainLoss, !trainLoss.isFinite { throw TrainerError.diverged }
            return Outcome(adapter: adapter,
                           trainLoss: trainLoss,
                           validationLoss: validationLoss.map { Double($0) },
                           iterations: losses.count,
                           trainSamples: train.count)
        }
    }
}
