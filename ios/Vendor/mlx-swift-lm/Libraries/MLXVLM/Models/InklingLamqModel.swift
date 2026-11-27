// [agent-console patch D] Running a LookAtMyQuant packed Inkling 2 Mini build through
// `Inkling2` (mlx-swift-lm PR #821). Spec: docs/LAMQ-FORMAT-SPEC.md §8-9.
//
// How a packed directory loads (all through the normal VLMModelFactory path):
// 1. `VLMModelFactory._load` sees `manifest.json` with an `inkling2-packed-v1` format and hands
//    the directory to the model (`Inkling2.useLamqPackedDirectory`).
// 2. The only safetensors file is `steering.safetensors` (HF transformers names, ~122 MB).
//    `Inkling2.prepareCheckpoint` renames it to PR #821's layout (`InklingLamqSteering`), and
//    replaces the packed parts with the layers below: the six attention linears of each layer,
//    the routed experts, the embedding and the vocabulary head. Their placeholder weights are
//    `[rows, 0]` arrays, so strict validation passes and nothing large is ever materialised.
// 3. `materializeModelForInference` then evaluates the (small) parameters as usual.

import Foundation
import MLX
import MLXNN

public enum InklingLamqSteering {
    /// Whether `directory` holds a lamq packed build.
    public static func isPackedDirectory(_ directory: URL) -> Bool {
        let url = directory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let format = json["format"] as? String
        else { return false }
        return format.hasPrefix("inkling2-packed-v1")
    }

    /// Whether the checkpoint is a lamq `steering.safetensors` (HF names, no packed tensors).
    public static func isSteering(_ names: some Collection<String>) -> Bool {
        names.contains("model.layers.0.mlp.gate.weight")
            && !names.contains { $0.hasSuffix("q_proj.weight") || $0.contains(".experts.") }
    }

    /// PR #821's name for a steering tensor, or nil when it has no place in the model.
    ///
    /// | steering (HF transformers) | PR #821 |
    /// |---|---|
    /// | `model.embed_tokens.embed_norm.weight` | `language_model.embed_norm.weight` |
    /// | `model.{norm,final_input,message_input}.*` | `language_model.…` (same) |
    /// | `model.layers.N.self_attn.q_proj.norm.scale` | `….self_attn.q_l2_clip.scale` |
    /// | `….self_attn.relative_position_bias.norm.scale` | `….self_attn.rel_l2_clip.scale` |
    /// | `….self_attn.relative_position_bias.proj` | `….self_attn.rel_logits_proj.proj` |
    /// | `….self_attn.{k,v}_proj.sconv.X` | `….self_attn.{k,v}_sconv.X` |
    /// | everything else under `model.layers.N.` | same name |
    ///
    /// Shapes are unchanged except the depthwise convolutions (`*.conv1d.weight`), which are
    /// `[C, 1, K]` in the steering file and `[C, K, 1]` in PR #821 (same as its Thinking
    /// Machines conversion): see `transform`.
    public static func mappedName(_ name: String) -> String? {
        switch name {
        case "model.embed_tokens.embed_norm.weight": return "language_model.embed_norm.weight"
        case "model.norm.weight", "model.final_input.query", "model.message_input.query":
            return "language_model." + name.dropFirst("model.".count)
        default: break
        }
        guard name.hasPrefix("model.layers.") else { return nil }
        var rest = "language_model." + name.dropFirst("model.".count)
        let renames: [(String, String)] = [
            (".self_attn.q_proj.norm.scale", ".self_attn.q_l2_clip.scale"),
            (".self_attn.relative_position_bias.norm.scale", ".self_attn.rel_l2_clip.scale"),
            (".self_attn.relative_position_bias.proj", ".self_attn.rel_logits_proj.proj"),
            (".self_attn.k_proj.sconv.", ".self_attn.k_sconv."),
            (".self_attn.v_proj.sconv.", ".self_attn.v_sconv."),
        ]
        for (from, to) in renames where rest.contains(from) {
            rest = rest.replacingOccurrences(of: from, with: to)
            break
        }
        return rest
    }

    /// The tensor in PR #821's layout. The depthwise convolution weights are float32 in the
    /// steering file; PR #821's decode epilogue reads them as `const device T*` with T the
    /// activation dtype (bfloat16), so they are rounded to bfloat16, the dtype its other
    /// checkpoints carry. Everything else keeps its dtype.
    public static func transform(_ name: String, _ value: MLXArray) -> MLXArray {
        name.hasSuffix("conv1d.weight") ? value.swappedAxes(1, 2).asType(.bfloat16) : value
    }

    /// lamq linear name -> PR #821 attention module key.
    public static let attentionLinears: [(lamq: String, key: String)] = [
        ("self_attn.q_proj.linear", "q_proj"),
        ("self_attn.k_proj.linear", "k_proj"),
        ("self_attn.v_proj.linear", "v_proj"),
        ("self_attn.relative_position_bias.linear", "r_proj"),
        ("self_attn.o_proj.gate_proj", "gate_proj"),
        ("self_attn.o_proj.linear", "o_proj"),
    ]

    static func layerMeta(_ directory: URL, _ layer: Int) throws -> InklingLamqLayerMeta {
        try InklingLamqLayerMeta.load(
            directory.appendingPathComponent(String(format: "layer_%02d_meta.json", layer)))
    }

    static func layerFile(_ directory: URL, _ layer: Int) throws -> InklingLamqNpz {
        try InklingLamqNpz(
            url: directory.appendingPathComponent(String(format: "layer_%02d_packed.npz", layer)))
    }
}

/// A packed attention linear in place of PR #821's `Linear`. The weight is a `[out, 0]`
/// placeholder: callers only read `weight.dim(0)`.
final class InklingLamqLinearLayer: Linear {
    let packed: InklingLamqPackedLinear

    init(_ packed: InklingLamqPackedLinear) {
        self.packed = packed
        super.init(weight: MLXArray.zeros([packed.meta.m, 0], dtype: .bfloat16))
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        packed(x).asType(x.dtype)
    }
}

/// The packed embedding in place of PR #821's `Embedding`: rows decoded on demand.
final class InklingLamqEmbeddingLayer: Embedding {
    let buckets: InklingLamqBuckets

    init(_ buckets: InklingLamqBuckets) {
        self.buckets = buckets
        super.init(weight: MLXArray.zeros([buckets.order.count, 0], dtype: .bfloat16))
    }

    /// bfloat16 `[..., hidden]` rows of the token ids `x` (read on the host: one sync per call).
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ids = x.asType(.int32).asArray(Int32.self).map { Int($0) }
        return buckets.embed(ids).reshaped(x.shape + [buckets.hidden]).asType(.bfloat16)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        fatalError("lamq: the packed embedding is not tied to the head")
    }
}

/// The packed vocabulary head in place of PR #821's `lm_head`: float32 logits over the
/// unpadded vocabulary, in token order, before the soft cap (the model applies it).
final class InklingLamqHeadLayer: Linear {
    let buckets: InklingLamqBuckets

    init(_ buckets: InklingLamqBuckets) {
        self.buckets = buckets
        super.init(weight: MLXArray.zeros([buckets.order.count, 0], dtype: .bfloat16))
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let lead = Array(x.shape.dropLast())
        return buckets.logits(x).reshaped(lead + [buckets.order.count])
    }
}

/// Holds a layer's packed experts out of `Module` reflection.
final class InklingLamqExpertsBox: @unchecked Sendable {
    var value: InklingLamqPackedExperts?
}

/// Holds the packed directory a model loads from, out of `Module` reflection.
final class InklingLamqSourceBox: @unchecked Sendable {
    var directory: URL?
}
