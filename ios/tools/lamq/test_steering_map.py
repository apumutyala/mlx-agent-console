"""Checks the steering-name map of InklingLamqModel.swift against PR #821's parameter tree.

`Inkling2.prepareLamqCheckpoint` (patch D) renames every tensor of a lamq `steering.safetensors`
with `InklingLamqSteering.mappedName`/`transform`, adds `[rows, 0]` placeholders for the packed
layers, removes the routed experts, and then the factory runs a STRICT update (`verify: .all`):
every model parameter must be present with its exact shape, and no tensor may be left over.

This test ports the Swift map literally and enumerates the parameters `Inkling2.swift` builds
(module keys and init shapes read from the source), then requires the two to match exactly for
the real steering header of the 3.9 GB build (`steering_header_3.9GB.json`: names, dtypes,
shapes only, fetched with an HTTP range request; no weights).

    cd ios/tools/lamq && python -m unittest -v test_steering_map
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = {  # text_config of every lamq Inkling-2-Mini build (identical config.json across sizes)
    "hidden": 2304, "layers": 24, "heads": 24, "kv_heads": 8, "head_dim": 128, "d_rel": 16,
    "window": 1024, "rel_extent": 1024, "kernel": 4, "experts": 128, "inter": 1152, "block": 6,
    "core": range(6, 18), "rank": 64, "vocab": 201024, "unpadded": 200058,
    "local": {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22},
}


# ---------------------------------------------------------------- literal port of the Swift map

def mapped_name(name: str) -> str | None:
    if name == "model.embed_tokens.embed_norm.weight":
        return "language_model.embed_norm.weight"
    if name in ("model.norm.weight", "model.final_input.query", "model.message_input.query"):
        return "language_model." + name[len("model."):]
    if not name.startswith("model.layers."):
        return None
    rest = "language_model." + name[len("model."):]
    renames = [
        (".self_attn.q_proj.norm.scale", ".self_attn.q_l2_clip.scale"),
        (".self_attn.relative_position_bias.norm.scale", ".self_attn.rel_l2_clip.scale"),
        (".self_attn.relative_position_bias.proj", ".self_attn.rel_logits_proj.proj"),
        (".self_attn.k_proj.sconv.", ".self_attn.k_sconv."),
        (".self_attn.v_proj.sconv.", ".self_attn.v_sconv."),
    ]
    for a, b in renames:
        if a in rest:
            rest = rest.replace(a, b)
            break
    return rest


def transform_shape(name: str, shape: list[int]) -> list[int]:
    if name.endswith("conv1d.weight"):
        return [shape[0], shape[2], shape[1]]  # swappedAxes(1, 2)
    return shape


ATTENTION_LINEARS = {  # lamq linear -> PR #821 key, (m, n)
    "q_proj": (3072, 2304), "k_proj": (1024, 2304), "v_proj": (1024, 2304),
    "r_proj": (384, 2304), "gate_proj": (3072, 2304), "o_proj": (2304, 3072),
}


# ------------------------------------------------- PR #821's parameters (from Inkling2.swift init)

def pr821_parameters(c=CONFIG) -> dict[str, list[int]]:
    D, H, KV, HD, R, K = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["d_rel"], c["kernel"]
    p: dict[str, list[int]] = {}
    for i in range(c["layers"]):
        sliding = i in c["local"]
        L = f"language_model.layers.{i}."
        a = L + "self_attn."
        p[a + "q_proj.weight"] = [H * HD, D]
        p[a + "k_proj.weight"] = [KV * HD, D]
        p[a + "v_proj.weight"] = [KV * HD, D]
        p[a + "r_proj.weight"] = [H * R, D]
        p[a + "o_proj.weight"] = [D, H * HD]
        p[a + "gate_proj.weight"] = [H * HD, D]
        p[a + "sink_proj.weight"] = [H, D]
        p[a + "rel_logits_proj.proj"] = [R, c["window"] if sliding else c["rel_extent"]]
        p[a + "q_l2_clip.scale"] = [H, HD]
        p[a + "rel_l2_clip.scale"] = [H, R]
        for kv in ("k", "v"):
            if sliding:
                p[a + f"{kv}_sconv.conv1d.weight"] = [KV * HD, K, 1]
            else:
                p[a + f"{kv}_sconv.static_weight"] = [K, KV * HD]
                p[a + f"{kv}_sconv.weight_proj.weight"] = [KV * K, D]
        p[L + "mlp.gate.weight"] = [c["experts"], D]
        p[L + "mlp.gate.e_score_correction_bias"] = [c["experts"]]
        # mlp.experts: removed by patch D (optional module set to none)
        if i > 0:
            p[L + "attention_input.query"] = [D] if sliding else [3, D]
        p[L + "mlp_input.query"] = [D]
        p[L + "post_attention_layernorm.weight"] = [D]
        p[L + "post_attention_convolution.conv1d.weight"] = [D, K, 1]
        p[L + "post_mlp_convolution.conv1d.weight"] = [D, K, 1]
        if sliding:
            p[L + "input_layernorm.weight"] = [D]
        else:
            for n in ("q", "k", "v"):
                p[L + f"{n}_input_layernorm.weight"] = [D]
        if i in c["core"]:
            for g in ("q", "k", "v", "mlp"):
                p[L + f"{g}_message_gate.down_proj.weight"] = [c["rank"], 2 * D]
                p[L + f"{g}_message_gate.forget_gate_proj.weight"] = [D, c["rank"]]
                p[L + f"{g}_message_gate.message_gate_proj.weight"] = [D, c["rank"]]
    p["language_model.embed_tokens.weight"] = [c["vocab"], D]
    p["language_model.embed_norm.weight"] = [D]
    p["language_model.final_input.query"] = [D]
    p["language_model.norm.weight"] = [D]
    p["language_model.message_input.query"] = [len(c["core"]), 4, D]
    p["lm_head.weight"] = [c["vocab"], D]
    return p


def lamq_checkpoint(header: dict, c=CONFIG) -> dict[str, list[int]]:
    """What prepareLamqCheckpoint hands to the strict update: mapped steering + placeholders."""
    out: dict[str, list[int]] = {}
    for name, info in header.items():
        m = mapped_name(name)
        assert m is not None, f"unmapped steering tensor {name}"
        assert m not in out, f"collision on {m}"
        out[m] = transform_shape(name, info["shape"])
    for i in range(c["layers"]):
        for key, (m, _) in ATTENTION_LINEARS.items():
            out[f"language_model.layers.{i}.self_attn.{key}.weight"] = [m, 0]
    out["language_model.embed_tokens.weight"] = [c["vocab"], 0]
    out["lm_head.weight"] = [c["unpadded"], 0]
    return out


class SteeringMap(unittest.TestCase):
    def setUp(self):
        self.header = json.loads((HERE / "steering_header_3.9GB.json").read_text())

    def test_every_parameter_and_nothing_else(self):
        want = pr821_parameters()
        got = lamq_checkpoint(self.header)
        self.assertEqual(sorted(set(want) - set(got)), [], "parameters the checkpoint lacks")
        self.assertEqual(sorted(set(got) - set(want)), [], "tensors with no parameter")

    def test_shapes(self):
        want = pr821_parameters()
        got = lamq_checkpoint(self.header)
        placeholders = {k for k in got if got[k][-1] == 0 and len(got[k]) == 2}
        for k in sorted(set(want) & set(got) - placeholders):
            self.assertEqual(got[k], want[k], k)
        # Placeholders keep the leading dimension callers read (`r_proj.weight.dim(0)` etc.).
        for k in placeholders:
            if k.endswith("embed_tokens.weight") or k == "lm_head.weight":
                continue
            self.assertEqual(got[k][0], want[k][0], k)

    def test_counts(self):
        self.assertEqual(len(self.header), 499)
        self.assertEqual(len(lamq_checkpoint(self.header)), 499 + 24 * 6 + 2)

    def test_float32_steering_matches_pr821_float32_parameters(self):
        # PR #821 casts these to float32 on its own conversion; lamq ships them float32 already.
        for name, info in self.header.items():
            if name.endswith(("mlp.gate.weight", "forget_gate_proj.weight", "message_gate_proj.weight")):
                self.assertEqual(info["dtype"], "F32", name)


if __name__ == "__main__":
    unittest.main()
