"""Tests for expert_repack.py on small synthetic Inkling-shaped checkpoints.

    cd tools && python -m unittest -v test_expert_repack
"""
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import expert_repack as er  # noqa: E402

# Scaled-down Inkling: hidden 128, intermediate 64, 8 experts, group 32.
HIDDEN, INTER, EXPERTS, GROUP = 128, 64, 8, 32
PREFIX = "model.language_model.layers"
NP_DTYPES = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32, "U8": np.uint8}


def write_safetensors(path: Path, tensors: dict, metadata=None) -> None:
    """`tensors`: name -> (dtype, ndarray). Mirrors the safetensors writer (8-byte padded header)."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    blobs = []
    for name, (dtype, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def read_tensor(path: Path, name: str) -> tuple:
    shard = er.read_shard(path)
    t = shard.tensors[name]
    with path.open("rb") as f:
        f.seek(t.begin)
        return t, f.read(t.nbytes)


def quantized(rng, out_dims, in_dims, bits, experts=EXPERTS, affine=True, group=GROUP):
    parts = {
        "weight": ("U32", rng.integers(0, 2**32, (experts, out_dims, in_dims * bits // 32),
                                       dtype=np.uint64).astype(np.uint32)),
        "scales": ("BF16" if affine else "U8",
                   rng.integers(0, 2**16 if affine else 2**8, (experts, out_dims, in_dims // group),
                                dtype=np.uint64).astype(np.uint16 if affine else np.uint8)),
    }
    if affine:
        parts["biases"] = ("BF16", rng.integers(0, 2**16, (experts, out_dims, in_dims // group),
                                                dtype=np.uint64).astype(np.uint16))
    return parts


def make_model(root: Path, layers=3, bits=((2, 3), (2, 2), (3, 3)), split=True, nvfp4_layer=None):
    """A two-shard (or one-shard) model; layer L's gate_up uses bits[L][0], down bits[L][1].
    With `split`, layer 1's down_proj lives in the second shard (cross-shard layers)."""
    rng = np.random.default_rng(1234)
    root.mkdir(parents=True, exist_ok=True)
    shard_a, shard_b = {}, {}
    quant = {"group_size": GROUP, "bits": 2, "mode": "affine"}
    shard_a["model.language_model.embed_tokens.weight"] = (
        "BF16", rng.integers(0, 2**16, (50, HIDDEN), dtype=np.uint64).astype(np.uint16))
    for layer in range(layers):
        gb, db = bits[layer]
        prefix = f"{PREFIX}.{layer}.mlp.experts"
        target = shard_b if (split and layer == layers - 1) else shard_a
        target[f"{PREFIX}.{layer}.input_layernorm.weight"] = (
            "BF16", rng.integers(0, 2**16, (HIDDEN,), dtype=np.uint64).astype(np.uint16))
        target[f"{PREFIX}.{layer}.mlp.gate.weight"] = (
            "F32", rng.standard_normal((EXPERTS, HIDDEN)).astype(np.float32))
        if layer == nvfp4_layer:
            gu = quantized(rng, 2 * INTER, HIDDEN, 4, affine=False, group=16)
            dn = quantized(rng, HIDDEN, INTER, 4, affine=False, group=16)
            gu["global_scale"] = ("F32", rng.random((EXPERTS,)).astype(np.float32))
            dn["global_scale"] = ("F32", rng.random((EXPERTS,)).astype(np.float32))
        else:
            gu = quantized(rng, 2 * INTER, HIDDEN, gb)
            dn = quantized(rng, HIDDEN, INTER, db)
            quant[f"{prefix}.gate_up_proj"] = {"bits": gb, "group_size": GROUP, "mode": "affine"}
            quant[f"{prefix}.down_proj"] = {"bits": db, "group_size": GROUP, "mode": "affine"}
        for part, value in gu.items():
            target[f"{prefix}.gate_up_proj.{part}"] = value
        down_target = shard_b if (split and layer == 1) else target
        for part, value in dn.items():
            down_target[f"{prefix}.down_proj.{part}"] = value
    shard_b["lm_head.weight"] = (
        "BF16", rng.integers(0, 2**16, (50, HIDDEN), dtype=np.uint64).astype(np.uint16))
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    write_safetensors(root / names[0], shard_a, {"format": "mlx"})
    write_safetensors(root / names[1], shard_b, {"format": "mlx"})
    weight_map = {n: names[0] for n in shard_a} | {n: names[1] for n in shard_b}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    config = {
        "model_type": "inkling2_loop",
        "text_config": {"hidden_size": HIDDEN, "intermediate_size": INTER,
                        "n_routed_experts": EXPERTS, "num_experts_per_tok": 2,
                        "num_hidden_layers": layers},
        "quantization": quant,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"fake": true}')
    return shard_a, shard_b, weight_map


class RepackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.out = self.tmp / "out"
        self.quiet = lambda *a, **k: None

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_every_expert(self):
        shard_a, shard_b, weight_map = make_model(self.src)
        manifest = er.repack(self.src, self.out, "all", log=self.quiet)
        blob = (self.out / "experts.bin").read_bytes()
        self.assertEqual(len(blob), manifest["file_bytes"])
        source = shard_a | shard_b
        for layer in manifest["layers"]:
            self.assertEqual(layer["base_offset"] % er.RECORD_ALIGNMENT, 0)
            self.assertEqual(layer["record_stride"] % er.RECORD_ALIGNMENT, 0)
            self.assertGreaterEqual(layer["record_stride"], layer["record_bytes"])
            for e in range(EXPERTS):
                start = layer["base_offset"] + e * layer["record_stride"]
                self.assertEqual(start % er.RECORD_ALIGNMENT, 0)
                for f in layer["fields"]:
                    self.assertEqual(f["offset"] % er.FIELD_ALIGNMENT, 0)
                    dtype, arr = source[f["tensor"]]
                    want = np.ascontiguousarray(arr[e]).tobytes()
                    got = blob[start + f["offset"]:start + f["offset"] + f["nbytes"]]
                    self.assertEqual(got, want, f"layer {layer['layer']} expert {e} {f['name']}")
                    self.assertEqual(f["dtype"], dtype)
                    self.assertEqual(f["shape"], list(arr.shape[1:]))
            # Reassembling the stacked tensors from the records gives the source bytes.
            for f in layer["fields"]:
                stacked = b"".join(
                    blob[layer["base_offset"] + e * layer["record_stride"] + f["offset"]:][:f["nbytes"]]
                    for e in range(EXPERTS))
                self.assertEqual(stacked, np.ascontiguousarray(source[f["tensor"]][1]).tobytes())

    def test_manifest_quantization_and_hashes(self):
        make_model(self.src)
        manifest = er.repack(self.src, self.out, "sample", log=self.quiet)
        self.assertEqual(manifest["format"], er.FORMAT_NAME)
        self.assertEqual(manifest["version"], er.FORMAT_VERSION)
        self.assertEqual(manifest["experts"], EXPERTS)
        self.assertEqual(manifest["experts_per_token"], 2)
        self.assertEqual([l["layer"] for l in manifest["layers"]], [0, 1, 2])
        l0 = manifest["layers"][0]
        self.assertEqual(l0["projections"]["gate_up_proj"],
                         {"mode": "affine", "bits": 2, "group_size": GROUP,
                          "input_dims": HIDDEN, "output_dims": 2 * INTER})
        self.assertEqual(l0["projections"]["down_proj"]["bits"], 3)
        self.assertEqual(l0["tensor_prefix"], f"{PREFIX}.0.mlp.experts")
        self.assertEqual([f["name"] for f in l0["fields"]],
                         ["gate_up_proj.weight", "gate_up_proj.scales", "gate_up_proj.biases",
                          "down_proj.weight", "down_proj.scales", "down_proj.biases"])
        # Mixed bits change record sizes per layer; offsets are contiguous across layers.
        self.assertNotEqual(manifest["layers"][1]["record_bytes"], manifest["layers"][2]["record_bytes"])
        for prev, nxt in zip(manifest["layers"], manifest["layers"][1:]):
            self.assertEqual(nxt["base_offset"], prev["base_offset"] + EXPERTS * prev["record_stride"])
        blob = (self.out / "experts.bin").read_bytes()
        for l in manifest["layers"]:
            region = blob[l["base_offset"]:l["base_offset"] + EXPERTS * l["record_stride"]]
            self.assertEqual(hashlib.sha256(region).hexdigest(), l["sha256"])
        names = {s["name"]: s for s in manifest["source"]["files"]}
        self.assertEqual(names["config.json"]["sha256"],
                         hashlib.sha256((self.src / "config.json").read_bytes()).hexdigest())
        self.assertEqual(names["model-00001-of-00002.safetensors"]["bytes"],
                         (self.src / "model-00001-of-00002.safetensors").stat().st_size)
        self.assertEqual(manifest["expert_bytes_total"],
                         sum(EXPERTS * l["record_bytes"] for l in manifest["layers"]))

    def test_core_has_no_experts_and_same_bytes(self):
        shard_a, shard_b, _ = make_model(self.src)
        er.repack(self.src, self.out, "none", log=self.quiet)
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        source = shard_a | shard_b
        core_names = {n for n in source if not er.EXPERT_RE.match(n)}
        self.assertEqual(set(index["weight_map"]), core_names)
        for name in core_names:
            t, data = read_tensor(self.out / index["weight_map"][name], name)
            dtype, arr = source[name]
            self.assertEqual(data, np.ascontiguousarray(arr).tobytes(), name)
            self.assertEqual(t.dtype, dtype)
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            s = er.read_shard(self.out / shard)
            self.assertEqual(s.metadata, {"format": "mlx"})
            self.assertEqual((8 + s.header_len) % 8, 0)
            self.assertFalse(any(er.EXPERT_RE.match(n) for n in s.tensors))
        # Non-weight files are copied unchanged.
        self.assertEqual((self.out / "config.json").read_bytes(), (self.src / "config.json").read_bytes())
        self.assertEqual((self.out / "tokenizer.json").read_text(), '{"fake": true}')
        self.assertFalse(list(self.out.glob("*.partial")))

    def test_verify_detects_corruption(self):
        make_model(self.src)
        manifest = er.repack(self.src, self.out, "none", log=self.quiet)
        er.verify(self.src, self.out, "all", log=self.quiet)
        layer = manifest["layers"][1]
        pos = layer["base_offset"] + 5 * layer["record_stride"] + layer["fields"][3]["offset"] + 7
        with (self.out / "experts.bin").open("r+b") as f:
            f.seek(pos)
            byte = f.read(1)
            f.seek(pos)
            f.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaises(er.RepackError):
            er.verify(self.src, self.out, "all", log=self.quiet)

    def test_nvfp4_global_scale_field(self):
        make_model(self.src, nvfp4_layer=0)
        manifest = er.repack(self.src, self.out, "all", log=self.quiet)
        l0 = manifest["layers"][0]
        self.assertEqual(l0["projections"]["gate_up_proj"]["mode"], "nvfp4")
        self.assertIn("gate_up_proj.global_scale", [f["name"] for f in l0["fields"]])
        gs = next(f for f in l0["fields"] if f["name"] == "down_proj.global_scale")
        self.assertEqual((gs["nbytes"], gs["shape"], gs["dtype"]), (4, [], "F32"))

    def test_single_shard_and_no_index(self):
        make_model(self.src, split=False)
        (self.src / "model.safetensors.index.json").unlink()
        (self.src / "model-00002-of-00002.safetensors").rename(self.src / "model-x.safetensors")
        manifest = er.repack(self.src, self.out, "all", log=self.quiet)
        self.assertEqual(len(manifest["layers"]), 3)

    def test_rejects_inconsistent_declared_bits(self):
        make_model(self.src)
        config = json.loads((self.src / "config.json").read_text())
        config["quantization"][f"{PREFIX}.0.mlp.experts.down_proj"]["bits"] = 4
        (self.src / "config.json").write_text(json.dumps(config))
        with self.assertRaises(er.RepackError):
            er.repack(self.src, self.out, "none", log=self.quiet)

    def test_real_dwq_header_plan(self):
        """The plan for the real LocallyInferred 3.1bpw-DWQ layer 0 / 9 shapes (header only)."""
        tensors = {}
        def add(layer, proj, part, dtype, shape):
            name = f"{PREFIX}.{layer}.mlp.experts.{proj}.{part}"
            n = int(np.prod(shape)) * er.DTYPE_SIZES[dtype]
            tensors[name] = er.TensorInfo(name, dtype, shape, 0, n, Path("x"))
        for layer, down_bits in ((0, 2), (9, 3)):
            add(layer, "gate_up_proj", "weight", "U32", [128, 2304, 144])
            add(layer, "gate_up_proj", "scales", "BF16", [128, 2304, 36])
            add(layer, "gate_up_proj", "biases", "BF16", [128, 2304, 36])
            add(layer, "down_proj", "weight", "U32", [128, 2304, 1152 * down_bits // 32])
            add(layer, "down_proj", "scales", "BF16", [128, 2304, 18])
            add(layer, "down_proj", "biases", "BF16", [128, 2304, 18])
        config = {"text_config": {"hidden_size": 2304, "intermediate_size": 1152,
                                  "n_routed_experts": 128, "num_experts_per_tok": 8}}
        layers = er.plan_layers(tensors, config)
        self.assertEqual(layers[0].record_bytes, 1658880 + 829440)   # 2.49 MB, 2-bit everywhere
        self.assertEqual(layers[1].record_bytes, 1658880 + 1161216)  # 3-bit down
        self.assertEqual(layers[1].projections["down_proj"].bits, 3)
        self.assertEqual(layers[0].record_stride % er.RECORD_ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
