#!/usr/bin/env python3
"""Repack an MLX mixture-of-experts build for SSD expert streaming.

Input: an MLX model directory whose safetensors shards hold stacked routed experts, with the
expert dimension outermost (`[E, out, packed]`). Two layouts are recognised per layer:

    inkling2    <prefix>.layers.<L>.mlp.experts.{gate_up_proj,down_proj}.{weight,scales,biases[,global_scale]}
                (Inkling 2, mlx-lm #2086 layout, as mlx-swift-lm PR #821 loads it)
    switch_glu  <prefix>.layers.<L>.{feed_forward,mlp}.switch_mlp.{gate_proj,up_proj,down_proj}.{...}
                (mlx-lm SwitchGLU: LFM2 MoE, Qwen MoE, ...)

`down*` projections take the MoE intermediate size as input, the others the hidden size.

Output (a new directory):

* the "core" model: the same shard names, every tensor except the routed experts, byte-copied, plus
  a rewritten `model.safetensors.index.json` and every other file of the source directory
  (config.json, tokenizer, chat template, ...) copied unchanged;
* `experts.bin`: layer-major, then expert. Expert `e` of layer `L` is one record at
  `layers[L].base_offset + e * layers[L].record_stride`; records start on 16 KiB boundaries and each
  field inside a record starts on a 256 B boundary. Field order: gate_up_proj.{weight,scales,biases,
  global_scale}, down_proj.{...} (only the fields the source has);
* `experts.json`: the manifest (offsets, sizes, dtypes, per-expert shapes, quantization per
  projection, per-layer SHA-256 of the layer's records, the source files' SHA-256).

The repack is byte-exact: the bytes of expert `e` in a record are bytes
`[e * n / E, (e + 1) * n / E)` of the stacked source tensor of `n` bytes. Nothing is requantized.
Memory use is bounded by one expert record (a few MB); shards are never read whole.

Python 3.9+ and the standard library only (numpy is optional and unused by the format itself).

    python tools/expert_repack.py SRC_DIR OUT_DIR [--verify sample|all|none] [--no-source-hash]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple

FORMAT_NAME = "inkling2-expert-stream"
FORMAT_VERSION = 1
RECORD_ALIGNMENT = 16 * 1024
FIELD_ALIGNMENT = 256
COPY_CHUNK = 8 * 1024 * 1024

EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.(?:mlp\.experts|(?:feed_forward|mlp)\.switch_mlp))\."
    r"(?P<proj>gate_up_proj|gate_proj|up_proj|down_proj)\.(?P<part>weight|scales|biases|global_scale)$")
# Layout name -> projections in record order.
LAYOUTS = {
    "inkling2": ("gate_up_proj", "down_proj"),
    "switch_glu": ("gate_proj", "up_proj", "down_proj"),
}
PARTS = ("weight", "scales", "biases", "global_scale")

DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


class RepackError(RuntimeError):
    pass


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


# MARK: - safetensors reading


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: List[int]
    begin: int  # absolute file offset
    end: int
    file: Path

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass
class Shard:
    path: Path
    header_len: int
    header: dict
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)

    @property
    def metadata(self) -> Optional[dict]:
        return self.header.get("__metadata__")


def read_shard(path: Path) -> Shard:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RepackError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 * 1024 * 1024:
            raise RepackError(f"{path}: header of {n} bytes")
        header = json.loads(f.read(n))
    size = path.stat().st_size
    shard = Shard(path, n, header)
    base = 8 + n
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        t = TensorInfo(name, info["dtype"], list(info["shape"]), base + b, base + e, path)
        count = 1
        for d in t.shape:
            count *= d
        if t.dtype in DTYPE_SIZES and count * DTYPE_SIZES[t.dtype] != t.nbytes:
            raise RepackError(f"{path}: {name} has {t.nbytes} bytes for {t.dtype}{t.shape}")
        if t.end > size:
            raise RepackError(f"{path}: {name} runs past the end of the file")
        shard.tensors[name] = t
    return shard


def model_shards(src: Path) -> List[Path]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(weight_map.values()))
        missing = [n for n in names if not (src / n).exists()]
        if missing:
            raise RepackError(f"index names missing shards: {missing}")
        return [src / n for n in names]
    shards = sorted(src.glob("model*.safetensors"))
    if not shards:
        raise RepackError(f"{src}: no model*.safetensors")
    return shards


def copy_range(src: BinaryIO, dst: BinaryIO, begin: int, length: int, hasher=None) -> None:
    src.seek(begin)
    left = length
    while left > 0:
        chunk = src.read(min(COPY_CHUNK, left))
        if not chunk:
            raise RepackError("unexpected end of file")
        dst.write(chunk)
        if hasher is not None:
            hasher.update(chunk)
        left -= len(chunk)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(COPY_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# MARK: - expert layout


@dataclass
class Projection:
    name: str
    mode: str
    bits: int
    group_size: int
    input_dims: int
    output_dims: int


@dataclass
class FieldSpec:
    name: str  # "gate_up_proj.weight"
    tensor: TensorInfo
    expert_bytes: int
    offset: int = 0  # inside the record

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "tensor": self.tensor.name,
            "dtype": self.tensor.dtype,
            "shape": self.tensor.shape[1:],
            "offset": self.offset,
            "nbytes": self.expert_bytes,
        }


@dataclass
class LayerSpec:
    layer: int
    prefix: str
    experts: int
    fields: List[FieldSpec]
    projections: Dict[str, Projection]
    layout: str = "inkling2"
    record_bytes: int = 0
    record_stride: int = 0
    base_offset: int = 0
    sha256: str = ""


def infer_projection(name: str, fields: Dict[str, TensorInfo], input_dims: int,
                     declared: Optional[dict]) -> Projection:
    w = fields["weight"]
    if len(w.shape) != 3:
        raise RepackError(f"{w.name}: expected [experts, out, packed], got {w.shape}")
    out = w.shape[1]
    if w.dtype != "U32":
        raise RepackError(f"{w.name}: unquantized experts ({w.dtype}) are not supported")
    bits_x_in = w.shape[2] * 32
    if bits_x_in % input_dims:
        raise RepackError(f"{w.name}: packed width {w.shape[2]} does not divide input {input_dims}")
    bits = bits_x_in // input_dims
    s = fields["scales"]
    group = input_dims // s.shape[-1]
    if "biases" in fields:
        mode = "affine"
    elif (bits, group) == (4, 16):
        mode = "nvfp4"
    elif (bits, group) == (4, 32):
        mode = "mxfp4"
    elif (bits, group) == (8, 32):
        mode = "mxfp8"
    else:
        raise RepackError(f"{w.name}: unknown quantization bits={bits} group={group}")
    if declared:
        if int(declared.get("bits", bits)) != bits or int(declared.get("group_size", group)) != group:
            raise RepackError(
                f"{w.name}: config declares {declared} but the shapes say bits={bits} group={group}")
        mode = declared.get("mode", mode)
    return Projection(name, mode, bits, group, input_dims, out)


def text_dims(config: dict) -> Tuple[int, int, int, int]:
    text = config.get("text_config", config)
    hidden = int(text["hidden_size"])
    inter = int(text.get("moe_intermediate_size") or text["intermediate_size"])
    experts = int(text.get("n_routed_experts") or text.get("num_experts") or 0)
    topk = int(text.get("num_experts_per_tok") or 0)
    return hidden, inter, experts, topk


def plan_layers(tensors: Dict[str, TensorInfo], config: dict) -> List[LayerSpec]:
    hidden, inter, n_experts, _ = text_dims(config)
    quant = config.get("quantization") or config.get("quantization_config") or {}
    grouped: Dict[int, Dict[str, Dict[str, TensorInfo]]] = {}
    prefixes: Dict[int, str] = {}
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if not m:
            continue
        layer = int(m["layer"])
        if prefixes.setdefault(layer, m["prefix"]) != m["prefix"]:
            raise RepackError(f"layer {layer}: two expert prefixes")
        grouped.setdefault(layer, {}).setdefault(m["proj"], {})[m["part"]] = t
    if not grouped:
        raise RepackError("no stacked expert tensors (mlp.experts.* or switch_mlp.*) found")

    layers = []
    for layer in sorted(grouped):
        projs = grouped[layer]
        fields: List[FieldSpec] = []
        projections: Dict[str, Projection] = {}
        experts = None
        layout = next((n for n, ps in LAYOUTS.items() if set(ps) == set(projs)), None)
        if layout is None:
            raise RepackError(f"layer {layer}: projections {sorted(projs)} match no known layout")
        for proj in LAYOUTS[layout]:
            if "weight" not in projs[proj] or "scales" not in projs[proj]:
                raise RepackError(f"layer {layer}: {proj} needs weight and scales")
            input_dims = inter if proj.startswith("down") else hidden
            declared = quant.get(f"{prefixes[layer]}.{proj}")
            projections[proj] = infer_projection(proj, projs[proj], input_dims,
                                                 declared if isinstance(declared, dict) else None)
            for part in PARTS:
                t = projs[proj].get(part)
                if t is None:
                    continue
                e = t.shape[0] if t.shape else 0
                experts = e if experts is None else experts
                if e != experts or e == 0 or t.nbytes % e:
                    raise RepackError(f"{t.name}: expert dimension {t.shape} disagrees ({experts})")
                fields.append(FieldSpec(f"{proj}.{part}", t, t.nbytes // e))
        if n_experts and experts != n_experts:
            raise RepackError(f"layer {layer}: {experts} experts, config says {n_experts}")
        offset = 0
        for f in fields:
            offset = align(offset, FIELD_ALIGNMENT)
            f.offset = offset
            offset += f.expert_bytes
        spec = LayerSpec(layer, prefixes[layer], experts, fields, projections, layout)
        spec.record_bytes = offset
        spec.record_stride = align(offset, RECORD_ALIGNMENT)
        layers.append(spec)
    return layers


# MARK: - writing


def write_core_shard(shard: Shard, keep: List[str], out: Path) -> Dict[str, str]:
    """Writes the tensors `keep` of `shard` to `out`, in source order. Returns name -> shard file."""
    keep_set = set(keep)
    ordered = sorted((t for n, t in shard.tensors.items() if n in keep_set), key=lambda t: t.begin)
    header: Dict[str, object] = {}
    if shard.metadata is not None:
        header["__metadata__"] = shard.metadata
    offset = 0
    for t in ordered:
        header[t.name] = {"dtype": t.dtype, "shape": t.shape,
                          "data_offsets": [offset, offset + t.nbytes]}
        offset += t.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (align(len(raw), 8) - len(raw))  # 8-byte aligned data start, as safetensors writes
    tmp = out.with_suffix(out.suffix + ".partial")
    with shard.path.open("rb") as src, tmp.open("wb") as dst:
        dst.write(struct.pack("<Q", len(raw)))
        dst.write(raw)
        for t in ordered:
            copy_range(src, dst, t.begin, t.nbytes)
    os.replace(tmp, out)
    return {t.name: out.name for t in ordered}


def write_experts(layers: List[LayerSpec], out: Path, log=print) -> int:
    tmp = out.with_suffix(out.suffix + ".partial")
    handles: Dict[Path, BinaryIO] = {}
    try:
        with tmp.open("wb") as dst:
            position = 0
            for spec in layers:
                spec.base_offset = position
                h = hashlib.sha256()
                for e in range(spec.experts):
                    record = bytearray(spec.record_stride)
                    for f in spec.fields:
                        src = handles.get(f.tensor.file)
                        if src is None:
                            src = handles[f.tensor.file] = f.tensor.file.open("rb")
                        src.seek(f.tensor.begin + e * f.expert_bytes)
                        data = src.read(f.expert_bytes)
                        if len(data) != f.expert_bytes:
                            raise RepackError(f"{f.tensor.name}: short read")
                        record[f.offset:f.offset + f.expert_bytes] = data
                    dst.write(record)
                    h.update(record)
                position += spec.experts * spec.record_stride
                spec.sha256 = h.hexdigest()
                log(f"layer {spec.layer}: {spec.experts} x {spec.record_bytes} B records "
                    f"(stride {spec.record_stride}) at {spec.base_offset}")
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        for f in handles.values():
            f.close()
    os.replace(tmp, out)
    return position


def manifest_json(layers: List[LayerSpec], file_bytes: int, config: dict,
                  sources: List[dict]) -> dict:
    hidden, inter, experts, topk = text_dims(config)
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "file": "experts.bin",
        "file_bytes": file_bytes,
        "record_alignment": RECORD_ALIGNMENT,
        "field_alignment": FIELD_ALIGNMENT,
        "experts": experts or (layers[0].experts if layers else 0),
        "experts_per_token": topk,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "expert_bytes_total": sum(l.experts * l.record_bytes for l in layers),
        "source": {"files": sources},
        "layers": [
            {
                "layer": l.layer,
                "tensor_prefix": l.prefix,
                "layout": l.layout,
                "experts": l.experts,
                "base_offset": l.base_offset,
                "record_bytes": l.record_bytes,
                "record_stride": l.record_stride,
                "sha256": l.sha256,
                "projections": {
                    p.name: {"mode": p.mode, "bits": p.bits, "group_size": p.group_size,
                             "input_dims": p.input_dims, "output_dims": p.output_dims}
                    for p in l.projections.values()
                },
                "fields": [f.to_json() for f in l.fields],
            }
            for l in layers
        ],
    }


# MARK: - verification


def verify(src_dir: Path, out_dir: Path, mode: str, seed: int = 0, log=print) -> int:
    """Compares the repacked bytes with the source. Returns the number of experts compared."""
    if mode == "none":
        return 0
    manifest = json.loads((out_dir / "experts.json").read_text(encoding="utf-8"))
    tensors: Dict[str, TensorInfo] = {}
    for p in model_shards(src_dir):
        tensors.update(read_shard(p).tensors)
    rng = random.Random(seed)
    checked = 0
    with (out_dir / manifest["file"]).open("rb") as blob:
        if os.fstat(blob.fileno()).st_size != manifest["file_bytes"]:
            raise RepackError("experts.bin size differs from the manifest")
        for layer in manifest["layers"]:
            n = layer["experts"]
            if mode == "all":
                chosen = range(n)
            else:
                chosen = sorted({0, n - 1, *rng.sample(range(n), max(1, n // 50))})
            for e in chosen:
                blob.seek(layer["base_offset"] + e * layer["record_stride"])
                record = blob.read(layer["record_bytes"])
                for f in layer["fields"]:
                    t = tensors[f["tensor"]]
                    with t.file.open("rb") as src:
                        src.seek(t.begin + e * f["nbytes"])
                        want = src.read(f["nbytes"])
                    got = record[f["offset"]:f["offset"] + f["nbytes"]]
                    if got != want:
                        raise RepackError(f"layer {layer['layer']} expert {e} {f['name']}: bytes differ")
                checked += 1
        if mode == "all":
            for layer in manifest["layers"]:
                h = hashlib.sha256()
                blob.seek(layer["base_offset"])
                left = layer["experts"] * layer["record_stride"]
                while left:
                    chunk = blob.read(min(COPY_CHUNK, left))
                    h.update(chunk)
                    left -= len(chunk)
                if h.hexdigest() != layer["sha256"]:
                    raise RepackError(f"layer {layer['layer']}: sha256 differs")
    # Core tensors: every non-expert tensor is present and byte-identical.
    core: Dict[str, TensorInfo] = {}
    for p in model_shards(out_dir):
        core.update(read_shard(p).tensors)
    for name, t in tensors.items():
        if EXPERT_RE.match(name):
            if name in core:
                raise RepackError(f"core still holds {name}")
            continue
        c = core.get(name)
        if c is None or c.dtype != t.dtype or c.shape != t.shape:
            raise RepackError(f"core tensor {name} missing or reshaped")
        if mode == "all" or rng.random() < 0.05:
            with t.file.open("rb") as a, c.file.open("rb") as b:
                a.seek(t.begin)
                b.seek(c.begin)
                left = t.nbytes
                while left:
                    k = min(COPY_CHUNK, left)
                    if a.read(k) != b.read(k):
                        raise RepackError(f"core tensor {name}: bytes differ")
                    left -= k
    log(f"verify ({mode}): {checked} experts and {len(core)} core tensors match the source")
    return checked


# MARK: - driver


def repack(src: Path, out: Path, verify_mode: str = "sample", hash_sources: bool = True,
           log=print) -> dict:
    src, out = src.resolve(), out.resolve()
    if src == out:
        raise RepackError("output must be a different directory")
    config_path = src / "config.json"
    if not config_path.exists():
        raise RepackError(f"{src}: no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shard_paths = model_shards(src)
    shards = [read_shard(p) for p in shard_paths]
    tensors: Dict[str, TensorInfo] = {}
    for s in shards:
        for name, t in s.tensors.items():
            if name in tensors:
                raise RepackError(f"{name} appears in two shards")
            tensors[name] = t
    layers = plan_layers(tensors, config)
    out.mkdir(parents=True, exist_ok=True)

    # Every non-weight file of the source, unchanged (config, tokenizer, templates, ...).
    shard_names = {p.name for p in shard_paths}
    for item in sorted(src.iterdir()):
        if item.is_file() and item.name not in shard_names and item.name not in (
                "model.safetensors.index.json", "experts.bin", "experts.json") \
                and not item.name.endswith(".safetensors"):
            shutil.copyfile(item, out / item.name)

    weight_map: Dict[str, str] = {}
    total_core = 0
    for s in shards:
        keep = [n for n in s.tensors if not EXPERT_RE.match(n)]
        if not keep:
            log(f"{s.path.name}: experts only, no core shard")
            continue
        weight_map.update(write_core_shard(s, keep, out / s.path.name))
        total_core += sum(s.tensors[n].nbytes for n in keep)
        log(f"{s.path.name}: {len(keep)} core tensors")
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_core}, "weight_map": dict(sorted(weight_map.items()))},
        indent=2), encoding="utf-8")

    file_bytes = write_experts(layers, out / "experts.bin", log=log)
    sources = []
    for p in shard_paths + [config_path]:
        entry = {"name": p.name, "bytes": p.stat().st_size}
        if hash_sources:
            entry["sha256"] = sha256_file(p)
        sources.append(entry)
    manifest = manifest_json(layers, file_bytes, config, sources)
    tmp = out / "experts.json.partial"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, out / "experts.json")
    log(f"experts.bin: {file_bytes / 1e9:.3f} GB in {len(layers)} layers; core {total_core / 1e9:.3f} GB")
    verify(src, out, verify_mode, log=log)
    return manifest


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="source model directory (PR #821-loadable MLX build)")
    ap.add_argument("out", type=Path, help="output directory (core shards + experts.bin/json)")
    ap.add_argument("--verify", choices=["sample", "all", "none"], default="sample")
    ap.add_argument("--no-source-hash", action="store_true",
                    help="skip SHA-256 of the source shards in the manifest")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        repack(args.src, args.out, args.verify, not args.no_source_hash)
    except RepackError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
