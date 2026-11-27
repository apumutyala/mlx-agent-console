"""Federated LoRA rounds over the compute fleet (FedAvg), stdlib only.

A round fans one `train_lora` task out to each participating node. Every node
fine-tunes LoRA adapters on its own data (local documents, or a shard of the
examples sent with the round), starting from the previous round's global
adapter, and uploads `adapter.safetensors`. When every task is finished the
coordinator averages the adapters, weighted by each node's example count, and
publishes the result as the next global adapter.

    POST /fed/rounds          {"base_model", "nodes"?, "iterations"?, "learning_rate"?, "rank"?,
                               "layers"?, "examples"?: [{"text"} | {"prompt","completion"}],
                               "use_local_data"?: bool, "parent_round"?: "latest" | id}
    GET  /fed/rounds, /fed/rounds/{id}, /fed/rounds/{id}/adapter, /fed/latest
"""
from __future__ import annotations

import json
import re
import struct
import threading
import time
import uuid
from array import array
from pathlib import Path

from fleet import TASK_TERMINAL, Fleet, handle as fleet_handle
from jobworker import now

ROUND_ID = re.compile(r"^r[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
ADAPTER = "adapter.safetensors"


# ------------------------------------------------------------ safetensors --

def read_safetensors(path: Path) -> tuple[dict[str, tuple[list[int], array]], dict]:
    """Return ({name: (shape, float32 array)}, metadata). F32/F16/BF16 only."""
    raw = Path(path).read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen])
    meta = header.pop("__metadata__", {}) or {}
    base = 8 + hlen
    out = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        buf = raw[base + b: base + e]
        dt = info["dtype"]
        if dt == "F32":
            vals = array("f", buf)
        elif dt == "F16":
            vals = array("f", struct.unpack(f"<{len(buf) // 2}e", buf))
        elif dt == "BF16":
            halves = array("H", buf)
            vals = array("f", array("I", (h << 16 for h in halves)).tobytes())
        else:
            raise ValueError(f"{name}: unsupported dtype {dt}")
        out[name] = (list(info["shape"]), vals)
    return out, meta


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], array]], metadata: dict | None = None):
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        shape, vals = tensors[name]
        data = array("f", vals).tobytes()
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    tmp = Path(str(path) + ".part")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for blob in blobs:
            f.write(blob)
    tmp.replace(path)


def fedavg(adapters: list[tuple[Path, float]]) -> dict[str, tuple[list[int], array]]:
    """Weighted average of adapter files. All must have identical tensor names/shapes."""
    if not adapters:
        raise ValueError("no adapters to average")
    total = sum(w for _, w in adapters)
    if total <= 0:
        raise ValueError("weights must be positive")
    acc: dict[str, tuple[list[int], array]] | None = None
    for path, w in adapters:
        tensors, _ = read_safetensors(path)
        scale = w / total
        if acc is None:
            acc = {k: (shape, array("f", (v * scale for v in vals))) for k, (shape, vals) in tensors.items()}
            continue
        if set(tensors) != set(acc):
            raise ValueError(f"{path.parent.parent.name}: adapter tensor names differ from the others")
        for k, (shape, vals) in tensors.items():
            ashape, avals = acc[k]
            if shape != ashape:
                raise ValueError(f"{k}: shape {shape} != {ashape}")
            for i, v in enumerate(vals):
                avals[i] += v * scale
    return acc


# ----------------------------------------------------------------- rounds --

class Federation:
    def __init__(self, fleet: Fleet, data_dir: Path):
        self.fleet = fleet
        self.root = Path(data_dir) / "fed" / "rounds"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rounds: dict[str, dict] = {}
        for d in sorted(self.root.iterdir()):
            p = d / "round.json"
            if p.exists():
                self.rounds[d.name] = json.loads(p.read_text(encoding="utf-8"))
        fleet.on_terminal(self._on_task_done)
        for rid, r in list(self.rounds.items()):  # finish rounds whose tasks ended while we were down
            if r["status"] == "running":
                self._maybe_aggregate(rid)

    def _save(self, r: dict) -> None:
        d = self.root / r["id"]
        d.mkdir(exist_ok=True)
        tmp = d / "round.json.tmp"
        tmp.write_text(json.dumps(r, indent=2), encoding="utf-8")
        tmp.replace(d / "round.json")

    def latest(self) -> dict | None:
        done = [r for r in self.rounds.values() if r["status"] == "aggregated"]
        return max(done, key=lambda r: r["finished_at"]) if done else None

    def get(self, rid: str) -> dict:
        if not ROUND_ID.match(rid or "") or rid not in self.rounds:
            raise LookupError("no such round")
        return self.rounds[rid]

    def adapter_path(self, rid: str) -> Path:
        r = self.get(rid)
        if r["status"] != "aggregated":
            raise LookupError("round has no aggregated adapter yet")
        return self.root / rid / ADAPTER

    def start(self, spec: dict) -> dict:
        base_model = spec.get("base_model")
        if not isinstance(base_model, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{3,128}", base_model):
            raise ValueError("base_model must be a Hugging Face repo id")
        nodes = spec.get("nodes") or self.fleet.online_nodes("train_lora")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("no online nodes can train (capability train_lora)")
        for n in nodes:
            if n not in self.fleet.nodes:
                raise LookupError(f"no such node: {n}")

        def num(key, default, lo, hi, cast):
            v = spec.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{key} must be in [{lo}, {hi}]")
            return cast(v)

        hp = {"iterations": num("iterations", 50, 1, 5000, int),
              "learning_rate": num("learning_rate", 1e-4, 1e-7, 1e-1, float),
              "rank": num("rank", 8, 1, 64, int),
              "layers": num("layers", 8, 1, 64, int),
              "batch_size": num("batch_size", 1, 1, 16, int)}
        examples = spec.get("examples") or []
        if not isinstance(examples, list) or not all(isinstance(e, dict) for e in examples):
            raise ValueError("examples must be a list of objects")
        use_local = bool(spec.get("use_local_data", not examples))
        parent = spec.get("parent_round")
        if parent == "latest":
            latest = self.latest()
            parent = latest["id"] if latest and latest["base_model"] == base_model else None
        if parent:
            if self.get(parent)["base_model"] != base_model:
                raise ValueError("parent round used a different base model")
            self.adapter_path(parent)

        rid = "r" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        r = {"id": rid, "status": "running", "base_model": base_model, "hyperparameters": hp,
             "use_local_data": use_local, "parent_round": parent, "created_at": now(),
             "finished_at": None, "tasks": {}, "result": None,
             "seed": int(spec.get("seed", 1234)) if not isinstance(spec.get("seed"), bool) else 1234}
        with self.lock:
            self.rounds[rid] = r
            for i, node in enumerate(nodes):
                shard = examples[i::len(nodes)]
                payload = {"round_id": rid, "base_model": base_model, **hp, "examples": shard,
                           "seed": r["seed"],
                           "use_local_data": use_local,
                           "init_adapter": f"fed/rounds/{parent}/adapter" if parent else None}
                task = self.fleet.create_task("train_lora", payload, node=node, round_id=rid)
                r["tasks"][task.id] = node
            self._save(r)
        return r

    def _on_task_done(self, meta: dict) -> None:
        rid = meta.get("round_id")
        if rid and rid in self.rounds:
            self._maybe_aggregate(rid)

    def _maybe_aggregate(self, rid: str) -> None:
        with self.lock:
            r = self.rounds[rid]
            if r["status"] != "running":
                return
            tasks = [self.fleet.tasks.get(t) for t in r["tasks"]]
            if any(t is None or t.meta["status"] not in TASK_TERMINAL for t in tasks):
                return
            inputs, contributors = [], []
            for t in tasks:
                if t.meta["status"] == "succeeded" and ADAPTER in t.meta.get("artifacts", []):
                    w = t.meta.get("metrics", {}).get("num_examples", 1)
                    w = float(w) if isinstance(w, (int, float)) and w > 0 else 1.0
                    inputs.append((t.root / "artifacts" / ADAPTER, w))
                    contributors.append({"task": t.id, "node": t.meta["assigned_to"], "weight": w,
                                         "final_loss": t.meta.get("metrics", {}).get("final_loss")})
            try:
                if not inputs:
                    raise ValueError("no node produced an adapter")
                avg = fedavg(inputs)
                write_safetensors(self.root / rid / ADAPTER, avg,
                                  {"round_id": rid, "base_model": r["base_model"],
                                   "contributors": len(inputs)})
                r.update(status="aggregated", result={"contributors": contributors,
                                                       "tensors": len(avg)})
            except Exception as e:
                r.update(status="failed", result={"error": str(e), "contributors": contributors})
            r["finished_at"] = now()
            self._save(r)


# ------------------------------------------------------------------ routes --

def make_router(data_dir: Path, offline_after: float = 75.0):
    fleet = Fleet(data_dir, offline_after=offline_after)
    fed = Federation(fleet, data_dir)

    def router(method, parts, query, read_json, read_raw):
        if parts[:1] == ["fed"]:
            try:
                if parts == ["fed", "rounds"]:
                    if method == "GET":
                        rs = sorted(fed.rounds.values(), key=lambda r: r["created_at"], reverse=True)
                        return 200, {"rounds": rs[:50]}
                    if method == "POST":
                        return 201, fed.start(read_json())
                if parts == ["fed", "latest"] and method == "GET":
                    return 200, {"round": fed.latest()}
                if len(parts) == 3 and parts[1] == "rounds" and method == "GET":
                    return 200, fed.get(parts[2])
                if len(parts) == 4 and parts[1] == "rounds" and parts[3] == "adapter" and method == "GET":
                    return 200, fed.adapter_path(parts[2])
            except LookupError as e:
                return 404, {"error": str(e).strip("'\"")}
            except ValueError as e:
                return 400, {"error": str(e)}
            return 404, {"error": "not found"}
        return fleet_handle(fleet, method, parts, query, read_json, read_raw)

    def sweeper():  # notice nodes that vanished even when nobody is polling
        while True:
            time.sleep(max(offline_after / 5, 0.2))
            try:
                fleet.sweep()
            except Exception:
                pass
    threading.Thread(target=sweeper, daemon=True, name="fleet-sweeper").start()

    router.fleet = fleet
    router.federation = fed
    return router, fleet
