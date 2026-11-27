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

from fleet import MAX_PAYLOAD, TASK_TERMINAL, Fleet, handle as fleet_handle
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
            vals = array("f", memoryview(buf).cast("e"))  # no per-element tuple (≈20x less RAM)
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

    def latest(self, base_model: str | None = None) -> dict | None:
        done = [r for r in self.rounds.values() if r["status"] == "aggregated"
                and (base_model is None or r["base_model"].lower() == base_model.lower())]
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
        """Validate everything first, then save the round, then dispatch. If dispatch
        fails partway, the tasks already created are cancelled and the round is failed."""
        base_model = spec.get("base_model")
        if not isinstance(base_model, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{3,128}", base_model):
            raise ValueError("base_model must be a Hugging Face repo id")
        nodes = spec.get("nodes") or self.fleet.online_nodes("train_lora")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("no online nodes can train (a device must enable 'Accept training tasks')")
        if len(nodes) > 64 or len(set(map(str, nodes))) != len(nodes):
            raise ValueError("nodes must be at most 64 unique ids")
        for n in nodes:
            node = self.fleet.nodes.get(n) if isinstance(n, str) else None
            if node is None:
                raise LookupError(f"no such node: {n}")
            if "train_lora" not in node.get("capabilities", {}).get("kinds", []):
                raise ValueError(f"node {n} does not accept training tasks")

        parent = spec.get("parent_round")
        if parent == "latest":
            latest = self.latest(base_model)
            parent = latest["id"] if latest else None
        parent_hp = {}
        if parent:
            pr = self.get(parent)
            if pr["base_model"].lower() != base_model.lower():
                raise ValueError("parent round used a different base model")
            self.adapter_path(parent)
            parent_hp = pr["hyperparameters"]

        def num(key, default, lo, hi, cast):
            v = spec.get(key, parent_hp.get(key, default))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{key} must be in [{lo}, {hi}]")
            return cast(v)

        hp = {"iterations": num("iterations", 50, 1, 5000, int),
              "learning_rate": num("learning_rate", 1e-4, 1e-7, 1e-1, float),
              "rank": num("rank", 8, 1, 64, int),
              "layers": num("layers", 8, 1, 64, int),
              "batch_size": num("batch_size", 1, 1, 16, int)}
        if parent and (hp["rank"], hp["layers"]) != (parent_hp.get("rank"), parent_hp.get("layers")):
            raise ValueError("rank and layers must match the parent round's adapter")
        deadline_s = num("deadline_seconds", 3600, 60, 86400, int)

        examples = spec.get("examples") or []
        if not isinstance(examples, list) or len(examples) > 2000:
            raise ValueError("examples must be a list of at most 2000 objects")
        for e in examples:
            if (not isinstance(e, dict) or set(e) - {"text", "prompt", "completion"}
                    or not all(isinstance(v, str) and len(v) <= 8000 for v in e.values())):
                raise ValueError("each example is {text} or {prompt, completion} with strings up to 8000 chars")
            if not (e.get("text") or (e.get("prompt") and e.get("completion"))):
                raise ValueError("each example needs text, or both prompt and completion")
        # Training on a device's private documents is opt-in twice: here, and on the device.
        use_local = spec.get("use_local_data", False) is True
        if not use_local and len(examples) < 2 * len(nodes):
            raise ValueError(f"need at least 2 examples per node ({2 * len(nodes)}), "
                             "or use_local_data=true (each device must also allow it)")
        seed = spec.get("seed", 1234)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be a non-negative integer")

        rid = "r" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        payloads = []
        for i, node in enumerate(nodes):
            payload = {"round_id": rid, "base_model": base_model, **hp, "examples": examples[i::len(nodes)],
                       "seed": seed, "use_local_data": use_local,
                       "init_adapter": f"fed/rounds/{parent}/adapter" if parent else None}
            if len(json.dumps(payload)) > MAX_PAYLOAD:
                raise ValueError("examples too large for one task; send fewer or shorter examples")
            payloads.append((node, payload))

        r = {"id": rid, "status": "running", "base_model": base_model, "hyperparameters": hp,
             "use_local_data": use_local, "parent_round": parent, "created_at": now(),
             "deadline_at": now() + deadline_s, "finished_at": None, "tasks": {}, "result": None,
             "seed": seed}
        with self.lock:
            self.rounds[rid] = r
            self._save(r)
            try:
                for node, payload in payloads:
                    task = self.fleet.create_task("train_lora", payload, node=node, round_id=rid)
                    r["tasks"][task.id] = node
            except Exception as e:
                for tid in r["tasks"]:
                    t = self.fleet.tasks.get(tid)
                    if t is not None:
                        self.fleet._finish(t, "cancelled", reason="round dispatch failed")
                r.update(status="failed", result={"error": f"dispatch failed: {e}"}, finished_at=now())
                self._save(r)
                raise
            self._save(r)
        return r

    def expire(self) -> None:
        """Past a round's deadline, stop waiting: cancel unfinished tasks and
        average whatever succeeded (a node that vanished can't stall a round)."""
        for rid, r in list(self.rounds.items()):
            if r["status"] != "running" or now() < r.get("deadline_at", float("inf")):
                continue
            for tid in r["tasks"]:
                t = self.fleet.tasks.get(tid)
                if t is not None:
                    self.fleet._finish(t, "cancelled", reason="round deadline passed")
            self._maybe_aggregate(rid)

    def _on_task_done(self, meta: dict) -> None:
        rid = meta.get("round_id")
        if rid and rid in self.rounds:
            # Averaging can take seconds; don't hold the node's /complete request.
            threading.Thread(target=self._maybe_aggregate, args=(rid,), daemon=True,
                             name=f"fedavg-{rid}").start()

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
                    bm = (query.get("base_model") or [None])[0]
                    return 200, {"round": fed.latest(bm)}
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
                fed.expire()
            except Exception:
                pass
    threading.Thread(target=sweeper, daemon=True, name="fleet-sweeper").start()

    router.fleet = fleet
    router.federation = fed
    return router, fleet
