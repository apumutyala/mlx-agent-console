"""Benchmark receipts from devices (the iPad app's Bench screen or fleet `bench` tasks).

A receipt is one JSON document per (device, model) run: workload set, per-repetition
timings, medians, spreads, memory, thermal state and an output fingerprint
(schema v1 is defined in ios/AgentConsole/Bench/BenchReceipt.swift).

    POST /receipts            receipt JSON (<= 256 KB)   -> 201 {"name", "bytes"}
    GET  /receipts?model=&limit=                         -> {"receipts": [summary, ...]} newest first
    GET  /receipts/{name}                                -> the stored receipt

Stored as data/receipts/<UTC timestamp>-<model id>[-n].json. Receipts are never
rewritten or deleted by the worker.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

MAX_RECEIPT = 256 * 1024
NAME = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[A-Za-z0-9._-]{1,80}\.json$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _model_id(receipt: dict) -> str:
    model = receipt.get("model")
    mid = model.get("id") if isinstance(model, dict) else None
    return str(mid) if isinstance(mid, str) and mid else "unknown"


def summarize(name: str, r: dict) -> dict:
    """Small per-receipt view for listings (the full receipt can be ~20-60 KB)."""
    model = r.get("model") if isinstance(r.get("model"), dict) else {}
    device = r.get("device") if isinstance(r.get("device"), dict) else {}
    workloads = []
    for w in r.get("workloads") or []:
        if not isinstance(w, dict):
            continue
        med = w.get("median") if isinstance(w.get("median"), dict) else {}
        workloads.append({k: v for k, v in {
            "id": w.get("id"), "prompt_tokens": w.get("prompt_tokens"),
            "ttft_s": med.get("ttft_s"), "prefill_tps": med.get("prefill_tps"),
            "decode_tps": med.get("decode_tps"), "peak_footprint_bytes": med.get("peak_footprint_bytes"),
            "output_hash": w.get("output_hash"), "deterministic": w.get("deterministic"),
        }.items() if v is not None})
    return {
        "name": name, "schema": r.get("schema"), "created_at": r.get("created_at"),
        "status": r.get("status"), "skip_reason": r.get("skip_reason"),
        "model": {k: model.get(k) for k in ("id", "repo", "display_name") if k in model},
        "device": {k: device.get(k) for k in ("machine", "os", "physical_memory_bytes") if k in device},
        "app_git_sha": r.get("app_git_sha"), "workload_set": r.get("workload_set"),
        "load": r.get("load"), "workloads": workloads,
    }


class ReceiptStore:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "receipts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def save(self, raw: bytes) -> dict:
        if len(raw) > MAX_RECEIPT:
            raise OverflowError(f"receipt too large ({len(raw)} bytes; max {MAX_RECEIPT})")
        try:
            receipt = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"receipt is not JSON: {e}")
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be a JSON object")
        if receipt.get("schema") != 1:
            raise ValueError("unsupported receipt schema (want schema: 1)")
        model = _SAFE.sub("_", _model_id(receipt))[:60].strip("._") or "unknown"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with self.lock:
            name, n = f"{stamp}-{model}.json", 1
            while (self.root / name).exists():
                n += 1
                name = f"{stamp}-{model}-{n}.json"
            tmp = self.root / (name + ".part")
            tmp.write_bytes(raw)
            tmp.replace(self.root / name)
        return {"name": name, "bytes": len(raw)}

    def get(self, name: str) -> dict:
        if not NAME.match(name or ""):
            raise LookupError("no such receipt")
        path = self.root / name
        if not path.is_file():
            raise LookupError("no such receipt")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, model: str | None = None, limit: int = 50) -> list[dict]:
        out = []
        for path in sorted(self.root.glob("*.json"), reverse=True):  # names start with a UTC stamp
            if not NAME.match(path.name):
                continue
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if model and _model_id(r) != model:
                continue
            out.append(summarize(path.name, r))
            if len(out) >= limit:
                break
        return out


def handle(store: ReceiptStore, method: str, parts: list[str], query: dict, read_raw):
    """Route /receipts requests; None if the path isn't ours."""
    if parts[:1] != ["receipts"]:
        return None
    q = {k: v[0] for k, v in query.items()}
    try:
        if parts == ["receipts"]:
            if method == "POST":
                return 201, store.save(read_raw(1_000_000))
            if method == "GET":
                limit = min(max(int(q.get("limit", "50")), 1), 200)
                return 200, {"receipts": store.list(q.get("model") or None, limit)}
        if len(parts) == 2 and method == "GET":
            return 200, store.get(parts[1])
    except OverflowError as e:
        return 413, {"error": str(e)}
    except LookupError as e:
        return 404, {"error": str(e).strip("'\"")}
    except ValueError as e:
        return 400, {"error": str(e)}
    return 404, {"error": "not found"}
