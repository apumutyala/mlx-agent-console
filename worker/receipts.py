"""Benchmark receipts from devices (the iPad app's Bench screen or fleet `bench` tasks).

A receipt is one JSON document per (device, model) run: workload set, per-repetition
timings, medians, spreads, memory, thermal state and an output fingerprint
(schema v1 is defined in ios/AgentConsole/Bench/BenchReceipt.swift).

    POST /receipts            UTF-8 JSON object, schema 1, <= 256 KB
                              -> 201 {"name", "bytes"}; 200 {"name", "duplicate": true} when
                                 a receipt with the same receipt_id is already stored
    GET  /receipts?model=&limit=   -> {"receipts": [summary, ...], "unreadable": [...]} newest first
    GET  /receipts/{name}          -> the stored receipt

Stored as data/receipts/<UTC stamp>-<seq:04d>-<model id>.json, written canonically
(re-serialized, UTF-8). Names sort newest-first as plain strings. Receipts are never
rewritten or deleted by the worker. Listing uses an in-memory index built once at
startup, so it does not re-read every file per request.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

MAX_RECEIPT = 256 * 1024
MAX_WORKLOADS = 32
NAME = re.compile(r"^([0-9]{8}T[0-9]{6}Z)-([0-9]{4})-([A-Za-z0-9._-]{1,60})\.json$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ReceiptTooLarge(Exception):
    pass


def _reject_constant(name):
    raise ValueError(f"receipt contains non-standard JSON constant {name}")


def _clip(v, n=200):
    """Bounded scalar for listings: strings clipped, non-finite / odd types dropped."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        return v[:n]
    return None


def validate(receipt) -> None:
    """Shape checks so a stored receipt can always be summarized."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a JSON object")
    if receipt.get("schema") != 1 or isinstance(receipt.get("schema"), bool):
        raise ValueError("unsupported receipt schema (want schema: 1)")
    for key in ("model", "device", "load", "environment", "params"):
        if key in receipt and receipt[key] is not None and not isinstance(receipt[key], dict):
            raise ValueError(f"{key} must be an object")
    model = receipt.get("model") or {}
    if not isinstance(model.get("id"), str) or not 0 < len(model["id"]) <= 128:
        raise ValueError("model.id must be a non-empty string")
    rid = receipt.get("receipt_id")
    if rid is not None and (not isinstance(rid, str) or not 0 < len(rid) <= 64):
        raise ValueError("receipt_id must be a string (max 64 chars)")
    workloads = receipt.get("workloads", [])
    if not isinstance(workloads, list) or len(workloads) > MAX_WORKLOADS:
        raise ValueError(f"workloads must be a list of at most {MAX_WORKLOADS} objects")
    for w in workloads:
        if not isinstance(w, dict):
            raise ValueError("each workload must be an object")
        for key in ("median", "min", "max"):
            if key in w and not isinstance(w[key], dict):
                raise ValueError(f"workload {key} must be an object")
        if "reps" in w and not isinstance(w["reps"], list):
            raise ValueError("workload reps must be a list")


def summarize(name: str, r: dict) -> dict:
    """Small, bounded per-receipt view for listings (a full receipt can be ~20-60 KB)."""
    model = r.get("model") if isinstance(r.get("model"), dict) else {}
    device = r.get("device") if isinstance(r.get("device"), dict) else {}
    load = r.get("load") if isinstance(r.get("load"), dict) else {}
    workloads = []
    for w in (r.get("workloads") if isinstance(r.get("workloads"), list) else [])[:MAX_WORKLOADS]:
        if not isinstance(w, dict):
            continue
        med = w.get("median") if isinstance(w.get("median"), dict) else {}
        row = {"id": _clip(w.get("id"), 64), "prompt_tokens": _clip(w.get("prompt_tokens")),
               "ttft_s": _clip(med.get("ttft_s")), "prefill_tps": _clip(med.get("prefill_tps")),
               "decode_tps": _clip(med.get("decode_tps")),
               "peak_footprint_bytes": _clip(med.get("peak_footprint_bytes")),
               "output_hash": _clip(w.get("output_hash"), 64), "deterministic": _clip(w.get("deterministic"))}
        workloads.append({k: v for k, v in row.items() if v is not None})
    out = {
        "name": name, "receipt_id": _clip(r.get("receipt_id"), 64), "created_at": _clip(r.get("created_at"), 40),
        "status": _clip(r.get("status"), 20), "skip_reason": _clip(r.get("skip_reason")),
        "error": _clip(r.get("error")),
        "model": {k: _clip(model.get(k), 128) for k in ("id", "repo", "display_name") if k in model},
        "device": {k: _clip(device.get(k), 64) for k in ("machine", "os", "physical_memory_bytes") if k in device},
        "app_git_sha": _clip(r.get("app_git_sha"), 40), "workload_set": _clip(r.get("workload_set"), 16),
        "engine": _clip(r.get("engine"), 16), "ab_step": _clip(r.get("ab_step"), 16),
        "jetsam_suspected": _clip(r.get("jetsam_suspected")),
        "load": {k: _clip(load.get(k)) for k in ("first_load_s", "warm_load_s", "download_observed")
                 if k in load},
        "workloads": workloads,
    }
    return {k: v for k, v in out.items() if v is not None}


class ReceiptStore:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "receipts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # name -> {"model": id, "receipt_id": str|None, "summary": dict}; names sort chronologically.
        self.index: dict[str, dict] = {}
        self.unreadable: list[str] = []
        for path in self.root.glob("*.json"):
            if not NAME.match(path.name):
                continue
            try:
                r = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant)
                validate(r)
                self._index(path.name, r)
            except (OSError, ValueError, TypeError, AttributeError) as e:
                self.unreadable.append(f"{path.name}: {str(e)[:200]}")

    def _index(self, name: str, r: dict) -> None:
        self.index[name] = {"model": r["model"]["id"], "receipt_id": r.get("receipt_id"),
                            "summary": summarize(name, r)}

    @staticmethod
    def parse(raw: bytes) -> dict:
        if len(raw) > MAX_RECEIPT:
            raise ReceiptTooLarge(f"receipt too large ({len(raw)} bytes; max {MAX_RECEIPT})")
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("receipt must be UTF-8 JSON without a byte-order mark")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("receipt must be UTF-8 JSON")
        try:
            receipt = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as e:
            raise ValueError(f"receipt is not JSON: {e}")
        validate(receipt)
        return receipt

    def save(self, raw: bytes) -> tuple[int, dict]:
        receipt = self.parse(raw)
        rid = receipt.get("receipt_id")
        model = _SAFE.sub("_", receipt["model"]["id"])[:60].strip("._") or "unknown"
        data = json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with self.lock:
            if rid:  # idempotent re-upload (e.g. after a timeout where the first POST did land)
                for name, meta in self.index.items():
                    if meta["receipt_id"] == rid:
                        return 200, {"name": name, "bytes": len(data), "duplicate": True}
            seq = 1 + max((int(m.group(2)) for n in self.index if (m := NAME.match(n)) and m.group(1) == stamp),
                          default=0)
            if seq > 9999:
                raise ValueError("too many receipts this second")
            name = f"{stamp}-{seq:04d}-{model}.json"
            tmp = self.root / (name + ".part")
            tmp.write_bytes(data)
            tmp.replace(self.root / name)
            self._index(name, receipt)
        return 201, {"name": name, "bytes": len(data)}

    def get(self, name: str) -> dict:
        if not NAME.match(name or "") or name not in self.index:
            raise LookupError("no such receipt")
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def list(self, model: str | None = None, limit: int = 50) -> dict:
        with self.lock:
            names = sorted(self.index, reverse=True)
            rows = [self.index[n]["summary"] for n in names if not model or self.index[n]["model"] == model]
        out = {"receipts": rows[:limit], "more": len(rows) > limit}
        if self.unreadable:
            out["unreadable"] = self.unreadable[:20]
        return out


def handle(store: ReceiptStore, method: str, parts: list[str], query: dict, read_raw):
    """Route /receipts requests; None if the path isn't ours."""
    if parts[:1] != ["receipts"]:
        return None
    q = {k: v[0] for k, v in query.items()}
    try:
        if parts == ["receipts"]:
            if method == "POST":
                try:
                    raw = read_raw(MAX_RECEIPT)   # refuses (unread) bodies over the cap
                except ValueError:
                    raise ReceiptTooLarge(f"receipt too large (max {MAX_RECEIPT} bytes)")
                return store.save(raw)
            if method == "GET":
                limit = min(max(int(q.get("limit", "50")), 1), 200)
                return 200, store.list(q.get("model") or None, limit)
        if len(parts) == 2 and method == "GET":
            return 200, store.get(parts[1])
    except ReceiptTooLarge as e:
        return 413, {"error": str(e)}
    except LookupError as e:
        return 404, {"error": str(e).strip("'\"")}
    except ValueError as e:
        return 400, {"error": str(e)}
    return 404, {"error": "not found"}
