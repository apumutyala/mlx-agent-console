#!/usr/bin/env python3
"""Offline analysis of an Inkling routing trace (design stage M1).

Input: the JSON-lines file the app writes with "Record routing trace" on
(Application Support/Diagnostics/inkling-routing.jsonl, shared from Settings). A line is either a
load header {"load": ..., "streaming": ..., "resident_fraction": ...} or one MoE call
{"f": forward, "v": visit, "l": layer, "r": rows, "ids": [rows * K expert ids]}.

For each load it reports, for slot counts S (default 32, 64, 96):
  * hit rate of the decayed-use policy the app uses (score +1 per use, x0.7 every 4 calls of the
    layer, never evict an expert of the current call), of plain LRU, and of an oracle (Belady);
  * loop-pass overlap: mean Jaccard of the expert sets of consecutive visits of the same layer in
    one forward (pass p vs p+1), decode forwards only;
  * expert-visits per decoded token and how many are repeats within the token.

    python tools/expert_trace_sim.py inkling-routing.jsonl [--slots 32 64 96] [--decode-only]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


def read_loads(lines: Iterable[str]) -> List[dict]:
    loads: List[dict] = []
    current: Optional[dict] = None
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if "load" in rec:
            current = {"header": rec, "calls": []}
            loads.append(current)
            continue
        if current is None:
            current = {"header": {}, "calls": []}
            loads.append(current)
        current["calls"].append(rec)
    return loads


def topk_of(call: dict) -> int:
    rows = max(1, call["r"])
    return len(call["ids"]) // rows


def simulate(calls: List[dict], slots: int, policy: str, experts: int = 128) -> dict:
    """Per-layer slot caches over the calls in order. A call's distinct experts that exceed the
    slots are processed in chunks of `slots` (like the app's chunked prefill)."""
    state: Dict[int, dict] = {}
    lookups = misses = 0
    # Oracle needs the future: per layer, the sequence of (call index, expert set).
    future: Dict[int, List[set]] = defaultdict(list)
    if policy == "oracle":
        for c in calls:
            future[c["l"]].append(set(c["ids"]))
    position: Dict[int, int] = defaultdict(int)
    for c in calls:
        layer = c["l"]
        st = state.setdefault(layer, {"resident": {}, "score": {}, "last": {}, "calls": 0})
        st["calls"] += 1
        needed = list(dict.fromkeys(c["ids"]))
        idx = position[layer]
        position[layer] += 1
        for start in range(0, len(needed), slots):
            chunk = needed[start:start + slots]
            protected = set(chunk)
            for e in chunk:
                lookups += 1
                if e in st["resident"]:
                    continue
                misses += 1
                if len(st["resident"]) >= slots:
                    candidates = [x for x in st["resident"] if x not in protected]
                    if policy == "lru":
                        victim = min(candidates, key=lambda x: st["last"][x])
                    elif policy == "oracle":
                        def next_use(x):
                            for j, s in enumerate(future[layer][idx + 1:]):
                                if x in s:
                                    return j
                            return 1 << 30
                        victim = max(candidates, key=next_use)
                    else:
                        victim = min(candidates, key=lambda x: (st["score"][x], st["last"][x]))
                    del st["resident"][victim]
                st["resident"][e] = True
                st["score"][e] = 0.0
            for e in chunk:
                st["score"][e] = st["score"].get(e, 0.0) + 1
                st["last"][e] = (st["calls"], start)
        if policy == "decayed" and st["calls"] % 4 == 0:
            for e in st["score"]:
                st["score"][e] *= 0.7
    return {"lookups": lookups, "misses": misses,
            "hit_rate": 1 - misses / lookups if lookups else 0.0}


def pass_overlap(calls: List[dict]) -> Optional[float]:
    by_forward: Dict[int, Dict[int, List[set]]] = defaultdict(lambda: defaultdict(list))
    for c in calls:
        if c["r"] == 1:
            by_forward[c["f"]][c["l"]].append(set(c["ids"]))
    scores = []
    for layers in by_forward.values():
        for visits in layers.values():
            for a, b in zip(visits, visits[1:]):
                scores.append(len(a & b) / len(a | b) if a | b else 1.0)
    return sum(scores) / len(scores) if scores else None


def token_reuse(calls: List[dict]) -> Optional[dict]:
    by_forward: Dict[int, List[dict]] = defaultdict(list)
    for c in calls:
        if c["r"] == 1:
            by_forward[c["f"]].append(c)
    if not by_forward:
        return None
    visits = repeats = 0
    for fc in by_forward.values():
        seen = set()
        for c in fc:
            for e in c["ids"]:
                visits += 1
                key = (c["l"], e)
                if key in seen:
                    repeats += 1
                seen.add(key)
    n = len(by_forward)
    return {"tokens": n, "expert_visits_per_token": visits / n, "repeat_fraction": repeats / visits}


def analyse(loads: List[dict], slot_counts: List[int], decode_only: bool) -> List[dict]:
    reports = []
    for load in loads:
        calls = [c for c in load["calls"] if not decode_only or c["r"] == 1]
        if not calls:
            continue
        report = {"header": load["header"], "calls": len(calls),
                  "top_k": topk_of(calls[0]), "policies": {}}
        for s in slot_counts:
            report["policies"][s] = {p: simulate(calls, s, p) for p in ("decayed", "lru", "oracle")}
        report["pass_jaccard"] = pass_overlap(calls)
        report["token_reuse"] = token_reuse(calls)
        reports.append(report)
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--decode-only", action="store_true", help="ignore prefill calls (rows > 1)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    with open(args.trace, encoding="utf-8") as f:
        reports = analyse(read_loads(f), args.slots, args.decode_only)
    if args.json:
        print(json.dumps(reports, indent=1))
        return 0
    for i, r in enumerate(reports):
        print(f"load {i}: {r['header']}  calls={r['calls']} K={r['top_k']}")
        for s, pol in r["policies"].items():
            cells = "  ".join(f"{p} {v['hit_rate'] * 100:5.1f}%" for p, v in pol.items())
            print(f"  S={s:3d}: {cells}")
        if r["pass_jaccard"] is not None:
            print(f"  loop-pass Jaccard (decode): {r['pass_jaccard']:.3f}")
        if r["token_reuse"]:
            t = r["token_reuse"]
            print(f"  decode tokens {t['tokens']}: {t['expert_visits_per_token']:.0f} expert-visits/token, "
                  f"{t['repeat_fraction'] * 100:.1f}% repeats within the token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
