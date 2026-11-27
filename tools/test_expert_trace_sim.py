"""Tests for expert_trace_sim.py.

    cd tools && python -m unittest -v test_expert_trace_sim
"""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expert_trace_sim as sim  # noqa: E402


def trace(tokens=50, layers=(0, 1, 1, 2), k=4, experts=16, seed=0, prefill_rows=0):
    rng = random.Random(seed)
    lines = [json.dumps({"load": "t", "streaming": True, "resident_fraction": 0.5})]
    f = 0
    if prefill_rows:
        f += 1
        for v, l in enumerate(layers):
            ids = [e for _ in range(prefill_rows) for e in rng.sample(range(experts), k)]
            lines.append(json.dumps({"f": f, "v": v, "l": l, "r": prefill_rows, "ids": ids}))
    for _ in range(tokens):
        f += 1
        chosen = {}
        for v, l in enumerate(layers):
            # Layer 1 is visited twice per token with the same experts (perfect pass overlap).
            ids = chosen.get(l) or rng.sample(range(experts), k)
            chosen[l] = ids
            lines.append(json.dumps({"f": f, "v": v, "l": l, "r": 1, "ids": ids}))
    return lines


class TraceSimTests(unittest.TestCase):
    def test_full_capacity_only_cold_misses(self):
        loads = sim.read_loads(trace())
        calls = loads[0]["calls"]
        r = sim.simulate(calls, 16, "decayed")
        distinct = len({(c["l"], e) for c in calls for e in c["ids"]})
        self.assertEqual(r["misses"], distinct)

    def test_oracle_bounds_policies(self):
        calls = sim.read_loads(trace(tokens=200, experts=32, seed=3))[0]["calls"]
        for s in (8, 16):
            oracle = sim.simulate(calls, s, "oracle")["hit_rate"]
            self.assertGreaterEqual(oracle + 1e-9, sim.simulate(calls, s, "lru")["hit_rate"])
            self.assertGreaterEqual(oracle + 1e-9, sim.simulate(calls, s, "decayed")["hit_rate"])

    def test_chunked_prefill_counts_every_expert(self):
        calls = sim.read_loads(trace(tokens=0, prefill_rows=20, experts=16))[0]["calls"]
        r = sim.simulate(calls, 4, "decayed")
        self.assertEqual(r["lookups"], sum(len(set(c["ids"])) for c in calls))

    def test_pass_overlap_and_reuse(self):
        loads = sim.read_loads(trace(tokens=10))
        self.assertAlmostEqual(sim.pass_overlap(loads[0]["calls"]), 1.0)
        reuse = sim.token_reuse(loads[0]["calls"])
        self.assertEqual(reuse["expert_visits_per_token"], 16)
        self.assertAlmostEqual(reuse["repeat_fraction"], 0.25)

    def test_report(self):
        reports = sim.analyse(sim.read_loads(trace()), [4, 8], decode_only=True)
        self.assertEqual(len(reports), 1)
        self.assertEqual(set(reports[0]["policies"][4]), {"decayed", "lru", "oracle"})
        self.assertEqual(reports[0]["top_k"], 4)


if __name__ == "__main__":
    unittest.main()
