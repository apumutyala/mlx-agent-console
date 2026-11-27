"""Tests for lamq_ref.py and for the index math of the Metal kernels in InklingLamqKernels.swift.

    cd ios/tools/lamq && python -m unittest -v test_lamq_ref

Needs only NumPy. With LAMQ_PACKED_DIR pointing at a downloaded packed build (the 5.4 GB repo at
the revision in lamq_digests.json), it also re-derives the recorded digests from the real files.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lamq_ref as ref  # noqa: E402

RNG = np.random.default_rng(20261127)


# ---------------------------------------------------------------------------------- oracles

def bigint_states(row_words: np.ndarray, n: int, k: int, g: int) -> list[int]:
    """Literal spec: the row is one integer B = sum(word[i] << 32 i); the group whose last column
    is `last` has state (B >> ((n - 1 - last) * k)) & 0xFFF."""
    b = 0
    for i, w in enumerate(row_words.tolist()):
        b |= int(w) << (32 * i)
    return [(b >> ((n - 1 - last) * k)) & 0xFFF for last in range(g - 1, n, g)]


def metal_state(rc: np.ndarray, wpr: int, n: int, k: int, g: int, gi: int) -> int:
    """Port, line for line, of the state extraction in `lamqTrellisStateSource` (uint32 math)."""
    M32 = 0xFFFFFFFF
    last = gi * g + g - 1
    bp = (n - 1 - last) * k
    wi, off = bp >> 5, bp & 31
    lo = int(rc[wi])
    hi = int(rc[wi + 1]) if wi + 1 < wpr else 0
    s = lo if off == 0 else ((lo >> off) | ((hi << (32 - off)) & M32))
    return s & 4095


def metal_index(s: int, log2k: int, hashed: bool) -> int:
    """Port of `lamq_index`: (s * 0x9E3779B1u) wraps mod 2^32, then >> (32 - LOGK)."""
    return ((s * ref.HASH_MULT_U32) & 0xFFFFFFFF) >> (32 - log2k) if hashed else s


def metal_dequant(codes, scales, lut, locs, rg_rows, n, fmt: ref.TrellisFormat) -> np.ndarray:
    """Emulates `lamq_trellis_dequant` thread by thread: grid (n/g, n_rg*RG); one group per thread."""
    wpr = n * fmt.k // 32
    flat_codes = codes.reshape(-1)
    flat_scales = scales.reshape(-1)
    lut = lut.reshape(-1)
    rows = len(locs) * rg_rows
    out = np.zeros((rows, n), np.float32)
    for row in range(rows):
        rg, rr = divmod(row, rg_rows)
        loc = int(locs[rg])
        base = (loc * rg_rows + rr) * wpr
        rc = flat_codes[base: base + wpr]
        sc = np.float32(flat_scales[loc * rg_rows + rr])
        for gi in range(n // fmt.g):
            idx = metal_index(metal_state(rc, wpr, n, fmt.k, fmt.g, gi), fmt.log2K, bool(fmt.hash_mult))
            for t in range(fmt.g):
                out[row, gi * fmt.g + t] = np.float32(lut[idx * fmt.g + t]) * sc
    return out


def metal_matvec(x, codes, scales, lut, sel, rg_rows, x_div, n, fmt: ref.TrellisFormat, lanes=32) -> np.ndarray:
    """Emulates `lamq_trellis_matvec`: one simdgroup per output row, lane-strided groups, simd_sum,
    sentinel rows write 0. Accumulation in float64 here (the kernel uses float32)."""
    slots, n_rg = sel.shape
    m = n_rg * rg_rows
    wpr = n * fmt.k // 32
    flat_codes, flat_scales, lut = codes.reshape(-1), scales.reshape(-1), lut.reshape(-1)
    y = np.zeros((slots, m), np.float64)
    for slot in range(slots):
        xs = x[slot // x_div]
        for row in range(m):
            rg, rr = divmod(row, rg_rows)
            loc = int(sel[slot, rg])
            if loc == 0xFFFFFFFF:
                continue
            base = (loc * rg_rows + rr) * wpr
            rc = flat_codes[base: base + wpr]
            lane_acc = [0.0] * lanes
            for lane in range(lanes):
                for gi in range(lane, n // fmt.g, lanes):
                    idx = metal_index(metal_state(rc, wpr, n, fmt.k, fmt.g, gi), fmt.log2K, bool(fmt.hash_mult))
                    for t in range(fmt.g):
                        lane_acc[lane] += float(lut[idx * fmt.g + t]) * float(xs[gi * fmt.g + t])
            y[slot, row] = sum(lane_acc) * float(flat_scales[loc * rg_rows + rr])
    return y


def metal_lm_dequant(codes_u8: np.ndarray, scales, levels, m: int, n: int, k: int) -> np.ndarray:
    """Emulates `lamq_lm_dequant`: one thread per 4 columns."""
    c = codes_u8.reshape(-1)
    bpr = n * k // 8
    out = np.zeros((m, n), np.float32)
    for row in range(m):
        sc = np.float32(scales[row])
        for q in range(n // 4):
            if k == 8:
                idx = [int(c[row * bpr + 4 * q + t]) for t in range(4)]
            else:
                p = row * bpr + 3 * q
                v = int(c[p]) | (int(c[p + 1]) << 8) | (int(c[p + 2]) << 16)
                idx = [(v >> (6 * t)) & 63 for t in range(4)]
            for t in range(4):
                out[row, 4 * q + t] = np.float32(levels[idx[t]]) * sc
    return out


def random_lut(fmt: ref.TrellisFormat) -> np.ndarray:
    return RNG.standard_normal(fmt.K * fmt.g).astype(np.float16)


# ------------------------------------------------------------------------------------- tests

class TrellisDecode(unittest.TestCase):
    def test_states_match_bigint_oracle(self):
        for name, f in ref.TRELLIS.items():
            for n in (64, 128, 1152, 2304):
                if (n * f.k) % 32:
                    continue
                codes = RNG.integers(0, 2**32, size=(3, n * f.k // 32), dtype=np.uint64).astype(np.uint32)
                got = ref.trellis_states(codes, n, f.k, f.g)
                for r in range(3):
                    self.assertEqual(got[r].tolist(), bigint_states(codes[r], n, f.k, f.g), f"{name} n={n} row {r}")

    def test_metal_state_math_matches_reference(self):
        for name, f in ref.TRELLIS.items():
            n = 2304
            codes = RNG.integers(0, 2**32, size=(2, n * f.k // 32), dtype=np.uint64).astype(np.uint32)
            want = ref.trellis_states(codes, n, f.k, f.g)
            for r in range(2):
                got = [metal_state(codes[r], n * f.k // 32, n, f.k, f.g, gi) for gi in range(n // f.g)]
                self.assertEqual(got, want[r].tolist(), name)

    def test_hash_index(self):
        f = ref.TRELLIS["ht2"]
        states = np.arange(4096, dtype=np.uint32)
        got = ref.trellis_indices(states, f)
        want = [metal_index(int(s), f.log2K, True) for s in states]
        self.assertEqual(got.tolist(), want)
        self.assertTrue(got.max() < f.K)
        self.assertEqual(ref.trellis_indices(states, ref.TRELLIS["rtr3"]).tolist(), list(range(4096)))

    def test_first_column_reads_highest_bits_last_column_lowest(self):
        # rtr4: column n-1 is the window at bit 0; column 0 the window at bit (n-1)*4.
        n, f = 64, ref.TRELLIS["rtr4"]
        codes = np.zeros((1, n * f.k // 32), np.uint32)
        codes[0, 0] = 0xABC            # bits 0..11
        st = ref.trellis_states(codes, n, f.k, f.g)[0]
        self.assertEqual(int(st[n - 1]), 0xABC)
        self.assertEqual(int(st[n - 2]), 0xAB)   # window at bit 4
        self.assertEqual(int(st[n - 3]), 0xA)
        self.assertEqual(int(st[0]), 0)

    def test_bits_past_end_read_zero(self):
        n, f = 64, ref.TRELLIS["rtr1"]
        codes = np.full((1, 2), 0xFFFFFFFF, np.uint32)
        st = ref.trellis_states(codes, n, f.k, f.g)[0]
        self.assertEqual(int(st[0]), 1)          # window at bit 63: one real bit, 11 zero bits
        self.assertEqual(int(st[n - 12]), 4095)

    def test_decode_is_exact_product(self):
        for name, f in ref.TRELLIS.items():
            n = 128
            codes = RNG.integers(0, 2**32, size=(4, n * f.k // 32), dtype=np.uint64).astype(np.uint32)
            scales = RNG.standard_normal(4).astype(np.float16)
            lut = random_lut(f)
            q = ref.decode_trellis(codes, scales, lut, n, name)
            idx = ref.trellis_indices(ref.trellis_states(codes, n, f.k, f.g), f)
            want = (lut.reshape(f.K, f.g).astype(np.float64)[idx].reshape(4, n) * scales.astype(np.float64)[:, None])
            self.assertTrue(np.array_equal(q.astype(np.float64), want), name)  # float32 product is exact

    def test_metal_dequant_emulation_matches(self):
        for name in ("ht2", "ht3", "rtr1", "rtr4"):
            f = ref.TRELLIS[name]
            n, rg_rows, slab = 128, 4, 5
            codes = RNG.integers(0, 2**32, size=(slab, rg_rows, n * f.k // 32), dtype=np.uint64).astype(np.uint32)
            scales = RNG.standard_normal((slab, rg_rows)).astype(np.float16)
            lut = random_lut(f)
            locs = np.array([3, 0, 4], np.uint32)
            got = metal_dequant(codes, scales, lut, locs, rg_rows, n, f)
            want = np.concatenate([ref.decode_trellis(codes[l], scales[l], lut, n, name) for l in locs])
            self.assertEqual(got.tobytes(), want.tobytes(), name)

    def test_metal_matvec_emulation_matches(self):
        for name in ("ht2", "ht3", "rtr1"):
            f = ref.TRELLIS[name]
            n, rg_rows, slab = 128, 4, 6
            codes = RNG.integers(0, 2**32, size=(slab, rg_rows, n * f.k // 32), dtype=np.uint64).astype(np.uint32)
            scales = RNG.standard_normal((slab, rg_rows)).astype(np.float16)
            lut = random_lut(f)
            x = RNG.standard_normal((2, n)).astype(np.float32)
            S = 0xFFFFFFFF
            sel = np.array([[5, S, 1], [0, 2, S], [S, S, S], [4, 3, 2]], dtype=np.uint64)  # 4 slots, x_div 2
            y = metal_matvec(x, codes, scales, lut, sel, rg_rows, 2, n, f)
            for slot in range(4):
                for rg in range(3):
                    loc = sel[slot, rg]
                    blk = y[slot, rg * rg_rows:(rg + 1) * rg_rows]
                    if loc == S:
                        self.assertTrue(np.all(blk == 0))
                        continue
                    w = ref.decode_trellis(codes[loc], scales[loc], lut, n, name).astype(np.float64)
                    np.testing.assert_allclose(blk, w @ x[slot // 2].astype(np.float64), rtol=1e-9, atol=1e-9)


class LloydMax(unittest.TestCase):
    def test_lm6_packing(self):
        m, n = 3, 64
        idx = RNG.integers(0, 64, size=(m, n))
        q = idx.reshape(-1, 4)
        v = q[:, 0] | (q[:, 1] << 6) | (q[:, 2] << 12) | (q[:, 3] << 18)
        packed = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], -1).astype(np.uint8).reshape(-1)
        self.assertEqual(ref.lm_indices(packed, m, n, 6).tolist(), idx.tolist())
        levels = RNG.standard_normal(64).astype(np.float32)
        scales = RNG.standard_normal(m).astype(np.float16)
        got = metal_lm_dequant(packed, scales, levels, m, n, 6)
        self.assertEqual(got.tobytes(), ref.decode_lm(packed, scales, levels, m, n, 6).tobytes())

    def test_lm8(self):
        m, n = 2, 32
        codes = RNG.integers(0, 256, size=m * n).astype(np.uint8)
        levels = RNG.standard_normal(256).astype(np.float32)
        scales = RNG.standard_normal(m).astype(np.float16)
        got = metal_lm_dequant(codes, scales, levels, m, n, 8)
        self.assertEqual(got.tobytes(), ref.decode_lm(codes, scales, levels, m, n, 8).tobytes())


class NVFP4(unittest.TestCase):
    def test_e4m3_table(self):
        t = ref.e4m3_table()
        self.assertEqual(t[0x38], 1.0)          # exp 7, mantissa 0
        self.assertEqual(t[0x7E], 448.0)        # max normal
        self.assertEqual(t[0x01], 2.0 ** -9)    # smallest subnormal
        self.assertEqual(t[0xB8], -1.0)
        self.assertEqual(t[0x7F], 0.0)          # NaN code reads as 0

    def test_decode(self):
        m, n = 2, 32
        codes = RNG.integers(0, 256, size=(m, n // 2)).astype(np.uint8)
        scales = np.full((m, n // 16), 0x38, np.uint8)   # 1.0
        w = ref.decode_nvfp4(codes, scales, np.float32(2.0))
        self.assertEqual(w[0, 0], ref.E2M1[codes[0, 0] & 15] * 2)
        self.assertEqual(w[0, 1], ref.E2M1[codes[0, 0] >> 4] * 2)


class Hadamard(unittest.TestCase):
    SIZES = (2304, 1152, 3072, 1024, 384, 128)

    def test_factor_shapes(self):
        # Shapes the Swift port (InklingLamqFormat.swift) must reproduce.
        want = {2304: (32, 72), 1152: (16, 72), 3072: (256, 12), 1024: (32, 32), 384: (32, 12), 128: (8, 16)}
        for n, (a, b) in want.items():
            A, B = ref.hadamard_factors(n)
            self.assertEqual((A.shape[0], B.shape[0]), (a, b), n)

    def test_is_hadamard(self):
        for n in self.SIZES:
            A, B = ref.hadamard_factors(n)
            H = np.kron(A, B)
            self.assertTrue(np.array_equal(H @ H.T, n * np.eye(n)), n)
            self.assertTrue(set(np.unique(H)) <= {-1.0, 1.0})

    def test_rht_definitions(self):
        for n in (2304, 384):
            A, B = ref.hadamard_factors(n)
            H = np.kron(A, B)
            s = RNG.standard_normal(n)          # f16 "signs" carry folded column scales
            R = H * s[None, :] / math.sqrt(n)
            x = RNG.standard_normal((2, n))
            np.testing.assert_allclose(ref.rht_in(x, s), x @ R.T, atol=1e-9)
            np.testing.assert_allclose(ref.rht_out(x, s), x @ R, atol=1e-9)

    def test_effective_weight(self):
        n, m = 384, 1024
        Q = RNG.standard_normal((m, n))
        si, so = RNG.standard_normal(n), np.sign(RNG.standard_normal(m))
        x = RNG.standard_normal((3, n))
        W = ref.effective_weight(Q, si, so)
        np.testing.assert_allclose(x @ W.T, ref.rht_out(ref.rht_in(x, si) @ Q.T, so), atol=1e-8)


class SwiftHadamardPort(unittest.TestCase):
    """The Swift factor construction (ported literally in rht_precision.py) is the reference's, and
    the on-device self-check's tolerances follow from TF32 GEMMs, not from the matrices."""

    def test_swift_factors_identical(self):
        import rht_precision as rp
        for n in (2304, 1152, 3072, 1024, 384):
            A, B = rp.swift_factors(n)
            RA, RB = ref.hadamard_factors(n)
            self.assertTrue(np.array_equal(A, RA) and np.array_equal(B, RB), n)

    def test_round_trip_precision_bounds(self):
        import rht_precision as rp
        n = 2304
        A, B = rp.swift_factors(n)
        rng = np.random.default_rng(3)
        s = (np.sign(rng.standard_normal(n)) * rng.uniform(0.982, 1.018, n)).astype(np.float32)
        x = rng.standard_normal((1, n)).astype(np.float32)
        want = x.astype(np.float64) * s.astype(np.float64) ** 2
        rel = {}
        for mode in ("f32", "tf32t"):
            back = rp.rot_out(rp.rot_in(x, s, A, B, mode), s, A, B, mode)
            rel[mode] = np.abs(back - want).max() / np.abs(want).max()
        self.assertLess(rel["f32"], 1e-6)
        self.assertLess(rel["tf32t"], 1e-2)      # the self-check's bound
        self.assertGreater(rel["tf32t"], 1e-4)   # TF32 is why the old 1e-4 absolute bound failed


class NpzLayout(unittest.TestCase):
    def test_member_offsets_allow_pread(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.npz"
            a = RNG.integers(0, 2**32, size=(4, 5), dtype=np.uint64).astype(np.uint32)
            b = RNG.standard_normal(7).astype(np.float16)
            np.savez(p, a=a, b=b)
            members = ref.npz_members(p)
            raw = p.read_bytes()
            for name, arr in (("a", a), ("b", b)):
                off, size = members[name]
                got = np.load(io.BytesIO(raw[off: off + size]))
                self.assertEqual(got.tobytes(), arr.tobytes())


class ExpertAssembly(unittest.TestCase):
    def test_row_groups(self):
        """Synthetic row-group map mixing two formats: each expert assembles to [groups * RG, n]."""
        n, rg_rows = 64, 4
        f2, f1 = ref.TRELLIS["ht2"], ref.TRELLIS["rtr1"]
        ht2 = RNG.integers(0, 2**32, size=(2, rg_rows, n * 2 // 32), dtype=np.uint64).astype(np.uint32)
        rtr1 = RNG.integers(0, 2**32, size=(4, rg_rows, n // 32), dtype=np.uint64).astype(np.uint32)
        s2 = RNG.standard_normal((2, rg_rows)).astype(np.float16)
        s1 = RNG.standard_normal((4, rg_rows)).astype(np.float16)
        luts = {"ht2": random_lut(f2), "rtr1": random_lut(f1)}
        emap = np.array([[[1, 0], [3, 2]], [[3, 1], [1, 1]], [[3, 0], [3, 3]]])
        for e in range(3):
            parts = []
            for fid, loc in emap[e]:
                fmt = ref.ID_TO_FMT[int(fid)]
                codes, sc = (ht2, s2) if fmt == "ht2" else (rtr1, s1)
                parts.append(ref.decode_trellis(codes[loc], sc[loc], luts[fmt], n, fmt))
            w = np.concatenate(parts)
            self.assertEqual(w.shape, (2 * rg_rows, n))


@unittest.skipUnless(os.environ.get("LAMQ_PACKED_DIR"), "set LAMQ_PACKED_DIR to a downloaded packed build")
class RealFiles(unittest.TestCase):
    def test_recorded_digests(self):
        d = Path(os.environ["LAMQ_PACKED_DIR"])
        rec = json.loads((HERE / "lamq_digests.json").read_text())
        luts = ref.load_luts(d)
        for k, v in rec["luts"].items():
            self.assertEqual(__import__("hashlib").sha256(luts[k].tobytes()).hexdigest(), v, k)
        for li, lrec in rec["layers"].items():
            if not (d / f"layer_{int(li):02d}_packed.npz").exists():
                continue
            pl = ref.PackedLayer(d, int(li), luts)
            for name, r in lrec["linears"].items():
                self.assertEqual(ref.digest(pl.linear_q(name)), r["sha256"], f"layer {li} {name}")
            for side in ("gu", "dn"):
                for e in (0, 1, 63, 127):
                    self.assertEqual(ref.digest(pl.expert_q(side, e)[0]), lrec["experts"][side][e], f"layer {li} {side} {e}")


if __name__ == "__main__":
    unittest.main()
