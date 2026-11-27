"""NumPy reference decoder for LookAtMyQuant ("lamq") packed Inkling 2 Mini builds.

Spec: docs/LAMQ-FORMAT-SPEC.md. The packed format and the Hadamard construction come from
LookAtMyQuant (https://github.com/ShinMegamiBoson/lookatmyquant, MIT, (c) 2026 Shin Megami Boson;
see LICENSE-lookatmyquant.txt next to this file). This file is our own implementation of that
spec. It is checked bit-for-bit against lamq's own PyTorch decoders by `crosscheck_lamq_torch.py`
(receipt: lamq_digests.json) and against a big-integer bitstream oracle by `test_lamq_ref.py`.

Everything here is exact, not fast: decoded weights are float32 `lut[idx] * scale` (a product of
two float16 values, which float32 represents exactly), so any correct decoder -- NumPy, PyTorch,
Metal -- produces the same bits. The Hadamard rotations are ordinary float math (not bit-exact
across implementations).

Usage:
    python lamq_ref.py <packed_dir> [layer]    # decode one layer and print digests
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HASH_MULT_U32 = 0x9E3779B1
STATE_BITS = 12
STATE_MASK = (1 << STATE_BITS) - 1

# Format ids as stored in `{gu,dn}_rg_map[..., 0]` / `{gu,dn}_map[..., 0]` and manifest.json `fmt_ids`.
ID_TO_FMT = {0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4", 7: "nvfp4"}
FMT_TO_ID = {v: k for k, v in ID_TO_FMT.items()}
NVFP4_FID = 7


@dataclass(frozen=True)
class TrellisFormat:
    name: str
    k: int          # bits per weight (stream advance per column)
    g: int          # weights produced by one 12-bit state
    K: int          # table entries (rows of the LUT)
    hash_mult: int  # 0: index = state; else index = (state * mult mod 2^32) >> (32 - log2 K)

    @property
    def log2K(self) -> int:
        return int(round(math.log2(self.K)))


TRELLIS = {
    "ht1": TrellisFormat("ht1", 1, 4, 512, HASH_MULT_U32),
    "ht2": TrellisFormat("ht2", 2, 4, 512, HASH_MULT_U32),
    "ht3": TrellisFormat("ht3", 3, 2, 1024, HASH_MULT_U32),
    "rtr1": TrellisFormat("rtr1", 1, 1, 4096, 0),
    "rtr2": TrellisFormat("rtr2", 2, 1, 4096, 0),
    "rtr3": TrellisFormat("rtr3", 3, 1, 4096, 0),
    "rtr4": TrellisFormat("rtr4", 4, 1, 4096, 0),
}

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


@lru_cache(maxsize=1)
def e4m3_table() -> np.ndarray:
    """float32[256] for FP8 E4M3 (bias 7, subnormals; 0x7F/0xFF, the NaN codes, read as 0)."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, mnt = (b >> 3) & 0xF, b & 0x7
        if e == 15 and mnt == 7:
            t[b] = 0.0
        elif e == 0:
            t[b] = sign * (mnt / 8.0) * 2.0 ** -6
        else:
            t[b] = sign * (1.0 + mnt / 8.0) * 2.0 ** (e - 7)
    return t


# ------------------------------------------------------------------------------------- trellis

def trellis_states(codes: np.ndarray, n: int, k: int, g: int) -> np.ndarray:
    """12-bit states `[..., m, n // g]` (uint32) from codes `[..., m, n*k/32]` (uint32).

    Each row is one little-endian bitstream (word i holds bits 32i..32i+31). Column n-1 sits at
    the low end: the group of g columns whose last column is `last` reads the 12-bit window that
    starts at bit `(n - 1 - last) * k`. Bits past the end of the row read as zero.
    """
    codes = np.ascontiguousarray(codes).view(np.uint32) if codes.dtype != np.uint32 else codes
    wpr = codes.shape[-1]
    if wpr * 32 != n * k:
        raise ValueError(f"codes have {wpr} words per row; n={n}, k={k} needs {n * k // 32}")
    if n % g:
        raise ValueError(f"n={n} not divisible by g={g}")
    w = np.concatenate([codes.astype(np.uint64), np.zeros(codes.shape[:-1] + (1,), np.uint64)], axis=-1)
    last = np.arange(g - 1, n, g, dtype=np.int64)
    bp = (n - 1 - last) * k
    wi = bp >> 5
    off = (bp & 31).astype(np.uint64)
    pair = w[..., wi] | (w[..., wi + 1] << np.uint64(32))
    return ((pair >> off) & np.uint64(STATE_MASK)).astype(np.uint32)


def trellis_indices(states: np.ndarray, fmt: TrellisFormat) -> np.ndarray:
    if not fmt.hash_mult:
        return states.astype(np.int64)
    prod = (states.astype(np.uint64) * np.uint64(fmt.hash_mult)) & np.uint64(0xFFFFFFFF)
    return (prod >> np.uint64(32 - fmt.log2K)).astype(np.int64)


def decode_trellis(codes: np.ndarray, scales: np.ndarray, lut: np.ndarray, n: int, fmt: str | TrellisFormat) -> np.ndarray:
    """Dense float32 `Q * scale` `[..., m, n]` in the rotated basis. `lut` is float16 `[K, g]` or flat."""
    f = TRELLIS[fmt] if isinstance(fmt, str) else fmt
    idx = trellis_indices(trellis_states(codes, n, f.k, f.g), f)
    table = np.asarray(lut, dtype=np.float16).reshape(f.K, f.g).astype(np.float32)
    vals = table[idx].reshape(*idx.shape[:-1], n)
    return vals * np.asarray(scales, dtype=np.float16).astype(np.float32)[..., None]


# --------------------------------------------------------------------------------- Lloyd-Max

def lm_indices(codes: np.ndarray, m: int, n: int, k: int) -> np.ndarray:
    """Row-major Lloyd-Max indices `[m, n]`. lm8: one byte each. lm6: 4 indices in 3 bytes,
    little-endian 24-bit word, first index in the low 6 bits."""
    c = np.ascontiguousarray(codes).reshape(-1).view(np.uint8).astype(np.uint32)
    if k == 8:
        return c.reshape(m, n).astype(np.int64)
    if k != 6:
        raise ValueError(f"Lloyd-Max k={k}")
    t = c.reshape(-1, 3)
    v = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    idx = np.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], axis=-1)
    return idx.reshape(m, n).astype(np.int64)


def decode_lm(codes: np.ndarray, scales: np.ndarray, levels: np.ndarray, m: int, n: int, k: int) -> np.ndarray:
    lv = np.asarray(levels).astype(np.float32).reshape(-1)
    return lv[lm_indices(codes, m, n, k)] * np.asarray(scales, dtype=np.float16).astype(np.float32)[:, None]


# ------------------------------------------------------------------------------------- NVFP4

def decode_nvfp4(codes: np.ndarray, scales: np.ndarray, outer) -> np.ndarray:
    """Dense float32 `[..., m, n]` from E2M1 nibbles `[..., m, n/2]` (low nibble = even column),
    E4M3 block scales `[..., m, n/16]`, and one float32 outer scale per leading index. NVFP4
    experts multiply the UNROTATED activation."""
    c = np.asarray(codes).view(np.uint8)
    vals = np.stack([E2M1[c & 0xF], E2M1[c >> 4]], axis=-1).reshape(*c.shape[:-1], c.shape[-1] * 2)
    bs = e4m3_table()[np.asarray(scales).view(np.uint8)]
    w = (vals.reshape(*bs.shape, 16) * bs[..., None]).reshape(vals.shape)
    o = np.asarray(outer, dtype=np.float32)
    return w * o.reshape(o.shape + (1,) * (w.ndim - o.ndim))


# ---------------------------------------------------------------------------------- Hadamard

def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, math.isqrt(q) + 1))


def _sylvester(a: int) -> np.ndarray:
    h = np.ones((1, 1))
    for _ in range(a):
        h = np.block([[h, h], [h, -h]])
    return h


def _paley1(q: int) -> np.ndarray:
    """Paley construction I: a Hadamard matrix of order q + 1 for a prime q = 3 (mod 4)."""
    squares = {(x * x) % q for x in range(1, q)}
    chi = np.array([0.0] + [1.0 if a in squares else -1.0 for a in range(1, q)])
    jacobsthal = chi[(np.arange(q)[None, :] - np.arange(q)[:, None]) % q]  # Q[i, j] = chi(j - i)
    s = np.zeros((q + 1, q + 1))
    s[0, 1:] = 1.0
    s[1:, 0] = -1.0
    s[1:, 1:] = jacobsthal
    return s + np.eye(q + 1)


@lru_cache(maxsize=32)
def hadamard_factors(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) with H = kron(A, B) of order n. Takes the largest power of two 2^a dividing n whose
    cofactor p is 1, 2, or q + 1 for a prime q = 3 (mod 4); A = Sylvester(2^a), B = Paley(p)
    (or [[1,1],[1,-1]] for p = 2). For a pure power of two (p = 1, a > 1) the Sylvester order is
    split in two halves, A = Sylvester(2^(a//2)), B = Sylvester(2^(a - a//2))."""
    for a in range(int(math.log2(n)), -1, -1):
        if n % (1 << a):
            continue
        p = n // (1 << a)
        if p in (1, 2) or (_is_prime(p - 1) and (p - 1) % 4 == 3):
            if p == 1 and a > 1:
                return _sylvester(a // 2), _sylvester(a - a // 2)
            if p == 1:
                return _sylvester(a), np.ones((1, 1))
            if p == 2:
                return _sylvester(a), np.array([[1.0, 1.0], [1.0, -1.0]])
            return _sylvester(a), _paley1(p - 1)
    raise ValueError(f"no Hadamard construction for n={n}")


def rht_in(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """`x @ R.T` with R = kron(A, B) diag(signs) / sqrt(n) (float64)."""
    n = x.shape[-1]
    A, B = hadamard_factors(n)
    xs = np.asarray(x, np.float64) * np.asarray(signs, np.float64)
    y = A @ xs.reshape(-1, A.shape[0], B.shape[0]) @ B.T
    return (y / math.sqrt(n)).reshape(x.shape)


def rht_out(y: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """`y @ R` with R = kron(A, B) diag(signs) / sqrt(m) (float64)."""
    m = y.shape[-1]
    A, B = hadamard_factors(m)
    z = (A.T @ np.asarray(y, np.float64).reshape(-1, A.shape[0], B.shape[0]) @ B).reshape(y.shape)
    return z * (np.asarray(signs, np.float64) / math.sqrt(m))


def effective_weight(Q: np.ndarray, signs_in: np.ndarray, signs_out: np.ndarray | None = None) -> np.ndarray:
    """Dense W (`y = x @ W.T`) = R_out.T @ Q @ R_in (R_out omitted when absent), float64."""
    n = Q.shape[-1]
    wt = rht_in(np.eye(n), signs_in) @ np.asarray(Q, np.float64).T   # R_in.T @ Q.T
    if signs_out is not None:
        wt = rht_out(wt, signs_out)
    return wt.T


# ------------------------------------------------------------------------------- packed files

def npz_members(path: Path) -> dict[str, tuple[int, int]]:
    """{array name: (byte offset of the .npy payload's header in the file, member size)}. The
    released files are STORED (no compression), so a loader can pread payloads directly."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path.name}:{info.filename} is compressed")
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                hdr = f.read(30)
                name_len = int.from_bytes(hdr[26:28], "little")
                extra_len = int.from_bytes(hdr[28:30], "little")
            out[info.filename[:-4]] = (info.header_offset + 30 + name_len + extra_len, info.file_size)
    return out


class PackedLayer:
    """One `layer_XX_packed.npz` plus its meta."""

    def __init__(self, packed_dir: Path, index: int, luts: dict[str, np.ndarray] | None = None):
        self.dir = Path(packed_dir)
        self.index = index
        self.meta = json.loads((self.dir / f"layer_{index:02d}_meta.json").read_text())
        self.arrays = np.load(self.dir / f"layer_{index:02d}_packed.npz")
        self.luts = luts if luts is not None else load_luts(self.dir)

    def linear_q(self, name: str) -> np.ndarray:
        """Decoded `Q * scale` `[m, n]` float32 of an attention linear (rotated basis)."""
        lm = self.meta["linears"][name]
        s, m, n, fmt = lm["safe"], int(lm["m"]), int(lm["n"]), lm["fmt"]
        a = self.arrays
        if fmt.startswith("lm"):
            return decode_lm(a[f"lin_{s}_codes"], a[f"lin_{s}_scales"], a[f"lin_{s}_levels"], m, n, int(lm["k"]))
        return decode_trellis(a[f"lin_{s}_codes"].reshape(m, -1), a[f"lin_{s}_scales"], self.luts[fmt], n, fmt)

    def linear_signs(self, name: str) -> tuple[np.ndarray, np.ndarray | None]:
        lm = self.meta["linears"][name]
        s = lm["safe"]
        so = self.arrays[f"lin_{s}_signs_out"].astype(np.float32) if lm.get("out_rot") else None
        return self.arrays[f"lin_{s}_signs_in"].astype(np.float32), so

    def expert_map(self, side: str) -> np.ndarray:
        """[E, n_rg, 2] (fid, slab index). Legacy per-expert maps become n_rg = 1."""
        if f"{side}_rg_map" in self.arrays:
            return np.asarray(self.arrays[f"{side}_rg_map"], dtype=np.int64)
        return np.asarray(self.arrays[f"{side}_map"], dtype=np.int64).reshape(-1, 1, 2)

    def expert_q(self, side: str, e: int, hidden: int = 2304, inter: int = 1152) -> tuple[np.ndarray, bool]:
        """(dense float32 weight of expert e, whether it reads the ROTATED activation).

        side "gu": [2*inter, hidden] (gate rows first, then up rows). side "dn": [hidden, inter].
        Row group r (rows r*R .. r*R+R-1, R = rows / n_rg) comes from slab `{side}_{fmt}_codes[loc]`.
        """
        n = hidden if side == "gu" else inter
        parts = []
        for fid, loc in self.expert_map(side)[e]:
            fmt = ID_TO_FMT[int(fid)]
            if fmt == "nvfp4":
                a = self.arrays
                return decode_nvfp4(a[f"{side}_nvfp4_codes"][loc], a[f"{side}_nvfp4_scales"][loc], a[f"{side}_nvfp4_outer"][loc]), False
            parts.append(decode_trellis(self.arrays[f"{side}_{fmt}_codes"][loc], self.arrays[f"{side}_{fmt}_scales"][loc],
                                        self.luts[fmt], n, fmt))
        return (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)), True


def load_luts(packed_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(packed_dir) / "luts.npz")
    return {k: np.asarray(z[k], dtype=np.float16) for k in z.files}


class PackedBuckets:
    """`embed_packed.npz` / `head_packed.npz`: vocabulary rows split into precision buckets.

    `order[o]` is the token id of bucket-ordered row o (buckets concatenated in meta order).
    Embedding: row e(token) = rht_out(Q_b[r], signs_b), i.e. stored rows are rotated rows.
    Head: logits[order[o]] = (Q_b @ rht_in(h, signs_b))[r] for o in bucket b.
    """

    def __init__(self, packed_dir: Path, kind: str, luts: dict[str, np.ndarray] | None = None, hidden: int = 2304):
        self.meta = json.loads((Path(packed_dir) / f"{kind}_meta.json").read_text())
        self.arrays = np.load(Path(packed_dir) / f"{kind}_packed.npz")
        self.luts = luts if luts is not None else load_luts(packed_dir)
        self.hidden = hidden
        self.order = np.asarray(self.arrays["order"], dtype=np.int64)
        self.inv = np.empty_like(self.order)
        self.inv[self.order] = np.arange(len(self.order))
        self.bounds = np.cumsum([0] + [int(b["rows"]) for b in self.meta])

    def bucket_q(self, b: int, rows: slice | None = None) -> np.ndarray:
        bm = self.meta[b]
        kb, opt, m, n = bm["kb"], bm["opt"], int(bm["rows"]), self.hidden
        a = self.arrays
        if opt.startswith("lm"):
            q = decode_lm(a[f"b{kb}_codes"], a[f"b{kb}_scales"], a[f"b{kb}_levels"], m, n, int(bm["k"]))
            return q if rows is None else q[rows]
        codes = a[f"b{kb}_codes"].reshape(m, -1)
        scales = a[f"b{kb}_scales"]
        if rows is not None:
            codes, scales = codes[rows], scales[rows]
        return decode_trellis(codes, scales, self.luts[opt], n, opt)

    def embedding_row(self, token: int) -> np.ndarray:
        o = int(self.inv[token])
        b = int(np.searchsorted(self.bounds, o, side="right") - 1)
        r = o - int(self.bounds[b])
        q = self.bucket_q(b, slice(r, r + 1))
        return rht_out(q, self.arrays[f"b{self.meta[b]['kb']}_signs_in"].astype(np.float64))[0]


def digest(a: np.ndarray) -> str:
    """SHA-256 of the little-endian float32 bytes (the bit-exactness receipt)."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


def main(argv: list[str]) -> int:
    d = Path(argv[1])
    layer = int(argv[2]) if len(argv) > 2 else 0
    pl = PackedLayer(d, layer)
    for name in pl.meta["linears"]:
        q = pl.linear_q(name)
        print(f"layer {layer} {name} {pl.meta['linears'][name]['fmt']} {q.shape} {digest(q)}")
    for side in ("gu", "dn"):
        for e in (0, 127):
            q, rot = pl.expert_q(side, e)
            print(f"layer {layer} {side} expert {e} rotated={rot} {q.shape} {digest(q)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
