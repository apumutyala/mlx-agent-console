"""Bit-exact cross-check of lamq_ref.py against LookAtMyQuant's own decoders, on real packed files.

LookAtMyQuant's PyTorch decoders (`runtime_cuda/inkling/formats.py`) are plain PyTorch and run on
the CPU; its MLX runtime needs Apple silicon. This script decodes every attention linear and every
routed expert of the given layers, and the embedding/head buckets, with both decoders and
requires identical float32 bits. It writes the per-tensor SHA-256 digests (the receipt the iPad
self-check compares against) to --out.

    uv run --no-project --with torch --with numpy python crosscheck_lamq_torch.py \
        --lamq-src <lookatmyquant checkout> --packed <packed build dir> --layers 0 7 \
        --out lamq_digests.json

The lamq checkout is only imported, never modified; keep it outside this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lamq_ref as ref  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lamq-src", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[0])
    ap.add_argument("--buckets", nargs="*", default=["embed"], help="embed and/or head (needs the npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.lamq_src).resolve()))
    import torch
    from runtime_cuda.inkling import formats as lf  # LookAtMyQuant's decoders

    def t(a: np.ndarray) -> torch.Tensor:
        a = np.ascontiguousarray(a)
        if a.dtype == np.uint32:
            a = a.view(np.int32)
        return torch.from_numpy(a)

    def same(name: str, mine: np.ndarray, theirs: torch.Tensor) -> str:
        th = theirs.numpy()
        if mine.shape != th.shape or mine.dtype != th.dtype:
            raise SystemExit(f"MISMATCH {name}: shape/dtype {mine.shape} {mine.dtype} vs {th.shape} {th.dtype}")
        if mine.tobytes() != th.tobytes():
            diff = np.flatnonzero(mine.view(np.uint32) != th.view(np.uint32))
            raise SystemExit(f"MISMATCH {name}: {diff.size} words differ, first at {diff[:5]}")
        return ref.digest(mine)

    d = Path(args.packed)
    luts = ref.load_luts(d)
    luts_t = {k: t(v) for k, v in luts.items()}
    out: dict = {
        "repo": args.repo_id,
        "manifest": json.loads((d / "manifest.json").read_text()),
        "method": "float32 bytes, little-endian, SHA-256; lamq_ref.py == lookatmyquant runtime_cuda.inkling.formats (torch CPU)",
        "torch": torch.__version__,
        "luts": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in luts.items()},
        "layers": {},
        "buckets": {},
        "hadamard": {},
    }
    checked = 0
    t0 = time.time()

    for n in (2304, 1152, 3072, 1024, 384, 128):
        A, B = ref.hadamard_factors(n)
        LA, LB = lf.hadamard_factors_np(n)
        if not (np.array_equal(A, LA) and np.array_equal(B, LB)):
            raise SystemExit(f"MISMATCH hadamard factors n={n}")
        out["hadamard"][str(n)] = [list(A.shape), list(B.shape)]

    for li in args.layers:
        pl = ref.PackedLayer(d, li, luts)
        rec: dict = {"linears": {}, "experts": {"gu": [], "dn": []}, "maps": {}}
        for name, lm in pl.meta["linears"].items():
            s, m, n, fmt = lm["safe"], int(lm["m"]), int(lm["n"]), lm["fmt"]
            a = pl.arrays
            if fmt.startswith("lm"):
                theirs = lf.decode_lm(t(a[f"lin_{s}_codes"]), t(a[f"lin_{s}_scales"]), t(a[f"lin_{s}_levels"]), m, n, int(lm["k"]))
            else:
                theirs = lf.decode_trellis(t(a[f"lin_{s}_codes"]).reshape(m, -1), t(a[f"lin_{s}_scales"]), luts_t[fmt], n, fmt)
            rec["linears"][name] = {"fmt": fmt, "shape": [m, n], "sha256": same(f"L{li} {name}", pl.linear_q(name), theirs)}
            checked += 1
            # The rotation is float math; check it agrees to float32 tolerance on one random input.
            si, so = pl.linear_signs(name)
            x = np.random.default_rng(li).standard_normal((2, n)).astype(np.float32)
            mine_y = ref.rht_in(x, si) @ pl.linear_q(name).astype(np.float64).T
            if so is not None:
                mine_y = ref.rht_out(mine_y, so)
            their_y = lf.rht_in(torch.from_numpy(x), torch.from_numpy(si)) @ theirs.T
            if so is not None:
                their_y = lf.rht_out(their_y, torch.from_numpy(so))
            err = float(np.abs(mine_y - their_y.double().numpy()).max() / (np.abs(mine_y).max() + 1e-30))
            if err > 1e-4:
                raise SystemExit(f"MISMATCH L{li} {name} rotated output rel err {err}")
            rec["linears"][name]["rot_rel_err"] = err
        for side in ("gu", "dn"):
            key = f"{side}_rg_map" if f"{side}_rg_map" in pl.arrays else f"{side}_map"
            raw = np.asarray(pl.arrays[key], dtype=np.int64)
            rec["maps"][side] = {"key": key, "shape": list(raw.shape), "sha256": hashlib.sha256(np.ascontiguousarray(pl.arrays[key]).tobytes()).hexdigest()}
            emap = raw.reshape(raw.shape[0], -1, 2)
            n = 2304 if side == "gu" else 1152
            for e in range(emap.shape[0]):
                parts = []
                for fid, loc in emap[e]:
                    fmt = ref.ID_TO_FMT[int(fid)]
                    if fmt == "nvfp4":
                        parts = [lf.decode_nvfp4(t(pl.arrays[f"{side}_nvfp4_codes"][loc]), t(pl.arrays[f"{side}_nvfp4_scales"][loc]), t(pl.arrays[f"{side}_nvfp4_outer"][loc]))]
                        break
                    parts.append(lf.decode_trellis(t(pl.arrays[f"{side}_{fmt}_codes"][loc]), t(pl.arrays[f"{side}_{fmt}_scales"][loc]), luts_t[fmt], n, fmt))
                theirs = parts[0] if len(parts) == 1 else torch.cat(parts)
                mine, _ = pl.expert_q(side, e)
                rec["experts"][side].append(same(f"L{li} {side} e{e}", mine, theirs))
                checked += 1
        out["layers"][str(li)] = rec
        print(f"layer {li}: {len(rec['linears'])} linears + {len(rec['experts']['gu']) + len(rec['experts']['dn'])} experts identical ({time.time() - t0:.0f}s)", flush=True)

    for kind in args.buckets:
        if not (d / f"{kind}_packed.npz").exists():
            continue
        pb = ref.PackedBuckets(d, kind, luts)
        brec = {"order_sha256": hashlib.sha256(np.ascontiguousarray(pb.arrays["order"]).tobytes()).hexdigest(), "buckets": []}
        for b, bm in enumerate(pb.meta):
            kb, opt, m = bm["kb"], bm["opt"], int(bm["rows"])
            a = pb.arrays
            h = hashlib.sha256()
            for r0 in range(0, m, 8192):
                r1 = min(m, r0 + 8192)
                mine = pb.bucket_q(b, slice(r0, r1))
                if opt.startswith("lm"):
                    theirs = lf.decode_lm(t(a[f"b{kb}_codes"]), t(a[f"b{kb}_scales"]), t(a[f"b{kb}_levels"]), m, 2304, int(bm["k"]))[r0:r1]
                else:
                    theirs = lf.decode_trellis(t(a[f"b{kb}_codes"].reshape(m, -1)[r0:r1]), t(a[f"b{kb}_scales"][r0:r1]), luts_t[opt], 2304, opt)
                same(f"{kind} b{kb} rows {r0}:{r1}", mine, theirs)
                h.update(np.ascontiguousarray(mine, dtype="<f4").tobytes())
            brec["buckets"].append({"kb": kb, "opt": opt, "rows": m, "sha256": h.hexdigest()})
            checked += 1
        out["buckets"][kind] = brec
        print(f"{kind}: {len(pb.meta)} buckets identical ({time.time() - t0:.0f}s)", flush=True)

    out["checked_tensors"] = checked
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"OK: {checked} tensors bit-identical; digests -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
