"""Why the on-device "Hadamard 2304 round trip" check failed at 5.2e-3: TF32 matmuls, not a bad matrix.

On Apple GPUs with neural accelerators (M5 family), MLX runs float32 GEMMs on them at TF32
precision unless MLX_ENABLE_TF32=0 (`mlx/utils.h: enable_tf32()` defaults to 1;
`backend/metal/matmul.cpp`: `use_nax = is_nax_available() && (enable_tf32() || dtype != f32)`).
Matrix-vector products (M = 1) take the gemv path and stay full float32.

This script
1. ports the Swift Hadamard construction (InklingLamqFormat.swift) literally and checks it
   equals lamq_ref.hadamard_factors (and therefore LookAtMyQuant's, see crosscheck) exactly;
2. runs the self-check's round trip `output(input(x, s), s) - x * s^2` with the same op order as
   InklingLamqRotation in float64, float32, TF32-emulated float32 and bfloat16-emulated inputs;
3. measures the transform output itself against the float64 reference in each precision;
4. measures the attention-style check (`rht_in(x) @ Q.T`, M = 2 GEMM) the same way.

    python rht_precision.py [--signs-from <layer_07_packed.npz>]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lamq_ref as ref  # noqa: E402


# ------------------------------------------------------------- literal port of the Swift factors

def swift_sylvester(a: int) -> np.ndarray:
    h, order = [1.0], 1
    for _ in range(a):
        o2 = order * 2
        nxt = [0.0] * (o2 * o2)
        for i in range(order):
            for j in range(order):
                v = h[i * order + j]
                nxt[i * o2 + j] = v
                nxt[i * o2 + j + order] = v
                nxt[(i + order) * o2 + j] = v
                nxt[(i + order) * o2 + j + order] = -v
        h, order = nxt, o2
    return np.array(h).reshape(order, order)


def swift_paley(q: int) -> np.ndarray:
    squares = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0.0 if a == 0 else (1.0 if a in squares else -1.0)  # noqa: E731
    o = q + 1
    s = [0.0] * (o * o)
    for j in range(1, o):
        s[j] = 1.0
    for i in range(1, o):
        s[i * o] = -1.0
    for i in range(q):
        for j in range(q):
            s[(i + 1) * o + (j + 1)] = chi(((j - i) % q + q) % q)
    for i in range(o):
        s[i * o + i] += 1.0
    return np.array(s).reshape(o, o)


def swift_factors(n: int) -> tuple[np.ndarray, np.ndarray]:
    a = 0
    while (n >> (a + 1)) << (a + 1) == n:
        a += 1
    while a >= 0:
        if n % (1 << a) == 0:
            p = n // (1 << a)
            if p in (1, 2) or (ref._is_prime(p - 1) and (p - 1) % 4 == 3):
                if p == 1 and a > 1:
                    return swift_sylvester(a // 2), swift_sylvester(a - a // 2)
                if p == 1:
                    return swift_sylvester(a), np.ones((1, 1))
                if p == 2:
                    return swift_sylvester(a), np.array([[1.0, 1.0], [1.0, -1.0]])
                return swift_sylvester(a), swift_paley(p - 1)
        a -= 1
    raise ValueError(n)


# ------------------------------------------------------------------------ precision emulation

def tf32(x: np.ndarray) -> np.ndarray:
    """float32 -> TF32 (10 explicit mantissa bits), round to nearest even, kept in float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 13) & 1
    u = (u + 0x0FFF + lsb) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32)


def tf32_trunc(x: np.ndarray) -> np.ndarray:
    """float32 -> TF32 by truncation (the low 13 mantissa bits dropped). Matches the device."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return (u & np.uint32(0xFFFFE000)).view(np.float32)


def bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def mm(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """Matmul as the device runs it: inputs rounded per `mode`, float32 accumulate (float64 for 'f64')."""
    if mode == "f64":
        return np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    if mode == "tf32":
        a, b = tf32(a), tf32(b)
    elif mode == "tf32t":
        a, b = tf32_trunc(a), tf32_trunc(b)
    elif mode == "bf16":
        a, b = bf16(a), bf16(b)
    return (np.asarray(a, np.float32) @ np.asarray(b, np.float32)).astype(np.float32)


def rot_in(x, s, A, B, mode):
    """InklingLamqRotation.input: matmul(matmul(A, (x*s).reshape(-1,a,b)), B.T) * 1/sqrt(n)."""
    n = x.shape[-1]
    dt = np.float64 if mode == "f64" else np.float32
    xs = (x.astype(dt) * s.astype(dt)).reshape(-1, A.shape[0], B.shape[0])
    y = np.stack([mm(mm(A, xi, mode), B.T, mode) for xi in xs])
    return (y * dt(1 / math.sqrt(n))).reshape(x.shape)


def rot_out(y, s, A, B, mode):
    """InklingLamqRotation.output: matmul(matmul(A.T, y.reshape(-1,a,b)), B) * (s/sqrt(n))."""
    n = y.shape[-1]
    dt = np.float64 if mode == "f64" else np.float32
    ys = y.astype(dt).reshape(-1, A.shape[0], B.shape[0])
    z = np.stack([mm(mm(A.T, yi, mode), B, mode) for yi in ys]).reshape(y.shape)
    return z * (s.astype(dt) * dt(1 / math.sqrt(n)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs-from", help="a layer npz (uses its gu_signs_in, as the self-check does)")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    print("1. Swift Hadamard port vs lamq_ref (== LookAtMyQuant, crosscheck_lamq_torch.py):")
    for n in (2304, 1152, 3072, 1024, 384):
        A, B = swift_factors(n)
        RA, RB = ref.hadamard_factors(n)
        H = np.kron(A, B)
        print(f"   n={n}: A {A.shape[0]}x{A.shape[0]} B {B.shape[0]}x{B.shape[0]}  identical={np.array_equal(A, RA) and np.array_equal(B, RB)}"
              f"  H@H.T==n*I: {np.array_equal(H @ H.T, n * np.eye(n))}")

    n = 2304
    A, B = swift_factors(n)
    if args.signs_from:
        s = np.load(args.signs_from)["gu_signs_in"].astype(np.float32)
        src = f"gu_signs_in of {Path(args.signs_from).name}"
    else:
        s = (np.sign(rng.standard_normal(n)) * rng.uniform(0.982, 1.018, n)).astype(np.float32)
        src = "random f16-like signs (|s| in 0.982..1.018)"
    x = rng.standard_normal((1, n)).astype(np.float32)   # MLXRandom.normal([1, 2304])
    print(f"\n2. Round trip output(input(x, s), s) - x*s^2, x ~ N(0,1) [1,2304], signs: {src}")
    want = x.astype(np.float64) * s.astype(np.float64) ** 2
    for mode in ("f64", "f32", "tf32", "tf32t", "bf16"):
        back = rot_out(rot_in(x, s, A, B, mode), s, A, B, mode)
        print(f"   {mode:>4}: max abs err {np.abs(back - want).max():.3e}")
    print("   on device (M5, MLX fp32 GEMM): 5.185e-03")

    print("\n3. Transform output itself vs float64, rel err (max|d| / max|ref|), 32 inputs:")
    X = rng.standard_normal((32, n)).astype(np.float32)
    y64 = rot_in(X, s, A, B, "f64")
    for mode in ("f32", "tf32", "tf32t", "bf16"):
        y = rot_in(X, s, A, B, mode)
        print(f"   {mode:>4}: {np.abs(y - y64).max() / np.abs(y64).max():.3e}")
    assert np.allclose(y64, ref.rht_in(X, s), atol=1e-9), "rot_in port disagrees with lamq_ref.rht_in"

    print("\n4. Attention check: rht_in(x) @ Q.T, x [2,2304], Q a random rtr4-like [3072,2304]:")
    Q = (rng.standard_normal((3072, n)) * 0.02).astype(np.float32)
    x2 = rng.standard_normal((2, n)).astype(np.float32)
    xr = rot_in(x2, s, A, B, "f32")
    exact = xr.astype(np.float64) @ Q.astype(np.float64).T
    fused = (xr @ Q.T).astype(np.float32)               # the matvec kernel: f32 FMA chains
    for name, got in (("fused f32 matvec", fused), ("MLX GEMM tf32 RNE (M=2)", mm(xr, Q.T, "tf32")),
                      ("MLX GEMM tf32 truncating (M=2)", mm(xr, Q.T, "tf32t"))):
        print(f"   {name}: rel err vs exact {np.abs(got - exact).max() / np.abs(exact).max():.3e}")
    d = np.abs(fused - mm(xr, Q.T, "tf32t")).max() / np.abs(exact).max()
    print(f"   fused vs tf32 GEMM (what the self-check compares): {d:.3e}  (device: 6.5e-4 .. 8.5e-4)")
    xe = rng.standard_normal((1, n)).astype(np.float32)
    e_exact = xe.astype(np.float64) @ Q.astype(np.float64).T
    print(f"   experts check is M=1 (gemv, full f32): rel err {np.abs((xe @ Q.T) - e_exact).max() / np.abs(e_exact).max():.3e}"
          "  (device: 1.7e-7 .. 2.9e-7)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
