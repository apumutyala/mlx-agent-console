"""NumPy reference for Inkling 2's fused decode expert kernels (affine 2/3/4/8-bit).

Two independent halves:

* ``mlx_quantize`` / ``dequantize_bitstream``: MLX's affine quantization (a port of
  ``quantize`` in mlx/backend/cpu/quantized.cpp at the mlx commit mlx-swift 0.33.0 bundles,
  36969c0) and a dequantizer that reads each row as one little-endian bit stream. They do not
  share packing code, so a packing mistake in either shows up as a mismatch.
* ``kernel_gate_up`` / ``kernel_down``: the Metal kernels ``inkling2_expert_gate_up`` and
  ``inkling2_expert_down`` in ios/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Inkling2Kernels.swift,
  ported line by line (same grid loops, lane loop, byte offsets and rounding points) onto flat
  byte / scale / bias buffers, so an indexing bug in the Metal source shows up here too.

``dense_reference`` is the plain math (dequantize, matmul, softcapped SwiGLU, weighted sum,
short convolution, residual) that both must equal.
"""

from __future__ import annotations

import numpy as np

SIMD = 32


# ---------------------------------------------------------------------------- rounding


def bf16(x):
    """Round float32 to bfloat16 (nearest even), returned as float32. ``T(...)`` in Metal."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    out = u.astype(np.uint32).view(np.float32)
    return out if out.shape else np.float32(out)


def identity(x):
    return np.asarray(x, dtype=np.float32) if np.ndim(x) else np.float32(x)


# ------------------------------------------------------------------- MLX affine packing


def pack_factor(bits: int) -> int:
    """Values per pack (``get_pack_factor<bits, 32>``)."""
    return 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)


def bytes_per_pack(bits: int) -> int:
    """``get_bytes_per_pack(bits)`` with its default ``wsize = 8``: 1 for powers of two, whose
    ``quantize`` loop then counts uint32 words; 3 bytes for 3-bit packs."""
    return 1 if bits & (bits - 1) == 0 else (5 if bits == 5 else 3)


def mlx_quantize(w: np.ndarray, bits: int, group_size: int, scale_round=bf16):
    """MLX affine ``quantize`` (CPU), rows of ``w`` ``[..., K]``.

    Returns ``(packed uint32 [..., K * bits / 32], scales [..., K / gs], biases [..., K / gs])``.
    Scales and biases are rounded with ``scale_round`` (bf16 for bfloat16 weights) *after*
    the codes are chosen, as MLX stores ``static_cast<T>(scale)``.
    """
    w = np.asarray(w, dtype=np.float32)
    lead, k = w.shape[:-1], w.shape[-1]
    assert k % group_size == 0 and (k * bits) % 32 == 0
    flat = w.reshape(-1)
    n_bins = float((1 << bits) - 1)
    eps = 1e-7
    el_per_int = pack_factor(bits)
    bpp = bytes_per_pack(bits)
    int_per_group = group_size * bpp // el_per_int  # uint32 words (pow2) or bytes per group
    pow2 = bits & (bits - 1) == 0
    n_groups = flat.size // group_size
    if pow2:
        out = np.zeros(n_groups * int_per_group, dtype=np.uint32)
    else:
        out = np.zeros(n_groups * int_per_group, dtype=np.uint8)
    scales = np.zeros(n_groups, dtype=np.float32)
    biases = np.zeros(n_groups, dtype=np.float32)
    for i in range(n_groups):
        g = flat[i * group_size:(i + 1) * group_size]
        w_max, w_min = float(g.max()), float(g.min())
        mask = abs(w_min) > abs(w_max)
        scale = max((w_max - w_min) / n_bins, eps)
        scale = scale if mask else -scale
        edge = w_min if mask else w_max
        q0 = float(np.rint(edge / scale))
        bias = 0.0
        if q0 != 0:
            scale = edge / q0
            bias = edge
        base = i * int_per_group
        for j in range(int_per_group // bpp):
            el = 0
            for kk in range(el_per_int):
                v = float(np.rint((float(g[j * el_per_int + kk]) - bias) / scale))
                v = int(min(max(v, 0.0), n_bins))
                el |= v << (kk * bits)
            if pow2:
                out[base + j] = el
            else:
                for b in range(bpp):
                    out[base + bpp * j + b] = (el >> (8 * b)) & 0xFF
        scales[i] = scale
        biases[i] = bias
    if not pow2:
        out = out.view(np.uint32)  # little endian, as the uint32 checkpoint tensor
    packed = out.reshape(*lead, k * bits // 32)
    return (packed, scale_round(scales).reshape(*lead, k // group_size),
            scale_round(biases).reshape(*lead, k // group_size))


def unpack_bitstream(packed: np.ndarray, bits: int, k: int) -> np.ndarray:
    """Codes ``[..., K]`` of ``packed`` (uint32 words or raw uint8 bytes) read as one
    little-endian bit stream per row."""
    lead = packed.shape[:-1]
    if packed.dtype != np.uint8:
        packed = np.ascontiguousarray(packed, dtype="<u4").view(np.uint8)
    raw = packed.reshape(*lead, -1)
    bitsarr = np.unpackbits(raw, axis=-1, bitorder="little")[..., :k * bits]
    bitsarr = bitsarr.reshape(*lead, k, bits).astype(np.int64)
    return (bitsarr << np.arange(bits)).sum(-1)


def dequantize_bitstream(packed, scales, biases, bits, group_size, k):
    q = unpack_bitstream(packed, bits, k).astype(np.float64)
    s = np.repeat(np.asarray(scales, np.float64), group_size, axis=-1)
    b = np.repeat(np.asarray(biases, np.float64), group_size, axis=-1)
    return q * s + b


# ------------------------------------------------------------ the Metal kernels, ported


def unit_dot(w: np.ndarray, off: int, x: np.ndarray, bits: int) -> float:
    """``inkling2_unit_dot<BITS, true>``: 16 codes at byte ``off`` of ``w`` (uint8) with ``x``."""
    s = np.float32(0.0)
    if bits == 2:
        q = int(w[off]) | int(w[off + 1]) << 8 | int(w[off + 2]) << 16 | int(w[off + 3]) << 24
        for i in range(16):
            s = np.float32(s + x[i] * np.float32((q >> (2 * i)) & 3))
        return s
    if bits == 3:
        for h in range(2):
            q = int(w[off + 3 * h]) | (int(w[off + 3 * h + 1]) << 8) | (int(w[off + 3 * h + 2]) << 16)
            for i in range(8):
                s = np.float32(s + x[8 * h + i] * np.float32((q >> (3 * i)) & 7))
        return s
    if bits == 4:
        qx = int.from_bytes(bytes(w[off:off + 4]), "little")
        qy = int.from_bytes(bytes(w[off + 4:off + 8]), "little")
        for i in range(8):
            n = (qx if i < 4 else qy) >> ((i % 4) * 8)
            s = np.float32(s + x[2 * i] * np.float32(n & 15) + x[2 * i + 1] * np.float32((n >> 4) & 15))
        return s
    assert bits == 8
    for i in range(16):
        s = np.float32(s + x[i] * np.float32(w[off + i]))
    return s


def rows_dot(w, w_off, scales, s_off, biases, b_off, x, x_off, lane, acc, *, bits, gs, k, r_rows):
    """``inkling2_rows_dot<BITS, GS, true, K, R>``: lane ``lane``'s share, added into ``acc``."""
    for u in range(lane, k // 16, 32):
        xv = np.asarray(x[x_off + u * 16:x_off + u * 16 + 16], dtype=np.float32)
        xs = np.float32(0.0)
        for i in range(16):
            xs = np.float32(xs + xv[i])
        g = u * 16 // gs
        for r in range(r_rows):
            dot = unit_dot(w, w_off + (r * k + u * 16) * bits // 8, xv, bits)
            acc[r] = np.float32(acc[r] + np.float32(scales[s_off + r * (k // gs) + g]) * dot)
            acc[r] = np.float32(acc[r] + np.float32(biases[b_off + r * (k // gs) + g]) * xs)


def kernel_gate_up(x, w, scales, biases, indices, caps, *, bits, gs, din, inter, ke, nsg=4, T=bf16):
    """``inkling2_expert_gate_up`` over its whole grid. ``x`` ``[B, DIN]``; returns ``[B, KE, I]``.

    ``w`` is the flat uint8 view of ``[E, 2I, DIN * bits / 32]`` uint32; scales/biases flat.
    """
    x = np.asarray(x, np.float32).reshape(-1)
    flat_idx = np.asarray(indices).reshape(-1)
    rows_k = flat_idx.size
    act = np.zeros(rows_k * inter, dtype=np.float32)
    for bj in range(rows_k):                       # threadgroup_position_in_grid.y
        for tg in range(inter // nsg):              # threadgroup_position_in_grid.x
            for sg in range(nsg):                   # simdgroup_index_in_threadgroup
                c = tg * nsg + sg
                e = int(flat_idx[bj])
                row = e * 2 * inter + 2 * c
                lanes = []
                for lane in range(SIMD):
                    acc = [np.float32(0.0), np.float32(0.0)]
                    rows_dot(w, row * (din * bits // 8), scales, row * (din // gs), biases,
                             row * (din // gs), x, (bj // ke) * din, lane, acc,
                             bits=bits, gs=gs, k=din, r_rows=2)
                    lanes.append(acc)
                g = np.float32(sum(a[0] for a in lanes))  # simd_sum
                u = np.float32(sum(a[1] for a in lanes))
                s = np.float32(1.0)                       # HAS_GS is false for affine
                gate = np.float32(T(g * s))
                up = np.float32(T(u * s))
                y = np.float32(1.0 / (1.0 + np.exp(np.float32(abs(gate)))))
                sig = y if gate < 0 else np.float32(1.0 - y)
                gc = np.float32(np.tanh(gate / caps[0]) * caps[0])
                uc = np.float32(np.tanh(up / caps[1]) * caps[1])
                act[bj * inter + c] = T(gc * sig * uc)
    return act.reshape(*np.shape(indices), inter)


def epilogue_tail(m, row, d, *, D, state, conv_weight, divisor, residual, hidden, state_out, T):
    """``expertEpilogueTail`` for one ``(row, d)``."""
    mb = np.float32(T(m))
    sb = row * 3 * D + d
    s0, s1, s2 = (np.float32(state[sb]), np.float32(state[sb + D]), np.float32(state[sb + 2 * D]))
    cw = conv_weight[d * 4:d * 4 + 4]
    a = np.float32(s0 * cw[0] + s1 * cw[1] + s2 * cw[2] + mb * cw[3])
    y = np.float32(T((np.float32(T(a)) + mb) / divisor))
    i = row * D + d
    hidden[i] = T(residual[i] + y)
    state_out[sb] = state[sb + D]
    state_out[sb + D] = state[sb + 2 * D]
    state_out[sb + 2 * D] = mb


def kernel_down(act, w, scales, biases, indices, weights, residual, state, conv_weight, *,
                bits, gs, k, d_out, ke, rows_tile=2, divisor=np.float32(np.sqrt(5.0)), T=bf16):
    """``inkling2_expert_down`` over its whole grid. Returns ``(hidden [B, D], state [B, 3, D])``."""
    act = np.asarray(act, np.float32).reshape(-1)
    flat_idx = np.asarray(indices).reshape(-1)
    weights = np.asarray(weights, np.float32).reshape(-1)
    residual = np.asarray(residual, np.float32).reshape(-1)
    state = np.asarray(state, np.float32).reshape(-1)
    conv_weight = np.asarray(conv_weight, np.float32).reshape(-1)
    n_rows = flat_idx.size // ke
    hidden = np.zeros(n_rows * d_out, np.float32)
    state_out = np.zeros(n_rows * 3 * d_out, np.float32)
    for row in range(n_rows):                          # threadgroup_position_in_grid.y
        for tgx in range(d_out // rows_tile):          # threadgroup_position_in_grid.x
            d0 = tgx * rows_tile
            outputs = np.zeros(ke * rows_tile, np.float32)  # threadgroup memory
            for j in range(ke):                        # simdgroup_index_in_threadgroup
                e = int(flat_idx[row * ke + j])
                s = np.float32(1.0)
                for r in range(rows_tile):
                    n = e * d_out + d0 + r
                    lanes = []
                    for lane in range(SIMD):
                        acc = [np.float32(0.0)]
                        rows_dot(w, n * (k * bits // 8), scales, n * (k // gs), biases,
                                 n * (k // gs), act, (row * ke + j) * k, lane, acc,
                                 bits=bits, gs=gs, k=k, r_rows=1)
                        lanes.append(acc[0])
                    o = np.float32(sum(lanes))
                    outputs[j * rows_tile + r] = np.float32(T(o * s))
            for t in range(rows_tile):                 # threads past ROWS return
                d = d0 + t
                m = np.float32(0.0)
                for kk in range(ke):
                    m = np.float32(m + outputs[kk * rows_tile + t] * weights[row * ke + kk])
                epilogue_tail(m, row, d, D=d_out, state=state, conv_weight=conv_weight,
                              divisor=divisor, residual=residual, hidden=hidden,
                              state_out=state_out, T=T)
    return hidden.reshape(n_rows, d_out), state_out.reshape(n_rows, 3, d_out)


# ------------------------------------------------------------------ the plain math


def dense_reference(x, gate_up_w, down_w, indices, weights, residual, state, conv_weight, caps,
                    divisor=np.sqrt(5.0), T=bf16):
    """The same layer from dense (dequantized) weights: ``gate_up_w`` ``[E, 2I, D]`` with gate and
    up rows alternating, ``down_w`` ``[E, D, I]``. Rounds to ``T`` where the kernels do."""
    x = np.asarray(x, np.float64)
    b, ke = np.shape(indices)
    d = down_w.shape[1]
    hidden = np.zeros((b, d), np.float32)
    state_out = np.zeros((b, 3, d), np.float32)
    acts = np.zeros((b, ke, gate_up_w.shape[1] // 2), np.float32)
    for row in range(b):
        outs = []
        for j in range(ke):
            e = int(indices[row, j])
            gu = gate_up_w[e] @ x[row]
            gate, up = T(gu[0::2].astype(np.float32)), T(gu[1::2].astype(np.float32))
            sig = 1.0 / (1.0 + np.exp(-gate.astype(np.float64)))
            act = T((np.tanh(gate / caps[0]) * caps[0] * sig * np.tanh(up / caps[1]) * caps[1])
                    .astype(np.float32))
            acts[row, j] = act
            outs.append(T((down_w[e] @ act.astype(np.float64)).astype(np.float32)))
        m = sum(np.float64(weights[row, j]) * outs[j].astype(np.float64) for j in range(ke))
        mb = T(m.astype(np.float32))
        st = np.asarray(state[row], np.float64)
        cw = np.asarray(conv_weight, np.float64).reshape(d, 4)
        a = st[0] * cw[:, 0] + st[1] * cw[:, 1] + st[2] * cw[:, 2] + mb * cw[:, 3]
        y = T(((T(a.astype(np.float32)).astype(np.float64) + mb) / divisor).astype(np.float32))
        hidden[row] = T((np.asarray(residual[row], np.float64) + y).astype(np.float32))
        state_out[row, 0], state_out[row, 1], state_out[row, 2] = state[row, 1], state[row, 2], mb
    return acts, hidden, state_out
