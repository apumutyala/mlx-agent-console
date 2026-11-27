// [agent-console patch D] LookAtMyQuant ("lamq") packed Inkling 2 Mini: Metal kernels.
// Not part of mlx-swift-lm PR #821. Format: docs/LAMQ-FORMAT-SPEC.md. The format follows
// LookAtMyQuant (MIT, (c) 2026 Shin Megami Boson); this code is our own implementation.
//
// The index math below is ported line for line into ios/tools/lamq/test_lamq_ref.py
// (`metal_state`, `metal_index`, `metal_dequant`, `metal_matvec`, `metal_lm_dequant`), which
// checks it against the reference decoder. `ios/tools/kernels/check_lamq_metal_kernels.py`
// rebuilds the source MLX generates from these strings and compiles it with `xcrun metal`.
//
// Kernels:
// - `lamq_trellis_dequant`: trellis rows -> dense `[rows, N]` (float32 for checks, float16 for
//   prefill tiles). Rows are gathered through `locs` (row groups of `params[0]` rows each).
// - `lamq_trellis_matvec`: fused decode matvec, one simdgroup per output row, rows selected per
//   slot and row group through `sel` (0xFFFFFFFF = row group belongs to another format: write 0).
// - `lamq_lm_dequant` / `lamq_lm_matvec`: the same for Lloyd-Max `lm6` / `lm8`.

import Foundation
import MLX

let lamqHeader = """
    // 12-bit trellis state whose window starts at bit `bp` of the row's little-endian bitstream.
    // Bits past the row's last word read as zero.
    inline uint lamq_state(const device uint32_t* rc, uint wpr, uint bp) {
        uint wi = bp >> 5;
        uint off = bp & 31u;
        uint lo = rc[wi];
        uint hi = (wi + 1u < wpr) ? rc[wi + 1u] : 0u;
        uint s = (off == 0u) ? lo : ((lo >> off) | (hi << (32u - off)));
        return s & 4095u;
    }

    // Table index of a state: itself (rtr*), or the top LOGK bits of state * 0x9E3779B1 mod 2^32 (ht*).
    template <int LOGK, bool HASHED>
    inline uint lamq_index(uint s) {
        if (HASHED) {
            return (s * 0x9E3779B1u) >> uint(32 - LOGK);
        }
        return s;
    }

    """

// Grid (N / G, rows, 1). One thread decodes one state group (G weights) of one output row.
// params[0] = rows per row group (RG). Output row `row` is row `row % RG` of slab entry
// `locs[row / RG]`.
let lamqTrellisDequantSource = """
        constexpr uint WPR = uint(N * KB / 32);
        constexpr uint NG = uint(N / G);
        uint gi = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (gi >= NG) {
            return;
        }
        uint rg_rows = params[0];
        uint rg = row / rg_rows;
        uint rr = row - rg * rg_rows;
        ulong base = ulong(locs[rg]) * rg_rows + rr;
        const device uint32_t* rc = codes + base * WPR;
        uint last = gi * uint(G) + uint(G) - 1u;
        uint s = lamq_state(rc, WPR, (uint(N) - 1u - last) * uint(KB));
        uint idx = lamq_index<LOGK, HASHED>(s);
        float sc = float(scales[base]);
        ulong o = ulong(row) * uint(N) + gi * uint(G);
        for (int t = 0; t < G; t++) {
            out[o + t] = T(float(lut[idx * uint(G) + uint(t)]) * sc);
        }
    """

// Threadgroups (ceil(M / NSG), slots); NSG simdgroups of 32 lanes each, one output row per
// simdgroup. params = [RG, n_rg, x_div]; M = RG * n_rg. Slot `slot` reads activation row
// `slot / x_div` (8 expert slots per token on gate/up, 1 on down) and slab entry
// `sel[slot, row / RG]` for its row group.
let lamqTrellisMatvecSource = """
        constexpr uint WPR = uint(N * KB / 32);
        constexpr uint NG = uint(N / G);
        constexpr uint LUTN = uint((1 << LOGK) * G);
        threadgroup half tlut[LUTN];
        uint tid = thread_index_in_threadgroup;
        for (uint i = tid; i < LUTN; i += uint(NSG) * 32u) {
            tlut[i] = lut[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint rg_rows = params[0];
        uint n_rg = params[1];
        uint x_div = params[2];
        uint m = rg_rows * n_rg;
        uint slot = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.x * uint(NSG) + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        if (row >= m) {
            return;
        }
        uint rg = row / rg_rows;
        uint loc = sel[ulong(slot) * n_rg + rg];
        ulong yi = ulong(slot) * m + row;
        if (loc == 0xFFFFFFFFu) {
            if (lane == 0u) {
                y[yi] = 0.0f;
            }
            return;
        }
        ulong base = ulong(loc) * rg_rows + (row - rg * rg_rows);
        const device uint32_t* rc = codes + base * WPR;
        const device float* xs = x + ulong(slot / x_div) * uint(N);
        float acc = 0.0f;
        for (uint gi = lane; gi < NG; gi += 32u) {
            uint last = gi * uint(G) + uint(G) - 1u;
            uint s = lamq_state(rc, WPR, (uint(N) - 1u - last) * uint(KB));
            uint idx = lamq_index<LOGK, HASHED>(s);
            for (int t = 0; t < G; t++) {
                acc += float(tlut[idx * uint(G) + uint(t)]) * xs[gi * uint(G) + uint(t)];
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            y[yi] = acc * float(scales[base]);
        }
    """

// Grid (N / 4, rows, 1). One thread decodes 4 consecutive weights of row `rows_sel[row]`.
let lamqLMDequantSource = """
        constexpr uint BPR = uint(N * KB / 8);
        uint q = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (q >= uint(N / 4)) {
            return;
        }
        uint src = rows_sel[row];
        const device uint8_t* p = codes + ulong(src) * BPR;
        uint i0, i1, i2, i3;
        if (KB == 8) {
            p += 4u * q;
            i0 = p[0];
            i1 = p[1];
            i2 = p[2];
            i3 = p[3];
        } else {
            p += 3u * q;
            uint v = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
            i0 = v & 63u;
            i1 = (v >> 6) & 63u;
            i2 = (v >> 12) & 63u;
            i3 = (v >> 18) & 63u;
        }
        float sc = float(scales[src]);
        ulong o = ulong(row) * uint(N) + 4u * q;
        out[o] = T(levels[i0] * sc);
        out[o + 1] = T(levels[i1] * sc);
        out[o + 2] = T(levels[i2] * sc);
        out[o + 3] = T(levels[i3] * sc);
    """

// Threadgroups (ceil(M / NSG), T); one output row per simdgroup. params[0] = M.
let lamqLMMatvecSource = """
        constexpr uint BPR = uint(N * KB / 8);
        constexpr uint NQ = uint(N / 4);
        constexpr uint NLEV = uint(1 << KB);
        threadgroup float tlev[NLEV];
        uint tid = thread_index_in_threadgroup;
        for (uint i = tid; i < NLEV; i += uint(NSG) * 32u) {
            tlev[i] = levels[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint m = params[0];
        uint tok = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.x * uint(NSG) + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        if (row >= m) {
            return;
        }
        const device uint8_t* rowp = codes + ulong(row) * BPR;
        const device float* xs = x + ulong(tok) * uint(N);
        float acc = 0.0f;
        for (uint q = lane; q < NQ; q += 32u) {
            uint i0, i1, i2, i3;
            if (KB == 8) {
                const device uint8_t* p = rowp + 4u * q;
                i0 = p[0];
                i1 = p[1];
                i2 = p[2];
                i3 = p[3];
            } else {
                const device uint8_t* p = rowp + 3u * q;
                uint v = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
                i0 = v & 63u;
                i1 = (v >> 6) & 63u;
                i2 = (v >> 12) & 63u;
                i3 = (v >> 18) & 63u;
            }
            uint c = 4u * q;
            acc += tlev[i0] * xs[c] + tlev[i1] * xs[c + 1] + tlev[i2] * xs[c + 2] + tlev[i3] * xs[c + 3];
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            y[ulong(tok) * m + row] = acc * float(scales[row]);
        }
    """

/// Simdgroups per threadgroup in the matvec kernels (rows per threadgroup).
let lamqMatvecSimdgroups = 8

final class InklingLamqKernels: Sendable {
    static let shared = InklingLamqKernels()

    let trellisDequant: MLXFast.MLXFastKernel
    let trellisMatvec: MLXFast.MLXFastKernel
    let lmDequant: MLXFast.MLXFastKernel
    let lmMatvec: MLXFast.MLXFastKernel

    private init() {
        trellisDequant = MLXFast.metalKernel(
            name: "lamq_trellis_dequant",
            inputNames: ["codes", "scales", "lut", "locs", "params"],
            outputNames: ["out"],
            source: lamqTrellisDequantSource, header: lamqHeader)
        trellisMatvec = MLXFast.metalKernel(
            name: "lamq_trellis_matvec",
            inputNames: ["x", "codes", "scales", "lut", "sel", "params"],
            outputNames: ["y"],
            source: lamqTrellisMatvecSource, header: lamqHeader)
        lmDequant = MLXFast.metalKernel(
            name: "lamq_lm_dequant",
            inputNames: ["codes", "scales", "levels", "rows_sel"],
            outputNames: ["out"],
            source: lamqLMDequantSource)
        lmMatvec = MLXFast.metalKernel(
            name: "lamq_lm_matvec",
            inputNames: ["x", "codes", "scales", "levels", "params"],
            outputNames: ["y"],
            source: lamqLMMatvecSource)
    }
}

/// The template arguments of a trellis format, in the kernels' order.
private func trellisTemplate(_ format: InklingLamqTrellisFormat, n: Int) -> [(String, any KernelTemplateArg)] {
    [("KB", format.k), ("G", format.g), ("LOGK", format.log2K), ("HASHED", format.hashed), ("N", n)]
}

/// Dense `Q * scale` rows of a trellis slab (rotated basis).
///
/// - Parameters:
///   - codes: uint32 `[entries * rowsPerGroup, n * k / 32]` (any shape with that many words).
///   - scales: float16 `[entries * rowsPerGroup]`.
///   - lut: float16 `[K * g]`.
///   - locs: uint32 `[groups]`, the slab entry of each output row group.
///   - rowsPerGroup: rows per slab entry (128 for expert slabs; `m` for one dense matrix).
///   - dtype: `.float32` (bit-exact checks) or `.float16` (prefill tiles).
/// - Returns: `[groups * rowsPerGroup, n]`.
func lamqTrellisDequant(
    codes: MLXArray, scales: MLXArray, lut: MLXArray, locs: MLXArray,
    rowsPerGroup: Int, n: Int, format: InklingLamqTrellisFormat, dtype: DType = .float32
) -> MLXArray {
    let rows = locs.dim(0) * rowsPerGroup
    let groups = n / format.g
    return InklingLamqKernels.shared.trellisDequant(
        [codes, scales, lut, locs, MLXArray([UInt32(rowsPerGroup)])],
        template: trellisTemplate(format, n: n) + [("T", dtype)],
        grid: (groups, rows, 1),
        threadGroup: (min(256, groups), 1, 1),
        outputShapes: [[rows, n]],
        outputDTypes: [dtype]
    )[0]
}

/// Fused decode matvec: `y[slot, row] = Q_sel(slot)[row] . x[slot / xDiv] * scale`, float32.
///
/// - Parameters:
///   - x: float32 `[slots / xDiv, n]`, rotated activations.
///   - sel: uint32 `[slots, groups]`; 0xFFFFFFFF marks a row group of another format (output 0).
/// - Returns: float32 `[slots, groups * rowsPerGroup]`.
func lamqTrellisMatvec(
    x: MLXArray, codes: MLXArray, scales: MLXArray, lut: MLXArray, sel: MLXArray,
    rowsPerGroup: Int, xDiv: Int, n: Int, format: InklingLamqTrellisFormat
) -> MLXArray {
    let slots = sel.dim(0)
    let groups = sel.dim(1)
    let m = groups * rowsPerGroup
    let nsg = lamqMatvecSimdgroups
    return InklingLamqKernels.shared.trellisMatvec(
        [x.asType(.float32), codes, scales, lut, sel,
         MLXArray([UInt32(rowsPerGroup), UInt32(groups), UInt32(xDiv)])],
        template: trellisTemplate(format, n: n) + [("NSG", nsg)],
        grid: ((m + nsg - 1) / nsg * nsg * 32, slots, 1),
        threadGroup: (nsg * 32, 1, 1),
        outputShapes: [[slots, m]],
        outputDTypes: [.float32]
    )[0]
}

/// Dense `levels[idx] * scale` rows `rowsSel` of a Lloyd-Max matrix `[rows, n]`.
func lamqLMDequant(
    codes: MLXArray, scales: MLXArray, levels: MLXArray, rowsSel: MLXArray, n: Int, k: Int,
    dtype: DType = .float32
) -> MLXArray {
    let rows = rowsSel.dim(0)
    return InklingLamqKernels.shared.lmDequant(
        [codes, scales, levels, rowsSel],
        template: [("KB", k), ("N", n), ("T", dtype)],
        grid: (n / 4, rows, 1),
        threadGroup: (min(256, n / 4), 1, 1),
        outputShapes: [[rows, n]],
        outputDTypes: [dtype]
    )[0]
}

/// `y[t, row] = (levels[codes[row]] . x[t]) * scale[row]` for a Lloyd-Max matrix `[m, n]`, float32.
func lamqLMMatvec(x: MLXArray, codes: MLXArray, scales: MLXArray, levels: MLXArray, m: Int, n: Int, k: Int)
    -> MLXArray
{
    let t = x.dim(0)
    let nsg = lamqMatvecSimdgroups
    return InklingLamqKernels.shared.lmMatvec(
        [x.asType(.float32), codes, scales, levels, MLXArray([UInt32(m)])],
        template: [("KB", k), ("N", n), ("NSG", nsg)],
        grid: ((m + nsg - 1) / nsg * nsg * 32, t, 1),
        threadGroup: (nsg * 32, 1, 1),
        outputShapes: [[t, m]],
        outputDTypes: [.float32]
    )[0]
}
