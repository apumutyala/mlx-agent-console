"""Unit tests for the Inkling 2 fused-expert NumPy reference.

Run: ``cd ios/tools/kernels && python -m unittest -v test_inkling2_expert_ref`` (numpy only).
"""

import unittest

import numpy as np

import inkling2_expert_ref as ref


def hand_pack(codes, bits):
    """MLX's layout written out by hand: 2-bit four codes per byte, 3-bit eight per 3 bytes,
    the first code in the lowest bits of the first byte; returned as uint8."""
    out = []
    if bits == 2:
        for i in range(0, len(codes), 4):
            out.append(codes[i] | codes[i + 1] << 2 | codes[i + 2] << 4 | codes[i + 3] << 6)
    elif bits == 3:
        for i in range(0, len(codes), 8):
            c = codes[i:i + 8]
            out.append((c[0] | c[1] << 3 | c[2] << 6) & 0xFF)
            out.append((c[2] >> 2 | c[3] << 1 | c[4] << 4 | c[5] << 7) & 0xFF)
            out.append((c[5] >> 1 | c[6] << 2 | c[7] << 5) & 0xFF)
    else:
        raise ValueError(bits)
    return np.array(out, dtype=np.uint8)


class PackingTests(unittest.TestCase):
    def test_hand_made_bytes_3bit(self):
        codes = list(range(8)) + [7, 0, 5, 2, 3, 6, 1, 4]
        packed = hand_pack(codes, 3)
        # 0..7 in 3-bit little-endian: 0b111_110_101_100_011_010_001_000 = 0xFAC688
        self.assertEqual(list(packed[:3]), [0x88, 0xC6, 0xFA])
        x = np.arange(1, 17, dtype=np.float32)
        self.assertEqual(ref.unit_dot(packed, 0, x, 3), float(np.dot(codes, x)))
        self.assertEqual(list(ref.unpack_bitstream(packed[None], 3, 16)[0]), codes)

    def test_hand_made_bytes_2bit(self):
        codes = [0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 0, 0, 3]
        packed = hand_pack(codes, 2)
        self.assertEqual(packed[0], 0b11100100)
        x = np.linspace(-2, 3, 16).astype(np.float32)
        self.assertAlmostEqual(ref.unit_dot(packed, 0, x, 2), float(np.dot(codes, x)), places=5)
        self.assertEqual(list(ref.unpack_bitstream(packed[None], 2, 16)[0]), codes)

    def test_quantize_packs_like_the_hand_layout(self):
        """The bytes ``quantize`` writes are the hand layout of the codes it chose. MLX picks a
        negative scale when |max| >= |min|, so the codes are recomputed from its scale/bias."""
        rng = np.random.default_rng(0)
        for bits in (2, 3):
            for gs in (64, 128):
                w = rng.standard_normal((3, 256)).astype(np.float32)
                packed, scales, biases = ref.mlx_quantize(w, bits, gs, scale_round=ref.identity)
                self.assertEqual(packed.shape, (3, 256 * bits // 32))
                s = np.repeat(scales, gs, axis=-1)
                b = np.repeat(biases, gs, axis=-1)
                codes = np.clip(np.rint((w - b) / s), 0, (1 << bits) - 1).astype(int)
                raw = packed.view(np.uint8)
                for r in range(3):
                    np.testing.assert_array_equal(raw[r], hand_pack(list(codes[r]), bits))
                np.testing.assert_allclose(
                    ref.dequantize_bitstream(packed, scales, biases, bits, gs, 256),
                    codes * s + b, rtol=1e-6, atol=1e-6)

    def test_round_trip_error_is_small(self):
        """MLX snaps one end of each group onto the grid, so the other end may clip: bound the
        mean error, not the maximum."""
        rng = np.random.default_rng(1)
        for bits in (2, 3, 4, 8):
            w = rng.standard_normal((4, 512)).astype(np.float32)
            packed, scales, biases = ref.mlx_quantize(w, bits, 64, scale_round=ref.identity)
            deq = ref.dequantize_bitstream(packed, scales, biases, bits, 64, 512)
            step = np.repeat(np.abs(scales), 64, axis=-1)
            self.assertLess(float(np.mean(np.abs(deq - w) / step)), 0.35, bits)

    def test_a_wrong_3bit_layout_is_caught(self):
        """Mutation check: decoding 3-bit as if each byte held whole codes disagrees."""
        codes = [5, 3, 7, 1, 6, 2, 4, 0] * 2
        packed = hand_pack(codes, 3)
        x = np.ones(16, np.float32)
        wrong = sum(float((b >> s) & 7) for b in packed for s in (0, 3))
        self.assertNotEqual(wrong, ref.unit_dot(packed, 0, x, 3))


class FusedKernelTests(unittest.TestCase):
    """The ported Metal kernels equal the dense math, for uniform and mixed bits per projection."""

    def setUp(self):
        self.caps = np.array([4.0, 25.0], np.float32)  # Inkling 2 Mini's mlp gate/up softcaps

    def layer(self, rng, *, d, inter, experts, rows, ke, gate_up_bits, down_bits, gs):
        gu_w = (rng.standard_normal((experts, 2 * inter, d)) * 0.05).astype(np.float32)
        dn_w = (rng.standard_normal((experts, d, inter)) * 0.05).astype(np.float32)
        gu = ref.mlx_quantize(gu_w, gate_up_bits, gs)
        dn = ref.mlx_quantize(dn_w, down_bits, gs)
        x = ref.bf16(rng.standard_normal((rows, d)).astype(np.float32))
        indices = np.stack([rng.choice(experts, ke, replace=False) for _ in range(rows)]).astype(np.uint32)
        weights = rng.random((rows, ke)).astype(np.float32)
        residual = rng.standard_normal((rows, d)).astype(np.float32)
        state = ref.bf16(rng.standard_normal((rows, 3, d)).astype(np.float32))
        conv = ref.bf16((rng.standard_normal((d, 4)) * 0.5).astype(np.float32))
        return gu, dn, x, indices, weights, residual, state, conv

    def run_case(self, *, gate_up_bits, down_bits, gs, d=256, inter=128, experts=6, rows=2, ke=4,
                 T=ref.bf16, seed=0):
        rng = np.random.default_rng(seed)
        gu, dn, x, indices, weights, residual, state, conv = self.layer(
            rng, d=d, inter=inter, experts=experts, rows=rows, ke=ke,
            gate_up_bits=gate_up_bits, down_bits=down_bits, gs=gs)
        act = ref.kernel_gate_up(
            x, gu[0].view(np.uint8).reshape(-1), gu[1].reshape(-1), gu[2].reshape(-1), indices,
            self.caps, bits=gate_up_bits, gs=gs, din=d, inter=inter, ke=ke, T=T)
        hidden, state_out = ref.kernel_down(
            act, dn[0].view(np.uint8).reshape(-1), dn[1].reshape(-1), dn[2].reshape(-1), indices,
            weights, residual, state, conv, bits=down_bits, gs=gs, k=inter, d_out=d, ke=ke, T=T)
        gu_dense = ref.dequantize_bitstream(*gu, gate_up_bits, gs, d)
        dn_dense = ref.dequantize_bitstream(*dn, down_bits, gs, inter)
        r_act, r_hidden, r_state = ref.dense_reference(
            x, gu_dense, dn_dense, indices, weights, residual, state, conv, self.caps, T=T)
        return (act, hidden, state_out), (r_act, r_hidden, r_state)

    def assert_close(self, got, want, *, exact):
        names = ("act", "hidden", "state_out")
        for name, g, w in zip(names, got, want):
            scale = float(np.max(np.abs(w))) or 1.0
            err = float(np.max(np.abs(g - w)))
            # Without intermediate rounding only float32 summation order differs. With bf16
            # rounding, an intermediate can land on the other side of a rounding boundary.
            tol = 2e-5 * scale if exact else 2e-2 * scale
            self.assertLessEqual(err, tol, f"{name}: max |diff| {err:.3g} > {tol:.3g}")

    def test_cases_without_intermediate_rounding(self):
        """With T = identity the only difference is float summation order: an indexing slip
        (wrong row, group, byte offset or lane split) would be orders of magnitude off."""
        for gub, dnb, gs in [(2, 2, 64), (3, 3, 64), (2, 3, 64), (3, 2, 64), (2, 3, 128),
                             (4, 4, 64), (8, 8, 128)]:
            with self.subTest(gate_up_bits=gub, down_bits=dnb, gs=gs):
                got, want = self.run_case(gate_up_bits=gub, down_bits=dnb, gs=gs, T=ref.identity)
                self.assert_close(got, want, exact=True)

    def test_cases_with_bf16_rounding(self):
        for gub, dnb, gs in [(2, 3, 64), (3, 3, 128)]:
            with self.subTest(gate_up_bits=gub, down_bits=dnb, gs=gs):
                got, want = self.run_case(gate_up_bits=gub, down_bits=dnb, gs=gs, seed=3)
                self.assert_close(got, want, exact=False)

    def test_odd_lane_split(self):
        """K / 16 not a multiple of 32: lanes get unequal unit counts (DIN 2304 -> 144 units on
        Inkling 2 Mini). A 2304-like split at small size: D = 416 -> 26 units, I = 48 -> 3 units (group size 16 is not
        an MLX affine size; it only exercises the group index)."""
        got, want = self.run_case(gate_up_bits=3, down_bits=2, gs=16, d=416, inter=48,
                                  T=ref.identity, experts=5, ke=3, rows=1)
        self.assert_close(got, want, exact=True)


if __name__ == "__main__":
    unittest.main()
