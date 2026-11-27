# LookAtMyQuant ("lamq") packed Inkling 2 Mini: format spec and iPad plan

Status: 2026-11-27, branch `inkling-lamq`.
- The spec below is verified bit-for-bit against LookAtMyQuant's own decoders on real files.
- The Metal kernels are written but not yet compiled (CI freeze) and have not run on a device.
- **The 3.9 GB build is wired into PR #821's Inkling 2 forward pass and is pickable** (fully resident). This is compile-unverified; see §8a.
- The 4.7 and 5.4 GB builds wait on PR #9's slot cache (§11).

## Sources and licences

| Item | Where | Licence | Notes |
|---|---|---|---|
| lamq runtime and compiler | github.com/ShinMegamiBoson/lookatmyquant @ `main` (tarball, 2026-11-27; `pyproject` version 0.3.0) | **MIT**, © 2026 Shin Megami Boson (root `LICENSE`, plus `runtime_cuda/LICENSE`) | We copy no code. `ios/tools/lamq/lamq_ref.py` and the Metal kernels are our own implementation of the format. The MIT notice is kept in `ios/tools/lamq/LICENSE-lookatmyquant.txt` because the format and the Hadamard construction follow lamq. |
| Packed weights: 3.9 / 4.7 / 5.4 / 8.9 / 11.2 / 12.8 GB repos | huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-* | **None declared.** The card has no `license` field and the repo has no LICENSE file. | The base model, `thinkingmachines/Inkling-2-Mini`, is Apache-2.0. The weights are a derivative and are never redistributed by us: the app downloads them from Hugging Face on the device, like every other model. Asking the author to add a licence would be an upstream action; it is not done. |

Revisions checked:
- 5.4 GB @ `d470c9c15fdccf7380c401c7fd72e291d92c6b0d`
- 4.7 GB @ `d865ec500e2e88da2240ef41a15eb892cb3f1a2f`
- 3.9 GB @ `fbee46a43565d8efac84e89edc196add5f28b254`
- 8.9 GB @ `ce249fc36862904b879988f341b9a596cb7565d2`
- 11.2 GB @ `7191658901060d7c80871eb27413010c3e563b8c`

## 1. Files

Every build has 62 files:
- `layer_00_packed.npz` … `layer_23_packed.npz`, each with a `layer_XX_meta.json`
- `embed_packed.npz` + `embed_meta.json`
- `head_packed.npz` + `head_meta.json`
- `luts.npz`
- `steering.safetensors`
- `manifest.json`
- `config.json`
- tokenizer and chat template files

`manifest.json` has:
- `"format": "inkling2-packed-v1-sqyfr"`
- `"subformat": "rg128-f16signs-v3"`, on the 3.9, 4.7 and 5.4 GB builds only
- `fmt_ids`: `{ht1:0, ht2:1, ht3:2, rtr1:3, rtr2:4, rtr3:5, rtr4:6, nvfp4:7}`

**npz.** Every `.npz` is a ZIP with **STORED** members (compression type 0, no zip64 extras). Each member is a `.npy` file (version 1.0, little-endian). A loader can find a member's payload offset from its local header and read it with `pread` with no decompression (`lamq_ref.npz_members`). Experts can therefore be streamed straight from the downloaded files.

### Build survey (sizes from the HF tree API; formats from all 24 `layer_*_meta.json`)

| Build | Total files GB | Layers GB | Expert formats (row-group counts, gu + dn) | Attention linears | Embed / head buckets | NVFP4 |
|---|---:|---:|---|---|---|---|
| 3.9 GB | 3.848 | 3.488 | rtr1 104,606, ht2 5,986 | ht3 125, rtr4 19 | lm6/rtr1–3 · lm6/rtr2–4 | none |
| 4.7 GB | 4.718 | 4.333 | rtr1 75,863, ht2 33,486, ht3 1,243 | rtr4 108, ht3 31, lm6 5 | lm6/rtr1–4 · lm8/lm6/rtr2–4 | none |
| 5.4 GB | 5.379 | 4.990 | rtr1 55,872, ht2 48,960, ht3 5,760 | rtr4 105, ht3 25, lm6 14 | lm6/rtr1–4 · lm8/lm6/rtr2–4 | none |
| 8.9 GB | 8.877 | 8.382 | per-expert map: ht2, ht3, rtr4, **nvfp4 400** | rtr4, lm6, lm8, ht3 | lm8/lm6/rtr2–4 | yes |
| 11.2 GB | 11.182 | 10.649 | per-expert map: ht3, ht2, rtr4, **nvfp4 1,392** | lm8, lm6, rtr4, ht3 | lm8/lm6/rtr2–4 | yes |

Notes on the survey:
- **NVFP4 passthrough appears only in the 8.9 and 11.2 GB builds.** The 8.9 GB card's "23/192 hottest expert buckets" refers to that build. The three iPad-sized builds use no NVFP4 and are pure trellis plus Lloyd-Max.
- The iPad-sized builds use **row-group maps** (`{gu,dn}_rg_map`, 18 groups of 128 rows). The larger builds use legacy per-expert maps (`{gu,dn}_map`).

## 2. Trellis codes (`ht1..3`, `rtr1..4`)

Parameters per format: `k` bits per weight, `g` weights per state, `K` table entries, and a hash multiplier.

| fmt | k | g | K | hash | LUT (`luts.npz`, float16) |
|---|---|---|---|---|---|
| ht1 | 1 | 4 | 512 | 0x9E3779B1 | not shipped (unused) |
| ht2 | 2 | 4 | 512 | 0x9E3779B1 | `[512, 4]` |
| ht3 | 3 | 2 | 1024 | 0x9E3779B1 | `[1024, 2]` |
| rtr1..rtr4 | 1..4 | 1 | 4096 | none | `[4096]` each |

**Codes and scales.** A matrix `[m, n]` is stored as:
- codes: `uint32 [m, n*k/32]`. Each row is **one little-endian bitstream**: word `i` holds bits `32i…32i+31`.
- scales: one `float16` scale per row, `[m]`.

**State.** For the group of `g` columns `j … j+g-1`, let `last = j+g-1`. The 12-bit state is `(B >> ((n-1-last)·k)) & 0xFFF`, where `B` is the row's bitstream read as one integer.
- Bits past the end of the row read as zero.
- Column `n-1` sits at bit 0. The first column reads the highest window.
- Consecutive groups share `12 - g·k` bits. That overlap is the trellis.

**Index.**
- `rtr*`: `idx = state`.
- `ht*`: `idx = (state · 0x9E3779B1 mod 2^32) >> (32 - log2 K)`.

**Value.** `W[r, j+t] = lut[idx, t] · scale[r]`, for `t < g`. Computed in float32, this product of two float16 numbers is **exact**, so every correct decoder produces identical bits. The repo's SHA-256 receipts rely on this.

**Basis.** `W` here is `Q`, in the *rotated* basis (see §5).

## 3. Lloyd-Max codes (`lm6`, `lm8`)

These are used by some attention linears (for example `v_proj` in the 5.4 GB build) and by embed/head buckets.
- Codes are row-major `uint8`, one index per weight:
  - `lm8`: one byte per weight.
  - `lm6`: 4 indices in 3 bytes. Read the bytes as a little-endian 24-bit word `v`; index `t` is `(v >> 6t) & 63`.
- Value: `levels[idx] · scale[row]`.
  - In layers, `levels` is float32 `[64 or 256]`.
  - In buckets, `levels` is float32, and lamq's fast MLX path casts it to float16.

## 4. NVFP4 experts (format id 7; 8.9 and 11.2 GB builds only)

- `{side}_nvfp4_codes`: `uint8 [E', m, n/2]`, E2M1 nibbles, with the low nibble for the even column.
- `{side}_nvfp4_scales`: `uint8 [E', m, n/16]`, E4M3 block scales. The NaN codes 0x7F and 0xFF read as 0.
- `{side}_nvfp4_outer`: `float32 [E']`.
- Value: `e2m1 · e4m3(block) · outer`.
- **NVFP4 experts multiply the unrotated activation.** They are lossless copies of the original NVFP4 experts.
- lamq rejects NVFP4 inside a row-group map. So do we.

## 5. Randomized Hadamard rotations

For a vector length `n`, `H = kron(A, B)` is chosen as follows:
- `2^a` is the largest power of two dividing `n` whose cofactor `p` is 1, 2, or `q+1` for a prime `q ≡ 3 (mod 4)`.
- `A = Sylvester(2^a)`.
- `B = Paley-I(q)`, or `[[1,1],[1,-1]]` when `p = 2`.
- For a pure power of two, the Sylvester order is split into two halves.

Factor shapes used by Inkling:

| n | 2304 | 1152 | 3072 | 1024 | 384 | 128 |
|---|---|---|---|---|---|---|
| A × B | 32 × 72 | 16 × 72 | 256 × 12 | 32 × 32 | 32 × 12 | 8 × 16 |

**Rotation.** `R(s) = H · diag(s) / √n`. The "signs" `s` are **float16 values near ±1**, for example 0.982…1.018 in magnitude. The polished per-column input scales are folded into them ("f16signs"). Output signs (`signs_out`) are int8 ±1.
- `rht_in(x, s) = x @ R(s).T`. Reshape `x·s` to `[A, B]`, then compute `A @ X @ B.T / √n`.
- `rht_out(y, s) = y @ R(s)`. Compute `A.T @ Y @ B`, then multiply by `s / √n`.

**Linear.** `y = rht_out(rht_in(x, s_in) @ Q.T, s_out)`. The output rotation applies only when the meta says `"out_rot": true` (the `lm6` `v_proj` layers in the 5.4 GB build). The effective dense weight is `R_out.T · Q · R_in`.

**Embed, head and experts** use only the input rotation, as described in §6–7.

## 6. Layers

`layer_XX_meta.json` has three parts:
- `linears`: maps each name to `{fmt, m, n, k, [g, K, hash_mult], [out_rot], safe}`.
- `experts`: for each side, `{fmt: row-group count}`.
- `n_rg: 18` and `rg_rows: 128`.

**Attention linears.** There are six; their tensors are named `lin_{safe}_{codes|scales|signs_in|[signs_out]|[levels]}`.

| name | m × n |
|---|---|
| `self_attn.q_proj.linear` | 3072 × 2304 |
| `self_attn.k_proj.linear` | 1024 × 2304 |
| `self_attn.v_proj.linear` | 1024 × 2304 |
| `self_attn.relative_position_bias.linear` | 384 × 2304 (24 heads × d_rel 16) |
| `self_attn.o_proj.gate_proj` | 3072 × 2304 |
| `self_attn.o_proj.linear` | 2304 × 3072 |

**Routed experts.** There are 128 per layer, with top-8 routing and no shared expert. They are stored per side:
- `gu`: gate and up concatenated, `[2·1152, 2304]`. Gate is the first half. The activation is `tanh(g/4)·4 · sigmoid(g) · tanh(u/25)·25`.
- `dn`: `[2304, 1152]`.

Each side's rows split into 18 groups of 128 rows. Each group has its own format:
- `{side}_rg_map`: `int16 [128 experts, 18, 2]` = `(fid, loc)`.
- `{side}_{fmt}_codes`: `uint32 [count_fmt, 128, n·k/32]`.
- `{side}_{fmt}_scales`: `float16 [count_fmt, 128]`.

Rows `128·r … 128·r+127` of expert `e` are slab entry `loc` of format `fid`. Each side has one input rotation: `gu_signs_in` `[2304]` and `dn_signs_in` `[1152]`.

**Expert forward:**
```
x_rot = rht_in(x, gu_signs_in)
gate_up = x_rot @ Q_gu[e].T
act = softcapped_swiglu(gate_up)
y_e = rht_in(act, dn_signs_in) @ Q_dn[e].T
out = Σ_e w_e · y_e
```
The routing weights `w_e` come from the sigmoid top-8 router, normalised and multiplied by `route_scale = 8`; the router weights and bias are in `steering.safetensors`.

## 7. Embedding and LM head (`embed_*`, `head_*`)

Vocabulary rows (201,024 embed, 200,058 head) are split into 8 buckets. Each bucket has one format: `b{kb}_{codes, scales, signs_in, [levels]}`.

**Order.** `order[o]` is the token at bucket-ordered row `o`. Buckets are concatenated in meta order.

**Embedding.** `embedding(t) = rht_out(Q_b[r], signs_b)`, where `o = inv_order[t]` and bucket `b` contains row `o` at offset `r`. Then apply `embed_norm` (RMSNorm, weight in steering). Bucket trellis formats in released builds are `rtr*` only (`g = 1`); `ht*` appear only in layers.

**Head.**
- `logits[order[o]] = (Q_b @ rht_in(h, signs_b))[r]`.
- Then apply the soft cap: `30·tanh(z/30)`.
- Greedy argmax can skip the permutation.

## 8. `steering.safetensors`

It holds 499 tensors with HF-transformers names (`model.layers.N.…`): everything that is not packed.
- float32: routers (`mlp.gate.weight [128,2304]`, `e_score_correction_bias`), message-gate forget/gain projections, attention-residual queries, depthwise conv weights.
- bf16: norms, `q_proj.norm.scale`, `relative_position_bias.{norm.scale, proj}`, `sink_proj`, message `down_proj`, dynamic short-conv weights.

lamq loads them into `transformers.Inkling2LoopForCausalLM` with the packed modules substituted (`runtime_cuda/inkling/reference.py`).

**This is not the layout `Inkling2.swift` (mlx-swift-lm PR #821) loads.** That code takes Thinking Machines' `model.llm.*` names or its converted layout (`r_proj`, `gate_proj`, `k_sconv`, …). §8a gives the map.

## 8a. Steering → PR #821 map and wiring (patch D)

The map is a pure rename plus one transpose, with no reparametrisation. Comparing lamq's own forward (`runtime_mlx/inkling/model.py`, `PackedDecoderLayer.__call__`) with PR #821 term by term confirms the same math for each tensor:

| Term | lamq forward | PR #821 |
|---|---|---|
| q clip | `l2_clip(q · q_scale, τ)` | `Inkling2ScaledL2Clip.scale` |
| rel clip | `l2_clip(rel · rel_scale, τ)` | `rel_l2_clip` |
| rel logits | `rel @ rel_proj [16, 1024]` | `rel_logits_proj` |
| sinks | `q_in @ sink_w.T` | `sink_proj` |
| short convs | `[C,1,K]` conv weights | `[C,K,1]` conv weights |
| message gates | `(1 − tanh(f)) h + tanh(g) m` on RMS-normed `[h, m]` | same formula |
| messages order | `message_input.query[layer]` → (q, k, v, mlp) | same order |
| attention-residual queries | `[D]` sliding / `[3, D]` full | same shapes |

| steering (HF) | PR #821 |
|---|---|
| `model.embed_tokens.embed_norm.weight` | `language_model.embed_norm.weight` |
| `model.{norm.weight, final_input.query, message_input.query}` | `language_model.…` (same) |
| `model.layers.N.self_attn.q_proj.norm.scale` | `….self_attn.q_l2_clip.scale` |
| `….self_attn.relative_position_bias.norm.scale` | `….self_attn.rel_l2_clip.scale` |
| `….self_attn.relative_position_bias.proj` | `….self_attn.rel_logits_proj.proj` |
| `….self_attn.{k,v}_proj.sconv.{conv1d.weight, static_weight, weight_proj.weight}` | `….self_attn.{k,v}_sconv.…` |
| `….*.conv1d.weight` `[C,1,K]` f32 | swapped to `[C,K,1]`, **rounded to bf16**. PR #821's decode epilogue reads `const device T* conv_weight` with T = bf16. |
| everything else under `model.layers.N.` | same name |

**Packed parts replace PR #821 modules in `Inkling2.prepareCheckpoint`:**

| lamq | PR #821 module | Replacement |
|---|---|---|
| `self_attn.q_proj.linear` | `q_proj` | `InklingLamqLinearLayer: Linear` (placeholder weight `[m, 0]`) |
| `self_attn.k_proj.linear` | `k_proj` | same |
| `self_attn.v_proj.linear` | `v_proj` | same |
| `self_attn.relative_position_bias.linear` | `r_proj` | same |
| `self_attn.o_proj.gate_proj` | `gate_proj` | same |
| `self_attn.o_proj.linear` | `o_proj` | same |
| experts `gu`/`dn` | `mlp.experts` (made optional) | removed; `Inkling2SparseMoE` routes to `lamqExperts` |
| embed buckets | `embed_tokens` | `InklingLamqEmbeddingLayer: Embedding` |
| head buckets | `lm_head` | `InklingLamqHeadLayer: Linear` (logits over the 200,058 unpadded rows) |

**How the experts run in each path:**
- Decode (`Inkling2SparseMoE.decode`): fused per-slot matvecs, then PR #821's own `inkling2ExpertEpilogue` (weighted sum, post-MLP convolution, residual).
- Prefill and `callAsFunction`: lamq experts combine first, then PR #821's convolution fallback.

**Loading:**
- `VLMModelFactory._load` hands the directory to the model when `manifest.json` says `inkling2-packed-v1…`.
- The only safetensors file is `steering.safetensors`. After the map plus 146 placeholders, the factory's **strict** update (`verify: .all`) checks every remaining parameter.
- `ios/tools/lamq/test_steering_map.py` ports the map and PR #821's parameter tree. Against the real 3.9 GB steering header it finds **0 missing, 0 extra, 0 shape mismatches** over 645 entries.
- The fused-projection fast path is ineligible, since it requires exact `QuantizedLinear`, so PR #821's unfused path runs.

**App side:**
- `ModelCatalog.inkling2MiniLamq39`: pinned revision, `residentGB` 3.85, `headroomGB` 0.9, 4K context, 512-token prefill steps. `MemoryBudget.fit` and the benchmark skip now take a per-entry headroom.
- `MLXBackend.loadLarge` downloads the pinned snapshot with `*.npz` (the factory's patterns omit it), then loads it from the directory.

## 9. How the iPad build consumes it (this branch)

**Decode (T ≤ 8 tokens), fused, no dequantized copies:**
- `lamq_trellis_matvec<KB, G, LOGK, HASHED, N, NSG>`: one simdgroup per output row. Lanes stride over the row's state groups, read 2 words, compute the state and index, look up the threadgroup-resident LUT, multiply by `x_rot`, then `simd_sum` × the row scale.
- `sel [slots, n_rg]` holds each slot's slab entry for each row group, or `0xFFFFFFFF`. The same kernel serves:
  - routed experts (`slots = tokens × 8`; `x_div = 8` for gu, 1 for dn);
  - dense linears (`n_rg = m/128`, `sel = 0…n_rg-1`);
  - head buckets (one group of `rows`).
- It runs once per format present on a side (≤ 3), and the outputs are summed. Rows of other formats are written as 0.
- `lamq_lm_matvec<KB, N, NSG>`: the same scheme for `lm6`/`lm8`.

**Prefill (T > 8):** dequantize a tile to float16 (`lamq_trellis_dequant` / `lamq_lm_dequant`) and run the MLX matmul on rotated activations.
- Experts are processed one at a time over the tokens routed to them. Peak temp is about 10.6 MB per gu tile.
- The head is processed in 16K-row chunks (about 75 MB f16).

**Rotations:** two small MLX matmuls with `A`, `B` (the Swift port of §5).

**Embedding:** rows are decoded on demand. There is no dense 0.9 GB table, unlike lamq's PyTorch reference.

**Code map:**
- `ios/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/InklingLamqKernels.swift`: Metal sources and wrappers.
- `InklingLamqFormat.swift`: formats, Hadamard/RHT, STORED-npz `pread` reader, metas.
- `InklingLamqModules.swift`: packed linear, experts, embedding/head buckets.
- `InklingLamqSelfCheck.swift` + generated `InklingLamqDigests.swift`.
- `InklingLamqModel.swift`: steering map, `Linear`/`Embedding` subclasses, directory detection. Patch D in `Inkling2.swift` and `VLMModelFactory.swift`.
- App: `ModelCatalog.inkling2MiniLamq39` / `lamqBuilds`, `MLXBackend.loadLarge`, `LamqSelfCheck.swift`, `LamqSettingsSection.swift`.

**Prefill caveat:** the expert path dequantizes each routed expert's tiles, about 16 MB of f16 writes per expert per visit. It is correct but likely slow (TTFT). A grouped kernel that reads an expert once for all its tokens (lamq's `trellis_moe_grouped`) is the obvious next step, after a profile.

## 10. Memory and speed on the iPad (estimates, not measured)

The measured per-app limit is about 5.3 GB, with about 5.2 GB available before a load (free Apple ID sideload; see the receipts of 2026-11-27).

**Overhead beyond weights:**

| Component | Estimate |
|---|---|
| KV cache, fp16: 8 global-attention visits × 4 KiB/token, plus 40 sliding visits × 1024 × 4 KiB | ≈ 0.29 GB at 4K, 0.42 GB at 8K |
| Prefill scratch (512-token chunks, one expert tile, head chunk) | ≈ 0.2 GB |
| Metal/runtime/app | ≈ 0.3–0.4 GB |
| **Total** | **≈ 0.8–1.0 GB** |

lamq's own figure on an M5 Max is 6.41 GB peak for 5.34 GB active including a 179 MB drafter, which is consistent.

| Build | Weights | Resident need | Fits 5.3 GB? | Plan |
|---|---:|---:|---|---|
| 3.9 GB | 3.85 GB | ≈ 4.7 GB at 4K context | **yes, ~0.5 GB margin** | fully resident, 4K context budget |
| 4.7 GB | 4.72 GB | ≈ 5.6 GB | no | stream ≈ 0.5–0.7 GB of experts (≈ 12–16% of expert bytes) |
| 5.4 GB | 5.38 GB | ≈ 6.3 GB | no | stream ≈ 1.2–1.4 GB (≈ 25–30% of expert bytes) via the PR #9 slot cache (§11) |

**Decode bytes per token** (5.4 GB build):
- 48 visits × 8 experts × ≈ 1.5 MB ≈ 0.6 GB
- 48 × ≈ 12 MB of attention linears ≈ 0.6 GB
- head ≈ 0.15 GB
- Total ≈ 1.3 GB per token.

**Speed:**
- lamq reaches 66 tok/s on an M5 Max (≈ 86 GB/s effective, 14% of its bandwidth). That means trellis decode is ALU/LUT-bound, not bandwidth-bound.
- Scaling by GPU cores (40 → 10) gives ≈ 16 tok/s for an equally tuned kernel. Our first kernel is simpler.
- **Expect 5–12 tok/s resident on the iPad M5, and less when streaming.** All unmeasured.

## 11. Streaming the remainder (4.7 / 5.4 GB) with PR #9's slot cache

PR #9 (`inkling-stream`) keeps `S` slots inside each layer's expert tensors and `preadv`s misses from `experts.bin`. A lamq expert is 36 row-group slab entries in up to 3 formats per side, so its bytes are not one record.

The 4.7 and 5.4 GB builds are held until the PR #9 owner agrees an interface. PR #9 is not forked.

**What lamq needs from the slot cache.** Bookkeeping stays PR #9's: LRU per physical layer, revisits across the 3 loop passes, GPU sync before overwrite, statistics and trace. Only the record shape and the reader generalise.

1. **Records made of parts.**
   - A lamq expert is up to 36 slab entries (18 row groups × gu/dn). Each entry lives in a per-format slab, with codes `[128, n·k/32]` u32 plus scales `[128]` f16.
   - Interface: `struct ExpertRecordPart { side; format: Int; rowGroup: Int; codes: (file: URL, offset: UInt64, bytes: Int); scales: (file, offset, bytes) }`, and `records(layer:expert:) -> [ExpertRecordPart]`.
   - lamq computes the offsets from the STORED npz headers (`InklingLamqNpz.entries`): `payload + loc · entryBytes`.
2. **Slot storage per (layer, side, format).**
   - The pool owns one MLX slab per format, sized `S × maxEntriesPerExpert(format)`, where the maximum is over that layer's experts (often 18 for a single format; layers are usually uniform per side).
   - It also needs a raw writable pointer to a slab entry (or `preadv` straight into it), as PR #9 already does for `[S, out, packed]`.
3. **Remapped selection.**
   - `slots(for ids: [Int]) -> [Int]`, plus the misses to fill. That is the existing call.
   - lamq then builds `sel[slot, rg] = slotEntry(slot, rg)` on the host from `rg_map` and the slot's base. The kernels are unchanged: `sel` already indexes slab entries.
4. **Prefill.** `ub × top-k ≤ slots` still bounds micro-batches. lamq's prefill path dequantizes per expert, so each expert can be filled, used and released within a step.

Budget: 4.7 GB needs ≈ 0.6 GB streamed (S ≈ 110 of 128 experts per layer); 5.4 GB needs ≈ 1.3 GB (S ≈ 95).

## 12. Verification on this branch

| Check | How | Result |
|---|---|---|
| Decoder == lamq's own decoders, bit for bit | `ios/tools/lamq/crosscheck_lamq_torch.py` (torch 2.15.0 CPU) on 5.4 GB layer 0 and layer 7 (all 6 linears + all 128×2 experts each), all 8 embed buckets, and the Hadamard factors for 6 sizes. Formats covered: ht2, ht3, rtr1, rtr4 and lm6 with `out_rot` in the layers; lm6 and rtr1–4 in the embed buckets. Rotated outputs agree to ≤ 7.3e-7 relative error. | **532/532 tensors bit-identical** (686 s). Digests are in `ios/tools/lamq/lamq_digests.json`. |
| Spec oracles | `test_lamq_ref.py`: big-integer bitstream oracle, Metal index-math emulation (dequant, matvec, lm), hash, lm6 packing, E4M3 table, Hadamard identities, npz `pread` offsets; with `LAMQ_PACKED_DIR`, it re-derives the recorded digests | 19/19 pass locally, including the real-file test |
| Metal compiles | `ios/tools/kernels/check_lamq_metal_kernels.py --compile` (156 instantiations) in CI | sources generate locally; **compile not run (CI freeze)** |
| Metal == reference on device | Settings › Inkling 2 Mini · LookAtMyQuant › self-check. It downloads `layer_07_packed.npz` + meta + `luts.npz` (≈ 190 MB); layer 7 uses every layer format of the build. It checks the LUT bytes, GPU-dequantizes all 6 linears and the experts covering every format, and compares SHA-256 with the digests above. It also checks 4 rows per linear against a scalar Swift decode (bit for bit), fused matvec against dequant+matmul, and the Hadamard round trip. | **not run** |
| Steering map + strict-update coverage | `test_steering_map.py`: literal port of the Swift map and PR #821's parameter tree vs the real 3.9 GB steering header | 0 missing / 0 extra / 0 shape mismatches (4 tests pass) |
| 3.9 GB formats in the bit-exact check | 3.9 GB layer 0 + all 8 head buckets via `crosscheck_lamq_torch.py` | **270/270 tensors bit-identical** (585 s): layer 0, all 6 linears (ht3, rtr4) + 256 expert sides (ht2, rtr1), and all 8 head buckets (lm6, rtr2–4). Digests: `lamq_digests_3.9GB.json` |
| End-to-end generation (3.9 GB) | wired; needs a build and the device | **not run**: compile-unverified, no greedy-agreement gate yet |

## What is left (in order)

1. **Compile.** The coordinator build compiles the Swift and the 156 Metal instantiations.
2. **Device self-check.** Run it on the iPad. The SHA-256 digests must match.
3. **3.9 GB chat** on the iPad, then receipts: tok/s, TTFT, peak footprint at 1K/4K.
4. **Correctness gate.** Greedy-token agreement against lamq's CPU reference (`runtime_cuda.inkling.reference` on a rented box: it needs transformers from git plus about 20 GB RAM) on fixed prompts. Until then, "it answers sensibly" is the only end-to-end signal.
5. **Prefill speed.** Profile it, then add a grouped expert kernel.
6. **Slot-cache adapter** for 4.7 and 5.4 GB, once PR #9's interface is agreed (§11).
