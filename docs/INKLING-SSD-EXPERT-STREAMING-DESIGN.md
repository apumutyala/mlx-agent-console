# SSD expert streaming for Inkling 2 Mini on iPad Pro M5 (MLX Swift): design

Date: 2026-11-26. This is read-only research. No repo files were edited and no weights were downloaded; only HF READMEs, file trees and safetensors headers (via HTTP Range) were fetched. PR copy line numbers refer to `scratchpad\pr\*.swift` (mlx-swift-lm PR #821 @ 5507b3b).

## TL;DR

- **It is feasible, but not through mmap.** The approach that works is the one oMLX ships for Inkling 2 on macOS: a fixed per-layer **slot cache** held inside MLX-allocated expert tensors (`[capacity, out, packed]`). Expert ids are remapped to slot ids. A miss is filled with `F_NOCACHE` `pread`/`preadv` straight into the slot's unified-memory bytes. The PR's existing `gatherQuantizedMM` and fused decode kernels then run unchanged on the slot tensors, plus one small kernel change for NVFP4 global scales.
- **Zero-copy mmap'd MTLBuffers are a dead end on iPadOS.** (a) Metal pages in and wires the *whole* no-copy buffer on first GPU use, so residency is per buffer, not per page (mlx #4249, measured). (b) On an iPhone, Metal over mmap'd file pages re-faulted every token at about 15 s/token, and with residency on it ran out of memory (inkling-ios P6/P7). (c) On iOS the page cache "never warmed" (inkling-ios P9–P14). So file-backed clean memory cannot be relied on as a cache. The resident set has to be explicit dirty memory, which counts toward jetsam, and is sized honestly.
- **Expected results with the PR's ~2.5–3-bit experts** (iPad M5, effective SSD 4 GB/s, assumed 43 ms compute+sync floor; every number here is a model to be measured in M0):
  - 25% resident: about **4.5 GB at about 12 tok/s**.
  - 50% resident: about **6.6 GB at about 18 tok/s**.
  - 4-bit experts at 50% resident: about **9.2 GB at about 16 tok/s**. That is a quality upgrade that fits where today's resident 2/3-bit build (9.2–9.6 GB) barely fits.
- **Measured anchors:**
  - theblackcat102 on an M4 MacBook Air 16 GB: 7.9 tok/s at 4.6 GiB peak with 25% of experts resident, 10.0 tok/s at 6.6 GiB with 50%.
  - oMLX PR #5657 on an M4 Pro, Inkling oQ4e: 95.5% expert hit rate at 50% residency, 97.6% at 60%.
- **The hit rate is everything.** 20 tok/s needs ≥96–98% hits; 15 tok/s needs ≥88–95%. Below 25% residency, decode is miss-bound.
- **Entitlements:**
  - `extended-virtual-addressing` is **not** needed by this design, since there is no whole-file mapping. It is also **not available to free personal teams**: Apple's capability table lists it for ADP and ADEP only.
  - `increased-memory-limit` is not listed in that table at all. Whether a free team can get it is **unverified**, and it decides whether the budget is ~5 GB or ~12 GB. Streaming is exactly what makes the ~5 GB case work.

---

## 1. Prior art (what exists, what it measured)

### 1.1 theblackcat102/Inkling-2-Mini-oQ2.5e-mtp-offload + oMLX
- **Card:** https://huggingface.co/theblackcat102/Inkling-2-Mini-oQ2.5e-mtp-offload
- **Layout.** This is a repack of mlx-community oQ2.5e. Each expert is stored as **separate tensors, back to back**: `model.layers.<i>.mlp.experts.<e>.{w13,w2}.{weight,scales,biases}`. "Expert offload reads a missing expert with one `preadv` straight into its cache slot." The expert bytes are unchanged.
- **What the shard headers show** (I read them via Range requests):
  - `w13.weight` is U32 `[2304,144]`, i.e. 2-bit gate+up interleaved.
  - `w2.weight` is U32 `[2304,108]`, i.e. 3-bit down. Layers 3,4,5,20,22 use 2-bit down.
  - Scales and biases are BF16, group 64.
  - One expert is **2.820 MB** (2.488 MB in the 2-bit-down layers).
  - The routed experts total **8.451 GB**. Non-expert text is 1.35 GB (embed 0.26, head 0.26, other 0.83); vision is 0.36 GB.
  - Tensor offsets are not 16 KiB-aligned (only `w13.weight` happens to be a multiple of 16 KiB in size).
- **Speed** (M4 Air 16 GB, 1,426-token prompt, 48 greedy tokens, no MTP):
  - 25% resident: 7.9 tok/s at 4.6 GiB.
  - 50% resident: 10.0 tok/s at 6.6 GiB.
  - The old stacked layout at 25%: 4.36 tok/s at 5.30 GiB.
  - The card warns of up to 1.7x session-to-session variance from thermals on that fanless machine.
- **oMLX runtime** (https://github.com/jundot/omlx):
  - `omlx/patches/moe_slot_cache.py` (`SlotCache`): replaces each projection's `weight/scales/biases` with `mx.zeros((capacity, ...))` slot tensors. It keeps `slot_of` (LRU dict), `free`, and a device `map[n_experts] -> slot`. `ensure()` does `idx.tolist()` (a host sync), protects every expert in the current call, submits misses to a thread pool (≤512 MiB in flight), and installs them serially.
  - `omlx/patches/inkling2/moe_offload.py` (`OffloadedInkling2Experts`): runs the module's *own* forward on `mx.take(c.map, inds)`. An over-capacity prefill is chunked **expert-major, resident experts first**, so a miss never evicts an expert the same call still needs. Layers compile `_post_attn`/`_post_mlp` around an **eager MoE**.
  - PR #5657 (merged): Inkling 2 offload on an M4 Pro 24 GB with oQ4e. Fully loaded: 17.0 GB at 48.4 tok/s. At 0.5 residency: 10.2 GB, 20.4 tok/s, **95.5% hit rate**. At 0.6 residency: 11.8 GB, 25.9 tok/s, **97.6% hit rate**. At residency 1.0 the output is bit-identical. Below 1.0, only over-capacity prefill chunking changes rounding.
  - PR #5691 (open) is a bug to avoid. Per-miss writes to an NVFP4 per-slot `global_scale` table stayed pending in the MLX graph and leaked one Metal buffer per miss ("Resource limit (499000) exceeded"). The fix rewrites the whole table.
  - PR #4951 (merged, DeepSeek V4.1) profiled a decode cycle:
    - 46% waiting on reads.
    - **26% routing sync (`tolist`)**.
    - 11% install copies.
    - It then switched to **in-place reads into slot memory** via a host view and `mx.synchronize()` before overwriting victims (−10%).
    - It switched to **decayed-use eviction** (LFU with half-life), which cut misses 7.3% → 6.0%.
    - Net: −25–29% per cycle.
  - `docs/experimental/MoE_Expert_Offload.md` adds:
    - mmap gather vs positional reads: 0.23 vs 9.3 GB/s, because 16 KiB page faults kill throughput.
    - **Next-layer prediction prefetch was measured and rejected.** It covered 59% of misses, but wrong guesses competed for the SSD and made decode 2–15% slower.
    - Resident/miss **overlap** in decode helps only when reads are slow (USB SSD: 3.56 → 5.03 tok/s), because it keeps the GPU from down-clocking.
    - "Pinning hot experts" (dropping the rest) is catastrophic for accuracy (−91% gsm8k). Fetch-on-miss keeps the computation exact.

### 1.2 kevindechang/inkling-ios (llama.cpp, iPhone 15 Pro, CPU)
- **Repo:** https://github.com/kevindechang/inkling-ios (see `docs/ARCHITECTURE.md` and `docs/LOG.md`).
- **iOS facts measured on device:**
  - Stock mmap + `madvise(WILLNEED)` hung at load.
  - Plain mmap decode ran at 0.64 tok/s.
  - **"OS page cache never warmed on iOS"**.
  - **mmap-backed Metal buffers re-fault every token (~15 s/tok)**.
  - Metal with residency on hit OOM.
  - 16 KB-fault memcpy moved about 0.2 GB/s.
  - **`F_NOCACHE` parallel `pread`** (via `dispatch_apply`) moved about 0.8–1.2 GB/s on iPhone NAND. It gave 4.72 tok/s at the full 3-pass K8 shape with a 3000 MB LRU (≈36% residency), with about **69 MB/token missed (≈25 expert misses/token, ≈93–94% hit)**.
  - The iOS compressor cliff on an 8 GB iPhone is at about 4.45 GB footprint.
- **Ideas worth porting:**
  - **Known-route paired-down overlap.** Once routes are known, prefetch the `down` slices while gate_up computes. Prefetch hit rate is 100% by construction, and it gave −40% latency.
  - Hot-expert reordering (experts 0..39 = hot) and REAP-style tracing.
- **Not applicable to us:** its speed comes mostly from changing the model (1 pass, top-4, varloop sibling, distillation), and it is CPU-only.

### 1.3 Apple "LLM in a Flash" (Alizadeh et al., arXiv 2312.11514)
- Flash is read in large contiguous chunks. Its "row-column bundling" is the same idea as a per-expert contiguous record.
- A sliding-window DRAM cache reuses recently used neurons; for MoE this is the per-layer LRU/LFU slot cache.
- A predictor fetches ahead of need. oMLX's measurements say not to do *speculative* prefetch on a single SSD unless its precision is high.

### 1.4 MLX upstream
- **mlx #4249** "Add mmap_weights" (closed): a no-copy mmap'd array via `allocator::make_buffer`.
  - Key measured facts: "Metal wires the entire bytesNoCopy buffer on first kernel use, not touched pages … the unit of residency is the buffer."
  - "The wired ceiling applies to mapped buffers."
  - The maintainer replied that mmap is not being considered (discussion #615).
- **mlx #4883** (merged 2026-11-20): user-provided buffers (`array(void*, shape, dtype, deleter)`) are no longer added to residency sets. They still count as active memory. "A GPU command buffer that uses the buffer still pages all of it in."
- **mlx #3371** (layer streaming) was closed as incorrect, and issue #2878 (out-of-core streaming) is still open. There is **no** upstream MLX expert-offload feature.
- **mlx-swift-lm #708** (open) keeps embedding tables file-backed and gathers rows **on the CPU stream** only. That is the one case where mmap works: the GPU never touches the mapped buffer.

### 1.5 Expert locality: public data for Inkling (question 4)
- **Sadanie/Inkling-2-Mini-Memory-Sparse-MoE** (https://huggingface.co/Sadanie/Inkling-2-Mini-Memory-Sparse-MoE):
  - A "receptionist" (built on Qwen3-Embedding-0.6B) picks 64 of the 128 experts per layer from the question alone. Those picks "cover about 90% of the expert use of an answer", based on 18,271 recorded answers.
  - Without fine-tuning, restricting the model to that set costs real accuracy: MBPP 91.4 → 73.2, GSM8K 95.4 → 92.6, MMLU 78.0 → 75.4.
  - With router plus attention-LoRA fine-tuning it recovers: 89.1 / 95.3 / 76.0.
  - Runs in about 9 GB at about 10 tok/s on an M1.
  - **Implication:** a per-prompt static subset is a good *prefetch/warm-start* signal, about 90% coverage at 50%. Used as a hard restriction it changes the model. Our design uses it only as a warm-start hint (optional M5), never as a mask.
- **REAP** (Cerebras, arXiv 2510.13999) has community Inkling REAP12/25/50 MLX builds (pipenetwork/usterquant). These are global saliency prunes, i.e. a static global hot set. They are useful for **initial slot fill order and eviction priors**, not for exactness.
- **inkling-ios WSD-40:** distilling the router toward 40 hot experts reached a 99.6% hit rate but cut reasoning scores. Again, this changes the model.
- **Loop-pass overlap (pass 2/3 vs pass 1 of the same layer):** **no public measurement was found.** Indirect evidence: oMLX's 95.5% hit rate at 50% already includes revisits, and a revisit after only 12 layers is the friendliest case for LRU. The trace tool in M1 measures it directly as `J(pass1, pass2)`, `J(pass2, pass3)` per layer per token and the "intra-token reuse" fraction. That number sets how many of the 384 expert-visits per token are guaranteed hits.

---

## 2. MLX / mlx-swift specifics (question 2)

| Question | Answer | Source |
|---|---|---|
| Does `loadArrays`/safetensors mmap? | **No, it copies.** `load_safetensors` builds lazy arrays with a `Load` primitive over `io::ParallelFileReader` (`pread`). On eval, the bytes are read into a freshly allocated MTLBuffer. The PR's `Load.swift:9-18,174-262` relies on this: lazy header, I/O at `eval`, concurrent groups. | `mlx/io/safetensors.cpp:111-217`, `mlx/io/load.h:72-119` |
| Can MLXArray wrap a no-copy MTLBuffer over mmap pages? | **Technically yes**, through `MLXArray(rawPointer:_:dtype:finalizer:)`, which calls `mlx_array_new_data_managed_payload` and then `MetalAllocator::make_buffer`, i.e. `newBuffer(ptr,len,opts,nullptr)` = bytesNoCopy. The pointer and length must be page-aligned (16 KiB). **On iOS this is not usable for experts**, because the whole buffer is wired on GPU use (see 1.4) and it thrashed on iPhone (1.2). The only safe use is CPU-stream gathers (like #708), which our GPU expert kernels are not. | mlx-swift `Source/MLX/MLXArray+Init.swift:98-113`; mlx `backend/metal/allocator.cpp:228-241` |
| Cost of creating arrays per token? | Avoid it. Each no-copy wrap costs an MTLBuffer creation plus a residency/accounting update. oMLX #5691 shows per-miss pending *writes* (`slice_update`) accumulating Metal buffers to the 499k limit. The design creates **no** arrays per token. The slot tensors are allocated once, and misses are written into their bytes in place. The only per-step arrays are tiny id tensors. | oMLX PR #5691, #4951 |
| Can gather-QMM read experts from a small resident slot tensor via remapped ids? | **Yes.** This is exactly oMLX's design. `gatherQuantizedMM(x, weight[S,..], scales[S,..], biases[S,..], rhsIndices: slotIds)` works because each index only gathers rows (`Inkling2.swift:1123-1128`). The PR's fused decode kernels index `const uint e = indices[...]` and compute `row = e*2*I + …` against `[E, …]` tensors (`K.swift:1683-1739`), so passing slot ids against `[S, …]` tensors works unchanged. **One exception:** `global_scale[e]` (`K.swift:1696, 1716`, NVFP4 only) must also be slot-indexed, or the kernel must take original ids for scales. Our affine 2/3/4-bit builds have no global scale. | PR copy; `moe_slot_cache.py` |
| Writing into an MLX array's memory in place | Possible. After `eval`, `mlx_array_data_uint8(arr.ctx)` gives the base pointer of a shared-storage buffer, and `asMTLBuffer(noCopy:)` uses the same path (`MLXArray+Metal.swift`). This is outside MLX's functional model, so: (1) no queued or in-flight command buffer may still read the victim slot (call `Stream.gpu.synchronize()` / `MLX.synchronize()` before overwriting, or track the last command buffer that used the layer); (2) never alias the slot array with a donated buffer. The slot tensors are module parameters that stay referenced, so they are never donated. | oMLX #4951 ("mx.synchronize() drains the GPU so no queued kernel still reads a victim's slot") |
| `MLX.Memory` controls | `Memory.memoryLimit` (malloc waits when exceeded; default 1.5x the recommended working set), `Memory.cacheLimit` (**set low on iOS**, e.g. 256–512 MB; the buffer cache is dirty memory), `Memory.activeMemory/peakMemory/snapshot()`, and wired-limit scoping (`WiredMemory.swift`). Slot tensors are MLX-allocated, so they **count as active memory and footprint**. That is intended. | mlx-swift `Source/MLX/Memory.swift:175-307` |

---

## 3. iPadOS specifics (question 3)

- **Jetsam footprint.** Footprint = dirty + compressed memory. Clean, file-backed pages (mmap of an unmodified file) can be evicted and **do not count** (WWDC18 "iOS Memory Deep Dive", https://developer.apple.com/videos/play/wwdc2018/416; WWDC21 10180). But:
  1. Pages wired for GPU use are not evictable.
  2. On iOS, inkling-ios saw no useful page-cache reuse.
  3. We read with `F_NOCACHE`, which deliberately bypasses the page cache so streaming does not thrash it.

  So the design treats **slots = footprint** and gets nothing from the page cache. This is also why "lower residency is faster" (oMLX DSv4.1 on a 128 GB Mac, where the page cache served repeated misses) does **not** carry over to iPad.
- **Compressor.** Dirty slot pages that sit idle can be compressed by the iOS memory compressor. GPU access then stalls on decompression (inkling-ios: a 42 → 60 ms/token cliff on an 8 GB iPhone). Mitigations: keep the total footprint well under the limit, and either set a wired limit so MLX puts slot buffers in residency sets (wired, never compressed) or make sure every slot is touched regularly. Measure in M2.
- **Entitlements.** Apple's capability table (https://developer.apple.com/support/app-capabilities/, iOS table, columns ADP | ADEP | Apple Developer):
  - **Extended Virtual Addressing: ADP ✓, ADEP ✓, free ✗.** It is not needed here: we never map the 10–20 GB file. `pread` needs a file descriptor, not address space.
  - "Increased Memory Limit" has **no row** in the table, so free-team availability is **unverified**.
  - Historical per-app limits on a 16 GB M1 iPad Pro: about 5 GB without the entitlement, up to about 12 GB with it (MacRumors 2021).
  - Without the entitlement, only the 25% configurations (about 4.3–5.0 GB) fit. **M0 must log `os_proc_available_memory()` on the device** under the actual signing.
  - Free provisioning also expires after 7 days.
- **SSD bandwidth.** The iPad Pro M5 16 GB models are the 1 TB and 2 TB SKUs. Third-party app tests:
  - ITmedia (1 TB, AnTuTu): about 6.63 GB/s sequential read.
  - tbreak (Jazz Disk): about 5.75 GB/s.
  - Apple claims up to 2x the M4 generation.

  An expert record is a 2.5–4.5 MB contiguous read, so with 4–8 in flight it should run near sequential speed. For planning I use **4 GB/s effective**, with a 2.5–5 GB/s range. The iPhone figure of 0.8–1.2 GB/s is a lower bound.
- **Bytes per token.**
  - One expert holds 3×2304×1152 = 7.96M params. With affine group 64 and BF16 scale+bias, that is bits+0.5 bits per weight: **2-bit 2.49 MB, 3-bit 3.48 MB, 4-bit 4.48 MB, oQ2.5e mix 2.75 MB average.**
  - Visits per token = 6 + 12×3 + 6 = **48**, so 48×8 = **384 expert-visits per token**.
  - If every visit missed, that is 0.96 / 1.06 / 1.34 / 1.72 GB per token (2-bit / mix / 3-bit / 4-bit), i.e. 0.25–0.4 s per token at 4 GB/s. **Caching is mandatory.**
  - Model: `t = T_compute + T_sync + 384·(1−h)·S / BW`, with T_compute ≈ 33 ms and T_sync ≈ 10 ms (48 router readbacks at about 0.2 ms). **Both are assumptions until M0.**

**Hit rate h needed:**

| Experts | 10 tok/s @2.5 / 5 GB/s | 15 tok/s @2.5 / 5 | 20 tok/s @2.5 / 5 |
|---|---|---|---|
| 2-bit (2.49 MB) | 85.1% / 70.2% | 93.8% / 87.6% | 98.2% / 96.3% |
| oQ2.5e (2.75 MB) | 86.5% / 73.0% | 94.4% / 88.8% | 98.3% / 96.7% |
| 3-bit (3.48 MB) | 89.3% / 78.7% | 95.6% / 91.2% | 98.7% / 97.4% |
| 4-bit (4.48 MB) | 91.7% / 83.4% | 96.6% / 93.1% | 99.0% / 98.0% |

(Above 23 tok/s, the assumed 43 ms floor dominates and streaming is irrelevant.)

---

## 4. Memory budget vs expected tok/s (question 5, budget table)

Resident = 1.35 GB non-expert text (4-bit embed/head as in the offload repack; vision 0.36 GB loaded on demand) + 1.0 GB runtime (KV for 4 global layers + SWA ring, activations, MLX cache capped) + f × total expert bytes.

Hit-rate assumptions: 25% → 0.85 (inferred: inkling-ios ≈0.935 at ≈36%; gemma-26B 0.67 at 25%), 50% → 0.955 (**measured**, oMLX #5657), 75% → 0.985 (extrapolated). BW is 4 GB/s and reads are serial (no overlap).

| Experts (total expert GB) | 25% resident | 50% resident | 75% resident |
|---|---|---|---|
| 2-bit (7.64) | 4.26 GB, ~12.7 tok/s | 6.17 GB, ~18.6 | 8.08 GB, ~21.5 |
| oQ2.5e 2/3 (8.45) | 4.46 GB, ~12.1 | 6.58 GB, ~18.2 | 8.69 GB, ~21.3 |
| 3-bit (10.70) | 5.03 GB, ~10.7 | 7.70 GB, ~17.2 | 10.38 GB, ~20.8 |
| 4-bit (13.76) | 5.79 GB, ~9.3 | 9.23 GB, ~16.0 | 12.67 GB, ~20.2 (**doesn't fit 12 GB**) |

- **Sanity check:** the oQ2.5e row reproduces theblackcat102's measured peaks (4.6 / 6.6 GiB) closely. Their 7.9 / 10.0 tok/s on an M4 Air sit below these predictions, which is consistent with the Air's slower SSD and fanless thermals.
- **Treat every tok/s cell as ±40%** until M0 measures T_compute and T_sync on the iPad.
- **Recommended targets:**
  - (A) Smallest footprint: oQ2.5e at 25%, about 4.5 GB. This fits even without `increased-memory-limit`.
  - (B) Quality: **4-bit experts at 50%**, about 9.2 GB. That is resident-build memory with 4-bit experts.

---

## 5. Concrete design for our codebase

### 5.1 File layout: `experts.bin` + `experts.json`
- **One file per model, ordered layer-major, then expert.** Each expert is one **record** whose start is aligned to 16 KiB (the iOS page size; also good for `F_NOCACHE` DMA).
  ```
  record(L,e) = [gate_up.weight | gate_up.scales | gate_up.biases | down.weight | down.scales | down.biases]  (+ global_scale if NVFP4)
  ```
  - Each field is padded to 256 B inside the record, so destination offsets stay aligned.
  - Layer-major order makes an expert-major prefill of one layer an almost sequential sweep.
  - Records are fixed-size per layer, but layers may differ (mixed bit layers).
- **`experts.json`** holds: per-layer `{record_bytes, base_offset, fields:[{name, dtype, shape_per_expert, offset_in_record, nbytes}], format:{mode,bits,group}}`, plus a SHA-256 per layer for a cheap integrity check at first run.
- **Rest of the model:** the original safetensors with the `mlp.experts.*` tensors **removed** (a "trunk" file set), so the PR loader never materializes the experts.
  - Alternative with no trunk rewrite: pass `isRead` (`Load.swift:174`) to skip `*.mlp.experts.*`.
- **Why not per-expert safetensors tensors (blackcat layout)?** Those are not 16 KiB-aligned, and they cost 6 syscalls or iovecs over an unaligned file offset. A custom record allows exactly one `preadv` per miss, with an aligned file offset and length.

### 5.2 Resident cache: `Inkling2ExpertSlots` (one per layer)
- **Storage.** Slot tensors `gateUp.{weight,scales,biases}: [S, 2I, …]` and `down.{…}: [S, D, …]`, allocated once with `MLXArray.zeros` and `eval`'d. `S = clamp(round(f·128), min: 8 × (rows per step), max: 128)`. These replace the `[128, …]` parameters of `Inkling2ExpertProjection`.
- **Host state per layer:**
  - `slotOf: [Int16](128)` (−1 if not resident) and `expertOf: [Int16](S)`.
  - `score: [Float](S)` for decayed-use LFU: each use adds 1; multiply by 0.7 every 4 calls, as in oMLX. Ties fall back to LRU.
  - `lastUseVisit: [Int32](S)` for protection.
- **Protection rule:** never evict an expert used by the current call. Optionally also protect the experts used by this layer's previous pass in the current token, while a later pass is still to come (cheap, since loop revisits are 12 layers apart). The M1 trace decides this.
- **Warm start:** prefill naturally fills the slots. Optionally seed them with a REAP or receptionist prior (M5).

### 5.3 Async IO
- **Setup.** `open(O_RDONLY)` plus `fcntl(fd, F_NOCACHE, 1)`. Optionally `F_RDAHEAD 0`.
- **Readers.** A pool of 4–8 reader threads (`DispatchQueue.concurrentPerform` or a dedicated `OperationQueue`; `DispatchIO` adds little for fixed-offset reads). `pread`/`preadv` go into **slot memory directly**, with 6 iovecs, one per field destination:
  ```
  base(field) = mlx_array_data_uint8(slotArray.ctx) + slot * fieldBytes
  ```
- **In-flight limit:** ≤128 MB per call, which keeps the transient footprint bounded.
- **Error handling:** a short read or EINTR retries; on an error the slot goes back to `free` and the call throws, matching oMLX's contract.

### 5.4 Synchronization with MLX lazy eval (the critical part)
Every MoE visit becomes a **host sync point**: 48 per decoded token, and one per layer-visit per prefill chunk.

1. Build the router lazily (`Inkling2Router.callAsFunction`, `Inkling2.swift:1258-1270`), then `eval(indices)` and copy the `[rows, 8]` UInt32 ids to the host.
   - This evaluates the full dependency chain, *including all earlier layers' expert kernels*. When it returns, no in-flight GPU work reads any slot of any earlier layer.
   - The current layer's slots were last read at least one MoE visit ago (or on the previous token). So **victim slots are safe to overwrite after the eval**, with no extra `synchronize`.
   - **Exception:** anything evaluated asynchronously and *not* in the dependency chain, for example `asyncEval` of the next step in the token iterator (`Inkling2.swift:1960`). With offload on, disable cross-step async pipelining, or call `Stream.gpu.synchronize()` before the first overwrite of a step.
2. `ensure(ids)`: classify hits and misses, choose victims, issue reads, and wait. For decode this is ≤8 misses, about 2–6 ms at 4 GB/s in the worst case.
3. Translate ids to slot ids **on the host** (ids are already there) and upload a `[rows, 8]` UInt32 `MLXArray(slotIds)`. A device-side `take(map, indices)` would need its own map writes, which is the source of oMLX's pending-write leak.
4. Call the unchanged fused kernels with `indices: slotIds` against the slot tensors.

**Cost:** 48 command-buffer round trips per token. oMLX measured routing sync at 26% of a DSv4.1 cycle. Mitigations, measured in M3:
- (a) Build the attention half of the next visit before the readback returns.
- (b) Skip the readback when the layer is fully resident: `S == 128`, or a "warm" flag as in oMLX.
- (c) Stage 4: a **GPU-side miss check**. A tiny kernel writes `slotIds` and a miss bitmap; the host waits only on that bitmap. This is the same sync count but less work is serialized.

### 5.5 Exact changes in the PR copy
| Location | Change |
|---|---|
| `Inkling2ExpertProjection.init` (`Inkling2.swift:1098-1115`) | Take `experts: capacity` (S) instead of `config.experts`. Shapes stay `[S, out, packed]`. Keep a full `[128]` `globalScale` for NVFP4. |
| `Inkling2ExpertProjection.callAsFunction` (`:1123-1128`) | Unchanged, but receives slot ids. For NVFP4, pass a slot-ordered `globalScale` view that is rebuilt whole on change (lesson from oMLX #5691), or switch to affine builds. |
| `Inkling2Experts` (`:1172-1241`) | Add `slots: Inkling2ExpertSlots?`. In `outputs(_:_:)` (`:1203-1217`) and `decode(...)` (`:1221-1240`), call `let slotIds = slots?.ensure(indices) ?? indices` first. In `outputs`, an **over-capacity prefill** (distinct experts > S) goes to a new `chunkedOutputs`: sort routes by (resident first, expert id), cut chunks of ≤S distinct experts, ensure per chunk, `gatherQuantizedMM(sorted: true)`, `eval` each chunk, then unsort, as in oMLX `unreduced`. |
| `Inkling2SparseMoE.prefill/decode` (`:1293-1320`) | No logic change. They call `experts.outputs` / `experts.decode`, which now ensure. Keep `inkling2RowConvolution` on chunked outputs (the `order` semantics are unchanged once chunks are concatenated in sorted order). |
| `K.swift` expert kernels (`:1683-1739`, `Inkling2ExpertTensors :1742-1773`) | No change for affine. For NVFP4 `HAS_GS`, add a second index input `gidx` (original ids) for `global_scale[gidx]`, or pass a per-slot scale array. **Stage 4 option:** a single uint8 **record slab** `[S, recordBytes]` input with compile-time field offsets, so that one aligned `pread` fills a slot. |
| `Inkling2.prepareCheckpoint` (`:2023-2050`) | When `experts.json` exists, build `Inkling2Experts(text, gateUp:, down:, capacity: S)` from the manifest's format instead of `Inkling2ExpertFormat(path:weights:)`, which is shape-inferred and needs the stacked tensors. Attach `Inkling2ExpertSlots(layer, manifest, fd)`. |
| Loader (`Load.swift:174-262`) | Exclude `language_model.layers.*.mlp.experts.*` via `isRead` (or use the trunk files). |
| Token iterator / `Inkling2.prepare` (`:1942-1975`) | With offload on: no cross-step `asyncEval`, the drafter (MTP) is off at first, and `Memory.cacheLimit` is low. |

### 5.6 Prefetch policy (ordered by expected value)
1. **Within-visit overlap (free, always correct).** Issue the gate_up reads first and the down reads second for each miss. Run the resident experts' gate_up while the misses load. This is the on-GPU analogue of inkling-ios's known-route trick and oMLX's resident-first overlap. It needs a 2-phase fused decode, `inkling2ExpertGateUp` on the resident subset and then on the missing subset. Do it in M4 only if M3 shows the GPU idling on reads for more than about 1 ms per visit.
2. **Loop-pass carry-over (cheap, high precision).** When layer L pass p routes, mark those experts "protected until pass p+1 of L". This adds no reads, only eviction priority. Measure in M1.
3. **Speculative next-visit prefetch** (layer L+1's router on layer L's input, or pass p+1 = pass p's set). Off by default. oMLX measured a net loss on a single SSD. Re-test only if M1 shows a prediction precision above 90% for the pass-p → p+1 set.

### 5.7 Conversion script (`tools/make_expert_stream.py`): runs on the Windows Omen PC
- **Confirmed: no Apple silicon and no MLX needed.** Safetensors is an 8-byte little-endian header length, a JSON header, then raw bytes. The script uses plain Python + numpy, copies raw bytes (treating BF16 as uint16), and reads per-expert slices of stacked `[128, out, packed]` tensors by byte offset (the expert dimension is outermost and contiguous: `offset + e·(nbytes/128)`).
- **Steps:**
  1. Parse the index and headers.
  2. For each layer and expert, gather the 6 field slices, pad, and write the record at a 16 KiB-aligned offset.
  3. Write `experts.json`.
  4. Write the trunk safetensors (all non-expert tensors, byte-copied, plus the original `config.json`, so the PR's `quantization` metadata stays intact).
  5. Verify: re-read 1% of random experts and compare bytes to the source; check SHA per layer.
- **Inputs:**
  - Per-expert sources (blackcat) are also handled: `experts.<e>.w13.*` → gate_up, `w2` → down. **Check the naming and convention against `Inkling2Checkpoint.convertingOriginalLayout`.** The PR uses `gate_up_proj`/`down_proj` with interleaved gate/up rows, and w13 must be verified to be interleaved, not concatenated.
  - For 4-bit streamed experts, repack an existing 4-bit MLX build (e.g. mlx-community/Inkling-2-Mini-4bit or oQ4e). **Do not requantize in numpy** unless you reproduce MLX's affine quantizer bit-exactly. Repacking is exact by construction.
- **Cost:** a streaming copy of about 10–14 GB, a few minutes on NVMe. The output is device-ready and is copied into the app container (Files or Finder).

---

## 6. Risks
1. **Per-visit sync overhead** (48/token) may dominate, more than IO, on a short-context decode (oMLX: 26% of a cycle). This is the biggest unknown for the "about 18 tok/s @ 50%" claim. It is measured in M0/M3.
2. **Hit rate below assumptions** at 25% (Inkling routing locality is unmeasured; gemma was 0.67). At 0.67, 25% residency gives about 6 tok/s.
3. **Memory entitlement on a free team** is unknown. Without it, about 5 GB is the cap, so only 25% configs fit.
4. **Thermals.** Sustained SSD plus GPU on a fanless iPad. The blackcat card reports 1.7x variance on a fanless Air.
5. **Compressor stalls** on idle slots (see 3). Mitigate with wiring and headroom.
6. **Correctness drift.** Over-capacity prefill chunking changes rounding (oMLX: greedy diverges after a few hundred characters). Decode that fits in the cache must be **bit-identical** to resident. The gate: compare logits at f = 1.0 (exact) and f < 1.0 decode-only (exact), and accept prefill differences only within rounding bounds.
7. **In-place writes outside MLX's model.** Getting the async-eval boundary wrong creates a silent data race. M2 includes a stress test: random eviction at S = 8, compared against resident logits for 2k tokens.
8. **App lifecycle.** The FD must survive backgrounding. Re-open the file and revalidate the manifest SHA on foreground.
9. **F_NOCACHE plus unaligned iovec destinations** may fall back to slower paths. Field padding to 256 B and record alignment to 16 KiB address this. Verify the effective GB/s in M0.

## 7. Staged plan with measurable milestones
- **M0: device baselines (1 day).**
  - On the iPad: `os_proc_available_memory()` under the actual signing.
  - `F_NOCACHE` `preadv` throughput for 2.5/3.5/4.5 MB records at queue depth 1, 4 and 8.
  - Resident decode tok/s for the PR build. If the build does not fit, use a REAP25 or LookAtMyQuant-small proxy to get T_compute.
  - Per-sync cost: `eval` of a `[1,8]` array in a loop.
  - **Gate:** numbers logged; the model in §3/§4 recalibrated.
- **M1: routing trace (Mac or iPad, 1–2 days).** Instrument `Inkling2Router` to dump (token, visit, layer, pass, ids). Offline, simulate LRU / LFU / pass-protect at S = 32/64/96. Report the hit rate per policy, pass-overlap Jaccard, and the receptionist-style 64-set coverage. **Gate:** a chosen policy plus a predicted h for 25/50/75%.
- **M2: converter + slot cache, decode only (3–4 days).** `make_expert_stream.py`, the manifest loader, `Inkling2ExpertSlots`, `ensure` with synchronous reads, and slot ids through the fused decode kernels.
  - **Gate 1:** at f = 1.0, logits are bit-identical to resident.
  - **Gate 2:** at f = 0.25/0.5, decode is bit-identical for 512 tokens after the same prefill.
  - **Gate 3:** footprint matches the §4 prediction within ±10%.
- **M3: prefill + perf (3 days).** Over-capacity expert-major chunked prefill, decayed LFU, async reader pool, and the fully-resident fast path. **Gates:** the ten-token cosine/greedy gate versus resident; tok/s at 25/50% within ±25% of the recalibrated model; TTFT for a 2K prompt reported.
- **M4: overlap and sync reduction (optional).** The 2-phase gate_up/down overlap and the GPU-side miss bitmap. **Gate:** ≥10% decode gain, re-measured by a second run or agent.
- **M5: quality upgrade.** A 4-bit-expert stream at 50% (target about 9.2 GB). Compare quality against the resident 2/3-bit build (KL and greedy-match on a held-out set). Optionally test a REAP or receptionist warm-start prior (hit-rate gain in the first 64 tokens).

## 8. Feasibility statement
- **Feasible on iPadOS** with explicit slots and `F_NOCACHE` `pread`. Every primitive needed is available to a third-party app: `fcntl`, `preadv`, MLX unified-memory arrays, and custom Metal kernels.
- **Not feasible on iPadOS:**
  1. Transparent mmap streaming through Metal, because whole-buffer wiring and re-faulting were measured.
  2. Relying on the page cache.
  3. Whole-file mapping on a free team, because there is no EVA entitlement.
- **The real ceiling:** decode speed is bounded by hit rate × SSD bandwidth and by 48 host syncs per token. Expect about 10–20 tok/s depending on residency, not resident-level speed.

## Sources
- oMLX: https://github.com/jundot/omlx (`omlx/patches/moe_slot_cache.py`, `omlx/patches/inkling2/moe_offload.py`, `docs/experimental/MoE_Expert_Offload.md`); PRs #5657, #5691, #4951
- https://huggingface.co/theblackcat102/Inkling-2-Mini-oQ2.5e-mtp-offload (card plus shard headers)
- https://github.com/kevindechang/inkling-ios (`docs/ARCHITECTURE.md`, `docs/LOG.md`)
- https://huggingface.co/Sadanie/Inkling-2-Mini-Memory-Sparse-MoE
- REAP: https://arxiv.org/abs/2510.13999 ; community builds https://huggingface.co/pipenetwork/Inkling-MLX-REAP25-4bit
- LLM in a Flash: https://arxiv.org/abs/2312.11514
- MLX: https://github.com/ml-explore/mlx/pull/4249 , /pull/4883 , /pull/3371 , /issues/2878 , /discussions/615 ; `mlx/backend/metal/allocator.cpp`, `mlx/io/safetensors.cpp`, `mlx/io/load.h`
- mlx-swift: `Source/MLX/MLXArray+Init.swift`, `MLXArray+Metal.swift`, `Memory.swift`; mlx-swift-lm https://github.com/ml-explore/mlx-swift-lm/pull/708
- Apple: https://developer.apple.com/videos/play/wwdc2018/416 ; https://developer.apple.com/wwdc21/10180 ; https://developer.apple.com/support/app-capabilities/ ; https://developer.apple.com/forums/thread/685084
- iPad M5 SSD: https://www.itmedia.co.jp/pcuser/spv/2510/21/news129_2.html ; https://tbreak.com/ipad-pro-13-inch-m5-review-uae/ ; memory limits: https://www.macrumors.com/2021/09/17/ipados-15-up-to-12gb-ram-high-end-ipad-pro/
- Alternative (no streaming): https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB (1.18 bpw, about 4.6 GiB peak, its own runtime)
