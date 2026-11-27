# On-device engine and model-fit decision report: iPad Pro M5 (16 GB RAM, about 5.3 GB per-app limit)

## TL;DR
1. **Strata** is almost certainly **Niko1221/Strata**. It is MIT-licensed, macOS/Windows/Linux only, and has no iOS build. Don't integrate it. Copy its expert-streaming design into MLX instead.
2. **Xybrid already has a working on-device LLM path**: llama.cpp with Metal, loading from a memory-mapped file (mmap), and it supports the `lfm2moe` architecture. Run one A/B against MLX using **LFM2.5-8B-A1B only**. It cannot run Inkling 2 Mini.
3. **LFM2.5-8B-A1B** has two routes that need no new engine work:
   - **MLX 3-bit.** About 3.7 GB by my estimate, so it fits outright.
   - **Xybrid GGUF.** `UD-Q3_K_XL` (4.00 GB) or `UD-IQ4_XS` (4.26 GB).
   - Expected decode is about **50–90 tok/s**. This is my estimate, not measured.
4. **Inkling 2 Mini** fits only with real expert streaming. Even the smallest GGUF is 7.76 GB. The realistic route is a bounded expert-slot pool in MLX:
   - Hot experts stay in memory; cold experts are read from flash with `pread`.
   - Expected decode is about **10–25 tok/s**. This is an estimate and needs a routing-locality profile first.
5. **Storage**:
   - MLX weights currently sit in the purgeable `Library/Caches`. Move them to Application Support and exclude them from backup.
   - Load from local files first, and fix the benchmark's `weight_files_bytes: 76` bug.
   - A reinstall over the existing app with the same Apple ID and bundle ID keeps the weights. The engines are compiled into the app, so they are never downloaded.

---

## 1. Strata: what it is, and should we integrate it?

| | Evidence |
|---|---|
| **Identity (very likely)** | [Niko1221/Strata](https://github.com/niko1221/strata), HF org [strata-engine](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata), v0.2.22 (2026-11-25), MIT ([LICENSE](https://raw.githubusercontent.com/Niko1221/Strata/main/LICENSE)). |
| Engine | Its own Metal kernels (`engine/metal/kernels/*.metal`) plus a shim that maps CUDA runtime calls onto Metal. [linuxcompatible](https://www.linuxcompatible.org/story/strata-0216-runs-its-own-engine-drops-llamacpp-claims-identical-output/) says ggml and llama.cpp were dropped in 0.2.16. **Conflict:** the README and the M5 Ultra write-up still describe a llama.cpp lineage. The truth is probably "originally derived, now its own kernels". |
| Platforms | Windows, Linux and macOS 15+ (Metal). **No iOS, no xcframework, no Swift package, no Neural Engine path.** The only "ios" files in the repo are default Tauri icons. |
| Models | Inkling 2 Mini, Qwen3.x MoE and Hy4. **No LFM support.** |
| Inkling packs | SQ-10 is 9.75 GiB (10.47 GB, the same size in different units); SQ-15 and SQ-20 are larger. SQ-10's top token matches the original model 92.35% of the time. All are about 2x our memory budget. |
| Streaming design worth copying | `--mmap-experts`, `--resident-budget-gib`, `--expert-cache`, a hot-expert arena locked in memory, next-layer read-ahead, and a repacked `.experts` file so each expert is one read (8.62 vs 7.37 tok/s). Sources: [DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/master/docs/DETAILS.md), [MAC.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md). |
| Blocker for iPad | Its Inkling Metal backend assumes every expert is resident (`ink_metal_backend.mm:82`). It either copies weights into anonymous memory, which counts fully against the iOS memory-kill (jetsam) limit, or maps the whole file. It is driven from a CLI or HTTP server, so we would need our own bridge, tokenizer and chat templates, against near-daily releases. |
| Its speed claims | About 90% of peak bandwidth on M5 Max, against our ~70% with MLX. Its 3.4x win over MLX ([write-up](https://elitetestengineering.substack.com/p/strata-on-the-m5-ultra)) mixes in multi-token prediction and different quantization, so it is not like-for-like. |

Other candidates I ruled out:
- commoncompute/lokahi: a macOS Python fork of oLLM with 6 stars.
- ro99/strata: CUDA only.
- "Strata AI": a cloud keyboard app.
- kytheros/strata: a memory tool, not an engine.
- The EveryDev "GPU cluster manager" listing: unverified.

**Decision: don't adopt Strata.** Port its design, which is MIT-licensed, into MLX. Optionally spend at most 1 day cross-compiling `strata_inkling_metal`. It is a static library that links only Metal, Foundation and MPS. Expect jetsam to kill it.

## 2. Xybrid LLM path: verdict and A/B plan

**Facts** (from the v0.21.0 clone and main @ 9fbd8b1):
- Xybrid builds llama.cpp with Metal on for iOS devices.
- The pinned llama.cpp (`b1b770d9`) has `lfm2moe` but **not `inkling2_loop`**. The Inkling GGUFs need llama.cpp master.
- The wrapper sets only `n_gpu_layers`, which defaults to 99. Load mode stays AUTO, which means mmap.
- These cannot be set from Swift: tensor placement overrides, MoE experts on CPU, KV cache type (always F16), and load mode.
- Context length defaults to 4096.
- The Xybrid model registry has **no LFM2.5-8B-A1B and no Inkling**, so load with `ModelLoader.fromModelFile(url)`, which makes no network calls.
- Xybrid publishes no tok/s numbers for Apple devices.

**Verdict:** worth one A/B, for LFM2.5-8B-A1B only. Don't build Inkling on Xybrid.

**A/B protocol**. Everything runs in the same app, with greedy decoding, the same prompt, 256 output tokens, and 5 runs with the median reported.

Every run records:
- `os_proc_available_memory()` before and after load
- `task_vm_info.phys_footprint` at peak
- `MTLDevice.recommendedMaxWorkingSetSize` and `currentAllocatedSize`
- TTFT, prefill tok/s and decode tok/s. On Xybrid these come from `r.metrics.ttftMs`, `.prefillTps` and `.decodeTps`.
- whether jetsam killed the app

| # | Engine / weights | Size | Purpose |
|---|---|---|---|
| A1 | MLX 4-bit LFM2.5-1.2B vs Xybrid registry `lfm2.5-1.2b-instruct` Q4_K_M | ~0.7 GB | Pure engine speed against our measured 169 tok/s |
| A2 | Xybrid `unsloth/LFM2.5-8B-A1B-GGUF` UD-Q3_K_XL | 4.00 GB | Fit and speed |
| A3 | Xybrid UD-IQ4_XS | 4.26 GB | Quality/speed point |
| A4 | MLX 3-bit LFM2.5-8B-A1B (we convert it) | ~3.7 GB (est.) | Like-for-like against A2 |
| A5 | MLX 4-bit (`mlx-community/LFM2.5-8B-A1B-MLX-4bit`) | 4.85 GB | Expected to be killed by jetsam; confirms the limit |
| A6 | Xybrid Q4_0 | 4.84 GB | **mmap canary**: does a 4.84 GB mapped file push `phys_footprint` above about 5.3 GB? |
| A7 | A6 again with `setenv("GGML_METAL_NO_RESIDENCY","1",1)` set before load | 4.84 GB | Do pages that are not wired for the GPU escape the limit? Risk: page faults on every token (inkling-ios P6 saw about 15 s per token) |

**What A6 and A7 decide.**
- If the footprint stays well below the weight size, mmapped llama.cpp becomes our route past 5.3 GB. That would open Inkling UD-IQ1_S (7.76 GB) on a llama.cpp master build.
- If the footprint tracks the weight size, MLX streaming becomes the only route.
- **The evidence on this conflicts:**
  - [Liquid's docs](https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile) and the [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) say mapped, unmodified pages are not charged to the footprint.
  - [inkling-ios LOG](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/docs/LOG.md) hit a Metal out-of-memory crash with residency on, page faults every token with residency off, and an OS page cache that "never warmed".
  - The A6/A7 result settles it.

## 3. Engines ranked

**Goal A: run models bigger than 5.3 GB**

| Rank | Engine | Why | Evidence |
|---|---|---|---|
| 1 | **MLX + our own expert-slot pool** (Strata / RFC #23324 design) | Bounded, measurable memory footprint. MLX is already our fastest engine. Covers both models. | [RFC #23324](https://github.com/ggml-org/llama.cpp/discussions/23324): Qwen3-30B-A3B ran at 12–22 tok/s with 8–32 slots, against 38 vanilla, on M3 Pro. [MAC.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md) |
| 2 | **llama.cpp via mmap** (Xybrid now; `llama.xcframework` at master for Inkling) | Only works if A6/A7 show file-backed pages are not charged | See the conflict in section 2 |
| 3 | CPU-only llama.cpp with `pread` expert cache (inkling-ios) | Proven on an iPhone at 21–23 tok/s in a modified top-4 setup. Needs the increased-memory-limit entitlement, which is **not effective on our free Apple ID**. Running dense layers on Metal with experts on CPU was 2.8x slower. | [inkling-ios](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/README.md), [ARCHITECTURE.md](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/docs/ARCHITECTURE.md) |
| 4 | flash-moe (Anemll) | A proof of concept with `pread` streaming on iPhone (400B at 0.6 tok/s). No license stated, so use it as reference only. | [fork](https://github.com/Anemll/flash-moe) |
| – | Strata | No iOS build | Section 1 |
| – | Cactus, ZETIC MLange, MLC-LLM, ExecuTorch, ANEMLL, Core AI | No documented MoE or flash streaming for iOS | Section 3, goal B table |

**Goal B: maximum speed for models that fit**

| Rank | Engine | Evidence |
|---|---|---|
| 1 | **MLX (current)** | Measured on this iPad: 424/169/113/109 tok/s, about 70% of bandwidth |
| 2 | llama.cpp / Xybrid | About 0.8x MLX on iPhone 17 Pro for Gemma E2B (38.8 vs 49.1) ([bench](https://github.com/john-rocky/apple-silicon-llm-bench)). A1 measures it on our iPad. |
| 3 | Cactus | iPad M5, Gemma-4-E2B CQ4: 71 tok/s decode. iPhone: 50.0 vs 49.1 for MLX, but GSM8K was 3% "as shipped" ([GitHub](https://github.com/cactus-compute/cactus)) |
| 4 | Core AI (iOS 27) | 67.6 vs 65.1 for MLX on Qwen3-1.7B (iPhone); weaker on MoE (78.1 vs 100.2 for gpt-oss-20b on M4 Max). Watch it for dense models ([WWDC26](https://developer.apple.com/wwdc26/guides/ios/)). |
| – | Liquid LEAP SDK | **Deprecated**; it was a llama.cpp wrapper ([migration guide](https://docs.liquid.ai/deployment/on-device/llama-cpp/migrating-from-leap-sdk)) |
| – | Nexa SDK | The team joined Qualcomm; Qualcomm chips only ([Qualcomm blog](https://www.qualcomm.com/developer/blog/2026/06/geniex-developer-preview)) |
| – | ZETIC MLange | Needs keys and a download, only 1.1–1.5B models, performance section says "to be added" ([doc](https://zetic-ai.github.io/ZETIC_MLange_document/overview/zetic-mlange-llm-model.html)) |
| – | Strata (Mac) | About 90% of peak bandwidth on M5 Max; no iOS build. A possible 1.2–1.3x kernel-tuning target for MLX, inferred, not measured. |

Every engine you listed has now been assessed: Liquid/LEAP, Cactus, Nexa, Xybrid, ZETIC and Strata.

## 4. Running both models in about 5.3 GB

### LFM2.5-8B-A1B
- **Path 1 (fastest to land): MLX 3-bit**, or mixed 3/4-bit with attention and embeddings kept at 4-bit.
  - About 3.7 GB, which leaves room for the KV cache. The model has only about 6 attention layers.
  - Quality gate: against the 4-bit model on a Mac, top-1 agreement ≥90% over 256 greedy tokens and no perplexity regression above 5%. This threshold is my proposal.
- **Path 2: Xybrid GGUF** UD-Q3_K_XL or UD-IQ4_XS (A2/A3).
- **Path 3: full 4-bit MLX** (4.85 GB) with expert streaming (backlog T6).
- **Expected decode: about 50–90 tok/s** (estimate). For reference:
  - Liquid reports 253 tok/s on M5 Max with llama.cpp. Scaling by our 153.6 vs 614 GB/s bandwidth gives about 63 ([Liquid blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)).
  - Core AI int4 on iPhone 17 Pro runs about 32 tok/s ([HF card](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI)).
  - The active parameter count is unchecked; "A1B" suggests 1–1.5B.
- **Risks:**
  - 3-bit quality loss on a model with only about 1B active parameters.
  - Jetsam during the prefill burst. Cap prefill chunk size.

### Inkling 2 Mini (26B MoE)
Config: 24 layers, 128 experts with top-8 routing, expert intermediate size 1152 ([config](https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json)).

Budget (my arithmetic, unverified):

| Component | Estimate |
|---|---|
| All experts at about 2.5 bits per weight | ≈ 24.5B params ≈ 7.6 GB |
| Expert reads per token | ≈ 1.5B params ≈ 0.48 GB |
| Dense part (attention + LM head ≈ 0.9B) at 4-bit | ≈ 0.5 GB |
| Room left for resident experts in a 5.3 GB budget | ≈ 3.5–4 GB, about 45–50% of all experts |
| KV + Metal buffers + app | the remainder |

"Loop" may repeat layers. That is unverified, and if true it raises the bytes read per token.

| Route | Fits? | Expected tok/s | Risk |
|---|---|---|---|
| **MLX expert-slot pool** (recommended): repacked per-expert file, hottest experts pinned in memory, fixed slot pool per layer with LRU and parallel `pread` using `F_NOCACHE`, prefetch from the next layer's router | Yes, by construction | ~10–25 if the hit rate is 70–90% (estimate: misses of about 50–150 MB per token at unmeasured iPad flash speeds) | iPad flash read speed unknown. inkling-ios found the iOS page cache never warmed, so the anonymous memory pool must be ours. Memory-compressor cliff at about 4.4 GB on iPhone; the iPad threshold is unknown. Routing locality is unprofiled. |
| llama.cpp master via mmap (UD-IQ1_S 7.76 GB / IQ2_XXS 8.49 GB) | Only if A6/A7 pass | ≤ resident speed when the cache is warm; collapses when it is cold | Most likely outcome: a Metal out-of-memory crash or page faults every token, as inkling-ios saw |
| CPU `pread` (inkling-ios) | Needs the memory entitlement | 4.7–23 tok/s | Blocked on a free Apple ID |
| Strata SQ-10 | No (9.75 GiB, everything resident) | n/a | No iOS build |

**Profile before building.** Log Inkling's per-layer expert-routing histograms on a Mac or any MLX host over realistic prompts. Then compute the hit-rate curve against the resident budget. If 3.5 GB of resident experts gives less than a 60% hit rate, Inkling on this iPad is bound by flash speed and is likely under 10 tok/s.

## 5. Model storage: making sure nothing is downloaded twice

**Current state**
- MLX weights live at `Library/Caches/huggingface/hub`. iOS can purge this folder when storage is low.
- Xybrid speech models live in `Library/Application Support/Xybrid/Models`. They persist but are included in iCloud backup.
- Every MLX load makes one `listFiles` network call. It downloads nothing when the files are present, but it waits for the URLSession timeout when the network is bad.
  - Cause: we pass `"main"` as the revision, which disables the library's offline fast path.
- Receipts:
  - `download_observed` is a heuristic.
  - `weight_files_bytes: 76` is a bug: it measures the symlink, not the file it points to.

**Reinstall rules**
- **Weights are kept:** Sideloadly install over the existing app with the same Apple ID and bundle ID. The 7-day re-sign is fine.
- **Weights are lost:**
  - deleting the app first;
  - a different Apple ID or team (the upgrade is rejected) ([forum 126589](https://developer.apple.com/forums/thread/126589));
  - Sideloadly rewriting the bundle ID.
- **The inference engines** (MLX, its Metal library, the Xybrid xcframework and llama.cpp) are compiled into the app. Each install copies them over USB/Wi-Fi as part of the app; they are never fetched from the network.

**Fixes** (backlog T1)
1. Add `ModelStore.swift`: an explicit `HubCache(cacheDirectory: ApplicationSupport/Models/huggingface/hub)`, with `isExcludedFromBackup` set. Load via `loadModelContainer(from: #hubDownloader(ModelStore.hub), ...)`.
2. One-time migration: `moveItem` the old Caches `hub` folder. It is the same volume, so the move is instant and the relative symlinks stay valid. Guard it with the `store.migrated.v1` flag. Without this step, the LFM3 and Qwen weights already on the iPad would download again.
3. Local-first load:
   - resolve `refs/main` to a snapshot;
   - check `config.json`, the tokenizer and every shard against a size manifest;
   - set `cfg.id = .directory(snap)` so the registry's `toolCallFormat` is kept;
   - go to the network only if the check fails.
4. Free-space check before downloading (`volumeAvailableCapacityForImportantUsageKey`).
5. A "Downloaded models" screen in Settings listing MLX and Xybrid models, with size, commit and a confirmed delete. Delete only after the model is unloaded.
6. Exclude `Application Support/Xybrid` from backup.
7. **Never call `Xybrid.initSdkCacheDir`.** It sets `HF_HOME`, which silently moves the default MLX cache. The explicit `HubCache` removes this risk.
8. GGUF files for the Xybrid A/B go in `Application Support/Models/gguf/<model>/`, one directory per file, loaded with `fromModelFile`, with no network.
9. Benchmark fixes:
   - Resolve symlinks before measuring sizes.
   - Record blob bytes before and after each load.
   - Add a `local_snapshot_complete_before` flag next to `download_observed`.

## 6. Prioritized backlog (agent-sized)

The repo is read-only for research agents. Implementation agents work on the branches below. Every on-device receipt includes the device, iPadOS version, build SHA, model repo and commit, quantization, prompt hash, 5-run median with spread, `phys_footprint` peak, and `os_proc_available_memory` before and after.

| P | Task | Branch | Files | Acceptance |
|---|---|---|---|---|
| P0 | **T1 Model store + local-first load + migration** | `storage/model-store` | new `ios/AgentConsole/Inference/ModelStore.swift`; `Inference/MLXBackend.swift`; `Inference/ModelCatalog.swift`; Settings view | After a reinstall over the existing app, the LFM3/Qwen loads show blob-bytes delta = 0 and `first_load_s` < 1 s. A load in **airplane mode** succeeds with no network wait. The folder is under Application Support with `isExcludedFromBackup` = true. A receipt is committed under `docs/receipts/<date>/`. |
| P0 | **T2 Benchmark receipt fixes + memory probe** | `bench` (existing) | `ios/AgentConsole/Bench/BenchmarkRunner.swift`, new `Bench/MemoryProbe.swift` | `weight_files_bytes` equals the real blob sizes (within 1%). Receipts include `phys_footprint` peak, `os_proc_available_memory`, `recommendedMaxWorkingSetSize`, `currentAllocatedSize`, the blob delta and `local_snapshot_complete_before`. |
| P0 | **T3 Xybrid LLM backend + mmap canary (A1, A2, A3, A6, A7)** | `bench/xybrid-llm-ab` | new `Inference/XybridLLMBackend.swift`; `Bench/BenchmarkRunner.swift`; `project.yml` (no new dependency) | A1–A3 and A6/A7 receipts. A clear verdict on whether mmapped GGUF weights are charged to `phys_footprint` with residency on and off. LFM2.5-1.2B tok/s against MLX's 169. |
| P0 | **T4 LFM2.5-8B-A1B MLX 3-bit / mixed-bit conversion + quality gate** | `models/lfm25-8b-mlx-3bit` | conversion script outside the app (Apple Silicon or MLX-on-Linux host, **unverified**); `ModelCatalog.swift` entry | Weights ≤ 4.0 GB. Top-1 agreement with 4-bit ≥ 90% over 256 greedy tokens and perplexity regression ≤ 5% (my proposed thresholds). On iPad: no jetsam kill, `phys_footprint` peak < 5.0 GB, decode tok/s receipt (A4). |
| P1 | **T5 Inkling routing-locality profile** | `research/inkling-routing-profile` | offline script + JSON histograms (Mac/MLX host) | Hit-rate curve against a resident budget of 1–4 GB, per layer. A go/no-go number for T6 on Inkling: projected tok/s at the iPad budget. |
| P1 | **T6 MLX MoE expert-slot pool** (Strata/RFC #23324 design) | `mlx/moe-expert-stream` | new `Inference/ExpertStore.swift` (repacked per-expert file, pinned hot set, per-layer LRU slot pool, parallel `pread` with `F_NOCACHE`, next-layer prefetch); custom SwitchGLU hook for LFM2MoE first; repack script | LFM2.5-8B 4-bit runs on iPad with `phys_footprint` peak ≤ 5.0 GB. Outputs are bit-identical (greedy) to the fully resident model on a Mac. tok/s and hit-rate receipts. Then the same for Inkling 2 Mini at 2–3 bit under the T5 budget. |
| P1 | **T7 Inkling via llama.cpp master mmap probe** (only if T3/A6 shows mapped pages are not charged) | `bench/llamacpp-master-inkling` | `llama.xcframework` built from master (needs a Mac/Xcode CI host); thin backend | UD-IQ1_S 7.76 GB loads without jetsam. Warm and cold tok/s receipts. If it fails, a negative-result receipt. |
| P2 | **T8 Strata iOS cross-compile spike** (≤ 1 day) | `spike/strata-ios-metal` | external clone only | `strata_inkling_metal` builds for iOS arm64 (yes/no). If yes, an SQ-10 footprint receipt, expected to be a kill. |
| P2 | **T9 MLX kernel-efficiency check** against Strata's ~90% bandwidth claim | `research/mlx-bw-efficiency` | profiling notes | Measured achieved bandwidth per kernel on the iPad, with each gap to 90% attributed to a named kernel. |
| P3 | **T10 Core AI dense-model watch** | `research/coreai-watch` | notes | Qwen3-1.7B Core AI vs MLX on iPad, only if a free-Apple-ID build can use it. |

**Order:** T1, T2 and T4 can run in parallel now. T3 depends on T2 for the memory probes. T5 runs off-device in parallel. T6 starts after T4/T3 confirm the memory budget. T7 waits on the A6 result.

---

## Not answered by this research
- **When the next benchmark build is ready, and its path.** These reports contain no build or CI status. The orchestrator has to check the build pipeline and report the output path.
- **Unverified items to flag:**
  - whether wired, file-backed Metal pages count toward jetsam on iPadOS 27;
  - iPad M5 flash read throughput;
  - LFM2.5-8B active parameter count;
  - Inkling "loop" layer repetition;
  - the shipped swift-huggingface version (no `Package.resolved` in the repo);
  - Strata star count (the GitHub API was off-limits).

Source clones (read-only):
- `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\strata-src\s`
- `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\src`
- `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\lc`
- `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\src\`

---

# Adversarial critique

# Adversarial review of the on-device engine and model-fit report (iPad Pro M5, about 5.3 GB per-app limit)

## Summary

The report is mostly well-grounded. Its overall direction (no Strata on iOS, one Xybrid A/B for LFM2.5-8B only, MLX expert-slot pool for Inkling, storage fixes) holds up. The main problems:

- **Two load-bearing numbers rest on misread sources:**
  - The LFM2.5-8B speed estimate scales a CPU number by GPU bandwidth.
  - The Inkling budget ignores the layer loop, which its own config shows is real.
- **Some citations do not contain the claims attached to them:** the Core AI numbers, the Strata flags, and the 90% bandwidth figure.
- **Missing option:** an existing 5.38 GB Inkling quant.
- **Missing risk:** iOS virtual address space limits for large mmaps.

## A. Claims that are wrong or contradicted by sources

1. **The "253 tok/s on M5 Max with llama.cpp" scaling to about 63 tok/s is invalid.**
   - The Liquid blog gives 253 tok/s on Apple M5 Max as **CPU** decode, and 146 tok/s on a Ryzen AI Max+ 395. It does not name the engine. The device in its "about 30 tok/s on a phone" figure is not named either. Source: https://www.liquid.ai/blog/lfm2-5-8b-a1b
   - Scaling a CPU number by GPU bandwidth (153.6 vs 614 GB/s) is not like-for-like.
   - **Corrected estimate:** take about 1B active parameters plus the embedding and head reads, at about 3.5 effective bits. That is roughly 0.5–0.6 GB per token. At the measured ~70% efficiency (about 107 GB/s), the ceiling is about 180–200 tok/s.
   - Our dense LFM2.5-1.2B already runs at 169 tok/s on this iPad. A plausible range for LFM2.5-8B-A1B on MLX is therefore **about 80–150 tok/s, after an MoE gather penalty**. "50–90" is likely too pessimistic. Either way it is an estimate.
2. **"The active parameter count is unchecked; A1B suggests 1–1.5B" is now resolved.** Liquid states 8B total and 1B active (same blog). The BF16 GGUF is 16.95 GB, which is about 8.5B parameters. Source: https://huggingface.co/api/models/unsloth/LFM2.5-8B-A1B-GGUF/tree/main
3. **"'Loop' may repeat layers (unverified)" is wrong. The loop is confirmed, and it breaks the Inkling budget table.**
   - The config has `loop_config` with core layers 6–18, 3 passes, gate rank 64, and `is_varloop: false`. It also has a 3-layer MTP head and a 23-layer vision tower. Source: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
   - The LookAtMyQuant card independently refers to "all 48 logical visits", meaning 24 physical layers visited 48 times. Source: https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB
   - **Corrected budget:** expert reads per token are about 48 × 8 × 7.96M parameters ≈ 3.06B. That is **about 0.96 GB per token at 2.5 bits per weight, not 0.48 GB**. Per-expert size is 3 × 2304 × 1152 = 7.96M; the 24.5B total-expert figure is correct.
   - The looped layers re-use the same weights, so revisits within a token can hit the slot pool. That makes the hit rate more favourable for looped layers, but the miss traffic per token is roughly twice what the report assumes.
   - **The 10–25 tok/s estimate should be re-derived.** At equal hit rates it roughly halves, to about 5–12 tok/s.
   - The T5 profile must log routing **per logical visit**, not per physical layer.
4. **Inkling's original checkpoint is NVFP4 (ModelOpt).** Requantizing to 2–3 bit compounds error on top of a 4-bit source (same config URL). This risk is not mentioned.
5. **Strata's MAC.md is cited for things it does not contain.** Source: https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md
   - The flags `--mmap-experts`, `--resident-budget-gib` and `--expert-cache` are not in MAC.md. They are in DETAILS.md (verified).
   - The **8.62 vs 7.37 tok/s** figures are real. But they come from an M4 Mac mini with 32 GB, IQ3_XXS of a **47 GB model with a 12 GB arena**. That is not Inkling, which the report implies by context.
   - The "about 90% of peak bandwidth" claim appears in neither MAC.md nor the first 100K characters of DETAILS.md. It is attributable only to the Substack post, which I did not verify. Mark it unverified.
   - The report mixes `/main/` and `/master/` branch URLs. The local clone shows `main` only.
6. **"Its Inkling Metal backend assumes every expert is resident (`ink_metal_backend.mm:82`)" is weak evidence.**
   - Line 82 sits in `stats()`: `s.vram_experts_loaded = s.total_experts; // Unified zero-copy Apple Silicon memory!`. That is a reporting line, not proof of allocation behaviour.
   - MAC.md says Strata on Mac maps the model file, keeps an arena of hot experts, and writes an `.experts` copy for streaming Macs.
   - **Corrected statement:** the generic Mac engine streams. The Inkling-specific Metal backend appears to report everything as resident, and the allocation path is unconfirmed.
7. **The Core AI numbers (67.6 vs 65.1 on Qwen3-1.7B; 78.1 vs 100.2 on gpt-oss-20b) are unsupported by their citation.** The WWDC26 iOS guide contains no tok/s figures, no MLX and none of those models. Source: https://developer.apple.com/wwdc26/guides/ios/
   - Either find the real source (probably the john-rocky bench) or mark them unverified.
8. **The mmap jetsam evidence is overstated.**
   - Liquid's docs say weights "count as file-backed pages rather than app RSS" and that jetsam treats them "far more leniently". That is not "not charged". Source: https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile
   - Also, Metal residency or wired buffers made with `newBufferWithBytesNoCopy` can be charged. A6/A7 remain the right test, but restate the prior as "lenient, not free".
9. **The A6 "Xybrid Q4_0 4.84 GB" has no source listed.** It is not in unsloth's repo. It is `LiquidAI/LFM2.5-8B-A1B-GGUF` → `LFM2.5-8B-A1B-Q4_0.gguf`, 4,844,678,368 bytes. Source: https://huggingface.co/api/models/LiquidAI/LFM2.5-8B-A1B-GGUF/tree/main
10. **The constraint "`ub * n_expert_used <= n_slots`" in RFC #23324 is not carried into the design.**
    - The RFC numbers are verified, and the PoC requires `--no-mmap` and `--no-warmup`. Source: https://github.com/ggml-org/llama.cpp/discussions/23324
    - With top-8 routing, a 512-token prefill micro-batch needs up to 4096 slots per layer. So a bounded slot pool forces tiny prefill micro-batches or a separate streamed prefill path.
    - The report's 8K-context prefill or TTFT will be far worse than its decode. This is a missing risk for T6 on both models.

## B. Claims checked and confirmed

- **Unsloth GGUF sizes:** UD-Q3_K_XL is 4,003,114,592 bytes and UD-IQ4_XS is 4,264,996,448 bytes (unsloth tree URL above).
- **Strata's Hugging Face packs:**
  - SQ-10 is 10.47 GB with 92.35% top-1 agreement on test prompts and 89.79% on WikiText-2.
  - SQ-15 is 14.35 GB; SQ-20 is 19.35 GB.
  - The engine is MIT-licensed, and the platforms are Windows, Linux and Mac.
  - Source: https://huggingface.co/strata-engine/Inkling-2-Mini-Strata
  - **Nuance:** the card says a 4 GiB GPU is enough because the rest stays in system RAM. On PC, "everything resident" therefore means GPU plus RAM, not GPU only.
- **No iOS code in Strata.** In the local clone (HEAD `c2347740`, 2026-11-25), the only `ios` paths are Tauri `AppIcon` PNGs. There are no `.swift` files or xcframework.
- **Xybrid's pinned llama.cpp** has `src/models/lfm2moe.cpp` and no Inkling (grep of the local `xyb/lc`).
- **LEAP is deprecated.** Liquid describes it as "a Kotlin Multiplatform wrapper around llama.cpp" and the replacement is `llama.xcframework`. Source: https://docs.liquid.ai/deployment/on-device/llama-cpp/migrating-from-leap-sdk
- **Inkling config:** 24 layers, 128 experts, top-8 routing, expert intermediate size 1152, no shared experts. Hidden size is 2304 and the vocabulary is 201,024.

**Not checked by me:** the Qualcomm/Nexa, Cactus, ZETIC and linuxcompatible URLs, the Substack post, the inkling-ios LOG, and the apple-silicon-llm-bench numbers.

## C. Missing options and research angles

1. **An existing 5.38 GB Inkling quant is the biggest omission.**
   - `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB` is about 1.65 bits per weight, made with trellis-coded weights plus distillation-based training.
   - Reported results: GSM8K 92.5% (BF16 94.0%), HumanEval 76.8% (BF16 89.6%). On an M5 Max it peaks at about 6.41 GB of RAM and decodes at about 66 tok/s.
   - It needs the `lamq` runtime (Python, MLX/CUDA). There is no Swift port and no iPad claim.
   - It does not fit 5.3 GB as-is. But streaming only about 1–1.5 GB of it (the heavy layers 17–23 are about 1.9 GB) is a far smaller problem than streaming about 4–5 GB of a 2.5-bit model.
   - Source: https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB
   - Add an extreme low-bit quant plus a small streamed remainder as a ranked route, and add a task to port the LUT/trellis dequant to Metal.
2. **Other prebuilt quants exist, so T4 may not need a conversion.**
   - Inkling: `mlx-community/Inkling-2-Mini-oQ3e-mtp` (3-bit MLX) and `ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF`. The smallest GSQ-RCO file, IQ2_S, is 8.98 GB.
   - LFM2.5-8B: `JANGQ-AI` and `OsaurusAI` `LFM2.5-8B-A1B-JANG_2L` (about 2-bit MLX), and `OsaurusAI/...-MXFP4`.
   - Sources: https://huggingface.co/api/models?search=Inkling-2-Mini and https://huggingface.co/api/models?search=LFM2.5-8B-A1B
3. **Smaller LFM GGUFs are omitted** from the A/B table (sizes from the unsloth tree):

   | File | Size |
   |---|---|
   | UD-IQ3_XXS | 3.27 GB |
   | UD-IQ3_S | 3.57 GB |
   | UD-Q2_K_XL | 2.93 GB |
   | MXFP4_MOE | 5.35 GB (too big) |

4. **Address space on iOS is not covered.** Mapping a file of 4.8–7.8 GB may need `com.apple.developer.kernel.extended-virtual-addressing`, which may also be unavailable on a free Apple ID. This is a missing gate for A6, A7 and T7. Unverified.
5. **MLX memory knobs are not covered.**
   - `Memory.cacheLimit` / `GPU.set(cacheLimit:)` and the wired limit can inflate or bound `phys_footprint` independently of weight size.
   - MLX's safetensors loading reads into allocated buffers rather than true mmap. T6 therefore needs a custom loader, not just a SwitchGLU hook.
   - Add a run that shows MLX's buffer cache is not what pushes the 4.85 GB 4-bit model (A5) over the limit.
6. **Sideloadly and free Apple ID specifics are incomplete.**
   - Free accounts allow 3 sideloaded apps, and a provisioning profile lasts 7 days.
   - Sideloadly can rewrite the bundle ID, and the report lists that as a risk. The concrete check: record `Bundle.main.bundleIdentifier` in every receipt, and keep Sideloadly's bundle-ID setting fixed.
   - Weights survive only if the container persists. Add a receipt that confirms the container UUID path persists across reinstalls.
7. **MTP head.** Inkling ships a 3-layer MTP head. Strata's speed claims partly come from multi-token prediction. That counts as speculative-style decoding, so note it as a speed lever and flag whether it is in scope.
8. **The user's direct question is not answered.** The report punts on when the next benchmark build is ready and its path. The orchestrator must answer from CI or the local build output; research alone cannot. The request also asked whether every engine was explored: Strata and Xybrid were covered in depth, while Cactus, Nexa and ZETIC were assessed from documentation only, with no hands-on build.

## D. Corrected statements (drop-in)

- "LFM2.5-8B-A1B has 8B total and 1B active parameters (Liquid). The expected iPad MLX 3-bit decode is about 80–150 tok/s, based on bandwidth from 169 tok/s on the dense 1.2B. This is an estimate."
- "Inkling 2 Mini loops core layers 6–18 three times, for 48 logical layer visits per token. Expert reads are about 0.96 GB per token at 2.5 bits per weight, and slot-pool estimates roughly halve, to about 5–12 tok/s, unless re-visits hit the cache."
- "The Strata flags are documented in DETAILS.md. The 8.62/7.37 tok/s result is from a 47 GB model on an M4 Mac mini. The 90% bandwidth figure is unverified."
- "The Core AI vs MLX figures are unsourced. The WWDC26 guide contains no benchmarks."
- "Mapped GGUF pages are file-backed and treated 'more leniently' (Liquid docs), not proven uncharged. A6/A7 decide this, subject to the address-space entitlement."
- "The bounded slot pool constrains prefill batch size (`ub × top-k ≤ slots`). Prefill or TTFT needs its own design."

Sources checked: Liquid LFM2.5-8B blog; unsloth and LiquidAI GGUF trees; Inkling `config.json`; Strata MAC.md and DETAILS.md; Strata Hugging Face card; the WWDC26 iOS guide; llama.cpp discussion #23324; Liquid's mobile and LEAP-migration docs; Hugging Face model searches; LookAtMyQuant and ISTA-DASLab trees and cards. I also inspected the local clones at `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\strata-src\s` (the working tree is empty, so I read it through `git show`) and `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\lc`.

---

# Raw reports

## strata

# Strata: what it is and whether it fits the iPad app

**Short answer:** "Strata" is almost certainly **Niko1221/Strata**. It is an MIT-licensed local inference engine and desktop app with its own C++ engine and a native Apple Silicon Metal backend. It has a dedicated Inkling 2 Mini path. It ships for macOS, Windows and Linux only. There is no iOS or iPadOS build, no Swift package and no embeddable library. Its smallest Inkling 2 Mini pack is 9.75 GiB, and its Metal path for Inkling loads every expert into memory, so as shipped it does not run Inkling 2 Mini within our ~5.3 GB limit. Its bigger-than-RAM expert streaming (`--mmap-experts`, `--expert-cache`) is the part worth studying, and it is a pattern we could copy in MLX rather than a dependency to add. It does not support LFM models.

## Candidates, ranked by likelihood

| # | Candidate | What it is | iOS? | Evidence |
|---|---|---|---|---|
| 1 | **Niko1221/Strata** (very likely) | Local LLM engine and app for MoE models, spread across VRAM, RAM and SSD, with OpenAI/Anthropic-compatible APIs | No | [repo](https://github.com/niko1221/strata), [releases](https://github.com/niko1221/strata/releases), [M5 Ultra write-up](https://elitetestengineering.substack.com/p/strata-on-the-m5-ultra), [linuxcompatible 0.2.16](https://www.linuxcompatible.org/story/strata-0216-runs-its-own-engine-drops-llamacpp-claims-identical-output/), [v0.1.38 article](https://www.linuxcompatible.org/story/strata-v0138-runs-a-125billionmodel-llm-on-any-gaming-pc/) |
| 2 | commoncompute/lokahi ("strata-llm") | Python research engine for Apple Silicon Macs (a fork of oLLM), memory-tiered, MLX baseline. 6 stars; SSD paging listed as a goal, not shipped | No (macOS) | [repo](https://github.com/commoncompute/lokahi) |
| 3 | ro99/strata | C++20/CUDA MoE engine for models bigger than VRAM or RAM (VRAM + RAM + NVMe). Apache-2.0, 11 stars, NVIDIA only | No | [repo](https://github.com/ro99/strata) |
| 4 | "Strata AI" | AI keyboard for iPhone, cloud-based (Gemini). Not an engine | Cloud app | [Product Hunt](https://www.producthunt.com/products/strata-ai/reviews) |
| - | EveryDev listing | Describes a "Strata" as a GPU cluster manager, which conflicts with the others. Unverified | - | [everydev](https://www.everydev.ai/tools/strata) |

Candidate 1 is the best match because it is the only one with high visibility, a Metal backend, active Apple Silicon releases and explicit Inkling 2 Mini support.

## Niko1221/Strata details

- **License:** MIT, copyright 2026 Niko1221 and contributors ([LICENSE](https://raw.githubusercontent.com/Niko1221/Strata/main/LICENSE)).
- **Maturity:**
  - Latest release is 0.2.22 (25 Nov 2026), with near-daily releases from 0.2.15 to 0.2.22 between 16 and 25 Nov ([releases](https://github.com/niko1221/strata/releases)).
  - My shallow clone shows the last commit at 2026-11-25T23:59+01:00 and about 8,486 files.
  - Stars are about 45.7k and forks about 5.1k according to [linuxcompatible](https://www.linuxcompatible.org/story/strata-0216-runs-its-own-engine-drops-llamacpp-claims-identical-output/). I could not check this because the GitHub API was off-limits.
  - The project is mostly one developer.
- **Engine:**
  - Since 0.2.16 it uses its own engine written from scratch, replacing ggml and llama.cpp's matrix kernels (same linuxcompatible article). The M5 Ultra write-up still calls it "llama.cpp-derived," which describes the older lineage.
  - Backends are CUDA, HIP, SYCL and Metal.
  - The Metal port has its own kernels (`engine/metal/kernels/*.metal`: moe_mm, gemv, gdn, router, sample and others) plus a CUDA-to-Metal compatibility shim (`engine/metal/include/metal_compat/cuda_runtime.h`). This comes from my clone.
- **Platforms:** Windows x64, Linux x86_64/aarch64 and macOS arm64 (DMG, signed and notarized Metal engine packages, macOS 15 minimum).
  - The only "ios" files in the repo are default Tauri icon assets (`app/desktop/src-tauri/icons/ios/*`).
  - There is no iOS target, no xcframework and no Swift package.
  - There is no Apple Neural Engine (ANE) path.
- **Formats:** GGUF (including Unsloth UD-Q4_K_XL quants), its own ".strata"/SQ packs with `experts.bin`, and NVFP4 ([README](https://raw.githubusercontent.com/Niko1221/Strata/main/README.md), [DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/master/docs/DETAILS.md)). Native Metal quant types include IQ3_XXS, Q2_K, Q3_K, Q8_0 and NVFP4 (0.2.20 notes).
- **Models:**
  - Qwen3.8-Flash-Next (about 24,576 experts, about 10 active per token), Tencent Hy4-Flash, Qwen3.6-35B-A3B and Inkling 2 Mini.
  - **No LFM or Liquid AI support.** The only "liquid" file in the clone is a CSS file.
- **Bigger-than-memory features** ([DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/master/docs/DETAILS.md)):
  - `--mmap-experts`: experts are mapped from files and the OS page cache holds what the GPU doesn't.
  - `--resident-budget-gib N`: the hottest N GiB of experts are copied into RAM and the rest are read from files.
  - `--expert-cache N`: a GPU-resident expert cache (each extra GB holds about 700 experts).
  - There is also a conversation and KV cache that spills to SSD.
  - The README admits that with a small GPU, "most experts come from the SSD and it is much slower."
- **Inkling 2 Mini packs** (0.2.19 notes):
  - SQ-10 is 9.75 GiB, described as fitting "16 GiB Macs".
  - SQ-15 is 13.36 GiB and SQ-20 is 18.02 GiB.
  - The README also says "A 4 GiB card is enough to start" for the Unsloth packs, presumably with experts streamed from RAM or SSD. That is a desktop setup.
- **Apple benchmarks:**
  - Up to 526 GB/s (89.9% of peak bandwidth) and 6,769 tok/s prefill on M5 Max (0.2.20).
  - Inkling 2 Mini at 108.9–114.5 tok/s decode on M5 Max 128 GB (0.2.19), and 73.4 tok/s for SQ-15/SQ-20 on unnamed Apple Silicon (0.2.20).
  - On M5 Ultra, Strata (IQ3_S with MTP, a draft-token method) against MLX 4-bit gave 142.1 vs 41.7 tok/s decode and 1,139 vs 404 tok/s prefill ([write-up](https://elitetestengineering.substack.com/p/strata-on-the-m5-ultra)). This is not a like-for-like comparison: the quantizations differ and MTP is a speculative method.
- **Correctness history:** 0.2.15 computed partial layers on M1–M4 (first-token divergence 0.303, fixed to 0.0068 in 0.2.16). Apple results need to be checked per release.

## Can we integrate it into the iPad app?

**Not as a drop-in.** What the clone shows:
- `engine/src/inkling2/metal/` builds as a **static library** (`strata_inkling_metal`): Objective-C++ (`ink_metal_backend.mm`) with embedded `.metal` shaders, linking only Metal, Foundation and MetalPerformanceShaders. There is no AppKit dependency. Compiling it for iOS arm64 is plausible but nobody has tried it.
- The **Inkling Metal backend assumes everything is resident.** `stats()` hard-codes `vram_experts_loaded = total_experts` with the comment "Unified zero-copy Apple Silicon memory!" (`ink_metal_backend.mm:82`).
- The loader either maps the whole file read-only (`ink_metal_common.h:451`) or copies it into an anonymous arena (`:706`). The anonymous arena is dirty memory and would count fully against jetsam.
- The read-only mapping can only stay under 5.3 GB if iOS treats GPU-accessed file-backed pages as reclaimable. That is unverified, and there is no expert cache or prefetch on this path.
- The flags that do stream experts (`--mmap-experts`, `--expert-cache`) live in the main engine and Hy4 paths, which were designed around a desktop GPU-VRAM/RAM split and a 4 GiB "safety headroom elsewhere."
- The engine is driven through a CLI and HTTP server with a Node/Tauri host. We would have to write a C/Swift bridge, a tokenizer and chat-template handling ourselves, and keep up with near-daily releases.

**What it would add over MLX for our goals:**
1. **Inkling 2 Mini:** a second, independently tuned Metal implementation and an existing quant pipeline. But its smallest pack (9.75 GiB) is about 2x our limit, so nothing is gained unless we add expert streaming ourselves, and that is the same work we would do in MLX.
2. **LFM2.5-8B-A1B:** nothing, because it isn't supported.
3. **Speed:** its Metal kernels report about 90% of peak bandwidth on M5 Max, against our measured ~70% with MLX on M5 iPad. That is a plausible 1.2–1.3x decode ceiling for models that fit. The 3.4x figure from the write-up includes MTP and different quants, so we shouldn't plan on it.
4. **Design to copy:** the most useful takeaway is its architecture: a byte-budgeted resident set of hot experts, cold experts read from files on demand, and an expert cache sized in multiples of the largest expert blob. We can build that in our MLX path with mmap'd safetensors and per-layer expert gathering. Its `--ple-io ram` result (about 86% faster prefill when a hot table is kept in RAM instead of read from SSD) shows how costly SSD reads are.

**Recommendation:** don't adopt Strata as our engine. If we want to experiment, a time-boxed spike would be to cross-compile `strata_inkling_metal` for iOS arm64 and check whether the read-only mapping path stays under jetsam with SQ-10. Expect it not to, because GPU-touched pages are likely to count. In parallel, port its expert-residency and streaming design into our MLX path. That path is where both LFM2.5-8B-A1B (4.85 GB) and Inkling 2 Mini have a realistic route to fitting in 5.3 GB.

Source checkout (read only, outside the repo): `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\strata-src\s`

**Sources:** [Niko1221/Strata](https://github.com/niko1221/strata) · [releases](https://github.com/niko1221/strata/releases) · [README](https://raw.githubusercontent.com/Niko1221/Strata/main/README.md) · [DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/master/docs/DETAILS.md) · [LICENSE](https://raw.githubusercontent.com/Niko1221/Strata/main/LICENSE) · [Strata on the M5 Ultra](https://elitetestengineering.substack.com/p/strata-on-the-m5-ultra) · [linuxcompatible 0.2.16](https://www.linuxcompatible.org/story/strata-0216-runs-its-own-engine-drops-llamacpp-claims-identical-output/) · [linuxcompatible v0.1.38](https://www.linuxcompatible.org/story/strata-v0138-runs-a-125billionmodel-llm-on-any-gaming-pc/) · [commoncompute/lokahi](https://github.com/commoncompute/lokahi) · [ro99/strata](https://github.com/ro99/strata) · [Strata AI (Product Hunt)](https://www.producthunt.com/products/strata-ai/reviews) · [EveryDev listing](https://www.everydev.ai/tools/strata) · [DeepWiki](https://deepwiki.com/Niko1221/Strata) (returned 429, not read)

## xybrid-llm

# Xybrid LLM path deep-dive (v0.21.0 and main @ 9fbd8b1, 2026-11-25)

## Verdict
- **Add an A/B with LFM2.5-8B-A1B only.** Compare MLX 3-bit or 4-bit against Xybrid's llama.cpp running unsloth `UD-IQ4_XS` (4.26 GB) or `UD-Q3_K_XL` (4.00 GB). This is the one model where a GGUF fits under about 5.3 GB on llama.cpp's mmap + Metal zero-copy path. Use Q4_0 (4.84 GB) only as a stress probe.
- **Inkling 2 Mini cannot run through Xybrid.**
  - Its pinned llama.cpp has no `inkling2_loop` arch.
  - The Swift API has no CPU-expert or streaming controls.
  - Even the smallest GGUF (7.76 GB IQ1_S) is larger than the memory budget.
  - The only proven iOS route is a CPU-only, patched llama.cpp with `pread` expert streaming (kevindechang/inkling-ios). It needs the increased-memory-limit entitlement, which is not effective on our free-Apple-ID build.
- **Strata identified:** it is the Strata engine, github.com/Niko1221/Strata (HF org `strata-engine`). It is desktop only (Windows, Linux, macOS Metal), with no iOS build. Its macOS SSD expert-streaming design is a useful blueprint, not something we can integrate.

## 1. Xybrid's LLM backends
- **Version:** workspace is `0.21.0`. `Package.swift` pins `sdkVersion = "0.21.0"`, the platform is iOS 16+ only, and the prebuilt `XybridFFI.xcframework` ships ios-arm64 and simulator slices only (local clone `Package.swift`).
- **llama.cpp with Metal on device:**
  - `.bazelrc`: "Apple Gate 1: iOS device staticlib … llama.cpp builds with Metal ON".
  - `build.rs` sets `GGML_METAL=ON` for `target_os == "ios"`.
  - The simulator is forced to CPU (`effective_gpu_layers` returns 0 because "The iOS Simulator's Metal path produces invalid tokens").
- **Candle:** used for Whisper safetensors only. CHANGELOG 0.21.0 says "Candle left the platform presets in #476" and the MPS frameworks were dropped.
- **Other backends:** ONNX Runtime (vendor/ort-ios) and whisper.cpp, both for speech and other tasks.
- **Pinned llama.cpp:**
  - main is at submodule commit `b1b770d9`, dated 2026-11-13. CHANGELOG "[Unreleased]" lists bumps to llama.cpp v0.9.0 and then v1.0.0.
  - So v0.21.0, released 2026-11-22, ships an older llama.cpp.
  - `lfm2moe` is present (`src/llama-arch.cpp:140`). There is no `inkling2_loop`; I grepped all 177 `src/models/*` files.
  - The Inkling GGUFs (ggml-org, created 2026-11-20) use arch `inkling2_loop` (HF API `gguf.architecture`).
  - **Inference, not verified:** v0.21.0 cannot load Inkling GGUFs.
- **llama.cpp params Xybrid actually sets** (`crates/llama-cpp-sys/wrapper.cpp:182`, `llama_load_model_from_file_c`):
  - Only `n_gpu_layers` is set; everything else is `llama_model_default_params()`.
  - So `load_mode` is AUTO, which means mmap (`llama-model-loader.cpp:564`), and `lazy_mode` is AUTO.
  - **Not passed:** `tensor_buft_overrides` (`-ot`), n-cpu-moe, load_mode or no_host.
  - **Context:** `n_ctx`, `n_threads` (0 means `hardware_concurrency`), `n_batch` (default 512), `flash_attn` (default true). There is no KV cache type, so KV is F16.
  - **`gpu_layers`:** defaults to 99 (`runtime_adapter/types.rs:680`). The GGUF `model_metadata.json` template exposes only `model_file`, `chat_template`, `context_length` (default 4096) and `generation_params` (`execution/template/metadata.rs`). **CPU-only or partial offload cannot be chosen from Swift.**
- **Registry** (live `https://registry.xybrid.dev/v1/models`, 41 entries):
  - LLMs: lfm2.5-230m, lfm2.5-350m, lfm2.5-1.2b-instruct/-thinking, lfm2.5-vl-3b, lfm3-1.7b, lfm3-vl-2b, lfm2-vl-450m, d1-2b (decision), qwen2.5-0.5b, qwen3.5-0.8b/2b, gemma-3-1b, gemma-4-e2b/e4b, functiongemma-270m, llama-3.2-1b, ministral-3-3b, mistral-7b, phi4-mini, smollm2-360m. bonsai-27b is listed for macos-arm64 only.
  - **There is no LFM2.5-8B-A1B and no Inkling.**
  - **Variant format:** GGUF Q4_K_M from the vendor HF repo, e.g. `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (`/v1/models/lfm2.5-1.2b-instruct`).
  - Note that lfm3-1.7b and lfm3-vl-2b list no `universal` or iOS variant.
- **Published tok/s on Apple devices:** none. A repo-wide grep found nothing.

## 2. Swift text-generation API (`bindings/apple/Sources/Xybrid/Xybrid.swift`)
```swift
Xybrid.initialize()                                   // no apiKey = fully on-device
let url = appSupport.appending(path: "lfm25-8b/LFM2.5-8B-A1B-UD-IQ4_XS.gguf") // own dir, writable
let model = try await ModelLoader.fromModelFile(url).load()   // no download; writes model_metadata.json beside it (ctx capped 4096)
// alt: ModelLoader.fromHuggingFace("unsloth/LFM2.5-8B-A1B-GGUF:UD-IQ4_XS") (downloads+caches), .fromRegistry("lfm2.5-1.2b-instruct")
let opts = XybridRunOptions.make(generationConfig: .greedy(maxTokens: 256))
for try await t in model.streamTokens(envelope: .text(prompt), options: opts) { print(t.token, terminator: "") }
let r = try await model.runAsync(envelope: .text(prompt), options: opts)
r.metrics.ttftMs; r.metrics.prefillTps; r.metrics.decodeTps; r.metrics.tokensOut   // XybridInferenceMetrics
```
- **Multi-turn:** `XybridConversationContext()` with `.setSystem(envelope:)` and `.push(envelope:)`, then `runWithContext` or `runStreamWithContext`.
- **Tools:** `GenerationConfig.withTools([XybridToolDefinition(name:description:parametersJson:)])`. Read `result.toolCalls`, or the terminal `StreamToken.toolCalls`/`rawText`, then send `Envelope.toolResults(...)`.
- **Chat templates:** GGUF Jinja via its own HF-compatible renderer (`llama_cpp/jinja_template.rs`), with a special case for LFM3.
- **Memory:** `Xybrid.releaseMemory()`, `setAutoRelease(true)`, `model.unloadAsync()`.
- **Re-download avoidance:**
  - `fromModelFile` and `fromDirectory` do no network I/O. The doc says the app bundle is read-only, so give each GGUF its own directory.
  - Registry downloads use the SDK cache: check `Xybrid.hasCachedModelData(id)` or `cachedModelPath(id)` first.

## 3. iOS weight loading, mmap and jetsam: evidence
- **Zero-copy into Metal:** with mmap on unified memory, llama.cpp wraps the file mapping with `newBufferWithBytesNoCopy` (`ggml-metal-device.m:2206`; loader `llama-model.cpp:2052`). Weights are therefore file-backed pages, not anonymous dirty memory.
- **Residency sets are on for iOS SDK ≥ 18** (`ggml-metal-device.m:19-24`). They call `requestResidency` and run a keep-alive heartbeat thread (lines 933-1009), and can be disabled with the env var `GGML_METAL_NO_RESIDENCY`. Residency wires the pages for the GPU, which **defeats the "clean pages don't count" benefit**. Whether wired, file-backed GPU pages count toward jetsam `phys_footprint` is **unverified**; no authoritative source found.
- **Device evidence (inkling-ios LOG, iPhone 15 Pro, 9.1 GB GGUF)** ([LOG.md](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/docs/LOG.md)):
  - P0-P4: stock load hung because mmap issued `madvise(WILLNEED)` over the whole file; fixed with `LLAMA_MMAP_NO_PREFETCH=1`.
  - P6: Metal with residency off gave about 15 s/tok, because "host-pointer Metal buffers over file-backed pages faulted every token".
  - P7: Metal with residency on crashed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
  - P9-P14: "OS page cache never warmed on iOS"; mmap'd memcpy ran 869-961 ms/tok.
  - Fix: CPU-only, dense weights resident, an explicit anonymous LRU expert cache, and `F_NOCACHE` parallel `pread` (P15: 4.72 tok/s; final 21-23 tok/s, but in a modified 1-pass, top-4 shape).
  - It also found a memory-compressor cliff at about 4.42-4.45 GB footprint.
  - **Requires the increased-memory-limit entitlement** ([README](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/README.md)).
- **Hybrid split cost:** dense on Metal with experts on CPU caused about 900 graph splits per token and ran 2.8× slower than pure CPU (M2, Mac stand-in) ([ARCHITECTURE.md](https://raw.githubusercontent.com/kevindechang/inkling-ios/main/docs/ARCHITECTURE.md)). That argues against `-ot exps=CPU` on iPad even if Xybrid exposed it.
- **Lazy mode does not cover MoE experts:** llama.cpp `LLAMA_LAZY_MODE` ("read the rows of tensors marked by the arch on demand", `llama.h:220`) only marks Gemma-style `per_layer_tok_embd` (`llama-model.cpp:1991`).
- **Metal working-set limit:** a 12 GB iPhone Air reported `recommendedMaxWorkingSetSize` of 8192 MiB, and a 7.32 GB model failed to load (thread has no answers) ([Apple forum](https://developer.apple.com/forums/thread/805161)). I found no Metal-specific mmap measurements in [llama.cpp discussion #29347](https://github.com/ggml-org/llama.cpp/discussions/29347).

## 4. Fit on iPad (5.3 GB limit, 16 GB device)
- **LFM2.5-8B-A1B GGUF sizes** ([LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF), [unsloth](https://huggingface.co/unsloth/LFM2.5-8B-A1B-GGUF)):
  - Q4_0 4.84 GB, Q4_K_M 5.16 GB, UD-IQ4_XS 4.26 GB, UD-Q3_K_XL 4.00 GB, UD-IQ3_S 3.57 GB.
  - Add KV for 6 attention layers (small) plus Metal compute buffers plus the app.
  - IQ4_XS and Q3_K_XL should fit. Q4_0 and Q4_K_M are borderline or likely over if residency counts. **Unverified; measure.**
- **Other on-device data points for this model:** about 32 tok/s on iPhone 17 Pro, on Core AI int4 (4.7 GB), not llama.cpp ([HF card](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI)). Liquid AI claims about 30 tok/s "on a phone" without naming device or runtime, and 253 tok/s on an M5 Max with llama.cpp ([Liquid blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)).
- **MLX side:** [mlx-community/LFM2.5-8B-A1B-MLX-4bit](https://huggingface.co/mlx-community/LFM2.5-8B-A1B-MLX-4bit) (about 4.85 GB) is borderline, as you already know. Convert a 3-bit MLX build for a like-for-like comparison against IQ3 and Q3_K_XL.
- **Inkling 2 Mini:** smallest GGUF is unsloth UD-IQ1_S at 7.76 GB; UD-Q2_K_XL is 9.14 GB ([unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)). Strata's SQ-10 pack is 10.47 GB. Neither fits without streaming.

## 5. Strata
- It is the HF org [strata-engine](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata), with engine repo [Niko1221/Strata](https://raw.githubusercontent.com/Niko1221/Strata/main/README.md). It is MIT licensed, v0.2.22, built with llama.cpp/ggml parts.
- **Hardware:** NVIDIA, AMD, Intel Arc, and Apple Silicon M1+ on macOS 15+. **No iOS or iPadOS.**
- **Inkling packs:** SQ-10, SQ-15 and SQ-20 in a custom `.strata` format, not GGUF. SQ-20 is reported identical to the original; SQ-10 matches the original's top token 92.35% of the time. It decodes 27.6 tok/s on an M4 Mac mini 32 GB with SQ-10.
- **Mac streaming design** ([docs/MAC.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md)):
  - A hot-expert arena, wired with `mlock`.
  - SSD reads with next-layer read-ahead.
  - A repacked `.experts` file so each expert is one read: 8.62 vs 7.37 tok/s, and 31.5 vs 46.6 ms SSD wait per token.
  - Residency sets.
- A Leanpub book ([sample](https://s3.amazonaws.com/samples.leanpub.com/stratafastlocalinferenceforqwen38-flash-next125b-sample.pdf)) describes the same project. I found no other credible "Strata" inference engine.

## 6. Recommended A/B
1. **Same app, same prompt, greedy decoding, 256 output tokens, 5 runs, report the median.** Log `os_proc_available_memory()`, `task_vm_info.phys_footprint`, `MTLDevice.recommendedMaxWorkingSetSize`, `currentAllocatedSize`, and Xybrid's `metrics.decodeTps`/`prefillTps`/`ttftMs`.
2. **Models:**
   - LFM2.5-1.2B Q4_K_M (registry) vs MLX 4-bit, as a pure engine-speed comparison against your measured 169 tok/s.
   - LFM2.5-8B-A1B: `UD-IQ4_XS` and `UD-Q3_K_XL` via `fromModelFile`, vs MLX 3-bit and 4-bit.
   - Probe Q4_0 last, as a jetsam canary.
3. **Optional residency-off variant:** call `setenv("GGML_METAL_NO_RESIDENCY","1",1)` before the first load. It is read with `getenv` in the same process. This tests whether unwired file pages lower the footprint, at the risk of P6-style faulting.
4. **Skip the Xybrid route for Inkling.** That needs a custom llama.cpp build in the inkling-ios style (CPU-only `pread` expert cache), which is separate engine work and depends on the memory entitlement.

Source clone used for the code citations above (read-only, outside the repo): `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\src`. llama.cpp files at the pinned commit are in `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\lc`.

Other sources: [xybrid repo](https://github.com/xybrid-ai/xybrid), [dev.to/xybrid](https://dev.to/xybrid), [ggml-org Inkling GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF).

## other-engines

# Other on-device engines for running models larger than about 5.3 GB on the iPad M5: re-assessment (2026-11-26)

## Key findings

1. **Strata is Niko1221/Strata**, an open-source, MIT-licensed engine for large MoE models, first published 2026-09-24 and now at v0.2.22. It runs on Windows, Linux and macOS (Apple Silicon, with its own Metal engine). It has **no iOS or iPadOS support**. It matters because its Mac engine already does what we want:
   - streams experts from SSD out of an mmap'd file,
   - keeps a "hot-expert arena" in memory,
   - prefetches the next layer's likely experts,
   - can write an "experts' copy" file so each expert is one SSD read.

   It also ships Inkling 2 Mini packs (SQ-10 is 10.47 GB). Sources: [repo](https://github.com/niko1221/strata), [docs/MAC.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md), [HOW_IT_WORKS.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/HOW_IT_WORKS.md), [stratallm.org](https://stratallm.org/) (a community site, not the project's own), [wiki](https://ai.miraheze.org/wiki/Strata). Use it as a design reference and MIT code source, not as a drop-in. An unrelated project, [kytheros/strata](https://github.com/kytheros/strata), is a memory tool and not an engine.

2. **Getting past the 5.3 GB limit depends on whether weight memory counts toward the app's memory footprint.**
   - MLX copies weights into its own GPU buffers, and all of that counts toward the limit.
   - llama.cpp maps the model file into memory (mmap) by default. Liquid's docs say mapped weights "count as file-backed pages rather than app RSS," which iOS jetsam treats "far more leniently" ([Liquid mobile guide](https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile)).
   - The [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) notes that mapped "clean pages aren't charged to phys_footprint". It records llama.cpp at "191 MB mmap" for Gemma 4 E2B on iPhone 17 Pro.
   - **Unverified:** whether GPU-resident Metal no-copy buffers built over mapped pages stay uncharged at sizes of 5–10 GB on iPadOS 27. A [dev-forum thread](https://developer.apple.com/forums/thread/805161) shows a 13B llama.cpp load failing on a 12 GB iPhone near `recommendedMaxWorkingSetSize`, which was 8 GiB there. That thread has no answer.
   - **This is the single highest-value experiment.** The iPad has 16 GB of physical RAM, so a 4.8–9 GB file could sit fully in the OS page cache while the app's own footprint stays small.

3. **Xybrid, which is already in the app, ships llama.cpp with Metal.** I checked a shallow clone of xybrid-ai/xybrid, HEAD `9fbd8b1` (2026-11-25):
   - llama.cpp is pinned at `b1b770d`.
   - `crates/llama-cpp-sys/wrapper.cpp` uses `llama_model_default_params()`, which resolves to `LLAMA_LOAD_MODE_AUTO`. On Metal that means mmap (`src/llama-model.cpp:1701`, `llama-model-loader.cpp:564`).
   - The pinned llama.cpp **supports `lfm2moe`** (`src/llama-arch.cpp:140`).
   - It does **not** support Inkling 2 Mini. That model's GGUF architecture is `inkling2_loop`, which exists only in llama.cpp master (`LLM_ARCH_INKLING2_LOOP`).
   - So LFM2.5-8B-A1B can probably be tried through Xybrid's existing LLM path with no new dependency. Inkling needs a newer llama.cpp than Xybrid's pin.
   - The pin also has `llama_lazy_mode` ("read the rows of tensors marked by the arch on demand (requires mmap)"). That is upstream groundwork for on-demand reading.

4. **Model files** (from the Hugging Face API):
   - **LFM2.5-8B-A1B-GGUF:** Q4_0 is 4,844,678,368 bytes and Q4_K_M is 5,155,564,768 bytes ([HF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF)). The license is LFM Open License v1.0.
   - **Unsloth Inkling-2-Mini-GGUF** (architecture `inkling2_loop`, Apache-2.0): UD-IQ2_XXS is 8.49 GB and UD-Q2_K_XL is 9.14 GB ([HF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)).
   - **Inkling active size** (my estimate from its [config](https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json): 24 layers, hidden size 2304, 128 experts with top-8, expert intermediate size 1152). That works out to about 1.5B active expert parameters, plus about 0.45B attention and about 0.46B LM head, so roughly **2.4B parameters read per token**. The "loop" setting may repeat layers, which is unverified.
   - At about 2.5 bits per weight, that is roughly 0.75 GB per token. If it all stays in memory, the ceiling is about 140 tok/s at our measured ~108 GB/s. If experts come from flash on a cache miss, it falls to single digits per second (my estimate, unmeasured).

## Verdict per engine

| Engine | Can it exceed 5.3 GB? | Expected speed vs our MLX numbers | Verdict |
|---|---|---|---|
| **llama.cpp direct** (`llama.xcframework`) | **Likely**, via mmap; unverified on the GPU path | About 0.8× MLX when resident (Gemma E2B on iPhone 17 Pro: 38.8 vs 49.1, [bench](https://github.com/john-rocky/apple-silicon-llm-bench)) | **Top pick** |
| **Xybrid** (llama.cpp inside) | Same as llama.cpp; LFM MoE yes, Inkling no at the current pin | Same as llama.cpp | **Fastest to test** |
| **Strata** | Yes on Mac (SSD streaming); no iOS build | M5 Max: 108–120 tok/s resident, 46–51 tok/s streamed | Reference design to port |
| **Anemll / danveloper flash-moe** | Yes: SSD `pread` streaming, demonstrated on iPhone | 0.6 tok/s for a 400B model on iPhone; 12.9 tok/s on M5 Max | Proof that it works; Qwen3.5-397B only |
| **llama.cpp RFC #23324** | Yes: a fixed-size pool of expert slots | 22–30 tok/s vs 38 tok/s vanilla on M3 Pro | Not merged; use the patch |
| **Liquid LEAP SDK** | No advantage over llama.cpp | Same as llama.cpp | **Deprecated**; was a wrapper around llama.cpp |
| **Cactus** | Not documented for MoE or flash | 71 tok/s for Gemma-4-E2B CQ4 on iPad M5; no edge over MLX | Not for this goal |
| **Nexa SDK** | n/a | n/a | Team joined Qualcomm; Qualcomm-only now |
| **ZETIC MLange** | No evidence | No published figures ("to be added") | Needs keys and a download; skip |
| **MLC-LLM** | No evidence of flash or MoE streaming | Mac study: MLX has higher sustained throughput | Skip |
| **ExecuTorch** | No evidence | No iPad LLM figures found | Skip |
| **Core ML stateful / Core AI (iOS 27)** | Unknown; Core AI advertises "fine-grained control over inference memory" | Core AI 67.6 vs MLX 65.1 tok/s (Qwen3-1.7B, iPhone 17 Pro); MoE weaker (gpt-oss-20b on M4 Max: 78.1 vs 100.2) | Watch; not for MoE now |
| **ANEMLL (Neural Engine)** | No: Neural Engine is a poor fit for MoE | 27 tok/s for a 2B model on the Neural Engine (dense) | Skip for MoE |
| **Apple Foundation Models** | n/a (Apple's own ~3B model) | Not published | Usable test-only side option |

### Details

- **llama.cpp directly.** Liquid now tells iOS developers to use `llama.xcframework`, which is Metal-enabled for iOS 16.4+ ([migration guide](https://docs.liquid.ai/deployment/on-device/llama-cpp/migrating-from-leap-sdk)).
  - Through `tensor_buft_overrides`, the experts could be placed on the CPU and read straight from the mapped file. Those pages are certainly file-backed and clean. CPU bandwidth is lower than the GPU's, but only about 1B parameters are active per token. This is an untested option.
  - LFM2.5-8B-A1B at Q4_0 (4.84 GB) resident in the page cache: I estimate about 90–120 tok/s, against our MLX 1.2B dense at 169 tok/s. That assumes about 1.5B active parameters, which I did not check.

- **Xybrid.** It is Apache-2.0 ([repo](https://github.com/xybrid-ai/xybrid)) and lists "llama.cpp (GGUF)", running "Metal (llama.cpp) on devices".
  - It has the same mmap behaviour as llama.cpp, but its wrapper only exposes `n_gpu_layers`. It does not expose load mode, CPU placement of experts, or lazy mode.
  - It can test LFM2.5-8B-A1B today. Inkling 2 Mini needs llama.cpp bumped past `b1b770d`, which is a change to Xybrid's vendored code or a fork.

- **Strata (details).**
  - It uses its own Metal engine (not MLX or llama.cpp), with a `metal_compat` shim that maps CUDA runtime calls onto Metal. It requires macOS 15+.
  - It streams experts when "weights plus 12 GiB of overhead exceed Metal's share". It keeps an arena of hot experts, reads the rest in place from the mapped file, and prefetches the next layer.
  - Measured on M5 Max (MAC.md): 108–120 tok/s resident with multi-token prediction, 46–51 tok/s when streaming the UD-Q5_K_XL pack. M4 mini 32 GB streamed: 9–14 tok/s.
  - On the same M5 Max, llama.cpp ran the Q2_0 pack at 56.6 tok/s.
  - Its multi-token-prediction drafting is speculative decoding. That is fine for this app, which is not covered by the Trainium scope exclusion.
  - The CPU kernels are x86 AVX (HOW_IT_WORKS). Only the Metal path is relevant to us.

- **flash-moe (Anemll fork of danveloper/flash-moe).** It is pure C and Metal ([fork](https://github.com/Anemll/flash-moe)). It reads only the active experts with parallel page-aligned `pread` calls and leaves caching to the OS page cache.
  - mmap'd experts were tried and rejected as about 5× slower.
  - The non-expert weights (5.5 GB) are mapped; total active RAM is about 5.8 GB.
  - The iPhone 17 Pro demo ran a 400B model at 0.6 tok/s ([wccftech](https://wccftech.com/iphone-17-pro-successfully-runs-400b-llm-locally/)). The repo itself has no iOS target and states no license.

- **llama.cpp RFC #23324** ([discussion](https://github.com/ggml-org/llama.cpp/discussions/23324)). It keeps a fixed pool of expert slots per layer in Metal shared memory, with LRU replacement, `pread` for misses and `MTLSharedEvent` to signal the GPU.
  - Qwen3-30B-A3B on M3 Pro: 32 slots used about 10.6 GiB wired at 22.1 tok/s; 8 slots used about 6.3 GiB at 12 tok/s; vanilla was 38.1 tok/s. About 13 tok/s was reported on an M1 Pro 16 GB.
  - It requires `--no-mmap`. It is the cleanest bounded-footprint design. Status: open RFC, not merged.

- **Liquid LEAP SDK.** It is deprecated and gets no new releases. It was a Kotlin Multiplatform wrapper around llama.cpp, with ExecuTorch used for `.bundle` files ([migration guide](https://docs.liquid.ai/deployment/on-device/llama-cpp/migrating-from-leap-sdk)). It offers nothing over llama.cpp directly.

- **Cactus.** The license is Apache-2.0 per its README ([GitHub](https://github.com/cactus-compute/cactus), [docs v2.0](https://docs.cactuscompute.com/v2.0/)).
  - Its own quantization scheme (CQ, 4-bit down to 1-bit) and Metal or CPU backends. No MoE support or mmap streaming is documented.
  - iPad/Vision Pro M5, Gemma-4-E2B-CQ4: 1336 tok/s prefill and 71 tok/s decode, 703 MB peak.
  - iPhone 17 Pro, Gemma 4 E2B: Cactus 50.0 vs MLX 49.1 tok/s, but GSM8K was only 3% "as shipped" ([bench](https://github.com/john-rocky/apple-silicon-llm-bench)).
  - An older Cactus blog described mmap with page faults on demand (search snippet only; the page now returns 404).

- **Nexa SDK.** The team joined Qualcomm in March 2026, and the co-founder said they now support only Qualcomm chips ([Qualcomm blog](https://www.qualcomm.com/developer/blog/2026/06/geniex-developer-preview), [huxiu](https://www.huxiu.com/article/4875366.html)). The current README lists Windows ARM64, Android and Linux only.
  - The older iOS SDK was a beta, and only speech recognition and embeddings used the Neural Engine. I could not reach docs.nexa.ai to confirm its current state.

- **ZETIC MLange.** `ZeticMLangeLLMModel(personalKey, modelKey)` downloads a model chosen for the device. The docs say it supports "multiple LLM backends including LLaMA.cpp". The listed models are 1.1–1.5B, and the performance section is marked "to be added" ([doc](https://zetic-ai.github.io/ZETIC_MLange_document/overview/zetic-mlange-llm-model.html)).

- **MLC-LLM.** It is TVM-compiled with weights held in GPU memory. I found no MoE or flash-streaming evidence for iOS in 2026. A Mac study found MLX had higher sustained throughput ([arXiv 2511.05502](https://arxiv.org/pdf/2511.05502v1)).

- **ExecuTorch.** The iOS backends are now `backend_coreml`, `backend_mlx` and `backend_xnnpack`; MPS is not listed ([docs](https://docs.pytorch.org/executorch/main/using-executorch-ios.html)). An ExecuTorch MLX backend would not beat MLX Swift on memory.

- **Core AI and Core ML stateful models.** Core AI is new in iOS 27 and is reported as the Core ML successor ([9to5mac](https://9to5mac.com/2026/03/01/apple-replacing-core-ml-with-modernized-core-ai-framework-for-ios-27/)). Apple advertises "fine-grained control over inference memory, zero-copy data paths, and stateful execution" ([WWDC26 guide](https://developer.apple.com/wwdc26/guides/ios/)).
  - The bench found Core AI "755 MB mmap" for Gemma E2B, so it may also benefit from file-backed weights.
  - It is worth a later check for dense models. The Neural Engine is a poor fit for MoE; the NPUMoE paper says sparse execution is "fundamentally misaligned with NPU execution pipelines" ([arXiv 2604.18788](https://arxiv.org/pdf/2604.18788)).

- **ANEMLL.** It runs dense models on the Neural Engine; its README says LUT4 quality is low. On iPhone 17 Pro the Neural Engine is slower in bursts than the GPU but throttles less ([dev.to](https://dev.to/john-rocky/iphone-on-device-llm-the-gpu-wins-the-sprint-the-neural-engine-wins-the-marathon-42lo)). It is not useful for these MoE models.

- **Apple Foundation Models.** The iOS 27 framework gives access to Apple's own on-device model. It now accepts any provider through the Language Model protocol, including "MLX Language Models" ([FM page](https://developer.apple.com/foundation-models-framework/)).
  - The protocol could wrap our MLX or llama.cpp engines behind one API, but it gives no memory relief.
  - **Unverified:** whether a free-Apple-ID sideloaded app can use the on-device model. I know of no documented entitlement for it, and it needs Apple Intelligence enabled. A [forum report](https://developer.apple.com/forums/thread/856179) shows a "device not eligible" error on an iOS 27.3 beta. Free private cloud access is tied to the App Store Small Business Program, so it is not available to us.

## Recommended next experiments, in order

1. **LFM2.5-8B-A1B Q4_0 GGUF (4.84 GB) through llama.cpp with mmap on the iPad.** Use Xybrid's existing LLM path, or `llama.xcframework`.
   - Log `phys_footprint`, `os_proc_available_memory` and tok/s.
   - Pass criterion: the load does not jetsam, and footprint stays well below the size of the weights.
2. **If the GPU-mapped path is charged to the footprint:** place the experts on the CPU via `tensor_buft_overrides` and keep attention on Metal. Alternatively, port the fixed-slot expert pool from RFC #23324.
3. **Inkling 2 Mini UD-IQ2_XXS (8.49 GB)** on llama.cpp master, which supports `inkling2_loop`, using the same mmap test. Measure decode speed separately for a warm page cache and for flash misses.
4. **Longer term:** port Strata's hot-expert arena, next-layer prefetch and "experts' copy" file to our engine. Strata is MIT-licensed. flash-moe states no license, so treat it as a reference only.

Everything I checked locally is in `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\xyb\`: the Xybrid clone (`src\`) plus `arch.cpp`, `llama.h`, `llama-model.cpp`, `llama.cpp` and `llama-model-loader.cpp` from llama.cpp at `b1b770d`. I did not touch the mlx-integration repo.

## storage

# Model storage persistence: where weights live, whether they survive, and what to change

## Bottom line

- **Where MLX weights are today:** `<container>/Library/Caches/huggingface/hub/models--<org>--<repo>/`. That is a purgeable folder. iOS can delete it when the device runs low on storage, and the next load then downloads everything again.
- **Reinstalling over the app keeps the weights**, as long as it is the same bundle ID and the same Apple ID. A delete-then-install, or signing with a different Apple ID or team, loses them.
- **Xybrid speech models are already safe.** They go to `Library/Application Support/Xybrid/Models`, which iOS does not purge. They are included in iCloud backup, though.
- **Every load makes a network call even when the files are already there.** It does not re-download anything. Offline it falls back to the local files, but on a slow network it waits for the call first.
- **`download_observed` is roughly right, but it is a heuristic.** Separately, the receipt field `weight_files_bytes: 76` is a bug.
- **The fix** is to move the cache to Application Support, exclude it from backup, and load from the local folder first. Exact API is in section 7.

## 1. Where MLX weights land today

1. Our code calls `#huggingFaceLoadModelContainer(configuration:progress:)` (`ios/AgentConsole/Inference/MLXBackend.swift`).
   - In mlx-swift-lm 3.33.0 that expands to `loadModelContainer(from: #hubDownloader(), using: #huggingFaceTokenizerLoader(), ...)`.
   - `#hubDownloader()` with no argument becomes `HubClient()` (`Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift`, around lines 23 and 186, at commit ca661dc6).
2. `HubClient.init(session:userAgent:cache: HubCache? = .default)` (`Sources/HuggingFace/Hub/HubClient.swift`).
3. `HubCache.default` is `HubCache(location: .environment)`. It resolves the folder in this order:
   1. `HF_HUB_CACHE`
   2. `HF_HOME` + `/hub`
   3. otherwise, on iOS, `URL.cachesDirectory/huggingface/hub` (`Sources/HuggingFace/Shared/CacheLocationProvider.swift`, `defaultCacheDirectory`, the `#else` branch).
4. I checked this at swift-huggingface tag 0.13.0 (f0a0cb43) and at main (55df50a2). Our `project.yml` asks for `from: 0.9.0`, which resolves to the newest 0.x, so 0.13.0. There is no `Package.resolved` in the repo, so the exact resolved version on the shipped build is unverified.
5. Folder layout is the same as Python's: `blobs/<etag>`, `refs/main`, `snapshots/<commit>/<file>`, plus `.metadata/` and `.locks/` at the top. The snapshot files are relative symlinks (`../../blobs/<etag>`).

## 2. Can iOS purge them?

**Caches: yes.** Apple's documentation says `Library/Caches` may be deleted when the device is low on disk space and is not backed up. The 4.85 GB LFM2.5-8B download is exactly the kind of thing that would be purged on a full iPad. ([File System Programming Guide](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html))

**Application Support: no**, but it is backed up unless you set `isExcludedFromBackup`. ([Optimizing your app's data for iCloud backup](https://developer.apple.com/documentation/foundation/optimizing-your-app-s-data-for-icloud-backup), [QA1719](https://developer.apple.com/library/archive/qa/qa1719/_index.html))

Our other app data already lives in Application Support: `Memory`, `Documents` and `Batches`. Only `JobFeed` uses Caches, which is appropriate for it.

## 3. Does a Sideloadly reinstall keep the container?

**Same bundle ID and same Apple ID, installed over the existing app: the container is kept.**
- Sideloadly's FAQ says to install over the existing app with the same Apple ID, and to reuse the same bundle ID if you set a custom one ([sideloadly.io/faq](https://sideloadly.io/faq)).
- The 7-day expiry of a free-account signature does not delete data. Re-signing with the same Apple ID and installing over it restores the app with its data.

**Our receipts agree** (`docs/receipts/2026-11-27/`):

| Model | `download_observed` | `first_load_s` |
|---|---|---|
| LFM3 1.7B | false | 0.283 |
| Qwen3 1.7B | false | about 0.31 |
| LFM2.5 1.2B | true | 26.7 |
| LFM2.5 350M | true | 9.86 |

A load of about 0.3 s cannot include a 1 GB download, so those weights came from disk. Qwen3 was the previous default, which suggests it was downloaded by an earlier build and survived the reinstall. That is likely but not proven: chat may have loaded those models on this install before the benchmark ran.

**Deleting the app first wipes everything**: weights, memory and documents.

**If the signing team changes** (a different Apple ID, or moving to a paid team), the app identifier prefix changes. iOS then refuses the upgrade with "Upgrade's application-identifier entitlement string … does not match installed application's application-identifier string; rejecting upgrade" ([Apple forum 126589](https://developer.apple.com/forums/thread/126589), [QA1879](https://developer.apple.com/library/content/qa/qa1879/_index.html)). The old app has to be deleted, so the container is lost.

**If Sideloadly rewrites the bundle ID**, which it does for free accounts when an ID clashes with an App Store app, iOS treats it as a different app with an empty container.

To make the weights survive any of these, they would have to live outside the container. The options are an App Group, which needs the same team, so it does not help here, or importing from the Files app. That second option is the only way to copy weights over from a Mac.

## 4. Xybrid speech cache

- On iOS, `CacheManager::get_cache_dir()` uses `$HOME/Library/Application Support/Xybrid/Models` unless `init_sdk_cache_dir` has been called (`crates/xybrid-sdk/src/cache/cache_manager.rs`, about lines 140–170, tag v0.21.0, 491da33d). Our `SpeechService` calls only `Xybrid.initialize()`, so this default applies.
- It is not purgeable and it persists across reinstalls like the MLX weights. It is not excluded from backup (I found no `isExcludedFromBackup` in the SDK).
- `Xybrid.hasCachedModelData(_:)`, which our `needsDownload` uses, already skips the download when files are present.
- **Hazard:** if anything ever calls `Xybrid.initSdkCacheDir(cacheDir:)`, the SDK runs `export_cache_env`, which sets `HF_HOME=<dir>/huggingface` and `HF_HUB_OFFLINE=1` (`crates/xybrid-sdk/src/lib.rs`, about lines 453–500). `HubCache.default` reads `HF_HOME` the first time it is used. That would silently move our MLX cache and force every model to download again. An explicit `HubCache(cacheDirectory:)`, as in section 7, removes this risk. swift-huggingface does not read `HF_HUB_OFFLINE`.

## 5. What happens on a load that is already cached

In `HubClient+Files.swift` at 0.13.0:

1. `downloadSnapshot` has a fast path, `cachedSnapshotPath`, that skips the network. It only works when the revision is a commit hash and `.metadata/<commit>.json` exists. That metadata is written only when the revision is a commit hash. We always pass `"main"`, so the fast path never runs.
2. So every load calls `listFiles` over the network, which is one Hub API request.
3. Each file is then found through `cache.cachedFilePath(revision: "main")`, which reads `refs/main`. These are not downloaded again.
4. If `listFiles` throws, the code falls back to `cachedSnapshotPathForLocalFilesOnly`. That fallback only checks that the files exist; it does not check sizes or hashes.
5. On a captive portal or a stalled network, the load waits up to the URLSession timeout before falling back.
6. `useLatest` is ignored by the macro's `HubBridge`, and `localFilesOnly` cannot be reached through `#hubDownloader`.
7. Blobs are written to an `.incomplete` file and renamed when finished, so a half-written blob cannot pass the "file exists" check.

## 6. Can `download_observed` be wrong?

It is set in the `bench` branch, `ios/AgentConsole/Bench/BenchmarkRunner.swift`, around line 114: `if f > 0 && f < 1 { sawPartial.set() }`, using the download progress the library samples every 100 ms.

- **False negative:** a small download, such as a single missing JSON file, can finish between samples and never show a fraction between 0 and 1.
- **Overcount:** a sidecar or tokenizer download while the weights are already cached still sets it to true. It means "some bytes were fetched", not "the weights were downloaded".
- **Rare false positive:** cached files complete their progress one at a time, so a sample could land on a fraction between 0 and 1 even with no download. This is unlikely because those steps take microseconds.
- **`weight_files_bytes: 76` is a real bug.** The benchmark reads `.fileSizeKey` on the snapshot symlink. 76 bytes is the length of `../../blobs/<64 hex characters>`. The fix is `url.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey])`.
- **More reliable detection:**
  - Record the total blob size of the repo before and after the load. Any growth means bytes were downloaded.
  - Or record whether a complete local snapshot existed before the load (section 7, step 2).
  - Or log `URLSessionTaskMetrics` byte counts.
  - Keep `first_load_s` as a cross-check.

## 7. Recommended code changes

These are recommendations only; nothing was edited.

**1. Own the cache location.** Add a new file `ios/AgentConsole/Inference/ModelStore.swift`:
```swift
import HuggingFace
enum ModelStore {
    static let root = URL.applicationSupportDirectory
        .appending(path: "Models/huggingface/hub", directoryHint: .isDirectory)
    static let cache: HubCache = {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var dir = root.deletingLastPathComponent()           // .../Models/huggingface
        var v = URLResourceValues(); v.isExcludedFromBackup = true
        try? dir.setResourceValues(v)
        return HubCache(cacheDirectory: root)
    }()
    static let hub = HubClient(cache: cache)
}
```
Load with `loadModelContainer(from: #hubDownloader(ModelStore.hub), using: #huggingFaceTokenizerLoader(), configuration: cfg, progress: .download { … })` instead of `#huggingFaceLoadModelContainer`.

**2. Migrate the existing cache once,** before the first load:
- If `URL.cachesDirectory/huggingface/hub` exists and `ModelStore.root` does not, call `FileManager.default.moveItem(at:to:)` on the whole `hub` folder.
- It is the same disk, so this is instant. The relative symlinks stay valid.
- Guard it with a `UserDefaults` flag such as `"store.migrated.v1"`.
- Without this step, the LFM3 and Qwen weights already on the iPad would be downloaded again.

**3. Load from local files first, with no network:**
- `let commit = ModelStore.cache.resolveRevision(repo: id, kind: .model, ref: "main")`
- `let snap = try ModelStore.cache.snapshotPath(repo: id, kind: .model, commitHash: commit)`
- Check that the required files exist:
  - `config.json`
  - `tokenizer.json` or `tokenizer_config.json`
  - every shard listed in `model.safetensors.index.json` under `weight_map`, or `model.safetensors`
- Check each resolved file's size against a manifest we store when the download finishes, for example `Models/manifest.json` mapping repo to commit, file sizes and etags. Optionally also check SHA-256 against the LFS oid; it is slow on 5 GB, so only do it after a crash or failed load.
- If everything checks out:
  1. Start from the registry config, `var cfg = Self.configuration(for: entry)`.
  2. Set `cfg.id = .directory(snap)`. This keeps `toolCallFormat` and the other registry settings.
  3. Call `loadModelContainer(from: #hubDownloader(ModelStore.hub), using: …, configuration: cfg, …)`. For a `.directory` ID it does not call the downloader at all (same pattern as `resolveForModelConversion` in `ModelConversion.swift`).
  - Alternatively, use `loadModelContainer(from: snap, using: #huggingFaceTokenizerLoader(), progress:)` from `ModelFactory.swift:535`. That one does not take a configuration, so it would lose the registry `toolCallFormat`.
- Only go to the network when the check fails.
- Optional: call `ModelStore.hub.downloadSnapshot(of:revision:matching:localFilesOnly: true, …)` through a small custom `MLXLMCommon.Downloader` conformance. That also gives `useLatest` a real meaning.
- Also an option: pin a commit with `ModelConfiguration(id:revision:)`. That turns on the library's own no-network fast path once `.metadata` has been written. **Only pin the commit already in `refs/main` on the device**; a different commit triggers a full download.

**4. Check free space before downloading:**
- Read `URL.volumeAvailableCapacityForImportantUsageKey` and refuse a download larger than what is free plus a margin.
- Show download size from `entry.downloadGB`, or better, from the sizes `listFiles` returns.

**5. Add a "Downloaded models" screen in Settings:**
- List each `models--*` folder under `ModelStore.root`.
- Size: add up `totalFileAllocatedSizeKey` over `blobs/` (not `snapshots/`, which holds symlinks).
- Also show the commit from `refs/main` and the last-used date, stored in `UserDefaults`.
- Delete: `FileManager.default.removeItem(at: cache.repoDirectory(repo:kind: .model))` plus `cache.metadataDirectory(repo:kind:)`. Do it only after unloading the model (`reconcileSelection` / `MLX.Memory.clearCache()`), and use a confirmation dialog.
- Add a row for Xybrid using `Xybrid.modelCacheEntries()` and `Xybrid.cachedModelPath(_:)` from `bindings/apple/Sources/Xybrid/Xybrid.swift`.
- Show the total and `volumeAvailableCapacityForImportantUsageKey`.

**6. Exclude Xybrid from backup:** set `isExcludedFromBackup` on `URL.applicationSupportDirectory/Xybrid` once the SDK has created it.

**7. Never call `Xybrid.initSdkCacheDir`,** or call it only after `ModelStore.cache` exists, because of the `HF_HOME` hazard in section 4.

**8. Fix the benchmark:**
- Resolve symlinks when measuring `weight_files_bytes`.
- Replace or augment `download_observed` with a before/after blob-size difference, plus a `local_snapshot_complete_before` flag.

## About the inference engine part of the question

The MLX engine, its Metal library and the Xybrid binaries are compiled into the app itself. Sideloadly sends the whole app on every install; that is a local USB/Wi-Fi transfer and cannot be avoided. What must not be fetched again is the model data in the container, and that is covered by the reinstall rule in section 3 and the changes in section 7.

Sources:
- [huggingface/swift-huggingface (CacheLocationProvider.swift, HubCache.swift, HubClient+Files.swift)](https://github.com/huggingface/swift-huggingface)
- [ml-explore/mlx-swift-lm 3.33.0 (MLXHuggingFaceMacros, ModelFactory.swift, Downloader.swift)](https://github.com/ml-explore/mlx-swift-lm/tree/3.33.0)
- [xybrid-ai/xybrid v0.21.0 (cache_manager.rs, lib.rs, Xybrid.swift)](https://github.com/xybrid-ai/xybrid/tree/v0.21.0)
- [Sideloadly FAQ](https://sideloadly.io/faq)
- [Apple forum: application-identifier mismatch rejects upgrade](https://developer.apple.com/forums/thread/126589)
- [Apple QA1879](https://developer.apple.com/library/content/qa/qa1879/_index.html)
- [Apple File System Programming Guide](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html)
- [Apple: Optimizing your app's data for iCloud backup](https://developer.apple.com/documentation/foundation/optimizing-your-app-s-data-for-icloud-backup)
- [Apple QA1719](https://developer.apple.com/library/archive/qa/qa1719/_index.html)

Source clones are in `C:\Users\apumu\AppData\Local\Temp\claude\958251b97b3e10b1\0d088ca9-22d8-4cc0-a946-7b5a50311863\scratchpad\src\`. Files read in the repo:
- `C:\Users\apumu\research\CHLabs\mlx-integration\ios\AgentConsole\Inference\MLXBackend.swift`
- `C:\Users\apumu\research\CHLabs\mlx-integration\ios\AgentConsole\Inference\ModelCatalog.swift`
- `C:\Users\apumu\research\CHLabs\mlx-integration\ios\project.yml`
- `C:\Users\apumu\research\CHLabs\mlx-integration\ios\AgentConsole\Speech\SpeechService.swift`
- `C:\Users\apumu\research\CHLabs\mlx-integration\docs\receipts\2026-11-27\`
- `ios/AgentConsole/Bench/BenchmarkRunner.swift` on the `bench` branch