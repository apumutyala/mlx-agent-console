# Engineering log: Agent Console (iPad MLX + compute fleet)

**Owner / integrator:** the coordinating Claude session. It owns `integration`, reviews every PR, and merges.
**Human approver:** Apuroop. Nothing goes upstream to open-source repositories without his explicit approval, plus a measured, reproduced improvement.

New entries go at the top of [Log](#log). Every agent appends an entry when it opens or updates a PR.

## Goal

1. **Run models on the iPad.** The iPad Pro 13" M5 has 16 GB of RAM. The best-quality local models should run there, with inference fully optimized: Liquid LFM family, Inkling 2 Mini, Qwen fallback.
2. **Make the agent capable on the device.**
   - Tools: documents, jobs, fleet.
   - Persistent, self-improving memory.
   - Voice: Xybrid speech-to-text and text-to-speech.
3. **Make the iPad a compute node.** It takes inference and federated LoRA tasks from the worker on the Omen. Claude drives everything through the MCP connector.
4. **Contribute findings upstream** (mlx-swift-lm, mlx-lm, Xybrid, …), only after Apuroop approves and only with a receipt for a real improvement.

## Branch / PR map

| Branch | Purpose | Base | State |
|---|---|---|---|
| `main` | last released build | — | protected: merges come only from `integration` via PR |
| `integration` | staging; everything that CI builds together | `fleet` | open PR → `main` |
| `fleet` | worker: compute fleet, FedAvg, MCP fleet tools | `main` | merged into `integration` |
| `liquid-models` | LFM3 default, model catalog, native tool calling | `fleet` | merged into `integration` (CI green 45949641772) |
| `ios-node` | iPad compute node, federated LoRA trainer, GPU gate | `fleet` | merged into `integration` (CI green 45949683774) |
| `xybrid-speech` | Xybrid Whisper STT + Kokoro TTS | `fleet` | merged into `integration` (CI green 45948595221) |
| `inkling2` | Inkling 2 Mini via vendored mlx-swift-lm PR #821 + Fix A | `integration` | PR open, CI pending |
| `inkling-kernels` | O2: 2/3-bit affine in Inkling's fused decode expert kernels; Metal compile check in CI | `inkling2` | PR open (depends on PR #2) |
| `memory-harness` | durable chats, memory store, playbook, document tools | `liquid-models` | in progress |

**Rules for agents:**
- Work only on your own branch, and open a PR into `integration`.
- Commits are authored as Apuroop Mutyala, with no AI trailers.
- Never push to `main`, `integration`, or another agent's branch.
- The only compiler is GitHub Actions macOS (`gh workflow run build.yml --ref <branch>`), and its minutes cost 10× on this private repo. Batch your fixes.
- Put the CI run URL and anything unverified in the PR body.

## Status (keep current)

| Area | Built in CI | Ran on device | Notes |
|---|---|---|---|
| Qwen3 1.7B chat (v1) | yes | **yes** (device test 2026-11-26) | hallucinates; replaced as the default |
| LFM3 1.7B default + model picker | yes | no | needs swift-transformers ≥ 1.4.0 |
| Native tool calling (LFM2 pythonic, LFM3 xml, Qwen json) | yes | no | text-parser fallback kept |
| Compute node + GPU gate | yes | no | pauses in background |
| Federated LoRA (device) / FedAvg (worker) | yes / tests pass | no | shared seed for the initial adapter |
| Xybrid speech | yes | no | +18 MB IPA; telemetry opted out |
| QR pairing (`worker/pair.ps1`) | not yet | no | VisionKit scanner |
| Inkling 2 Mini | in progress | no | see findings below |
| Memory harness + document tools | in progress | no | |

## Findings

### Inkling 2 Mini on a 16 GB iPad (scout report, 2026-11-26)

**Model and library support**
- The model is 26.4B parameters, a looped MoE with about 3.4B active per token. Routed experts make up about 24.5B of the parameters, so memory tracks total size, not active size.
- mlx-swift-lm 3.33.0 has no Inkling support. Open upstream PR ml-explore/mlx-swift-lm#821 (head `5507b3b`) adds it.
  - The PR loads the original Thinking Machines layout and the mlx-lm #2086 layout.
  - It does not load the mlx-vlm, oMLX/oQ or JANG layouts.
  - Its fused decode kernels cover 4/8-bit experts only. 2/3-bit experts use the slower unfused path.

**Fixes needed (candidate upstream contributions)**
- **Fix A** (found by reading the code; not run): `prepareCheckpoint` strips the `model.` prefix from weight names but not from the per-layer `quantization` keys. Mixed-bit checkpoints then fail the shape checks.
- **Fix B**: map mlx-vlm-layout names and interleave `gate_proj`/`up_proj` rows into `gate_up_proj`. This is lossless.

**Memory budget**
- Resident memory is roughly text weights + 1.0–1.8 GB. That means about 9.5 GB of text weights for a 16 GB iPad.

**Candidate builds**
1. `LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ`: ~9.6 GB text, needs Fix A only. This is the first target.
2. `netprojects/Inkling-2-Mini-LayerFit-MLX-2bit`: ~9.2 GB, KL 0.107, needs Fix B.
3. `VisualInference/Inkling-2-Mini-MLX-2bit`: ~9.4 GB; tools 19/19, GSM8K 47/50; needs Fix B.

**Other options**
- **Better builds that MLX Swift can't load:** LookAtMyQuant trellis 8.9 GB (KL 0.053), JANGH2, EXL3. Their quality per GB is higher.
- **Expert streaming from SSD** is the most promising future work: 4.6–6.6 GiB peak at 8–10 tok/s on a 16 GB M4 Air (`theblackcat102/...-offload`).
- No ternary or distilled Inkling 2 Mini exists yet.
- **Own-build plan:** GPTQ, then a knapsack over NVFP4 / 3-bit / 2-bit precision per layer, then DWQ healing. Targets ~8–10 GB, or ~6.9 GB with REAP-96.

### Device facts
- iPad Pro 13" M5, 1 TB storage, 16 GB RAM, iPadOS 27.2.1, sideloaded with a free Apple ID (7-day signing).
- The per-app memory limit under free provisioning is still unknown. The app will display `os_proc_available_memory()`.

## Inference optimization backlog (profile before optimizing)

| # | Item | Expected lever | Status |
|---|---|---|---|
| O1 | Measure tok/s, TTFT and peak memory per model on the iPad (in-app benchmark screen, receipts saved to the log) | baseline for everything below | todo |
| O2 | Add 2/3-bit branches to Inkling's fused expert kernel (`inkling2_unit_dot`) | large (2/3-bit experts currently unfused) | PR from `inkling-kernels`; device A/B pending; upstream candidate |
| O3 | Inkling expert streaming / mmap of routed experts | fits 8–12 GB devices, frees RAM | research |
| O4 | KV cache quantization / smaller prefill step on iPad | memory headroom, longer context | todo |
| O5 | Speculative / MTP heads (DSpark for LFM2, Inkling MTP builds) | decode speed | not in scope for the Trainium work; allowed here, pending a decision |
| O6 | Our own Inkling quant (REAP + mixed bits + DWQ) on a rented box | quality per GB | plan written |

## Upstream contribution queue (needs Apuroop's approval + receipts)

| Target | Change | Evidence required | State |
|---|---|---|---|
| ml-explore/mlx-swift-lm#821 | Fix A (quantization key prefix) | Mixed-bit checkpoint fails before the fix and loads after it; tests pass | waiting for the `inkling2` branch |
| ml-explore/mlx-swift-lm#821 | Fix B (load mlx-vlm layout) | Same, for a 2-bit mlx-vlm build | waiting |
| ml-explore/mlx-swift-lm | 2/3-bit fused expert kernel | Before/after tok/s with the same output (greedy match) | after O1/O2 |

## Log

### 2026-11-27 (later): O2, 2/3-bit fused expert kernels (`inkling-kernels`)
- **Change:** `inkling2_unit_dot` reads 2-bit (16 codes per `uint`) and 3-bit (8 codes per 3 bytes) MLX affine weights; `supportsFusedDecode` accepts affine 2/3/4/8. The default DWQ build (gate/up 2-bit ×22 + 3-bit ×2 layers, down 2-bit ×14 + 3-bit ×10, all group 64; read from its config.json) now decodes through the fused kernels instead of `gatherQuantizedMM` + epilogue. Signatures and indexing unchanged (safe for `inkling-stream`'s slot tensors).
- **Packing verified** against mlx@36969c0 (the mlx in mlx-swift 0.33.0): CPU `quantize` packs 3-bit as 8 values into 24 bits little-endian (`bytes_per_pack` 3), Metal `qdot` reads the same; 2-bit four per byte.
- **Verified locally:** `ios/tools/kernels/test_inkling2_expert_ref.py` (numpy): hand-packed bytes, a port of MLX `quantize`, and a line-by-line port of both kernels against the dense math for gate/up×down bits {2,3}×{2,3}, 4, 8 and groups 64/128; a mutated 3-bit offset fails it. Runs in CI (`worker-tests`).
- **New CI step:** `ios/tools/kernels/check_metal_kernels.py` regenerates the Metal source MLX builds for `MLXFast.metalKernel` (header, signature, template instantiation) and compiles it with `xcrun metal` against MLX's `utils.h`: 122 instantiations of the three expert kernels. Add specs to cover other runtime kernels.
- **Device self-check:** Settings → Memory → "Inkling kernel self-check on load" runs fused vs unfused on a random Mini-shaped layer for each format before the weights load and shows max |Δ| per case.
- **PR / CI:** [PR #8](https://github.com/apumutyala/mlx-agent-console/pull/8) (depends on #2); CI green, run 45956519987: numpy 8 OK, app build OK, Metal check 122/122 compiled against MLX `utils.h`.
- **Not verified:** nothing ran on the iPad: speed and greedy equivalence need the A/B in the PR. The app limit on the user's iPad is ~5.3 GB, so the A/B runs on the streamed tiers, which use the same kernels.

### 2026-11-27
- **Inkling 2 (`inkling2` → PR into `integration`):** vendored aleroot/mlx-swift-lm@5507b3b (PR #821) under `ios/Vendor/mlx-swift-lm` without tests; Fix A (per-layer quantization keys renamed with the `model.` strip) and a DSpark-namespace exclusion. Fix B (mlx-vlm layout remap) deferred: that layout renames/reparametrizes ~30 tensors per layer (`q_scale`/`rel_scale`/`rel_proj`, `controllers.*`, `message_queries`) and LayerFit-2bit mixes gate 3-bit with up 2-bit, so gate/up can't be fused losslessly into `gate_up_proj`; not verifiable without weights.
  - Tiers (PR layout only): `LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ` 9.60 GB text weights (default), `roman220220/...-gptq-mlx-jang-small` 10.45 GB, `roman220220/...-gptq-mlx-jang` 12.49 GB. Sizes from safetensors headers, vision excluded.
  - Runtime gate: needs weights + 1.5 GiB ≤ `os_proc_available_memory()` + MLX-held bytes; Settings shows the readout, a load report and a killed-load breadcrumb. Compiled only; nothing ran on the iPad.

### 2026-11-26 (later): goal set, slots filled
- **Goal:** the user set the standing goal, which is recorded in the Goal section.
- **Agent slots (5):**
  1. adversarial review workflow (fleet / ios-node / xybrid);
  2. `inkling2`;
  3. `memory-harness`;
  4. `bench`: on-device profiler and receipts (O1). Receipts are uploaded to the worker, readable by the MCP tool `list_receipts`, and runnable as a fleet `bench` task;
  5. design research for Inkling expert streaming (O3), read-only.
- **Process:** [PR #1](https://github.com/apumutyala/mlx-agent-console/pull/1) is open, `integration` → `main`. Agents open PRs into `integration`. The coordinator reviews and merges.

### 2026-11-26
- **Fleet (worker):** added `fleet.py` (pull-based nodes, streaming tasks, artifacts, requeue) and `federated.py` (FedAvg over safetensors), plus 7 MCP fleet tools. 40 worker tests pass. The worker is reachable on the tailnet only, at `:8443`.
- **Integration branch:** merged `liquid-models`, `ios-node` and `xybrid-speech`; CI green (run 45950481109).
  - Conflicts were in `ModelBackend`, `MLXBackend`, `LocalAgent`, `SettingsView` and `project.yml`.
  - Chat now holds the GPU gate only while streaming, and tool calls run outside it.
  - Node `generate` tasks stream only `.text` and get extra token budget when the model thinks.
  - Xybrid telemetry is opted out.
- **QR pairing:** added `worker/pair.ps1` + `pair.py` and a VisionKit scanner. The QR flow has not run on a device.
- **Review:** an adversarial review workflow (5 dimensions × 3 skeptics) is running on fleet, ios-node and xybrid-speech.
- **Inkling:** the scout report is above. The `inkling2` agent was told to vendor PR #821 privately with Fixes A and B, rather than publishing a public fork.
- **Blocked actions (permission classifier):** reading the iPad's model over USB (pymobiledevice3) and reading the worker token for the user. Both were stopped, not rerouted.
