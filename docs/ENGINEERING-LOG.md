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
| `inkling2` | Inkling 2 Mini via vendored mlx-swift-lm PR #821 + fixes | `liquid-models` | in progress |
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

### Device receipts, round 1 (2026-11-27, iPad17,3 = iPad Pro 13" M5, 16 GB, iPadOS 27.2.1)

- **Build:** app `71391b9` (bench branch), mlx-swift-lm 3.33.0, workload set `w1`.
- **Method:** raw token prompts, greedy decoding, 3 reps, median shown.
- **Conditions:** plugged in, thermal nominal throughout.
- **Receipts:** `docs/receipts/2026-11-27/`.

| Model (4-bit, g64) | Weights GB | Decode tok/s, short / 1K / 4K | TTFT 1K / 4K (s) | Prefill tok/s at 4K | Peak footprint GB | Decode MBU* |
|---|---|---|---|---|---|---|
| LFM2.5 350M | 0.22 | **424** / 349 / 197 | 0.09 / 0.38 | 10,660 | 1.50 | ~61% |
| LFM2.5 1.2B Instruct | 0.66 | **169** / 160 / 139 | 0.20 / 0.80 | 5,095 | 1.45 | ~73% |
| LFM3 1.7B (default) | 0.97 | **113** / 109 / 101 | 0.31 / 1.26 | 3,262 | 1.75 | ~72% |
| Qwen3 1.7B (fallback) | 0.97 | **109** / 89 / 48 | 0.29 / 1.56 | 2,629 | 2.10 | ~69% |
| LFM2.5 8B-A1B (MoE) | 4.85 | skipped | — | — | — | needs ~6.5 GB, only 5.2 GB available |

\*MBU = weight bytes × short-prompt decode tok/s ÷ 153.6 GB/s, the M5 iPad memory bandwidth spec. KV reads are ignored, so the true figure is slightly higher. Short-prompt decode is about 70% of the memory-bandwidth roofline for all 1–2B models, which is typical for MLX. Headroom per token is about 30%.

**Key findings**

1. **The per-app memory limit is about 5.3–5.5 GB.** `os_proc_available_memory()` reads about 5.2 GB before load, at a 0.1–0.3 GB baseline footprint, on a 16 GB device.
   - So the `increased-memory-limit` entitlement is **not** in effect under the free-Apple-ID sideload.
   - Consequences:
     - The LFM2.5 8B MoE (4.85 GB) does not fit.
     - All resident Inkling tiers (9.6–12.5 GB) are impossible on this device as signed.
     - Inkling can only run via SSD expert streaming, within a footprint budget of about 4.6 GB.
   - Unverified alternative: a paid Apple Developer account grants the entitlement. That is a cost decision for Apuroop.
2. **Hybrid conv/attention (LFM) holds up at long context; Qwen does not.**
   - LFM3 decode drops only 10% from short to 4K.
   - Qwen3, which is all-attention, drops 56% (109 → 48 tok/s), with high rep-to-rep variance at 4K (53 → 44). Its KV reads (~0.47 GB at 4K) plus an unfused attention path look like the cause. This needs a profile.
3. **The first rep of 1K/4K includes compile warm-up.** For example, LFM3 1K prefill is 864 tok/s on rep 1 versus 3,341 after. Medians are fine, but the warm-up should cover every prompt length (workload w2).
4. **Determinism:** every model produced identical output across all reps (fingerprints in the receipts). The 350M emits end-of-text almost immediately on raw prompts, as expected without a chat template; its throughput numbers are still valid.

**Optimization targets this sets**
- LFM3 decode at 1–2B scale is 72% of roofline, so at most about 1.35× is left from kernels.
- The bigger levers are quality per GB within 5.3 GB, and Inkling via streaming.

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
| O2 | Add 2/3-bit branches to Inkling's fused expert kernel (`inkling2_unit_dot`) | large (2/3-bit experts currently unfused) | todo; upstream candidate |
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

### 2026-11-26 (night): `fix-node`, review fixes for the iPad compute node
- **Scope:** the 20 confirmed findings in `docs/review/2026-11-26-ios-node.json`, plus the device half of the HIGH privacy finding in `docs/review/2026-11-26-worker.json` (federated.py:171). The worker Python side is a separate agent's job; no `worker/*.py` edits here.
- **Protocol (needs the worker fix):** a paused node now *releases* its task: it stops and never calls `/complete`. The worker requeues the task when the same node polls again or when the sweep runs. Finished results go to a persisted outbox (`node.pendingCompletions`). The outbox retries with backoff until the worker answers 2xx, 404 or 409, and it is flushed before every poll. Metrics are NaN/Inf-safe. Heartbeats carry a new key, `capabilities.train_on_documents`.
- **Privacy:** two new settings, both off by default: "Accept training tasks" (without it, `train_lora` isn't advertised and train tasks fail) and "Allow federated training on my documents" (without it, `use_local_data` tasks fail and documents are never read).
- **Chat weights:** a train task snapshots the chat's adapter and restores it afterwards, including on failure or cancel. Only "Use federated adapter" changes what the chat runs on.
- **GPU gate:** `GPUGate.acquire` is now `async throws` and responds to cancellation. **Other branches that call `await GPUGate.shared.acquire(...)` must change it to `try await`.** `ModelConversation` gains `synchronize()`. The gate is released only after a cancelled generation has really stopped.
- **Status:** CI build only (see the PR); nothing has run on a device.
- **PR #6 review (3 MEDIUM findings, `docs/review/2026-11-27-pr6.json`):**
  1. A deferred adapter restore used to release the gate while training weights were still loaded. Now the gate stays held until `applyPendingRestore` has restored the adapter; that runs on the next `.active` or the next loop pass.
  2. The node no longer pauses at `.inactive`, only at `.background`. Every release burns a server attempt, so two Control Center pulls could mark a task `lost`.
  3. `MLXBackend.load` is inline again, so it is cancellable. Every caller holds `GPUGate`, which already serializes loads, and a weight read that can't be stopped finishes while its caller still holds the gate. The diff to PR #2's `inflight`/`draining` load stays small.
- **Merged `integration`** (PR #3 memory harness, PR #4 worker). `MemoryHarness.withGPU` now uses `try await GPUGate.shared.acquire`.

### 2026-11-26 (night): `fix-speech`, speech review findings
- **Branch:** `fix-speech` from `integration`, with a PR into `integration`. It fixes all 8 items in `docs/review/2026-11-26-speech.json`. The only ChatView item in the ios-node review (#6, model load outside the GPU gate) is a node/backend race, so it is left to `fix-node`.
- **The voice was wrong.** `"af"` is not a Kokoro v1.0 voice. The Xybrid registry entry for kokoro-82m lists `default: af_heart` plus 54 ids, and Xybrid v0.21.0 matches ids exactly. The default is now `af_heart`, and each reply checks the voice against `model.voice(voiceId:)`, falling back to `defaultVoice()`. Settings has a voice picker built from `model.voices()`.
- **State fixes:**
  - Stop now ends the run as "Stopped.", because `LocalAgent` checks for cancellation after the stream loop. A truncated reply is no longer read aloud.
  - Loads are single-flight, and each load clears its own loading phase. Turning the speaker toggle off cancels the download with `XybridDownload.cancel()`. A Kokoro download no longer disables the mic.
  - `speak()` does nothing while recording or transcribing, and Send is disabled while the mic is live.
- **Audio session:**
  - An interruption stops playback and suspends the recording, keeping the audio.
  - An engine configuration change re-taps the mic and keeps the samples.
  - Unplugging headphones stops read-aloud.
  - The session is deactivated with `.notifyOthersOnDeactivation` when idle.
- **Recordings:** at the 2-minute cap, the recording stops, is transcribed and is sent. A failed ASR download keeps the recording, with Send recording / Discard buttons. Whisper is prefetched while the user talks.
- **Memory:** `Xybrid.releaseMemory()` and dropped model references on a memory warning, and when an LLM of 3 GB or more is selected. `setAutoRelease(true)` is also set.
- **Verification:** compiled in CI only; nothing has run on a device.
### 2026-11-26 (night): review → fixes → PRs
- **Review results:** the review workflow (149 agents) confirmed 43 findings: 6 high, 22 medium, 15 low. They are split by area in `docs/review/2026-11-26-*.json`, and 5 were rejected.
- **PRs:**
  - [#2](https://github.com/apumutyala/mlx-agent-console/pull/2) `inkling2`: vendored #821 + Fix A + draft exclusion. Tiers: DWQ 3.1bpw 9.6 GB (default), jang-small 10.45 GB, jang 12.49 GB (stretch). Memory gating, and a memory readout in Settings. Fix B (mlx-vlm layout) is parked because mixed gate/up bits can't fuse losslessly. CI green.
  - [#3](https://github.com/apumutyala/mlx-agent-console/pull/3) `memory-harness`: chats, memory store, BM25 + LFM2.5 embeddings, playbook with versions, document tools. CI green.
  - [#4](https://github.com/apumutyala/mlx-agent-console/pull/4) `fix-worker` (coordinator): 15 worker findings fixed. 49 worker tests pass.
- **In progress:** `fix-node` (20 iPad findings + training opt-in), `fix-speech` (8 findings, invalid Kokoro voice id), `bench` (profiler/receipts), `inkling-stream` (SSD expert streaming per `docs/INKLING-SSD-EXPERT-STREAMING-DESIGN.md`).
- **Streaming design takeaways:**
  - mmap does not work on iPadOS: Metal pins the whole buffer, and the page cache never warms.
  - Use per-layer slot tensors plus `preadv` into them.
  - Modelled: 2/3-bit at 25% resident ≈ 4.5 GB / ~12 tok/s; 4-bit at 50% ≈ 9.2 GB / ~16 tok/s. Hit rate and sync cost are unmeasured.
- **Process note:** the PR review workflow for #2–#4 is running. Merges wait on its confirmed findings.
### 2026-11-27: memory harness (`memory-harness`)
- **What:** `ios/AgentConsole/Memory/` adds saved chats (resume, rename, delete) and a typed memory store (fact, episode, lesson, world).
  - Retrieval is hybrid: BM25 always, plus optional LFM2.5-Embedding-350M-4bit through MLXEmbedders 3.33.0.
  - Each conversation starts with a budgeted system block: a situation report (time, device, model, documents, worker jobs, fleet nodes), the playbook, and the top-k memories. The default budget is 1200 tokens.
  - A versioned, self-maintained playbook can be rolled back.
  - A reflection pass runs after each chat or every 6 user turns. Its output is strict JSON, validated in Swift, and no tools run during it.
  - New tools: `remember`, `recall`, `forget` (asks the user to confirm), `create_document`, `append_to_document`, `list_documents`.
- **Safety:** chats that contain worker or fleet output produce `untrusted` memories. Playbook edits from those chats, and from imported files, wait for the user's approval.
- **GPU:** reflection and embedding take `GPUGate`, are cancelled before each chat turn, and never run in the background (no Metal there). The embedder is unloaded whenever `downloadGB + 0.8 > 0.45 × RAM`, so it is off for Inkling-size models.
- **Status:** CI is green; nothing has run on a device yet.

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
