# Inkling 2 Mini: SSD expert streaming (operator and measurement guide)

Each MoE layer keeps `S` of its 128 experts in memory, inside its usual expert tensors. The rest
are read from `experts.bin` on a miss, with `F_NOCACHE` `preadv` straight into the slot's bytes.
The design and its sources are in the research note `INKLING-SSD-EXPERT-STREAMING-DESIGN.md`
(stage plan M0 to M5).

| Tier (catalog id) | Slots per layer | Expert memory | Weights resident | Est. footprint | Fits a free sideload (~5.3 GB)? |
|---|---|---|---|---|---|
| `inkling2-mini-dwq3.1-stream19` | 24 | 1.54 GB | 2.90 GB | ~3.6 GB (4K) | yes |
| `inkling2-mini-dwq3.1-stream25` | 32 | 2.06 GB | 3.42 GB | ~4.1 GB (4K), ~4.3 GB (8K) | yes, the largest that does |
| `inkling2-mini-dwq3.1-stream50` | 64 | 4.12 GB | 5.48 GB | ~6.4 GB | no (needs increased-memory-limit) |
| `inkling2-mini-dwq3.1-stream100` | 128 | 8.24 GB | 9.60 GB | ~10.5 GB | no |
| `inkling2-mini-dwq3.1` (resident) | n/a | 8.24 GB | 9.60 GB | ~10.5 GB | no |

**Budget on the free-sideload iPad** (device receipt 2026-11-27): `os_proc_available_memory()` is
about 5.2 GB before the load, so the per-app limit is about 5.3–5.5 GB and the entitlement is not
in effect. The footprint target is ≤ 4.6 GB.

- One slot across all 24 layers costs 64.36 MB (the DWQ record sizes summed over the layers).
- Footprint = 1.36 GB core (embed, head, attention, norms; the vision tower is loaded only on
  demand) + S × 64.36 MB + KV (~0.2 GB at 4K, ~0.4 GB at 8K) + prefill scratch (~0.25 GB at
  256-token steps) + ~0.2 GB runtime.
- **S = 32 (25%) is the largest slot count that fits with headroom.** At 4K context S ≈ 36 would
  still fit; at 8K, S = 32 is the limit.
- The KV, scratch and runtime terms are estimates. Settings → Memory → *Last large load* shows the
  measured peak.

All of these use the same bytes: `LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ` at
`41debd25a0558fc9d127172dbb3da210cf7fb681`, repacked byte-exactly. The sizes come from its
safetensors headers.

## 1. Make the streamed build (on the Omen, once)

```powershell
# 1. Download the source build (~10.3 GB) with the HF CLI or a browser.
hf download LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ --revision 41debd25a0558fc9d127172dbb3da210cf7fb681 `
  --local-dir D:\models-src\Inkling-2-Mini-MLX-3.1bpw-DWQ
# 2. Repack (stdlib Python; ~10 GB written; a few minutes on NVMe).
python tools\expert_repack.py D:\models-src\Inkling-2-Mini-MLX-3.1bpw-DWQ `
  D:\models\Inkling-2-Mini-MLX-3.1bpw-DWQ-stream --verify all
```

- The output directory name must be `Inkling-2-Mini-MLX-3.1bpw-DWQ-stream`, which is what the
  catalog expects (`ModelCatalog.dwqStreamName`).
- The output holds the core shards (same names, no `mlp.experts.*`), a new
  `model.safetensors.index.json`, `experts.bin`, `experts.json`, and every other source file
  unchanged.
- `--verify all` compares every expert record and every core tensor with the source, and re-hashes
  each layer. `sample` (the default) checks about 2% of the experts.
- The manifest records each source shard's SHA-256.

## 2. Serve it read-only from the worker

```powershell
$env:WORKER_MODELS_DIR = "D:\models"     # or: python jobworker.py ... --models-dir D:\models
```

Restart the worker. It then answers these requests, behind the usual bearer token:

- `GET /models` lists the builds.
- `GET /models/<name>` lists the files with their sizes.
- `GET /models/<name>/<file>` returns one file, and supports `Range`.

Only regular files directly inside `<models-dir>/<name>/` are served. Symlinks, hidden files,
`*.partial` files and subdirectories are not.

## 3. On the iPad

1. **Point the app at the server.** Pair the worker (Settings → Remote worker). Or put another base
   URL in Settings → *Inkling expert streaming* → *Streamed models URL*. The worker token goes only
   to the worker's own host.
2. **Pick a tier.** Choose *Inkling 2 Mini · streamed 25% / 50% / 100% (check)*. The first load
   downloads the build into Application Support/StreamedModels, which is excluded from backup.
   The download resumes after interruption. The load then reads only the core and the slots.
3. **Check the load.** Settings → Memory → *Last large load* reports
   `streaming S/128 experts per layer (… of slots)`.

## 4. Correctness gate (offload on vs off)

This gate is designed but **not run yet**: nothing has run on a device.

On a free sideload, no resident or 100% tier fits, so the gate has two parts.

**(a) On device, offload changes nothing.** With *Exact prefill* on, prefill runs 3 tokens per step.
A call then needs at most 24 distinct experts, so nothing is chunked. Every streamed tier then
computes the same math a resident model would with that step; only the slot ids differ.

1. Turn on *Greedy decoding* and *Exact prefill*.
2. For each gate prompt, start a new chat on `stream19`, then on `stream25`.
3. **Pass:** every token is identical. The two tiers differ in which experts are resident, which
   ones miss and which slots get evicted, so this exercises the IO, the eviction and the remap.

**(b) Against a reference.** Run the same DWQ build in mlx-lm on a Mac or a Linux CPU box
(PR #2086) at temperature 0 with the same chat template and prefill step 3.
- **Pass:** the ten-token gate, greedy-matching tokens 1–9.
- Logit cosine is not exported by the app yet.
- Without *Exact prefill*, prefill is chunked expert-major whenever a call needs more than S
  distinct experts. That regroups matmul rows and may change rounding, so record the first
  mismatch index.

On a device where the resident build fits (entitlement granted), the original gate also applies:
`stream100` vs `inkling2-mini-dwq3.1`, identical tokens.

With the entitlement granted, the full gate is:

1. **Load the resident reference.** Turn on Settings → *Inkling expert streaming* → *Greedy
   decoding*. Load `inkling2-mini-dwq3.1` and send each gate prompt in a new chat. Save the
   transcripts.
2. **Load the streamed check tier.** Load `inkling2-mini-dwq3.1-stream100` and send the same
   prompts.
   - **Pass:** every generated token matches (greedy). With `S == 128`, slot `e` holds expert `e`,
     so the expert tensors and the computation are identical to the resident load. Any mismatch
     is a bug in the IO, binding or manifest, not rounding.
3. **Decode-only check at 50%.** Load `stream50` and send the same prompts.
   - **Pass:** at least the first 64 tokens match greedily on every prompt that has no prefill
     chunk over capacity.
   - Prompts longer than about 8 tokens make the prefill call need more than 64 distinct experts.
     The chunked prefill then regroups rows into different matmul segments, which may change
     rounding. For those prompts, report the first mismatch index instead of failing the gate.
     The oMLX analogue diverges after a few hundred characters.
4. **Record the evidence.** Keep the hit rate from Settings and the receipts.

## 5. Measurements wanted (design M0/M1)

Run on the iPad Pro M5 16 GB, plugged in, after a cold launch, with the 1K-token prompt from the
benchmark screen and 128 generated tokens:

| # | Run | Record |
|---|---|---|
| A | `stream25`, trace **on** | tok/s, TTFT, peak (Settings), counters, and the trace file. Routing does not depend on residency, so this trace serves every S in the simulator. |
| B | `stream19` | tok/s, TTFT, peak, counters |
| C | `stream25`, *Exact prefill* on, two prompts | gate (a) together with B |
| D | IO microbenchmark | `inkling_stream.read_gb_per_s` after A. It is the effective `preadv` bandwidth at 4 readers. |

- Between phases, use *Reset counters* (or `BenchMetricsRegistry.resetAll()`). The
  `inkling_stream.*` keys from `BenchMetricsRegistry.collect()` belong in each receipt.
- Analyse the trace from A offline, which costs no device time:

  ```
  python tools/expert_trace_sim.py inkling-routing.jsonl --slots 32 64 96 --decode-only
  ```

  This prints the hit rate of the app's decayed-use policy, LRU and the oracle at each `S`, plus
  the loop-pass Jaccard overlap and the expert-visits per token. Those numbers feed the hit-rate
  rows of the design's budget table.

## Known limits

- **Affine builds only.** NVFP4/MX experts would need slot-indexed global scales in the fused
  kernels, and the load refuses them with a clear error.
- **Read errors are fatal.** A read error during generation stops the app with a message
  (`fatalError`), because the model's forward pass can't throw and a silent zero expert would
  produce wrong text.
- **Syncs per token.** Every MoE call syncs the host to read its routes: 48 syncs per decoded token.
  The skip applies only when `S == 128`, every expert is resident and no trace is on.
- **No prefetch or read overlap yet.** Prefetch and overlap are design stages M4 and M5.
