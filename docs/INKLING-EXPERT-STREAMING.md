# Inkling 2 Mini: SSD expert streaming (operator and measurement guide)

Each MoE layer keeps `S` of its 128 experts in memory, inside its usual expert tensors. The rest
are read from `experts.bin` on a miss, with `F_NOCACHE` `preadv` straight into the slot's bytes.
The design and its sources are in the research note `INKLING-SSD-EXPERT-STREAMING-DESIGN.md`
(stage plan M0 to M5).

| Tier (catalog id) | Slots per layer | Expert memory | Weights resident |
|---|---|---|---|
| `inkling2-mini-dwq3.1-stream25` | 32 | 2.06 GB | 3.42 GB |
| `inkling2-mini-dwq3.1-stream50` | 64 | 4.12 GB | 5.48 GB |
| `inkling2-mini-dwq3.1-stream100` (gate) | 128 | 8.24 GB | 9.60 GB |
| `inkling2-mini-dwq3.1` (resident, reference) | n/a | 8.24 GB | 9.60 GB |

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
| A | resident `inkling2-mini-dwq3.1`, trace **on** | tok/s, TTFT, peak (Settings), and the trace file |
| B | `stream100` | tok/s, TTFT, peak; hit rate after warm-up must reach 100% |
| C | `stream50` | tok/s, TTFT, peak, hit rate, bytes read, read GB/s, routing wait, GPU sync |
| D | `stream25` | same as C |
| E | Settings → Memory, under the real signing | `os_proc_available_memory()` ("Available to app") |

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
