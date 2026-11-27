# Agent Console

An iPhone/iPad app that runs a small language model **on the device** with MLX, and also works as a **remote control** for long jobs that run on a Mac or Linux worker.

- **On-device** (`ios/`): chat with a local model that has typed tools. Batch-summarize imported files. A batch can keep running after you switch apps by using `BGContinuedProcessingTask` with background GPU access, but only where the device supports it.
- **Worker** (`worker/`): a stdlib-only Python HTTP service. It owns long-running processes, so a phone that gets suspended, force-quit or goes offline never kills a job.

## Layout

```
worker/
  jobworker.py          HTTP API, job store, process groups, approvals
  job_types.json        the only commands that can run (named types plus regex-checked params)
  examples/demo_harness.py   example harness that uses the approval protocol
  test_jobworker.py     end-to-end tests: real server, real subprocesses
ios/
  project.yml           XcodeGen spec (MLX package, entitlements, BG task IDs)
  AgentConsole/
    Inference/          ModelBackend protocol, MLXBackend (the only MLX file), Tools, LocalAgent loop
    Documents/          DocumentStore (imports copied into the app), checkpointed BatchSummarizer
    Background/         BGContinuedProcessingTask coordinator
    Remote/             WorkerClient + Keychain, JobFeed (cursor plus on-disk event cache)
    Intents/            Start Job / Get Job Status / Cancel Job App Intents + App Shortcuts
    Views/              Chat, Documents, Jobs, Settings
```

## Worker

```bash
cd worker
export WORKER_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
python jobworker.py --host 0.0.0.0 --port 8787 --data ./worker-data   # bind to your Tailscale IP in practice
```

| Route | Purpose |
|---|---|
| `GET /job-types` | named job types and their param regexes |
| `POST /jobs` `{"type", "params"}` | start a job (or put it in `waiting_approval` if the type needs approval) |
| `GET /jobs`, `GET /jobs/{id}` | status, exit code, pending approvals, `last_seq` |
| `GET /jobs/{id}/events?after=N&wait=S` | replay from the cursor; long-polls for up to 30 s |
| `POST /jobs/{id}/cancel` | SIGTERM the process group, then SIGKILL after a grace period (Windows: `taskkill /T`) |
| `POST /jobs/{id}/retry` | start a new job with the same type and params |
| `POST /jobs/{id}/approvals/{aid}` `{"decision"}` | answer a pending approval |

- **No shell endpoint.** A job is a named type from `job_types.json`. Each param must fill a whole argv element and fully match its regex, so a request can't inject flags or shell syntax. Unknown params are rejected.
- **Durability.** Each job keeps `job.json` (written atomically) and an append-only, fsynced `events.jsonl`. Events carry contiguous `seq` numbers, which serve as the reconnect cursor. If the worker restarts, jobs that were running are marked `lost`: the worker never claims a process it can no longer see.
- **Approvals mid-run.** Your harness prints `::approval:: {"id":"push","prompt":"Push to origin?"}` and then reads one JSON line from stdin. The job shows `waiting_approval` until someone answers from the phone. A type with `"requires_approval": true` needs approval before it starts.
- **Pause** is left out on purpose. Freezing an arbitrary subprocess isn't safe. Use cancel and retry, or build checkpoint and resume into the harness.

Plug in your own coding agent by editing `overnight_agent` in `job_types.json`.

### Tests (run on this machine)

```bash
cd worker && python -m unittest -v test_jobworker
```

There are 6 tests, all passing on Windows / Python 3.12. The main test follows the device scenario:

1. Start a job.
2. Stream a few events, then drop the client but keep its cursor (the "force-quit").
3. Reconnect and replay. It asserts the seqs are contiguous: no gaps, no duplicates.
4. Answer a mid-run approval.
5. Cancel, and assert that the **grandchild** process is dead too.

The other tests cover auth, param injection, success and retry, prompt long-poll wakeups, and worker restart recovery.

## iOS app

Requires macOS with Xcode 26 and a **physical** iPhone or iPad. MLX needs Metal; the simulator is not useful for this.

```bash
brew install xcodegen
cd ios && xcodegen && open AgentConsole.xcodeproj
```

Set your team and change `bundleIdPrefix`. In Settings, enter the worker URL and token. The token is stored in the Keychain.

How it is put together:

- **`ModelBackend`** keeps MLX behind one interface. `MLXBackend.swift` is the only file that imports MLX. It follows `LLMEval`: `#huggingFaceLoadModelContainer` loads the model, and a `ChatSession` holds each conversation.
- **Models** (`Inference/ModelCatalog.swift`, picked in Settings, saved in UserDefaults). Liquid AI models are the default:

  | Model | Download | Notes |
  |---|---|---|
  | **LFM3 1.7B** (`LiquidAI/LFM3-1.7B-MLX-4bit`) | ~1.0 GB | default; XML tool calls, `<think>` reasoning |
  | LFM2.5 1.2B Instruct (`LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit`) | ~660 MB | fast, no thinking; pythonic tool calls |
  | LFM2.5 350M (`LiquidAI/LFM2.5-350M-MLX-4bit`) | ~230 MB | tiny/fast, weakest at tools |
  | LFM2.5 8B-A1B MoE (`LiquidAI/LFM2.5-8B-A1B-MLX-4bit`) | ~4.9 GB | strongest; only listed when the device has a 16 GB-class RAM (`physicalMemory` ≥ 14 GiB) |
  | Qwen3 1.7B (`mlx-community/Qwen3-1.7B-4bit`) | ~1.0 GB | fallback (previous default) |

  Switching drops the loaded container, calls `MLX.Memory.clearCache()`, starts a new chat and loads the new model (first use downloads it). LFM3 needs swift-transformers ≥ 1.4.0 (tokenizer fixes), which `project.yml` requires.
- **Reasoning.** For thinking models mlx-swift-lm streams `<think>` content separately (`Generation.reasoning`, from each model's `reasoningConfig`). The chat shows it in a collapsed "Thinking" row; a fallback splitter hides any raw `<think>` tags the library didn't separate. Summaries drop reasoning.
- **Tools.** Tools are declared natively (`ChatSession(tools:)` with JSON-Schema `ToolSpec`s), so each model's own chat template renders them and mlx-swift-lm parses the model's own call dialect (LFM3 XML `<tool_call><function=…>`, LFM2/2.5 pythonic `<|tool_call_start|>[f(k='v')]<|tool_call_end|>`, Qwen JSON) into structured `.toolCall`s. There is no `toolDispatch`: the app runs calls itself and returns results as `tool`-role messages. If a call slips through as plain text, `ToolCallParser` (all three dialects) is the fallback. Swift then does the following in order:
  1. Parses the proposal and rejects unknown keys (`strictDecode`).
  2. Decodes it into a typed `Arguments` struct.
  3. Validates it semantically. For example, `fileID` must name an imported file.
  4. Asks the user first if the tool needs confirmation.
  5. Runs it.

  Errors go back to the model as an `ERROR: …` tool result; a call the library rejects (malformed or undeclared) is reported back with the original request. The tool loop stops after 4 steps. There are two tools: `summarizeImportedFile` (read-only) and `submitWorkerJob`, which only accepts named types and always asks the user first.
- **Batches** save their state after every file. When a batch expires, is cancelled or crashes, every finished summary is already saved, and Resume picks up at the next file. In the foreground on a device without background GPU, the batch stops at a file boundary when the app goes to the background. It doesn't try to run Metal work while suspended.
- **Continued processing.** The app first checks `BGTaskScheduler.supportedResources.contains(.gpu)`. If the GPU is supported, it submits `BGContinuedProcessingTaskRequest(... ).requiredResources = .gpu` with strategy `.fail`. While the batch runs, it reports progress and updates the subtitle. On expiration it cancels the batch, which keeps all finished files. If the GPU isn't supported, or the request is refused, the batch runs in the foreground only.
- **Job console.** `JobFeed` caches events on disk next to their cursor. After a force-quit you see what you already had immediately, and the app fetches only `after=<cursor>`. The feed long-polls (25 s), backs off on errors, and starts following again when the app returns to the foreground. Approve, Deny, Cancel (with a confirmation) and Retry are in the job screen.
- **App Intents.** Start Job, Get Job Status and Cancel Job are defined, and Cancel asks for confirmation. Job types come from the worker as `AppEntity` values. Shortcuts only offers types whose params all have defaults.

### Not verified: please check on a Mac

I wrote the Swift code on Windows, where no Swift toolchain was available. **It has not been compiled.** These are the places most likely to need small fixes:

1. **`mlx-swift-lm` API surface** (`MLXBackend.swift`). This includes `LLMRegistry.qwen3_1_7b_4bit`, the `ChatSession(_:instructions:generateParameters:)` initializer, `streamResponse(to:)` and `MLX.GPU.set(cacheLimit:)`. The package is set to `branch: main`; pin a tag once it builds. If the libraries moved, compare against the current `LLMEval` in `ml-explore/mlx-swift-examples`.
2. **BGContinuedProcessingTask details**:
   - registering with the wildcard ID `<bundle>.summarize.*` and submitting `<bundle>.summarize.<batchID>`
   - `task.updateTitle(_:subtitle:)`
   - `requiredResources = .gpu`

   Check these against the current BackgroundTasks docs.
3. **Swift 6 strictness.** The project uses language mode 5, so the non-Sendable `BGTask` captures produce warnings, not errors.

## Device test plan

The two tests from the design:

**A. Remote control outlives the app**
1. On the worker, set `DEMO_DELAY=2`, then from the iPad start `demo_agent` (steps 200) or `repo_tests`.
2. Watch a few log lines, then force-quit the app (swipe it away).
3. Wait about 30 s, then reopen it and go to the job. The log should continue with no gaps (the Events count matches the worker's `last_seq`).
4. Answer the "Apply the patch?" approval.
5. Cancel from the phone. On the worker, `ps -o pid,pgid,cmd` should show no `sleep(3600)` helper left.
6. Repeat step 1 through Siri: "Start a job in Agent Console." Then check it from the lock screen: "Check my job in Agent Console."

**B. Local agent in airplane mode**
1. Load the model once while online (it downloads to the app container).
2. Import 2 or 3 text or PDF files, then turn on airplane mode.
3. On the Agent tab, send "Summarize <file>". You should see a `summarizeImportedFile` tool call followed by a summary.
4. On the Documents tab, select 10 files and tap Summarize, then:
   - **Foregrounded:** the batch completes.
   - **Backgrounded:** switch apps. Watch the system progress UI if the device supports `.gpu`.
   - **Screen locked:** lock the device mid-batch.
   - **Expired:** if the system stops the batch, reopen the app. The batch shows `k of 10` with k summaries saved. Resume should start at file k+1.

Record the results for each device and OS build. A device that supports the capability isn't proof that MLX generation actually continues on that hardware.

## Claude app connector (iPad / iPhone / claude.ai)

[`worker/mcp_bridge.py`](worker/mcp_bridge.py) is a small MCP server, stdlib only, that uses the Streamable HTTP transport. It runs as its own process in front of the worker. Any Claude chat that has it enabled can:

- list the job types and jobs
- start a job
- read a job's log, page by page or by waiting for new lines
- cancel or retry a job
- answer an approval: by default it can only deny, and approving is left to you

It has no shell or file tools. Jobs keep running after the chat closes.

**1. Start both services** on the machine that should run the jobs:

```powershell
.\worker\start-connector.ps1            # Windows; -AllowApprove lets Claude approve, -ShowUrl prints the URL
```
```bash
./worker/start-connector.sh             # Mac/Linux; --allow-approve, --show-url
```

Both services listen only on localhost. Secrets are generated once and stored in `~/.job-worker/`, along with the logs and job data.

**2. Publish the bridge over HTTPS.** The Claude app connects to connectors from Anthropic's servers, so the bridge needs a public HTTPS URL. With Tailscale:

1. Allow Funnel for this machine in the Tailscale admin console.
2. Run:

   ```bash
   tailscale funnel --bg 8788
   ```

This publishes only the bridge on port 8788. The worker on port 8787 stays private.

**3. Add it in Claude:** go to claude.ai, then Settings → Connectors → *Add custom connector*.

- **Name:** Job Worker
- **URL:** `https://<machine>.<tailnet>.ts.net/mcp/<BRIDGE_SECRET>`. Run the start script with `-ShowUrl` / `--show-url` to print it.
- **Authentication: choose "No sign-in".** The secret in the URL is the key. The default, "Sign in now" with Claude's published identity (CIMD), fails with "Couldn't register". If you do pick an OAuth mode, [`oauth_shim.py`](worker/oauth_shim.py) can complete Dynamic Client Registration (DCR) instead, gated on the same secret.

The connector then shows up in the Claude app on your iPad and phone. Turn it on for a chat from the tools menu, then ask things like "start demo_agent with 50 steps" or "what's my overnight job doing?"

**For Claude Code on the same machine**, the public URL isn't needed:

```bash
claude mcp add --transport http job-worker http://127.0.0.1:8788/mcp --header "Authorization: Bearer <BRIDGE_SECRET>"
```

**Connector security model**
- The secret is part of the URL, because the "Add custom connector" dialog accepts only a URL. Anyone with the URL can use the tools, so treat it like a password. Rotate it by deleting `~/.job-worker/secrets.json` (or `secrets.env`), restarting, and re-adding the connector.
- Wrong or missing secrets get a 404, so scanners can't tell the endpoint exists. Browser `Origin` headers are rejected (the spec's DNS-rebinding guard), and the access log masks the secret.
- What the connector can do is limited by `job_types.json`: named types with regex-checked params. Keep the types that need approval (`requires_approval`) for anything sensitive. Claude can't approve them unless you start the bridge with `--allow-approve`.
- **Hardening defaults:**
  - OAuth endpoints are off unless you pass `--enable-oauth`.
  - Tool calls are limited to 60 a minute, and start/cancel/retry/approve to 10 a minute.
  - At most 32 connections at once, and each socket times out after 30 s.
  - Every tool call is appended to `~/.job-worker/audit.jsonl` (the secret is never logged).
  - Job log output is marked as untrusted data in tool results and in the server instructions.
- In claude.ai → connector → **Permissions**, set the write tools to always ask. That way prompt injection in a page or a job log can't start or cancel anything without your tap.
- Run `tailscale funnel --bg 8788 off` to take it offline. The jobs and the app keep working over your private tailnet.

**Tests:** run `uv run --no-project --with mcp python -m unittest -v test_mcp_bridge`. There are 12 tests, all passing:
- auth: secret path, bearer token, wrong secret, foreign Origin
- initialize with protocol-version negotiation
- the whole flow (start, follow the log with the cursor, deny an approval, cancel)
- deny-only enforcement, and approval when `--allow-approve` is set
- injection and unknown-argument rejection
- log paging that stays within the size budget and resumes exactly where it stopped
- a session driven by the **official MCP Python SDK client** (v2.3.0), as an independent compatibility check

## Compute fleet and federated training

The worker also coordinates a **fleet of devices** (the iPad, other Macs/iPhones) that run models
locally. Devices never accept inbound connections: each one sends heartbeats, long-polls for its
next task, streams tokens back, uploads artifacts, and reports a result. That works through NAT,
Tailscale, and iOS's networking rules (the app serves while it is open, or during a
user-started continued-processing task).

| | |
|---|---|
| Protocol + storage | [`worker/fleet.py`](worker/fleet.py): nodes, task queue, long-poll, cancel, requeue when a node vanishes (max 2 attempts), artifacts |
| Federated rounds | [`worker/federated.py`](worker/federated.py): one `train_lora` task per device, then a weighted **FedAvg** of the uploaded LoRA adapters (stdlib safetensors F32/F16/BF16) |
| Claude tools | `list_nodes`, `run_on_node`, `broadcast_prompt` (same prompt on every device, in parallel), `get_task`, `cancel_task`, `start_fed_round`, `get_fed_rounds` |
| Tests | `worker/test_fleet.py` (simulated nodes over HTTP, FedAvg numerics, restart/offline requeue); `test_mcp_bridge.test_fleet_tools` |

**Federated LoRA round**:
1. The coordinator sends each device the same seed, rank and layer count, plus the previous global adapter when there is one.
2. Each device trains on its own documents and/or a shard of examples. Raw data never leaves the device.
3. Each device uploads `adapter.safetensors`.
4. The coordinator averages the adapters, weighted by example count, and publishes the result at `/fed/rounds/{id}/adapter`.

The iPad reaches the worker over the tailnet only (`tailscale serve --https=8443 http://127.0.0.1:8787`). Only the MCP bridge is on the public Funnel.

**What iOS does not allow:**
- Controlling other apps, firmware, or which apps run in the background.
- A permanently running background server.

On a device that allows background GPU, a node serves in the foreground or inside a `BGContinuedProcessingTask` that you start. Everything in this repo stays inside those rules.

## Security notes

- Never expose the worker to the open internet. Use Tailscale or a VPN, or put it behind a TLS proxy. The app's ATS settings only allow plain `http` to local and IP addresses.
- The token is a single shared bearer secret, checked in constant time. For more than one person, put real auth in front of the worker.
- The model can't run arbitrary commands. It can only propose a named job type, and the user must approve it.
