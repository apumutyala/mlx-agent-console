"""Compute fleet: devices (iPad, Mac, ...) that pull model tasks from the worker.

Nodes never accept inbound connections. A node long-polls for its next task,
streams events back, uploads artifacts (e.g. a LoRA adapter), then reports a
result. This works through NAT, Tailscale, and iOS's foreground-only networking.

    POST /nodes/{node_id}/heartbeat     {"name", "kind", "capabilities": {"kinds": [...], ...}}
    GET  /nodes/{node_id}/next?wait=25  -> {"task": {...} | null}          (long-poll)
    POST /tasks/{id}/events             {"node_id", "events": [{"kind": "token", "text": ...}]}
                                         -> {"cancel": bool}
    PUT  /tasks/{id}/artifacts/{name}?node_id=...   raw bytes
    POST /tasks/{id}/complete           {"node_id", "status": "succeeded"|"failed", "output", "error", "metrics"}

    GET  /nodes                          fleet view (online = seen in the last 75 s)
    POST /tasks                          {"kind": "generate", "payload": {...}, "node": optional id}
    GET  /tasks, /tasks/{id}, /tasks/{id}/events?after=&wait=
    POST /tasks/{id}/cancel
    GET  /tasks/{id}/artifacts/{name}

Storage reuses the job layout: tasks/<id>/job.json + append-only events.jsonl,
so the same cursor/long-poll semantics apply.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from jobworker import Job, now

TASK_KINDS = {"generate", "train_lora"}
TASK_TERMINAL = {"succeeded", "failed", "cancelled", "lost"}
NODE_ID = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
TASK_ID = re.compile(r"^t[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
MAX_PAYLOAD = 512_000
MAX_ARTIFACT = 64 * 1024 * 1024  # LoRA adapters are a few MB


class Forbidden(Exception):
    pass


def new_task_id() -> str:
    return "t" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]


class Fleet:
    def __init__(self, data_dir: Path, offline_after: float = 75.0, max_attempts: int = 2,
                 queue_deadline: float = 1800.0):
        self.root = Path(data_dir) / "fleet"
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self.offline_after = offline_after
        self.max_attempts = max_attempts
        # A task pinned to one node expires (lost) if that node stays offline this long.
        self.queue_deadline = queue_deadline
        # RLock-based: _finish/_requeue may run while the caller already holds it.
        self.cond = threading.Condition(threading.RLock())
        self.nodes: dict[str, dict] = {}
        self.tasks: dict[str, Job] = {}
        self._listeners: list = []  # callables(task_meta) on terminal; must exist before _load
        self._load()

    # ------------------------------------------------------------ storage --
    def _load(self) -> None:
        nodes = self.root / "nodes.json"
        if nodes.exists():
            self.nodes = json.loads(nodes.read_text(encoding="utf-8"))
        for d in sorted((self.root / "tasks").iterdir()):
            meta_path = d / "job.json"
            if not meta_path.exists():
                continue
            task = Job(d, json.loads(meta_path.read_text(encoding="utf-8")))
            self.tasks[task.id] = task
        # In-flight tasks are left alone: a node that is still running one keeps
        # reporting to it. sweep() requeues them only if the owner stays offline.
        # Give every known node a fresh grace period instead of judging it by
        # timestamps from before the restart.
        for n in self.nodes.values():
            n["last_seen"] = now()

    def _save_nodes(self) -> None:
        tmp = self.root / "nodes.json.tmp"
        tmp.write_text(json.dumps(self.nodes, indent=2), encoding="utf-8")
        tmp.replace(self.root / "nodes.json")

    def on_terminal(self, fn) -> None:
        self._listeners.append(fn)

    def _finish(self, task: Job, status: str, **extra) -> bool:
        """Move a task to a terminal status exactly once. Returns False if it already was."""
        with self.cond:
            if task.meta["status"] in TASK_TERMINAL:
                return False
            task.set_status(status, **extra)
            node = self.nodes.get(task.meta.get("assigned_to") or "")
            if node and node.get("current_task") == task.id:
                node["current_task"] = None
            self.cond.notify_all()
        for fn in self._listeners:
            try:
                fn(task.meta)
            except Exception as e:  # a listener bug must not wedge the fleet
                task.append("log", stream="fleet", line=f"listener error: {e!r}")
        return True

    def _requeue_or_lose(self, task: Job, reason: str) -> None:
        with self.cond:
            if task.meta["status"] in TASK_TERMINAL:
                return
            if task.meta.get("cancel_requested"):
                self._finish(task, "cancelled", reason=reason)
            elif task.meta.get("attempts", 0) < self.max_attempts:
                node = self.nodes.get(task.meta.get("assigned_to") or "")
                if node and node.get("current_task") == task.id:
                    node["current_task"] = None
                task.set_status("queued", assigned_to=None, reason=reason)
                self.cond.notify_all()
            else:
                self._finish(task, "lost", reason=reason)

    # -------------------------------------------------------------- nodes --
    def heartbeat(self, node_id: str, info: dict) -> dict:
        if not NODE_ID.match(node_id or ""):
            raise ValueError("invalid node id")
        caps = info.get("capabilities") or {}
        if not isinstance(caps, dict):
            raise ValueError("capabilities must be an object")
        kinds = [k for k in caps.get("kinds", ["generate"]) if k in TASK_KINDS]
        with self.cond:
            node = self.nodes.setdefault(node_id, {"id": node_id, "first_seen": now(), "current_task": None})
            node.update(name=str(info.get("name") or node_id)[:64],
                        kind=str(info.get("kind") or "unknown")[:32],
                        capabilities={**{k: v for k, v in caps.items() if k != "kinds"}, "kinds": kinds},
                        last_seen=now())
            self._save_nodes()
            self.cond.notify_all()
            return dict(node)

    def _online(self, node: dict) -> bool:
        return time.time() - node.get("last_seen", 0) < self.offline_after

    def nodes_public(self) -> list[dict]:
        self.sweep()
        with self.cond:
            return [{**n, "online": self._online(n)} for n in
                    sorted(self.nodes.values(), key=lambda n: n.get("last_seen", 0), reverse=True)]

    def online_nodes(self, kind: str | None = None) -> list[str]:
        return [n["id"] for n in self.nodes_public()
                if n["online"] and (kind is None or kind in n["capabilities"]["kinds"])]

    def sweep(self) -> None:
        """Requeue work stuck on nodes that stopped polling, and expire tasks
        pinned to a node that has been gone longer than queue_deadline."""
        with self.cond:
            t_now = time.time()
            for t in list(self.tasks.values()):
                m = t.meta
                if m["status"] in ("assigned", "running"):
                    if not self._online(self.nodes.get(m.get("assigned_to") or "", {})):
                        self._requeue_or_lose(t, "node went offline")
                elif m["status"] == "queued" and m.get("target"):
                    target = self.nodes.get(m["target"], {})
                    gone_since = max(target.get("last_seen", 0), m.get("updated_at", 0))
                    if not self._online(target) and t_now - gone_since > self.queue_deadline:
                        self._finish(t, "lost", reason="target node offline past the queue deadline")

    # -------------------------------------------------------------- tasks --
    def create_task(self, kind: str, payload: dict, node: str | None = None, **extra) -> Job:
        if kind not in TASK_KINDS:
            raise ValueError(f"unknown task kind {kind!r}; known: {sorted(TASK_KINDS)}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if len(json.dumps(payload)) > MAX_PAYLOAD:
            raise ValueError("payload too large")
        if node is not None and node not in self.nodes:
            raise LookupError(f"no such node: {node}")
        if kind == "generate" and not isinstance(payload.get("prompt"), str):
            raise ValueError("generate needs payload.prompt (string)")
        tid = new_task_id()
        d = self.root / "tasks" / tid
        d.mkdir(parents=True)
        meta = {"id": tid, "kind": kind, "payload": payload, "target": node, "status": "queued",
                "created_at": now(), "updated_at": now(), "assigned_to": None, "attempts": 0,
                "output": None, "error": None, "metrics": {}, "artifacts": [], **extra}
        task = Job(d, meta)
        task.save_meta()
        task.append("created", task_kind=kind, target=node)
        with self.cond:
            self.tasks[tid] = task
            self.cond.notify_all()
        return task

    def get(self, task_id: str) -> Job:
        task = self.tasks.get(task_id) if TASK_ID.match(task_id or "") else None
        if task is None:
            raise LookupError("no such task")
        return task

    def public(self, task: Job, include_payload: bool = True) -> dict:
        m = dict(task.meta)
        if not include_payload:
            m.pop("payload", None)
        return {**m, "last_seq": task.next_seq - 1}

    def _claim(self, node_id: str) -> Job | None:
        node = self.nodes[node_id]
        kinds = node["capabilities"]["kinds"]
        for t in sorted(self.tasks.values(), key=lambda t: t.meta["created_at"]):
            m = t.meta
            if m["status"] == "queued" and m["kind"] in kinds and m["target"] in (None, node_id):
                return t
        return None

    def next_task(self, node_id: str, wait: float) -> dict | None:
        if node_id not in self.nodes:
            raise LookupError("unknown node; send a heartbeat first")
        self.sweep()
        deadline = time.monotonic() + wait
        with self.cond:
            # A node runs one task at a time and only polls between tasks, so:
            #  - a task it owns but never started (claim response lost) is re-delivered;
            #  - a task it owns and had started was abandoned (app paused / killed).
            for t in list(self.tasks.values()):
                if t.meta.get("assigned_to") != node_id:
                    continue
                if t.meta["status"] == "assigned":
                    self.nodes[node_id]["last_seen"] = now()
                    t.meta["deliveries"] = t.meta.get("deliveries", 1) + 1
                    if t.meta["deliveries"] <= 3:
                        return self.public(t)
                    # Handed out repeatedly but never started: the node can't run it.
                    self._requeue_or_lose(t, "node repeatedly failed to start the task")
                    continue
                if t.meta["status"] == "running":
                    self._requeue_or_lose(t, "node polled again without completing (abandoned)")
            while True:
                self.nodes[node_id]["last_seen"] = now()
                task = self._claim(node_id)
                if task is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(min(remaining, 5.0))
            # Claim atomically: nobody else can see this task as queued after this.
            task.meta["attempts"] = task.meta.get("attempts", 0) + 1
            task.meta["deliveries"] = 1
            self.nodes[node_id]["current_task"] = task.id
            task.set_status("assigned", assigned_to=node_id)
            return self.public(task)

    def _owned(self, task_id: str, node_id: str) -> Job:
        task = self.get(task_id)
        if task.meta.get("assigned_to") != node_id or task.meta["status"] in TASK_TERMINAL:
            raise PermissionError("task is not assigned to this node")
        return task

    def node_events(self, task_id: str, node_id: str, events: list) -> dict:
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        with self.cond:
            task = self._owned(task_id, node_id)
            if node_id in self.nodes:
                self.nodes[node_id]["last_seen"] = now()
            if task.meta["status"] == "assigned":
                task.set_status("running")
        for ev in events[:500]:
            if not isinstance(ev, dict):
                continue
            kind = ev.get("kind")
            if kind == "token":
                task.append("token", text=str(ev.get("text", ""))[:4000])
            elif kind == "progress":
                frac = ev.get("fraction")
                task.append("progress", fraction=float(frac) if isinstance(frac, (int, float)) else None,
                            message=str(ev.get("message", ""))[:500])
            else:
                task.append("log", stream="node", line=str(ev.get("text", ev))[:2000])
        return {"cancel": bool(task.meta.get("cancel_requested"))}

    def put_artifact(self, task_id: str, node_id: str, name: str, read_body) -> dict:
        """`read_body(limit)` reads the request body; it is called only after the
        caller is known to own the task, so strangers can't make us buffer uploads."""
        task = self._owned(task_id, node_id)
        if not ARTIFACT_NAME.match(name or "") or name.startswith("."):
            raise ValueError("invalid artifact name")
        data = read_body(MAX_ARTIFACT)
        if len(data) > MAX_ARTIFACT:
            raise ValueError("artifact too large")
        d = task.root / "artifacts"
        d.mkdir(exist_ok=True)
        tmp = d / (name + ".part")
        tmp.write_bytes(data)
        tmp.replace(d / name)
        with task.cond:
            if name not in task.meta["artifacts"]:
                task.meta["artifacts"].append(name)
            task.save_meta()
        task.append("artifact", name=name, bytes=len(data))
        return {"name": name, "bytes": len(data)}

    def artifact_path(self, task_id: str, name: str) -> Path:
        task = self.get(task_id)
        if task.meta.get("round_id"):
            # Per-device adapters from federated rounds stay on the coordinator;
            # only the averaged round adapter is published (/fed/rounds/{id}/adapter).
            raise Forbidden("per-device round artifacts are not downloadable")
        if name not in task.meta["artifacts"]:
            raise LookupError("no such artifact")
        return task.root / "artifacts" / name

    def complete(self, task_id: str, node_id: str, body: dict) -> dict:
        status = body.get("status")
        if status not in ("succeeded", "failed"):
            raise ValueError("status must be succeeded or failed")
        metrics = body.get("metrics") if isinstance(body.get("metrics"), dict) else {}
        output = body.get("output")
        with self.cond:
            task = self._owned(task_id, node_id)
            if task.meta.get("cancel_requested"):
                status = "cancelled"
            self._finish(task, status,
                         output=str(output)[:200_000] if output is not None else None,
                         error=str(body["error"])[:4000] if body.get("error") else None,
                         metrics={str(k)[:64]: v for k, v in list(metrics.items())[:50]
                                  if isinstance(v, (int, float, str, bool))
                                  and not (isinstance(v, float) and v != v)})
        return self.public(task)

    def cancel(self, task_id: str) -> dict:
        task = self.get(task_id)
        with self.cond:
            status = task.meta["status"]
            if status in ("queued", "assigned"):
                # Not started yet: the node (if it ever got it) is told via 409 on its first event.
                self._finish(task, "cancelled", reason="cancelled before start")
            elif status not in TASK_TERMINAL:
                with task.cond:
                    task.meta["cancel_requested"] = True
                    task.save_meta()
                task.append("cancel_requested")
        return self.public(task)

    def list_tasks(self, limit: int = 100) -> list[dict]:
        ts = sorted(self.tasks.values(), key=lambda t: t.meta["created_at"], reverse=True)
        return [self.public(t, include_payload=False) for t in ts[:limit]]


# ---------------------------------------------------------- HTTP routes ---

def handle(fleet: Fleet, method: str, parts: list[str], query: dict, body_reader, raw_reader):
    """Route a fleet request. Returns (status, json_body) or (status, Path) for
    file downloads, or None if the path isn't a fleet route."""
    q = {k: v[0] for k, v in query.items()}
    if not parts or parts[0] not in ("nodes", "tasks"):
        return None
    try:
        if parts == ["nodes"] and method == "GET":
            return 200, {"nodes": fleet.nodes_public()}
        if len(parts) == 3 and parts[0] == "nodes":
            if parts[2] == "heartbeat" and method == "POST":
                return 200, fleet.heartbeat(parts[1], body_reader())
            if parts[2] == "next" and method == "GET":
                wait = min(max(float(q.get("wait", "0")), 0.0), 30.0)
                return 200, {"task": fleet.next_task(parts[1], wait)}
        if parts == ["tasks"]:
            if method == "GET":
                return 200, {"tasks": fleet.list_tasks()}
            if method == "POST":
                b = body_reader()
                if b.get("kind") == "train_lora":
                    # Training only through /fed/rounds, which enforces the privacy
                    # opt-ins and keeps per-device adapters private.
                    raise ValueError("train_lora tasks are created by POST /fed/rounds")
                return 201, fleet.public(fleet.create_task(str(b.get("kind", "")), b.get("payload") or {},
                                                           b.get("node") or None))
        if len(parts) >= 2 and parts[0] == "tasks":
            tid = parts[1]
            if len(parts) == 2 and method == "GET":
                return 200, fleet.public(fleet.get(tid))
            sub = parts[2:]
            if sub == ["events"] and method == "GET":
                task = fleet.get(tid)
                after = int(q.get("after", "0"))
                wait = min(float(q.get("wait", "0")), 30.0)
                if wait > 0:
                    task.wait_for_events(after, wait)
                events = task.events_after(after)
                cursor = events[-1]["seq"] if events else after
                return 200, {"events": events, "cursor": cursor, "status": task.meta["status"],
                             "more": cursor < task.next_seq - 1}
            if sub == ["events"] and method == "POST":
                b = body_reader()
                return 200, fleet.node_events(tid, str(b.get("node_id", "")), b.get("events") or [])
            if sub == ["complete"] and method == "POST":
                b = body_reader()
                return 200, fleet.complete(tid, str(b.get("node_id", "")), b)
            if sub == ["cancel"] and method == "POST":
                return 202, fleet.cancel(tid)
            if len(sub) == 2 and sub[0] == "artifacts":
                if method == "PUT":
                    return 201, fleet.put_artifact(tid, q.get("node_id", ""), sub[1], raw_reader)
                if method == "GET":
                    return 200, fleet.artifact_path(tid, sub[1])
    except Forbidden as e:
        return 403, {"error": str(e)}
    except PermissionError as e:
        return 409, {"error": str(e)}
    except LookupError as e:
        return 404, {"error": str(e).strip("'\"")}
    except (ValueError, json.JSONDecodeError) as e:
        return 400, {"error": str(e)}
    return 404, {"error": "not found"}
