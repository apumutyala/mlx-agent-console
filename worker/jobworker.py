"""Job worker: owns long-running processes so the phone doesn't have to.

Stdlib only. Run on a Mac or Linux box (Windows works for development).

    WORKER_TOKEN=... python jobworker.py --config job_types.json --data ./data --port 8787

API (all routes require `Authorization: Bearer <WORKER_TOKEN>`):
    GET  /job-types
    POST /jobs                         {"type": "repo_tests", "params": {...}}
    GET  /jobs
    GET  /jobs/{id}
    GET  /jobs/{id}/events?after=<seq>&wait=<seconds>   (long-poll)
    POST /jobs/{id}/cancel
    POST /jobs/{id}/retry              -> new job with the same type/params
    POST /jobs/{id}/approvals/{aid}    {"decision": "approve" | "deny"}
    GET  /models[/<name>[/<file>]]     read-only model files (with --models-dir; see models.py)

Design rules:
  * Jobs are named types from a config file. There is no raw-shell endpoint.
  * Each job has a durable ID, a persisted status file, and an append-only
    events.jsonl. Event `seq` numbers are the reconnect cursor.
  * Each job runs in its own process group (POSIX: setsid; Windows: new
    process group + taskkill /T), so cancel stops grandchildren too.
  * The job's lifetime is not tied to any client connection.
  * Mid-run approvals: a harness prints a line
        ::approval:: {"id": "push", "prompt": "Push branch to origin?"}
    to stdout, then reads one JSON line from stdin:
        {"id": "push", "decision": "approve"}
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from models import parse_range  # single `Range: bytes=a-b` on file responses (resumable downloads)

IS_WINDOWS = os.name == "nt"
APPROVAL_PREFIX = "::approval::"
PARAM_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
TERMINAL = {"succeeded", "failed", "cancelled", "lost", "denied"}


def now() -> float:
    return round(time.time(), 3)


class ConfigError(ValueError):
    pass


class JobType:
    """A named, pre-approved command template.

    Config entry:
      "repo_tests": {
        "description": "Run the repo's test suite",
        "argv": ["python", "-m", "pytest", "{target}"],
        "cwd": "/srv/repo",
        "params": {"target": {"pattern": "^[\\w./-]{1,200}$", "default": "tests"}},
        "requires_approval": false,
        "env": {"PYTHONUNBUFFERED": "1"}
      }
    Params are substituted only into whole argv elements and must match their
    regex, so callers can't inject flags or shell syntax.
    """

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.description = spec.get("description", "")
        self.argv = list(spec["argv"])
        self.cwd = spec.get("cwd")
        self.env = dict(spec.get("env", {}))
        self.requires_approval = bool(spec.get("requires_approval", False))
        self.params = {}
        for pname, pspec in spec.get("params", {}).items():
            if not PARAM_NAME.match(pname):
                raise ConfigError(f"{name}: bad param name {pname!r}")
            self.params[pname] = {
                "pattern": re.compile(pspec["pattern"]),
                "default": pspec.get("default"),
            }

    def build_argv(self, params: dict) -> tuple[list[str], dict]:
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        resolved = {}
        for pname, pspec in self.params.items():
            value = params.get(pname, pspec["default"])
            if value is None:
                raise ValueError(f"missing param: {pname}")
            if not isinstance(value, str) or not pspec["pattern"].fullmatch(value):
                raise ValueError(f"param {pname} fails validation")
            resolved[pname] = value
        argv = []
        for part in self.argv:
            m = re.fullmatch(r"\{([a-z][a-z0-9_]*)\}", part)
            if m:
                argv.append(resolved[m.group(1)])
            elif "{" in part:
                raise ConfigError(f"{self.name}: params must be whole argv elements")
            else:
                argv.append(part)
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        return argv, resolved

    def public(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": {k: {"pattern": v["pattern"].pattern, "default": v["default"]}
                       for k, v in self.params.items()},
            "requires_approval": self.requires_approval,
        }


class Job:
    def __init__(self, root: Path, meta: dict):
        self.root = root
        self.meta = meta
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.proc: subprocess.Popen | None = None
        self.cancel_requested = False
        self.next_seq = 1
        events = root / "events.jsonl"
        if events.exists():
            with events.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.next_seq = json.loads(line)["seq"] + 1

    @property
    def id(self) -> str:
        return self.meta["id"]

    def save_meta(self) -> None:
        tmp = self.root / "job.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "job.json")

    def append(self, kind: str, **data) -> dict:
        with self.cond:
            event = {"seq": self.next_seq, "ts": now(), "kind": kind, **data}
            self.next_seq += 1
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.cond.notify_all()
            return event

    def set_status(self, status: str, **extra) -> None:
        with self.cond:
            self.meta["status"] = status
            self.meta["updated_at"] = now()
            self.meta.update(extra)
            self.save_meta()
        self.append("status", status=status, **extra)

    def events_after(self, after: int, limit: int = 500) -> list[dict]:
        out = []
        path = self.root / "events.jsonl"
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev["seq"] > after:
                    out.append(ev)
                    if len(out) >= limit:
                        break
        return out

    def wait_for_events(self, after: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self.cond:
            while self.next_seq - 1 <= after and self.meta["status"] not in TERMINAL:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.cond.wait(remaining)


class Worker:
    def __init__(self, data_dir: Path, job_types: dict[str, JobType], kill_grace: float = 5.0):
        self.data_dir = data_dir
        self.job_types = job_types
        self.kill_grace = kill_grace
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        (data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self._recover()

    def _recover(self) -> None:
        """Reload persisted jobs. Anything that was mid-run when the worker died
        is marked `lost`: we never pretend to still own a process we can't see."""
        for d in sorted((self.data_dir / "jobs").iterdir()):
            meta_path = d / "job.json"
            if not meta_path.exists():
                continue
            job = Job(d, json.loads(meta_path.read_text(encoding="utf-8")))
            self.jobs[job.id] = job
            if job.meta["status"] in ("running", "waiting_approval", "queued", "starting"):
                job.set_status("lost", reason="worker restarted while job was active")

    # ---- job lifecycle -------------------------------------------------
    def create(self, type_name: str, params: dict, retry_of: str | None = None) -> Job:
        jt = self.job_types.get(type_name)
        if jt is None:
            raise ValueError(f"unknown job type: {type_name}")
        argv, resolved = jt.build_argv(params or {})
        job_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        root = self.data_dir / "jobs" / job_id
        root.mkdir(parents=True)
        meta = {
            "id": job_id, "type": type_name, "params": resolved, "argv": argv,
            "status": "created", "created_at": now(), "updated_at": now(),
            "exit_code": None, "retry_of": retry_of, "pending_approvals": [],
        }
        job = Job(root, meta)
        job.save_meta()
        with self.lock:
            self.jobs[job_id] = job
        job.append("created", type=type_name, params=resolved)
        if jt.requires_approval:
            aid = "start"
            job.meta["pending_approvals"].append({"id": aid, "prompt": f"Start {type_name}?"})
            job.set_status("waiting_approval")
            job.append("approval_requested", approval_id=aid, prompt=f"Start {type_name}?")
        else:
            self._start(job)
        return job

    def _start(self, job: Job) -> None:
        jt = self.job_types[job.meta["type"]]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **jt.env, "JOB_ID": job.id}
        kwargs = dict(
            cwd=jt.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # own session + process group
        try:
            proc = subprocess.Popen(job.meta["argv"], **kwargs)
        except OSError as e:
            job.set_status("failed", reason=f"spawn failed: {e}")
            return
        job.proc = proc
        job.set_status("running", pid=proc.pid)
        pumps = [
            threading.Thread(target=self._pump, args=(job, proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=self._pump, args=(job, proc.stderr, "stderr"), daemon=True),
        ]
        for t in pumps:
            t.start()
        threading.Thread(target=self._reap, args=(job, pumps), daemon=True).start()

    def _pump(self, job: Job, stream, name: str) -> None:
        for line in iter(stream.readline, ""):
            line = line.rstrip("\r\n")
            if name == "stdout" and line.startswith(APPROVAL_PREFIX):
                self._approval_request(job, line[len(APPROVAL_PREFIX):])
            else:
                job.append("log", stream=name, line=line[:8000])
        stream.close()

    def _approval_request(self, job: Job, payload: str) -> None:
        try:
            req = json.loads(payload)
            aid = str(req["id"])[:64]
            prompt = str(req.get("prompt", ""))[:2000]
        except (ValueError, KeyError, TypeError):
            job.append("log", stream="stdout", line=APPROVAL_PREFIX + payload)
            return
        with job.cond:
            job.meta["pending_approvals"].append({"id": aid, "prompt": prompt})
        job.set_status("waiting_approval")
        job.append("approval_requested", approval_id=aid, prompt=prompt)

    def _reap(self, job: Job, pumps: list[threading.Thread]) -> None:
        code = job.proc.wait()
        # Drain trailing output before the terminal event. Bounded, because an
        # orphaned grandchild holding the pipe open would otherwise block us.
        for t in pumps:
            t.join(timeout=2.0)
        try:
            job.proc.stdin.close()
        except OSError:
            pass
        if job.cancel_requested:
            job.set_status("cancelled", exit_code=code)
        else:
            job.set_status("succeeded" if code == 0 else "failed", exit_code=code)

    def decide(self, job: Job, aid: str, decision: str) -> None:
        if decision not in ("approve", "deny"):
            raise ValueError("decision must be approve or deny")
        with job.cond:
            pending = [a for a in job.meta["pending_approvals"] if a["id"] == aid]
            if not pending:
                raise LookupError(f"no pending approval {aid}")
            job.meta["pending_approvals"].remove(pending[0])
            job.save_meta()
        job.append("approval_decided", approval_id=aid, decision=decision)
        if aid == "start" and job.proc is None:
            if decision == "approve":
                self._start(job)
            else:
                job.set_status("denied")
            return
        if job.proc and job.proc.poll() is None:
            try:
                job.proc.stdin.write(json.dumps({"id": aid, "decision": decision}) + "\n")
                job.proc.stdin.flush()
            except OSError:
                pass
            if not job.meta["pending_approvals"]:
                job.set_status("running")

    def cancel(self, job: Job) -> None:
        if job.meta["status"] in TERMINAL:
            return
        job.cancel_requested = True
        job.append("cancel_requested")
        proc = job.proc
        if proc is None:  # never started (awaiting start approval)
            job.set_status("cancelled")
            return
        threading.Thread(target=self._kill_group, args=(proc,), daemon=True).start()

    def _kill_group(self, proc: subprocess.Popen) -> None:
        if IS_WINDOWS:
            # taskkill /T walks the child tree; /F after the grace period.
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T"], capture_output=True)
            try:
                proc.wait(self.kill_grace)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(self.kill_grace)
        except subprocess.TimeoutExpired:
            pass
        try:  # also sweep survivors in the group even if the leader exited
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def retry(self, job: Job) -> Job:
        if job.meta["status"] not in TERMINAL:
            raise ValueError("can only retry a finished job")
        return self.create(job.meta["type"], job.meta["params"], retry_of=job.id)


def make_handler(worker: Worker, token: str, router=None):
    """`router(method, parts, query, read_json, read_raw)` may claim extra
    routes (the compute fleet); it returns None, (status, dict) or (status, Path)."""
    token_b = token.encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "jobworker/1"
        timeout = 90  # idle socket timeout; longer than the 30 s long-poll cap

        def log_message(self, fmt, *args):  # quiet; jobs have their own logs
            if os.environ.get("WORKER_ACCESS_LOG"):
                super().log_message(fmt, *args)

        def _send(self, status: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self) -> bool:
            got = self.headers.get("Authorization", "")
            if got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode(), token_b):
                return True
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n < 0 or n > 1_000_000:
                raise ValueError("bad or too large Content-Length")
            raw = self.rfile.read(n) if n else b"{}"
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            return body

        def _raw(self, limit: int = 64 * 1024 * 1024) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n < 0 or n > limit:
                raise ValueError("bad or too large Content-Length")
            return self.rfile.read(n) if n else b""

        def _send_file(self, path: Path) -> None:
            size = path.stat().st_size
            rng = parse_range(self.headers.get("Range"), size)
            if rng == "invalid":
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            length = end - start + 1 if size else 0
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = length
                while left > 0 and (chunk := f.read(min(1 << 20, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def _route(self, method: str) -> bool:
            if router is None:
                return False
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                out = router(method, parts, parse_qs(url.query), self._body, self._raw)
            except (ValueError, json.JSONDecodeError) as e:
                out = (400, {"error": str(e)})
            if out is None:
                return False
            status, body = out
            if method in ("PUT", "POST") and status >= 400:
                # The request body may be unread (e.g. refused for size); don't reuse this connection.
                self.close_connection = True
            if isinstance(body, Path):
                self._send_file(body)
            else:
                self._send(status, body)
            return True

        def do_PUT(self):
            if not self._auth():
                return
            if not self._route("PUT"):
                self._send(404, {"error": "not found"})

        def _job(self, job_id: str) -> Job | None:
            job = worker.jobs.get(job_id)
            if job is None:
                self._send(HTTPStatus.NOT_FOUND, {"error": "no such job"})
            return job

        def do_GET(self):
            if not self._auth() or self._route("GET"):
                return
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            q = parse_qs(url.query)
            if parts == ["job-types"]:
                return self._send(200, {"job_types": [jt.public() for jt in worker.job_types.values()]})
            if parts == ["jobs"]:
                jobs = sorted((j.meta for j in worker.jobs.values()),
                              key=lambda m: m["created_at"], reverse=True)
                return self._send(200, {"jobs": jobs[:100]})
            if len(parts) == 2 and parts[0] == "jobs":
                job = self._job(parts[1])
                if job:
                    self._send(200, {**job.meta, "last_seq": job.next_seq - 1})
                return
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "events":
                job = self._job(parts[1])
                if not job:
                    return
                try:
                    after = int(q.get("after", ["0"])[0])
                    wait = min(float(q.get("wait", ["0"])[0]), 30.0)
                except ValueError:
                    return self._send(400, {"error": "bad cursor"})
                if wait > 0:
                    job.wait_for_events(after, wait)
                events = job.events_after(after)
                cursor = events[-1]["seq"] if events else after
                return self._send(200, {"events": events, "cursor": cursor,
                                        "status": job.meta["status"],
                                        "more": cursor < job.next_seq - 1})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._auth() or self._route("POST"):
                return
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                body = self._body()
                if parts == ["jobs"]:
                    job = worker.create(str(body.get("type", "")), body.get("params") or {})
                    return self._send(201, job.meta)
                if len(parts) >= 3 and parts[0] == "jobs":
                    job = self._job(parts[1])
                    if not job:
                        return
                    if parts[2:] == ["cancel"]:
                        worker.cancel(job)
                        return self._send(202, job.meta)
                    if parts[2:] == ["retry"]:
                        return self._send(201, worker.retry(job).meta)
                    if len(parts) == 4 and parts[2] == "approvals":
                        worker.decide(job, parts[3], str(body.get("decision", "")))
                        return self._send(200, job.meta)
            except LookupError as e:
                return self._send(404, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return Handler


def load_job_types(path: Path) -> dict[str, JobType]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    types = {name: JobType(name, spec) for name, spec in raw.items()}
    for jt in types.values():  # relative (or missing) cwd = the config's directory
        jt.cwd = str((path.parent / (jt.cwd or ".")).resolve())
    return types


def serve(worker: Worker, token: str, host: str, port: int, router=None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(worker, token, router))
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("job_types.json"))
    ap.add_argument("--data", type=Path, default=Path("./worker-data"))
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; expose via Tailscale/VPN or a TLS proxy, not raw internet")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--models-dir", type=Path, default=os.environ.get("WORKER_MODELS_DIR"),
                    help="serve streamed model builds read-only at GET /models/<name>/<file>")
    args = ap.parse_args()
    token = os.environ.get("WORKER_TOKEN", "")
    if len(token) < 24:
        sys.exit("set WORKER_TOKEN to a random secret of at least 24 chars")
    worker = Worker(args.data, load_job_types(args.config))
    from federated import make_router  # compute fleet + federated rounds
    router, fleet = make_router(args.data)
    if args.models_dir:
        from models import chain, make_models_router
        router = chain(make_models_router(Path(args.models_dir)), router)
    server = serve(worker, token, args.host, args.port, router)
    print(f"jobworker listening on http://{args.host}:{args.port} "
          f"({len(worker.job_types)} job types, {len(worker.jobs)} persisted jobs, "
          f"{len(fleet.nodes)} known nodes)")
    server.serve_forever()


if __name__ == "__main__":
    main()
