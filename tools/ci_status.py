"""One shared, lightweight CI status poller for every agent working on this repo.

Why: GitHub's API allows 5,000 requests/hour per user, SHARED by every agent and
tool. Each `gh run watch` polls every 3 s, so six agents watching builds burn
the whole budget in minutes and everyone goes blind. Instead, exactly ONE
poller asks GitHub (one `gh run list` call every 2 minutes, ~30 calls/hour) and
writes the result to a local file. Agents wait on that file, which costs zero
API calls.

    # the coordinator starts it once (idempotent: exits if one is already running)
    python tools/ci_status.py poll &

    # agents: block until a run finishes (no GitHub calls; reads the local file)
    python tools/ci_status.py wait --run 45956596794            # exit 0 success, 1 failure
    python tools/ci_status.py wait --branch fix-node --after 2026-11-27T07:30:00Z

    # one-off look at the latest status (no GitHub calls)
    python tools/ci_status.py show [--branch fix-node]

Rules: never run `gh run watch`, never loop over `gh run view/list`. To get logs
of a FAILED run, call `gh run view <id> --log-failed` once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = "apumutyala/mlx-agent-console"
STATE_DIR = Path.home() / ".ci-status"
STATUS = STATE_DIR / "mlx-agent-console.json"
LOCK = STATE_DIR / "poller.pid"
POLL_SECONDS = 120          # one API call per interval
LOW_BUDGET = 500            # back off when the shared budget gets low


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gh_json(args: list[str]):
    out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300])
    return json.loads(out.stdout or "null")


def _remaining() -> tuple[int, int]:
    # GET /rate_limit does not count against the core budget.
    core = _gh_json(["api", "rate_limit", "--jq", ".resources.core"])
    return int(core["remaining"]), int(core["reset"])


def _write(doc: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, STATUS)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def poll() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            if _alive(int(LOCK.read_text().strip())):
                print("poller already running; nothing to do")
                return
        except ValueError:
            pass
    LOCK.write_text(str(os.getpid()))
    try:
        while True:
            sleep = POLL_SECONDS
            try:
                remaining, reset = _remaining()
                if remaining < LOW_BUDGET:
                    sleep = max(POLL_SECONDS, reset - int(time.time()) + 5)
                    _write({**_read(), "budget": {"remaining": remaining, "reset": reset},
                            "note": f"budget low; paused until reset", "updated_at": _now()})
                else:
                    runs = _gh_json(["run", "list", "-R", REPO, "-L", "40", "--json",
                                     "databaseId,headBranch,headSha,status,conclusion,createdAt,updatedAt,url,event"])
                    _write({"updated_at": _now(), "poll_seconds": POLL_SECONDS,
                            "budget": {"remaining": remaining - 1, "reset": reset}, "runs": runs})
            except Exception as e:  # keep the last good data; record the error
                _write({**_read(), "error": str(e), "updated_at": _now()})
            time.sleep(sleep)
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass


def _read() -> dict:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _match(runs, run_id=None, branch=None, after=None):
    for r in runs:  # newest first
        if run_id is not None and r["databaseId"] != run_id:
            continue
        if branch is not None and r["headBranch"] != branch:
            continue
        if after is not None and r["createdAt"] < after:
            continue
        return r
    return None


def wait(run_id=None, branch=None, after=None, timeout=7200) -> int:
    deadline = time.time() + timeout
    seen = None
    while time.time() < deadline:
        doc = _read()
        if not doc:
            print("no status file yet; is the poller running? (python tools/ci_status.py poll)", file=sys.stderr)
        r = _match(doc.get("runs", []), run_id, branch, after)
        if r and r["status"] == "completed":
            print(json.dumps(r))
            return 0 if r["conclusion"] == "success" else 1
        if r and r["status"] != seen:
            seen = r["status"]
            print(f"{r['databaseId']} {r['headBranch']}: {seen} (as of {doc.get('updated_at')})", flush=True)
        time.sleep(20)  # local file read only
    print("timed out waiting", file=sys.stderr)
    return 2


def show(branch=None) -> None:
    doc = _read()
    print(f"updated {doc.get('updated_at')}  budget {doc.get('budget')}  {doc.get('error', '')}")
    for r in doc.get("runs", [])[:15]:
        if branch and r["headBranch"] != branch:
            continue
        print(f"{r['databaseId']}  {r['headBranch']:<18} {r['status']:<11} {r.get('conclusion') or '':<10} {r['createdAt']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("poll")
    w = sub.add_parser("wait")
    w.add_argument("--run", type=int)
    w.add_argument("--branch")
    w.add_argument("--after", help="ignore runs created before this ISO time (e.g. when you pushed)")
    w.add_argument("--timeout", type=int, default=7200)
    s = sub.add_parser("show")
    s.add_argument("--branch")
    a = ap.parse_args()
    if a.cmd == "poll":
        poll()
    elif a.cmd == "wait":
        sys.exit(wait(a.run, a.branch, a.after, a.timeout))
    else:
        show(a.branch)


if __name__ == "__main__":
    main()
