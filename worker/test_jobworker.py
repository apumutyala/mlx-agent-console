"""End-to-end tests against a real HTTP server and real subprocesses.

The headline test mirrors the device test: start a run, "force-quit" the
client (drop it mid-stream, keep only its last cursor), reconnect, replay the
missed events with no gaps or duplicates, answer an approval, and cancel —
then prove the grandchild process is gone too.

    python -m unittest -v test_jobworker
"""
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jobworker  # noqa: E402

HERE = Path(__file__).parent


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie counts as dead
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


class Client:
    """Stand-in for the iOS WorkerClient."""

    def __init__(self, base: str, token: str):
        self.base, self.token = base, token

    def call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def events(self, job_id, after, wait=0):
        return self.call("GET", f"/jobs/{job_id}/events?after={after}&wait={wait}")[1]


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.pidfile = Path(self.tmp.name) / "child.pid"
        os.environ["DEMO_DELAY"] = "0.1"
        os.environ["DEMO_PIDFILE"] = str(self.pidfile)
        os.environ.pop("DEMO_NO_APPROVAL", None)
        self.token = secrets.token_urlsafe(32)
        self._start_server()

    def _start_server(self):
        types = jobworker.load_job_types(HERE / "job_types.json")
        self.worker = jobworker.Worker(self.data, types, kill_grace=2.0)
        self.server = jobworker.serve(self.worker, self.token, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}", self.token)

    def tearDown(self):
        for job in self.worker.jobs.values():
            self.worker.cancel(job)
        deadline = time.time() + 15  # let reapers write final status before rmtree
        while time.time() < deadline and any(
                j.proc and j.meta["status"] not in jobworker.TERMINAL for j in self.worker.jobs.values()):
            time.sleep(0.1)
        time.sleep(0.3)
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def wait_status(self, job_id, statuses, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.client.call("GET", f"/jobs/{job_id}")[1]["status"]
            if st in statuses:
                return st
            time.sleep(0.05)
        self.fail(f"job never reached {statuses}, last={st}")

    # ---------------------------------------------------------------
    def test_auth_required(self):
        bad = Client(self.client.base, "wrong")
        self.assertEqual(bad.call("GET", "/jobs")[0], 401)
        self.assertEqual(bad.call("POST", "/jobs", {"type": "demo_agent"})[0], 401)

    def test_rejects_unknown_type_and_bad_params(self):
        self.assertEqual(self.client.call("POST", "/jobs", {"type": "sh"})[0], 400)
        code, body = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "5; rm -rf /"}})
        self.assertEqual(code, 400, body)
        code, _ = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"extra": "x"}})
        self.assertEqual(code, 400)

    def test_force_quit_reconnect_replay_approve_cancel(self):
        code, job = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "400"}})
        self.assertEqual(code, 201, job)
        jid = job["id"]

        # Phone streams a bit, persists its cursor, then gets force-quit.
        seen, cursor = [], 0
        while len(seen) < 5:
            page = self.client.events(jid, cursor, wait=5)
            seen += page["events"]
            cursor = page["cursor"]
        saved_cursor = cursor
        del self.client  # the "app" is gone; the job must not care

        time.sleep(1.0)  # work keeps happening while the phone is away
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}", self.token)

        # Reopen: replay everything after the saved cursor.
        replay, cursor = [], saved_cursor
        while True:
            page = self.client.events(jid, cursor)
            replay += page["events"]
            cursor = page["cursor"]
            if not page["more"]:
                break
        seqs = [e["seq"] for e in seen + replay]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)), "gap or duplicate in replay")
        self.assertTrue(any(e["kind"] == "log" for e in replay), "missed logs not replayed")

        # Approval arrives mid-run; answer it from the "phone".
        self.wait_status(jid, {"waiting_approval"})
        pending = self.client.call("GET", f"/jobs/{jid}")[1]["pending_approvals"]
        self.assertEqual(pending[0]["id"], "apply")
        code, _ = self.client.call("POST", f"/jobs/{jid}/approvals/apply", {"decision": "approve"})
        self.assertEqual(code, 200)
        self.wait_status(jid, {"running"})

        child_pid = int(self.pidfile.read_text())
        self.assertTrue(pid_alive(child_pid))

        # Cancel from the phone: the leader AND its subprocess must die.
        code, _ = self.client.call("POST", f"/jobs/{jid}/cancel")
        self.assertEqual(code, 202)
        self.assertEqual(self.wait_status(jid, jobworker.TERMINAL), "cancelled")
        deadline = time.time() + 10
        while pid_alive(child_pid) and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(pid_alive(child_pid), "grandchild survived cancel")

        kinds = [e["kind"] for e in self.client.events(jid, 0)["events"]]
        self.assertIn("approval_decided", kinds)
        self.assertEqual(kinds[-1], "status")

    def test_success_and_retry(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        _, job = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "3"}})
        self.assertEqual(self.wait_status(job["id"], jobworker.TERMINAL), "succeeded")
        lines = [e["line"] for e in self.client.events(job["id"], 0)["events"] if e["kind"] == "log"]
        self.assertEqual(lines[-1], "done")
        code, retry = self.client.call("POST", f"/jobs/{job['id']}/retry")
        self.assertEqual(code, 201)
        self.assertEqual(retry["retry_of"], job["id"])
        self.assertEqual(self.wait_status(retry["id"], jobworker.TERMINAL), "succeeded")

    def test_long_poll_returns_promptly_on_new_event(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        _, job = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "50"}})
        last = self.client.call("GET", f"/jobs/{job['id']}")[1]["last_seq"]
        t0 = time.time()
        page = self.client.events(job["id"], last, wait=20)
        self.assertTrue(page["events"])
        self.assertLess(time.time() - t0, 5)

    def test_worker_restart_marks_active_jobs_lost_and_keeps_history(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        _, done = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "2"}})
        self.wait_status(done["id"], jobworker.TERMINAL)
        _, live = self.client.call("POST", "/jobs", {"type": "demo_agent", "params": {"steps": "500"}})
        self.wait_status(live["id"], {"running"})
        # Simulate a worker crash: drop the server, reload state from disk.
        old = self.worker
        self.server.shutdown()
        self.server.server_close()
        self._start_server()
        self.assertEqual(self.client.call("GET", f"/jobs/{live['id']}")[1]["status"], "lost")
        self.assertEqual(self.client.call("GET", f"/jobs/{done['id']}")[1]["status"], "succeeded")
        self.assertTrue(self.client.events(done["id"], 0)["events"])
        old.cancel(old.jobs[live["id"]])  # clean up the real process


if __name__ == "__main__":
    unittest.main()
