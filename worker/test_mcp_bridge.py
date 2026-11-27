"""Tests for the MCP bridge: real worker + real bridge over HTTP.

    python -m unittest -v test_mcp_bridge

If the official MCP Python SDK is importable (e.g. `uv run --with mcp python -m
unittest test_mcp_bridge`), an extra test drives the bridge with the SDK's
Streamable HTTP client to check protocol compatibility.
"""
import asyncio
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import federated  # noqa: E402
import jobworker  # noqa: E402
import mcp_bridge  # noqa: E402

HERE = Path(__file__).parent


def attr(obj, *names):
    """SDK 1.x uses camelCase fields, 2.x snake_case."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(names)

try:
    from mcp import ClientSession
    try:  # SDK 2.x
        from mcp.client.streamable_http import streamable_http_client as http_client
    except ImportError:  # SDK 1.x
        from mcp.client.streamable_http import streamablehttp_client as http_client
    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False


class BridgeTests(unittest.TestCase):
    allow_approve = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DEMO_DELAY"] = "0.1"
        os.environ["DEMO_PIDFILE"] = str(Path(self.tmp.name) / "child.pid")
        os.environ.pop("DEMO_NO_APPROVAL", None)
        token = secrets.token_urlsafe(32)
        self.secret = secrets.token_urlsafe(32)
        self.worker = jobworker.Worker(Path(self.tmp.name) / "data",
                                       jobworker.load_job_types(HERE / "job_types.json"), kill_grace=2.0)
        self.router, self.fleet = federated.make_router(Path(self.tmp.name) / "data")
        self.wserver = jobworker.serve(self.worker, token, "127.0.0.1", 0, self.router)
        self.wtoken = token
        threading.Thread(target=self.wserver.serve_forever, daemon=True).start()
        api = mcp_bridge.WorkerAPI(f"http://127.0.0.1:{self.wserver.server_port}", token)
        self.audit = Path(self.tmp.name) / "audit.jsonl"
        # Tests poll fast; only the rate-limit test runs with production limits.
        fast = self._testMethodName != "test_write_rate_limit_and_audit_log"
        self.guard = mcp_bridge.Guard(self.audit, calls_per_min=6000 if fast else 60)
        self.bserver = mcp_bridge.serve(mcp_bridge.Tools(api, self.allow_approve), self.secret, "127.0.0.1", 0,
                                        enable_oauth=self._testMethodName == "test_oauth_flow_like_claude",
                                        guard=self.guard)
        threading.Thread(target=self.bserver.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.bserver.server_port}"
        self.url = f"{self.base}/mcp/{self.secret}"
        self._id = 0

    def tearDown(self):
        for job in self.worker.jobs.values():
            self.worker.cancel(job)
        deadline = time.time() + 15
        while time.time() < deadline and any(
                j.proc and j.meta["status"] not in jobworker.TERMINAL for j in self.worker.jobs.values()):
            time.sleep(0.1)
        time.sleep(0.3)
        for s in (self.bserver, self.wserver):
            s.shutdown()
            s.server_close()
        self.tmp.cleanup()

    # -- raw JSON-RPC helpers
    def post(self, body, url=None, headers=None):
        req = urllib.request.Request(url or self.url, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None), dict(r.headers)
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, (json.loads(raw) if raw else None), dict(e.headers)

    def rpc(self, method, params=None):
        self._id += 1
        status, body, _ = self.post({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        self.assertEqual(status, 200, body)
        return body

    def tool(self, name, **args):
        res = self.rpc("tools/call", {"name": name, "arguments": args})["result"]
        return res, res.get("structuredContent")

    # -- tests
    def test_auth(self):
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        self.assertEqual(self.post(init, url=f"{self.base}/mcp/wrong")[0], 404)
        self.assertEqual(self.post(init, url=f"{self.base}/mcp")[0], 404)
        ok = self.post(init, url=f"{self.base}/mcp", headers={"Authorization": f"Bearer {self.secret}"})
        self.assertEqual(ok[0], 200)
        self.assertEqual(self.post(init, headers={"Origin": "https://evil.example"})[0], 403)

    def test_initialize_and_list(self):
        status, body, headers = self.post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("Mcp-Session-Id", headers)
        self.assertEqual(self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})[0], 202)
        names = {t["name"] for t in self.rpc("tools/list")["result"]["tools"]}
        self.assertEqual(names, {"list_job_types", "list_jobs", "get_job", "get_job_events",
                                 "start_job", "cancel_job", "retry_job", "decide_approval",
                                 "list_nodes", "run_on_node", "broadcast_prompt", "get_task", "cancel_task",
                                 "start_fed_round", "get_fed_rounds", "list_receipts"})
        self.assertIn("error", self.rpc("nope"))

    def test_full_flow_start_follow_approval_cancel(self):
        res, out = self.tool("start_job", type="demo_agent", params={"steps": "100"})
        self.assertFalse(res["isError"], res)
        jid = out["job"]["id"]

        # page the log with the cursor; no gaps
        cursor, lines = 0, []
        while len(lines) < 5:
            _, page = self.tool("get_job_events", job_id=jid, after=cursor, wait_seconds=5)
            lines += page["lines"]
            cursor = page["next_cursor"]
        self.assertTrue(any(l.startswith("step ") for l in lines))

        # approval shows up; connector can deny but not approve by default
        deadline = time.time() + 40
        while time.time() < deadline:
            _, out = self.tool("get_job", job_id=jid, tail=5)
            if out["job"]["pending_approvals"]:
                break
            time.sleep(0.1)
        self.assertEqual(out["job"]["pending_approvals"][0]["id"], "apply")
        res, _ = self.tool("decide_approval", job_id=jid, approval_id="apply", decision="approve")
        self.assertTrue(res["isError"])  # schema enum + server check both refuse
        res, _ = self.tool("decide_approval", job_id=jid, approval_id="apply", decision="deny")
        self.assertFalse(res["isError"], res)

        res, out = self.tool("cancel_job", job_id=jid)
        self.assertFalse(res["isError"])
        deadline = time.time() + 15
        while time.time() < deadline:
            _, out = self.tool("get_job", job_id=jid, tail=0)
            if out["job"]["status"] in jobworker.TERMINAL:
                break
            time.sleep(0.1)
        self.assertEqual(out["job"]["status"], "cancelled")

    def test_validation_errors_are_tool_errors(self):
        res, _ = self.tool("start_job", type="demo_agent", params={"steps": "1; rm -rf /"})
        self.assertTrue(res["isError"])
        self.assertIn("fails validation", res["content"][0]["text"])
        res, _ = self.tool("start_job", type="sh")
        self.assertTrue(res["isError"])
        res, _ = self.tool("get_job", job_id="../../etc/passwd")
        self.assertTrue(res["isError"])
        res, _ = self.tool("list_jobs", limit=5, sneaky=True)
        self.assertTrue(res["isError"])
        bad = self.rpc("tools/call", {"name": "run_shell", "arguments": {}})
        self.assertIn("error", bad)

    def test_log_paging_respects_budget_and_resumes(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        _, out = self.tool("start_job", type="demo_agent", params={"steps": "5"})
        jid = out["job"]["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            _, o = self.tool("get_job", job_id=jid, tail=0)
            if o["job"]["status"] in jobworker.TERMINAL:
                break
            time.sleep(0.1)
        old = mcp_bridge.MAX_TEXT
        mcp_bridge.MAX_TEXT = 40  # force several pages
        try:
            cursor, all_lines, pages = 0, [], 0
            while True:
                _, page = self.tool("get_job_events", job_id=jid, after=cursor)
                all_lines += page["lines"]
                cursor = page["next_cursor"]
                pages += 1
                if not page["more"]:
                    break
            self.assertGreater(pages, 2)
        finally:
            mcp_bridge.MAX_TEXT = old
        self.assertIn("done", all_lines)
        self.assertEqual(len(all_lines), self.worker.jobs[jid].next_seq - 1)

    def _get(self, path, follow=False):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        opener = urllib.request.build_opener(NoRedirect)
        try:
            r = opener.open(self.base + path, timeout=10)
            return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _form(self, path, data):
        req = urllib.request.Request(self.base + path, data=urllib.parse.urlencode(data).encode(), method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_oauth_off_by_default(self):
        for path in ("/.well-known/oauth-authorization-server", "/authorize",
                     f"/.well-known/oauth-protected-resource/mcp/{self.secret}"):
            self.assertEqual(self._get(path)[0], 404, path)
        self.assertEqual(self.post({"redirect_uris": ["https://claude.ai/cb"]}, url=self.base + "/register")[0], 404)

    def test_write_rate_limit_and_audit_log(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        results = [self.tool("start_job", type="demo_agent", params={"steps": "1"})[0] for _ in range(7)]
        errors = [r for r in results if r["isError"]]
        self.assertTrue(errors, "write burst should be rate limited")
        self.assertIn("rate limited", errors[0]["content"][0]["text"])
        self.tool("list_jobs")  # reads still allowed
        rows = [json.loads(l) for l in self.audit.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["tool"], "start_job")
        self.assertTrue(any(not r["ok"] and "rate limited" in r["detail"] for r in rows))
        self.assertNotIn(self.secret, self.audit.read_text(encoding="utf-8"))

    def test_log_output_marked_untrusted(self):
        os.environ["DEMO_NO_APPROVAL"] = "1"
        _, out = self.tool("start_job", type="demo_agent", params={"steps": "2"})
        _, page = self.tool("get_job_events", job_id=out["job"]["id"], after=0, wait_seconds=3)
        self.assertIn("not instructions", page["log_note"])
        _, job = self.tool("get_job", job_id=out["job"]["id"])
        self.assertIn("not instructions", job["log_note"])

    def _fake_node(self, nid, reply, stop):
        """A simulated iPad: heartbeat, long-poll, stream tokens, complete."""
        api = mcp_bridge.WorkerAPI(f"http://127.0.0.1:{self.wserver.server_port}", self.wtoken)
        api.call("POST", f"/nodes/{nid}/heartbeat", {"name": nid, "kind": "ipad",
                                                   "capabilities": {"kinds": ["generate"]}})
        while not stop.is_set():
            api.call("POST", f"/nodes/{nid}/heartbeat", {"name": nid, "kind": "ipad",
                                                       "capabilities": {"kinds": ["generate"]}})
            task = api.call("GET", f"/nodes/{nid}/next?wait=1")["task"]
            if not task:
                continue
            text = reply + ": " + task["payload"]["prompt"]
            for i in range(0, len(text), 4):
                api.call("POST", f"/tasks/{task['id']}/events",
                         {"node_id": nid, "events": [{"kind": "token", "text": text[i:i + 4]}]})
            api.call("POST", f"/tasks/{task['id']}/complete",
                     {"node_id": nid, "status": "succeeded", "output": text, "metrics": {"tokens": 3}})

    def test_fleet_tools(self):
        stop = threading.Event()
        nodes = [threading.Thread(target=self._fake_node, args=(n, n.upper(), stop), daemon=True)
                 for n in ("ipad-1", "mac-01")]
        for th in nodes:
            th.start()
        try:
            deadline = time.time() + 10
            while time.time() < deadline:
                _, listed = self.tool("list_nodes")
                if len(listed["nodes"]) == 2:
                    break
                time.sleep(0.1)
            self.assertTrue(all(n["online"] for n in listed["nodes"]))
            _, out = self.tool("run_on_node", prompt="hello", node="ipad-1", wait_seconds=20)
            self.assertEqual(out["status"], "succeeded", out)
            self.assertEqual(out["output"], "IPAD-1: hello")
            self.assertIn("note", out)
            _, b = self.tool("broadcast_prompt", prompt="sum", wait_seconds=20)
            self.assertEqual(sorted(r["output"] for r in b["results"]), ["IPAD-1: sum", "MAC-01: sum"])
            _, t = self.tool("get_task", task_id=out["id"])
            self.assertEqual(t["status"], "succeeded")
            res, _ = self.tool("run_on_node", prompt="x", node="../etc")
            self.assertTrue(res["isError"])
            res, _ = self.tool("start_fed_round", base_model="LiquidAI/LFM3-1.7B-MLX-4bit")
            self.assertTrue(res["isError"])  # no train_lora-capable node
            _, rounds = self.tool("get_fed_rounds")
            self.assertEqual(rounds["rounds"], [])
        finally:
            stop.set()
            for th in nodes:
                th.join(5)

    def test_list_receipts_tool(self):
        _, out = self.tool("list_receipts")
        self.assertEqual(out["receipts"], [])
        for mid in ("lfm3-1.7b", "qwen3-1.7b"):
            self.router.receipts.save(json.dumps({
                "schema": 1, "model": {"id": mid, "repo": "r/" + mid},
                "workloads": [{"id": "short-128", "median": {"decode_tps": 50.0}, "output_hash": "ab"}]}).encode())
        _, out = self.tool("list_receipts", model="qwen3-1.7b")
        self.assertEqual([r["model"]["id"] for r in out["receipts"]], ["qwen3-1.7b"])
        self.assertEqual(out["receipts"][0]["workloads"][0]["decode_tps"], 50.0)
        self.assertIn("note", out)
        _, full = self.tool("list_receipts", receipt=out["receipts"][0]["name"])
        self.assertEqual(full["receipt"]["model"]["repo"], "r/qwen3-1.7b")
        res, _ = self.tool("list_receipts", receipt="../../etc/passwd")
        self.assertTrue(res["isError"])
        res, _ = self.tool("list_receipts", limit=0)
        self.assertTrue(res["isError"])
        # read-only: not rate limited as a write
        self.assertNotIn("list_receipts", mcp_bridge.WRITE_TOOLS)

    def test_connection_cap(self):
        import socket
        socks = []
        try:
            for _ in range(40):  # idle connections hold slots until the 30 s socket timeout
                s = socket.create_connection(("127.0.0.1", self.bserver.server_port))
                socks.append(s)
            time.sleep(0.3)
            s = socket.create_connection(("127.0.0.1", self.bserver.server_port))
            s.settimeout(5)
            self.assertIn(b"503", s.recv(100))
            s.close()
        finally:
            for s in socks:
                s.close()

    def test_oauth_flow_like_claude(self):
        import base64, hashlib
        # discovery: path-suffixed resource metadata needs the secret; root AS metadata is public
        self.assertEqual(self._get("/.well-known/oauth-protected-resource/mcp/wrong")[0], 404)
        code, _, body = self._get(f"/.well-known/oauth-protected-resource/mcp/{self.secret}")
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)["resource"].endswith(f"/mcp/{self.secret}"))
        code, _, body = self._get("/.well-known/oauth-authorization-server")
        meta = json.loads(body)
        self.assertNotIn(self.secret, body.decode())
        # DCR: only Claude callback URLs
        cb = "https://claude.ai/api/mcp/auth_callback"
        req = lambda uris: self.post({"redirect_uris": uris, "client_name": "Claude"}, url=self.base + "/register")
        self.assertEqual(req(["https://evil.example/cb"])[0], 400)
        code, reg, _ = req([cb])
        self.assertEqual(code, 201)
        cid = reg["client_id"]
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        q = {"response_type": "code", "client_id": cid, "redirect_uri": cb, "code_challenge": challenge,
             "code_challenge_method": "S256", "state": "xyz"}
        # without the secret in `resource`: a form, no code
        code, headers, body = self._get("/authorize?" + urllib.parse.urlencode({**q, "resource": self.base + "/mcp/wrong"}))
        self.assertEqual(code, 200)
        self.assertIn(b"connector secret", body)
        # with it: immediate redirect back to Claude with a code
        q["resource"] = f"{self.base}/mcp/{self.secret}"
        code, headers, _ = self._get("/authorize?" + urllib.parse.urlencode(q))
        self.assertEqual(code, 302)
        loc = urllib.parse.urlparse(headers["Location"])
        self.assertEqual(f"{loc.scheme}://{loc.netloc}{loc.path}", cb)
        params = dict(urllib.parse.parse_qsl(loc.query))
        self.assertEqual(params["state"], "xyz")
        # token: PKCE enforced, code single-use
        bad = {"grant_type": "authorization_code", "code": params["code"], "client_id": cid,
               "redirect_uri": cb, "code_verifier": "wrong"}
        self.assertEqual(self._form("/token", bad)[0], 400)
        code, _, _ = self._get("/authorize?" + urllib.parse.urlencode(q))  # fresh code after burn
        loc = urllib.parse.urlparse(self._get("/authorize?" + urllib.parse.urlencode(q))[1]["Location"])
        good = {**bad, "code": dict(urllib.parse.parse_qsl(loc.query))["code"], "code_verifier": verifier}
        code, tok = self._form("/token", good)
        self.assertEqual(code, 200, tok)
        self.assertEqual(self._form("/token", good)[0], 400)  # replay refused
        # token works on the bare /mcp endpoint and with the secret URL
        init = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        self.assertEqual(self.post(init, url=self.base + "/mcp",
                                   headers={"Authorization": "Bearer " + tok["access_token"]})[0], 200)
        self.assertEqual(self.post(init, headers={"Authorization": "Bearer " + tok["access_token"]})[0], 200)
        self.assertEqual(self.post(init, url=self.base + "/mcp", headers={"Authorization": "Bearer nope"})[0], 404)
        # refresh rotates
        code, tok2 = self._form("/token", {"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                                           "client_id": cid})
        self.assertEqual(code, 200)
        self.assertEqual(self._form("/token", {"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                                               "client_id": cid})[0], 400)
        # typed-secret fallback form
        q2 = {k: v for k, v in q.items() if k != "resource"}
        req = urllib.request.Request(self.base + "/authorize", method="POST",
                                     data=urllib.parse.urlencode({**q2, "secret": self.secret}).encode())
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        try:
            urllib.request.build_opener(NoRedirect).open(req, timeout=10)
            self.fail("expected redirect")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 302)
            self.assertIn("code=", e.headers["Location"])

    @unittest.skipUnless(HAVE_SDK, "official MCP SDK not installed")
    def test_official_sdk_client(self):
        async def run():
            async with http_client(self.url) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as s:
                    init = await s.initialize()
                    self.assertEqual(attr(attr(init, "server_info", "serverInfo"), "name"), "job-worker")
                    tools = await s.list_tools()
                    self.assertIn("start_job", [t.name for t in tools.tools])
                    r = await s.call_tool("list_job_types", {})
                    self.assertFalse(attr(r, "is_error", "isError"))
                    self.assertIn("demo_agent", r.content[0].text)
                    r = await s.call_tool("start_job", {"type": "demo_agent", "params": {"steps": "3"}})
                    self.assertFalse(attr(r, "is_error", "isError"), r)
                    jid = attr(r, "structured_content", "structuredContent")["job"]["id"]
                    r = await s.call_tool("get_job_events", {"job_id": jid, "after": 0, "wait_seconds": 3})
                    self.assertFalse(attr(r, "is_error", "isError"), r)
                    await s.send_ping()
        asyncio.run(run())


class ApproveEnabledTests(BridgeTests):
    allow_approve = True

    def test_full_flow_start_follow_approval_cancel(self):
        _, out = self.tool("start_job", type="demo_agent", params={"steps": "40"})
        jid = out["job"]["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            _, out = self.tool("get_job", job_id=jid, tail=0)
            if out["job"]["pending_approvals"]:
                break
            time.sleep(0.1)
        res, _ = self.tool("decide_approval", job_id=jid, approval_id="apply", decision="approve")
        self.assertFalse(res["isError"], res)


if __name__ == "__main__":
    unittest.main()
