"""Fleet + federated rounds, end to end over HTTP with simulated nodes.

    python -m unittest -v test_fleet
"""
import json
import secrets
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import federated  # noqa: E402
import jobworker  # noqa: E402


class Client:
    def __init__(self, base, token):
        self.base, self.token = base, token

    def call(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                out = r.read()
                return r.status, (json.loads(out) if r.headers.get("Content-Type") == "application/json" else out)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


def adapter_bytes(values: dict[str, list[float]], dtype="F32") -> bytes:
    header, blobs, off = {}, [], 0
    for k, v in values.items():
        if dtype == "F32":
            b = array("f", v).tobytes()
        else:  # BF16: top half of the float32 bits
            b = array("H", (x >> 16 for x in array("I", array("f", v).tobytes()))).tobytes()
        header[k] = {"dtype": dtype, "shape": [len(v)], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.token = secrets.token_urlsafe(24)
        worker = jobworker.Worker(Path(self.tmp.name), {})
        self.router, self.fleet = federated.make_router(Path(self.tmp.name), offline_after=1.5)
        self.server = jobworker.serve(worker, self.token, "127.0.0.1", 0, self.router)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.c = Client(f"http://127.0.0.1:{self.server.server_address[1]}", self.token)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def node(self, nid, kinds=("generate",)):
        st, n = self.c.call("POST", f"/nodes/{nid}/heartbeat",
                            {"name": nid, "kind": "ipad", "capabilities": {"kinds": list(kinds), "model": "m"}})
        self.assertEqual(st, 200, n)
        return n

    def test_auth_required(self):
        bad = Client(self.c.base, "nope")
        self.assertEqual(bad.call("GET", "/nodes")[0], 401)

    def test_generate_roundtrip_with_streaming(self):
        self.node("ipad-1")
        st, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "hi"}})
        self.assertEqual(st, 201, t)
        st, nxt = self.c.call("GET", "/nodes/ipad-1/next?wait=2")
        self.assertEqual(nxt["task"]["id"], t["id"])
        self.assertEqual(nxt["task"]["payload"]["prompt"], "hi")
        st, r = self.c.call("POST", f"/tasks/{t['id']}/events",
                            {"node_id": "ipad-1", "events": [{"kind": "token", "text": "Hel"},
                                                              {"kind": "token", "text": "lo"}]})
        self.assertEqual(r, {"cancel": False})
        st, done = self.c.call("POST", f"/tasks/{t['id']}/complete",
                               {"node_id": "ipad-1", "status": "succeeded", "output": "Hello",
                                "metrics": {"tokens_per_second": 41.5}})
        self.assertEqual(done["status"], "succeeded")
        st, page = self.c.call("GET", f"/tasks/{t['id']}/events?after=0")
        toks = "".join(e["text"] for e in page["events"] if e["kind"] == "token")
        self.assertEqual(toks, "Hello")
        st, nodes = self.c.call("GET", "/nodes")
        self.assertTrue(nodes["nodes"][0]["online"])
        self.assertIsNone(nodes["nodes"][0]["current_task"])

    def test_wrong_node_cannot_report(self):
        self.node("ipad-1"), self.node("ipad-2")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}, "node": "ipad-1"})
        self.assertIsNone(self.c.call("GET", "/nodes/ipad-2/next?wait=0.2")[1]["task"])  # targeted elsewhere
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        st, _ = self.c.call("POST", f"/tasks/{t['id']}/complete", {"node_id": "ipad-2", "status": "succeeded"})
        self.assertEqual(st, 409)

    def test_long_poll_wakes_on_new_task(self):
        self.node("ipad-1")
        got = {}
        th = threading.Thread(target=lambda: got.update(r=self.c.call("GET", "/nodes/ipad-1/next?wait=10")))
        th.start()
        time.sleep(0.3)
        t0 = time.time()
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        th.join(10)
        self.assertEqual(got["r"][1]["task"]["id"], t["id"])
        self.assertLess(time.time() - t0, 6)

    def test_cancel_running_task(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        self.c.call("POST", f"/tasks/{t['id']}/events", {"node_id": "ipad-1", "events": []})  # running
        self.assertEqual(self.c.call("POST", f"/tasks/{t['id']}/cancel")[0], 202)
        _, r = self.c.call("POST", f"/tasks/{t['id']}/events", {"node_id": "ipad-1", "events": []})
        self.assertTrue(r["cancel"])
        _, done = self.c.call("POST", f"/tasks/{t['id']}/complete", {"node_id": "ipad-1", "status": "failed"})
        self.assertEqual(done["status"], "cancelled")

    def test_offline_node_task_is_requeued_then_lost(self):
        self.node("ipad-1"), self.node("ipad-2")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        time.sleep(2.2)  # ipad-1 vanishes (offline_after=1.5)
        self.node("ipad-2")
        st, nxt = self.c.call("GET", "/nodes/ipad-2/next?wait=2")
        self.assertEqual(nxt["task"]["id"], t["id"])  # requeued to a live node
        time.sleep(2.2)
        self.c.call("GET", "/nodes")  # triggers sweep
        _, final = self.c.call("GET", f"/tasks/{t['id']}")
        self.assertEqual(final["status"], "lost")  # max_attempts=2

    def test_restart_keeps_inflight_with_owner(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        fleet2 = federated.Fleet(Path(self.tmp.name))   # worker restart
        self.assertEqual(fleet2.tasks[t["id"]].meta["status"], "assigned")
        self.assertEqual(fleet2.next_task("ipad-1", 0)["id"], t["id"])  # owner gets it back

    def test_restart_with_exhausted_inflight_task_does_not_crash(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        task = self.fleet.tasks[t["id"]]
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        task.meta["attempts"] = 99
        task.save_meta()
        fleet2 = federated.Fleet(Path(self.tmp.name), offline_after=0.01)
        time.sleep(0.05)
        fleet2.sweep()   # owner offline -> attempts exhausted -> lost, with listeners present
        self.assertEqual(fleet2.tasks[t["id"]].meta["status"], "lost")

    def test_lost_claim_response_is_redelivered(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.fleet.next_task("ipad-1", 0)            # claimed; response "lost"
        st, nxt = self.c.call("GET", "/nodes/ipad-1/next?wait=0")
        self.assertEqual(nxt["task"]["id"], t["id"])
        self.assertEqual(self.fleet.tasks[t["id"]].meta["attempts"], 1)

    def test_unstartable_task_is_not_redelivered_forever(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        for _ in range(12):   # node keeps polling but never starts it
            self.fleet.next_task("ipad-1", 0)
        self.assertEqual(self.fleet.tasks[t["id"]].meta["status"], "lost")

    def test_train_lora_only_via_rounds(self):
        self.node("ipad-1", kinds=("train_lora",))
        st, _ = self.c.call("POST", "/tasks", {"kind": "train_lora", "payload": {"use_local_data": True}})
        self.assertEqual(st, 400)

    def test_poll_while_owning_running_task_requeues_it(self):
        self.node("ipad-1"), self.node("ipad-2")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        self.c.call("POST", f"/tasks/{t['id']}/events", {"node_id": "ipad-1", "events": []})  # running
        # The app paused/restarted and polled again: the abandoned task is requeued
        # (burning an attempt) and handed out fresh, here to the same idle node.
        _, again = self.c.call("GET", "/nodes/ipad-1/next?wait=0")
        self.assertEqual(again["task"]["id"], t["id"])
        self.assertEqual(again["task"]["status"], "assigned")
        self.assertEqual(again["task"]["attempts"], 2)

    def test_cancel_assigned_task_is_immediate(self):
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        self.c.call("GET", "/nodes/ipad-1/next?wait=1")
        _, c = self.c.call("POST", f"/tasks/{t['id']}/cancel")
        self.assertEqual(c["status"], "cancelled")
        st, _ = self.c.call("POST", f"/tasks/{t['id']}/events", {"node_id": "ipad-1", "events": []})
        self.assertEqual(st, 409)   # the node learns to stop

    def test_concurrent_polls_never_double_claim(self):
        nodes = [f"ipad-{i}" for i in range(6)]
        for n in nodes:
            self.node(n)
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}})
        got = []
        ths = [threading.Thread(target=lambda n=n: got.append(self.fleet.next_task(n, 0.5))) for n in nodes]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        self.assertEqual(sum(1 for g in got if g and g["id"] == t["id"]), 1)

    def test_pinned_task_expires_when_target_stays_offline(self):
        self.fleet.queue_deadline = 0.5
        self.node("ipad-1")
        _, t = self.c.call("POST", "/tasks", {"kind": "generate", "payload": {"prompt": "x"}, "node": "ipad-1"})
        time.sleep(2.2)          # offline_after=1.5, then past the deadline
        self.fleet.sweep()
        self.assertEqual(self.fleet.tasks[t["id"]].meta["status"], "lost")

    def test_negative_content_length_rejected(self):
        import http.client
        host, port = self.server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.putrequest("POST", "/tasks")
        conn.putheader("Authorization", f"Bearer {self.token}")
        conn.putheader("Content-Length", "-5")
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()

    def test_validation(self):
        self.assertEqual(self.c.call("POST", "/tasks", {"kind": "shell", "payload": {}})[0], 400)
        self.assertEqual(self.c.call("POST", "/tasks", {"kind": "generate", "payload": {}})[0], 400)
        self.assertEqual(self.c.call("POST", "/nodes/x/heartbeat", {})[0], 400)
        self.assertEqual(self.c.call("GET", "/tasks/../../etc")[0], 404)

    def test_federated_round_averages_adapters(self):
        a = {"layers.0.lora_a": [1.0, 2.0], "layers.0.lora_b": [0.0, 4.0]}
        b = {"layers.0.lora_a": [3.0, 6.0], "layers.0.lora_b": [4.0, 0.0]}
        for n in ("ipad-1", "mac-1"):
            self.node(n, kinds=("generate", "train_lora"))
        st, r = self.c.call("POST", "/fed/rounds", {"base_model": "mlx-community/LFM2-1.2B-4bit",
                                                    "iterations": 5,
                                                    "examples": [{"text": str(i)} for i in range(4)]})
        self.assertEqual(st, 201, r)
        self.assertEqual(len(r["tasks"]), 2)
        for nid, vals, n_ex, dtype in (("ipad-1", a, 1, "F32"), ("mac-1", b, 3, "BF16")):
            _, nxt = self.c.call("GET", f"/nodes/{nid}/next?wait=2")
            task = nxt["task"]
            self.assertEqual(task["kind"], "train_lora")
            self.assertEqual(len(task["payload"]["examples"]), 2)
            st, _ = self.c.call("PUT", f"/tasks/{task['id']}/artifacts/adapter.safetensors?node_id={nid}",
                                raw=adapter_bytes(vals, dtype))
            self.assertEqual(st, 201)
            self.c.call("POST", f"/tasks/{task['id']}/complete",
                        {"node_id": nid, "status": "succeeded", "metrics": {"num_examples": n_ex}})
        done = self.wait_round(r["id"])
        self.assertEqual(done["status"], "aggregated", done)
        # per-device adapters are never downloadable; only the average is
        st, _ = self.c.call("GET", f"/tasks/{next(iter(r['tasks']))}/artifacts/adapter.safetensors")
        self.assertEqual(st, 403)
        st, blob = self.c.call("GET", f"/fed/rounds/{r['id']}/adapter")
        path = Path(self.tmp.name) / "avg.safetensors"
        path.write_bytes(blob)
        tensors, meta = federated.read_safetensors(path)
        # weights 1:3 -> 0.25*a + 0.75*b
        self.assertEqual(list(tensors["layers.0.lora_a"][1]), [2.5, 5.0])
        self.assertEqual(list(tensors["layers.0.lora_b"][1]), [3.0, 1.0])
        self.assertEqual(meta["contributors"], "2")
        # next round continues from it
        st, bad = self.c.call("POST", "/fed/rounds", {"base_model": "mlx-community/LFM2-1.2B-4bit",
                                                      "nodes": ["ipad-1"], "parent_round": "latest", "rank": 16,
                                                      "use_local_data": True})
        self.assertEqual(st, 400, bad)   # rank must match the parent's adapter
        st, r2 = self.c.call("POST", "/fed/rounds", {"base_model": "MLX-community/lfm2-1.2B-4bit",
                                                     "nodes": ["ipad-1"], "parent_round": "latest",
                                                     "use_local_data": True})
        self.assertEqual(st, 201, r2)
        _, nxt = self.c.call("GET", "/nodes/ipad-1/next?wait=2")
        self.assertEqual(nxt["task"]["payload"]["init_adapter"], f"fed/rounds/{r['id']}/adapter")
        self.assertTrue(nxt["task"]["payload"]["use_local_data"])

    def wait_round(self, rid, timeout=10):
        deadline = time.time() + timeout
        while True:
            _, r = self.c.call("GET", f"/fed/rounds/{rid}")
            if r["status"] != "running" or time.time() > deadline:
                return r
            time.sleep(0.05)

    def test_round_fails_cleanly_without_adapters(self):
        self.node("ipad-1", kinds=("train_lora",))
        _, r = self.c.call("POST", "/fed/rounds", {"base_model": "m/x-1",
                                                   "examples": [{"text": "a"}, {"text": "b"}]})
        _, nxt = self.c.call("GET", "/nodes/ipad-1/next?wait=2")
        self.c.call("POST", f"/tasks/{nxt['task']['id']}/complete",
                    {"node_id": "ipad-1", "status": "failed", "error": "oom"})
        self.assertEqual(self.wait_round(r["id"])["status"], "failed")

    def test_round_validation_happens_before_dispatch(self):
        self.node("ipad-1", kinds=("train_lora",))
        self.node("ipad-9", kinds=("generate",))
        base = {"base_model": "m/x-1", "examples": [{"text": "a"}, {"text": "b"}]}
        cases = [
            {**base, "nodes": ["ipad-9"]},                                   # can't train
            {"base_model": "m/x-1"},                                         # no data, no local opt-in
            {**base, "examples": [{"text": "x" * 8001}, {"text": "b"}]},     # example too long
            {**base, "examples": [{"text": "a", "evil": "b"}, {"text": "c"}]},
            {**base, "examples": [{"text": "y" * 7000} for _ in range(120)]},  # payload too large
        ]
        for spec in cases:
            st, _ = self.c.call("POST", "/fed/rounds", spec)
            self.assertEqual(st, 400, spec.get("nodes") or list(spec))
        self.assertEqual(self.fleet.tasks, {})   # nothing was dispatched
        _, rounds = self.c.call("GET", "/fed/rounds")
        self.assertEqual(rounds["rounds"], [])

    def test_round_deadline_aggregates_what_succeeded(self):
        a = {"w": [2.0]}
        for n in ("ipad-1", "ipad-2"):
            self.node(n, kinds=("train_lora",))
        _, r = self.c.call("POST", "/fed/rounds", {"base_model": "m/x-1", "deadline_seconds": 60,
                                                   "examples": [{"text": str(i)} for i in range(4)]})
        _, nxt = self.c.call("GET", "/nodes/ipad-1/next?wait=2")
        tid = nxt["task"]["id"]
        self.c.call("PUT", f"/tasks/{tid}/artifacts/adapter.safetensors?node_id=ipad-1", raw=adapter_bytes(a))
        self.c.call("POST", f"/tasks/{tid}/complete", {"node_id": "ipad-1", "status": "succeeded"})
        fed = self.router.federation
        fed.rounds[r["id"]]["deadline_at"] = 0     # ipad-2 never shows up; deadline passes
        fed.expire()
        done = self.wait_round(r["id"])
        self.assertEqual(done["status"], "aggregated", done)
        self.assertEqual(len(done["result"]["contributors"]), 1)

    # ------------------------------------------------------------- bench --
    @staticmethod
    def receipt(model="lfm3-1.7b", decode=61.2, pad=0, rid=None):
        return {"schema": 1, "receipt_id": rid or secrets.token_hex(8), "app_git_sha": "abc123",
                "created_at": "2026-11-26T10:00:00Z", "status": "ok",
                "device": {"machine": "iPad17,3", "os": "iPadOS 27.2.1", "physical_memory_bytes": 16 << 30},
                "model": {"id": model, "repo": "LiquidAI/LFM3-1.7B-MLX-4bit"},
                "workload_set": "w1", "load": {"warm_load_s": 1.2, "junk": "x" * 5000},
                "workloads": [{"id": "short-128", "prompt_tokens": 23, "output_hash": "deadbeef",
                               "deterministic": True,
                               "median": {"ttft_s": 0.05, "prefill_tps": 460.0, "decode_tps": decode,
                                          "peak_footprint_bytes": 1_400_000_000}}],
                "padding": "x" * pad}

    def post_receipt(self, body):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return self.c.call("POST", "/receipts", raw=raw)

    def test_receipts_post_list_get(self):
        self.assertEqual(self.c.call("GET", "/receipts"), (200, {"receipts": [], "more": False}))
        st, a = self.post_receipt(self.receipt(decode=50.0))
        self.assertEqual(st, 201, a)
        self.assertRegex(a["name"], r"^\d{8}T\d{6}Z-\d{4}-lfm3-1\.7b\.json$")
        names = [a["name"]]
        for d in (60.0, 70.0):  # same second: never overwritten, still ordered newest first
            st, b = self.post_receipt(self.receipt(decode=d))
            self.assertEqual(st, 201, b)
            names.append(b["name"])
        self.assertEqual(len(set(names)), 3)
        self.post_receipt(self.receipt(model="../../qwen3"))
        stored = sorted(p.name for p in (Path(self.tmp.name) / "receipts").iterdir())
        self.assertEqual(len(stored), 4)
        self.assertTrue(all("/" not in n and ".." not in n for n in stored))
        st, page = self.c.call("GET", "/receipts?model=lfm3-1.7b")
        self.assertEqual(st, 200)
        self.assertEqual([r["workloads"][0]["decode_tps"] for r in page["receipts"]], [70.0, 60.0, 50.0])
        s = page["receipts"][0]
        self.assertEqual(s["model"]["id"], "lfm3-1.7b")
        self.assertEqual(s["device"]["machine"], "iPad17,3")
        self.assertNotIn("padding", s)
        self.assertNotIn("junk", s["load"])  # summaries are bounded to known fields
        _, one = self.c.call("GET", "/receipts?limit=1")
        self.assertEqual((len(one["receipts"]), one["more"]), (1, True))
        st, full = self.c.call("GET", f"/receipts/{a['name']}")
        self.assertEqual(st, 200)
        self.assertEqual(full["workloads"][0]["median"]["decode_tps"], 50.0)
        self.assertEqual(self.c.call("GET", "/receipts/..%2F..%2Fnodes.json")[0], 404)
        self.assertEqual(self.c.call("GET", "/receipts/20261126T000000Z-0001-nope.json")[0], 404)
        # a restarted worker rebuilds the same index from disk
        again = federated.ReceiptStore(Path(self.tmp.name))
        self.assertEqual([r["name"] for r in again.list()["receipts"]],
                         [r["name"] for r in self.c.call("GET", "/receipts")[1]["receipts"]])

    def test_receipts_dedupe_by_receipt_id(self):
        body = self.receipt(rid="0f0f0f0f-aaaa")
        st, a = self.post_receipt(body)
        self.assertEqual(st, 201)
        st, b = self.post_receipt(body)  # retry after an ambiguous failure
        self.assertEqual(st, 200, b)
        self.assertEqual((b["name"], b["duplicate"]), (a["name"], True))
        self.assertEqual(len(self.c.call("GET", "/receipts")[1]["receipts"]), 1)

    def test_receipt_summary_carries_engine_ab_fields(self):
        import receipts
        r = {**self.receipt(), "engine": "xybrid", "ab_step": "A6", "jetsam_suspected": True,
             "status": "killed", "identity": {"bundle_identifier": "x"}}
        s = receipts.summarize("n.json", r)
        self.assertEqual((s["engine"], s["ab_step"], s["jetsam_suspected"], s["status"]),
                         ("xybrid", "A6", True, "killed"))
        self.assertNotIn("engine", receipts.summarize("m.json", self.receipt()))   # old receipts unchanged

    def test_receipts_validation(self):
        good = json.dumps(self.receipt())
        bad = [b"not json", b"[1, 2]", json.dumps({"schema": 2}).encode(),
               json.dumps({**self.receipt(), "workloads": 5}).encode(),
               json.dumps({**self.receipt(), "workloads": [3]}).encode(),
               json.dumps({**self.receipt(), "model": "m"}).encode(),
               json.dumps({**self.receipt(), "model": {}}).encode(),
               json.dumps({**self.receipt(), "workloads": [{"median": 1}]}).encode(),
               good.encode("utf-16"),
               b"\xef\xbb\xbf" + good.encode(),
               good.replace('"decode_tps": 61.2', '"decode_tps": NaN').encode()]
        for raw in bad:
            self.assertEqual(self.post_receipt(raw)[0], 400, raw[:80])
        self.assertEqual(self.c.call("GET", "/receipts?limit=x")[0], 400)
        self.assertEqual(self.c.call("GET", "/receipts")[1], {"receipts": [], "more": False})
        self.assertEqual(Client(self.c.base, "nope").call("GET", "/receipts")[0], 401)

    def test_receipt_too_large_is_refused_unread_and_closed(self):
        import socket
        host, port = self.server.server_address[:2]
        with socket.create_connection((host, port), timeout=10) as s:
            s.sendall((f"POST /receipts HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {self.token}\r\n"
                       "Content-Type: application/json\r\nContent-Length: 2000000\r\n\r\n").encode())
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break  # closed: the unread body is never parsed as the next request
                data += chunk
        self.assertTrue(data.startswith(b"HTTP/1.1 413"), data[:100])

    def test_unreadable_receipt_file_does_not_break_listing(self):
        self.post_receipt(self.receipt())
        d = Path(self.tmp.name) / "receipts"
        (d / "20200101T000000Z-0001-broken.json").write_bytes(
            b'{"schema": 1, "model": {"id": "m"}, "workloads": 5}')
        page = federated.ReceiptStore(Path(self.tmp.name)).list()  # as after a worker restart
        self.assertEqual(len(page["receipts"]), 1)
        self.assertIn("20200101T000000Z-0001-broken.json", page["unreadable"][0])

    def test_bench_task_roundtrip(self):
        self.node("ipad-1")  # generate-only node never gets bench work
        self.node("ipad-2", kinds=("generate", "bench"))
        st, t = self.c.call("POST", "/tasks", {"kind": "bench", "payload": {
            "model_id": "lfm3-1.7b", "workload_ids": ["short-128"], "repetitions": 1}})
        self.assertEqual(st, 201, t)
        self.assertIsNone(self.c.call("GET", "/nodes/ipad-1/next?wait=0.2")[1]["task"])
        _, nxt = self.c.call("GET", "/nodes/ipad-2/next?wait=2")
        self.assertEqual(nxt["task"]["kind"], "bench")
        self.assertEqual(nxt["task"]["payload"]["workload_ids"], ["short-128"])
        receipt = json.dumps(self.receipt())
        _, done = self.c.call("POST", f"/tasks/{t['id']}/complete",
                              {"node_id": "ipad-2", "status": "succeeded", "output": receipt,
                               "metrics": {"short-128.decode_tps": 61.2}})
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(json.loads(done["output"])["model"]["id"], "lfm3-1.7b")
        _, nodes = self.c.call("GET", "/nodes")
        self.assertIn("bench", next(n for n in nodes["nodes"] if n["id"] == "ipad-2")["capabilities"]["kinds"])
        for bad in ({"model_id": 3}, {"workload_ids": "short-128"}, {"workload_ids": []},
                    {"repetitions": 0}, {"repetitions": True}):
            self.assertEqual(self.c.call("POST", "/tasks", {"kind": "bench", "payload": bad})[0], 400, bad)
        self.assertEqual(self.c.call("POST", "/tasks", {"kind": "bench", "payload": {}})[0], 201)


if __name__ == "__main__":
    unittest.main()
