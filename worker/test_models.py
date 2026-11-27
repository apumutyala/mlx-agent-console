"""The read-only model file route against a real HTTP server.

    python -m unittest -v test_models
"""
import http.client
import json
import os
import secrets
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jobworker  # noqa: E402
import models  # noqa: E402


class ModelsRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.models = root / "models"
        build = self.models / "Inkling-2-Mini-DWQ-stream"
        build.mkdir(parents=True)
        self.blob = os.urandom(300_000)
        (build / "experts.bin").write_bytes(self.blob)
        (build / "experts.json").write_text('{"format": "inkling2-expert-stream"}')
        (build / ".hidden").write_text("no")
        (build / "experts.bin.partial").write_text("unfinished")
        (build / "sub").mkdir()
        (build / "sub" / "x.txt").write_text("nested")
        (root / "secret.txt").write_text("outside")
        self.symlinked = False
        try:
            (build / "link.txt").symlink_to(root / "secret.txt")
            (self.models / "linked").symlink_to(root, target_is_directory=True)
            self.symlinked = True
        except (OSError, NotImplementedError):
            pass  # Windows without symlink rights
        self.token = secrets.token_hex(16)
        worker = jobworker.Worker(root / "data", {})
        router = models.chain(models.make_models_router(self.models), None)
        self.server = jobworker.serve(worker, self.token, "127.0.0.1", 0, router)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def get(self, path, headers=None, auth=True):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if auth:
            h["Authorization"] = f"Bearer {self.token}"
        c.request("GET", path, headers=h)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, dict(r.getheaders()), body

    def test_requires_token(self):
        status, _, _ = self.get("/models", auth=False)
        self.assertEqual(status, 401)
        status, _, _ = self.get("/models/Inkling-2-Mini-DWQ-stream/experts.bin", auth=False)
        self.assertEqual(status, 401)

    def test_lists_builds_and_files(self):
        status, _, body = self.get("/models")
        self.assertEqual(status, 200)
        listed = json.loads(body)["models"]
        self.assertEqual([m["name"] for m in listed], ["Inkling-2-Mini-DWQ-stream"])
        status, _, body = self.get("/models/Inkling-2-Mini-DWQ-stream")
        self.assertEqual(status, 200)
        files = json.loads(body)["files"]
        self.assertEqual([f["name"] for f in files], ["experts.bin", "experts.json"])
        self.assertEqual(files[0]["size"], len(self.blob))

    def test_whole_file(self):
        status, headers, body = self.get("/models/Inkling-2-Mini-DWQ-stream/experts.bin")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.blob)
        self.assertEqual(headers["Accept-Ranges"], "bytes")

    def test_ranges(self):
        path = "/models/Inkling-2-Mini-DWQ-stream/experts.bin"
        status, headers, body = self.get(path, {"Range": "bytes=1000-1999"})
        self.assertEqual((status, body), (206, self.blob[1000:2000]))
        self.assertEqual(headers["Content-Range"], f"bytes 1000-1999/{len(self.blob)}")
        status, _, body = self.get(path, {"Range": "bytes=299000-"})
        self.assertEqual((status, body), (206, self.blob[299000:]))
        status, _, body = self.get(path, {"Range": "bytes=-10"})
        self.assertEqual((status, body), (206, self.blob[-10:]))
        status, _, body = self.get(path, {"Range": "bytes=100-999999999"})
        self.assertEqual((status, body), (206, self.blob[100:]))
        status, headers, _ = self.get(path, {"Range": f"bytes={len(self.blob)}-"})
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], f"bytes */{len(self.blob)}")

    def test_rejects_traversal_and_unlisted_files(self):
        bad = [
            "/models/Inkling-2-Mini-DWQ-stream/..%2F..%2Fsecret.txt",
            "/models/Inkling-2-Mini-DWQ-stream/../../secret.txt",
            "/models/../secret.txt",
            "/models/Inkling-2-Mini-DWQ-stream/.hidden",
            "/models/Inkling-2-Mini-DWQ-stream/experts.bin.partial",
            "/models/Inkling-2-Mini-DWQ-stream/sub",
            "/models/Inkling-2-Mini-DWQ-stream/sub/x.txt",
            "/models/Inkling-2-Mini-DWQ-stream/link.txt",
            "/models/linked/secret.txt",
            "/models/linked",
            "/models/nope/experts.bin",
            "/models/Inkling-2-Mini-DWQ-stream/%2e%2e",
        ]
        for path in bad:
            status, _, body = self.get(path)
            self.assertIn(status, (400, 404), path)
            self.assertNotIn(b"outside", body, path)
            self.assertNotIn(b"nested", body, path)

    def test_read_only(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/models/Inkling-2-Mini-DWQ-stream/experts.bin", body=b"{}",
                  headers={"Authorization": f"Bearer {self.token}", "Content-Length": "2"})
        r = c.getresponse()
        r.read()
        c.close()
        self.assertEqual(r.status, 405)
        self.assertEqual((self.models / "Inkling-2-Mini-DWQ-stream" / "experts.bin").read_bytes(),
                         self.blob)

    def test_other_routes_fall_through(self):
        status, _, _ = self.get("/job-types")
        self.assertEqual(status, 200)


class ParseRangeTests(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(models.parse_range(None, 10))
        self.assertIsNone(models.parse_range("items=0-1", 10))
        self.assertIsNone(models.parse_range("bytes=0-1,4-5", 10))
        self.assertEqual(models.parse_range("bytes=2-4", 10), (2, 4))
        self.assertEqual(models.parse_range("bytes=2-", 10), (2, 9))
        self.assertEqual(models.parse_range("bytes=-3", 10), (7, 9))
        self.assertEqual(models.parse_range("bytes=-30", 10), (0, 9))
        self.assertEqual(models.parse_range("bytes=5-2", 10), "invalid")
        self.assertEqual(models.parse_range("bytes=10-", 10), "invalid")
        self.assertEqual(models.parse_range("bytes=-", 10), "invalid")


if __name__ == "__main__":
    unittest.main()
