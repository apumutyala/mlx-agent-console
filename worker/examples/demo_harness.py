"""Demo harness speaking the worker's approval protocol.

Prints progress, starts a long-lived child (to prove cancel kills the whole
process group), and halfway through asks for approval on stdout and waits for
the decision on stdin.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--delay", type=float, default=float(os.environ.get("DEMO_DELAY", "0.5")))
ap.add_argument("--pidfile", default=os.environ.get("DEMO_PIDFILE"))
ap.add_argument("--no-approval", action="store_true", default=bool(os.environ.get("DEMO_NO_APPROVAL")))
args = ap.parse_args()

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
if args.pidfile:
    with open(args.pidfile, "w") as f:
        f.write(str(child.pid))
print(f"started helper subprocess pid={child.pid}", flush=True)

for i in range(1, args.steps + 1):
    print(f"step {i}/{args.steps}", flush=True)
    if i == args.steps // 2 and not args.no_approval:
        print("::approval:: " + json.dumps({"id": "apply", "prompt": "Apply the patch?"}), flush=True)
        reply = json.loads(sys.stdin.readline())
        print(f"approval {reply['id']} -> {reply['decision']}", flush=True)
        if reply["decision"] != "approve":
            print("skipping patch", file=sys.stderr, flush=True)
    time.sleep(args.delay)

child.terminate()
print("done", flush=True)
