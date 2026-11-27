"""Show a QR code that pairs the iPad app with this worker (Settings -> Scan pairing code).

The code contains the worker URL and WORKER_TOKEN, so it is shown only on this
screen and the image file is deleted when you press Enter. Run it yourself:

    uv run --no-project --with "qrcode[pil]" python pair.py            # opens an image
    uv run --no-project --with qrcode python pair.py --ascii           # prints in the terminal

The URL defaults to this machine's Tailscale name on port 8443 (the tailnet-only
`tailscale serve --https=8443 http://127.0.0.1:8787`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def tailnet_url(port: int) -> str:
    out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, check=True).stdout
    host = json.loads(out)["Self"]["DNSName"].rstrip(".")
    return f"https://{host}:{port}"


def payload(url: str, token: str) -> str:
    if not url.startswith(("https://", "http://")):
        raise SystemExit("url must start with https:// or http://")
    if not 24 <= len(token) <= 256 or any(c.isspace() for c in token):
        raise SystemExit("worker token looks wrong (need 24-256 non-space characters)")
    return json.dumps({"v": 1, "url": url, "token": token}, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secrets", type=Path, default=Path.home() / ".job-worker" / "secrets.json")
    ap.add_argument("--url", help="worker URL the iPad should use (default: this machine's tailnet name)")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--ascii", action="store_true", help="print the code in the terminal instead of an image")
    args = ap.parse_args()

    token = json.loads(args.secrets.read_text(encoding="utf-8-sig"))["worker_token"]
    data = payload(args.url or tailnet_url(args.port), token)

    import qrcode  # provided by `uv run --with qrcode[pil]`
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    if args.ascii:
        sys.stdout.reconfigure(encoding="utf-8")  # block characters on Windows consoles
        qr.print_ascii(invert=True)
        input("Scan it on the iPad, then press Enter to clear... ")
        print("\033[2J\033[H", end="")
        return
    fd, path = tempfile.mkstemp(prefix="pair-", suffix=".png")
    os.close(fd)
    try:
        qr.make_image(fill_color="black", back_color="white").save(path)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - opens the image viewer
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        print(f"Pairing code for {json.loads(data)['url']} is open in your image viewer.")
        input("Scan it on the iPad (Settings -> Scan pairing code), then press Enter to delete it... ")
    finally:
        try:
            os.remove(path)
        except OSError:
            print(f"Close the image viewer and delete {path}")


if __name__ == "__main__":
    main()
