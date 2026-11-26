#!/usr/bin/env bash
# Mac/Linux: start the job worker (127.0.0.1:8787) and MCP bridge (127.0.0.1:8788).
# Usage: ./start-connector.sh [--allow-approve] [--show-url]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
state="$HOME/.job-worker"
mkdir -p "$state" && chmod 700 "$state"
secrets="$state/secrets.env"
if [[ ! -f "$secrets" ]]; then
  umask 077
  printf 'WORKER_TOKEN=%s\nBRIDGE_SECRET=%s\n' \
    "$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" \
    "$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" > "$secrets"
fi
set -a; . "$secrets"; set +a

bridge_args=(); show=0
for a in "$@"; do
  case "$a" in --allow-approve) bridge_args+=(--allow-approve);; --show-url) show=1;; esac
done

up() { python3 -c "import socket,sys;s=socket.socket();sys.exit(s.connect_ex(('127.0.0.1',$1)))"; }
up 8787 || nohup python3 "$here/jobworker.py" --data "$state/data" --port 8787 >"$state/worker.log" 2>&1 &
up 8788 || nohup python3 "$here/mcp_bridge.py" --port 8788 "${bridge_args[@]}" >"$state/bridge.log" 2>&1 &
sleep 2
up 8787 && echo "worker: up" || echo "worker: DOWN (see $state/worker.log)"
up 8788 && echo "bridge: up" || echo "bridge: DOWN (see $state/bridge.log)"
if [[ $show == 1 ]]; then echo "Connector URL once published: https://<your-host>.ts.net/mcp/$BRIDGE_SECRET"
else echo "Run with --show-url to print the connector URL (contains the secret)."; fi
