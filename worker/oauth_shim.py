"""Minimal OAuth 2.1 shim so Claude's "Connect" flow completes for this connector.

Claude's connector client runs OAuth discovery and Dynamic Client Registration
even for this server. Real authorization is still *knowledge of the connector
secret*:

  * /authorize auto-approves only when the `resource` parameter (RFC 8707, sent
    by MCP clients) is this server's secret URL. Otherwise it shows a form
    asking for the secret.
  * Redirects only go to Claude's own callback hosts, so the flow can't be used
    to send a code anywhere else.
  * Codes are single-use, last 5 minutes, and require PKCE S256.
  * Access tokens are random, stored hashed, and checked in constant time.
    They are an alternative to the secret path, never a way around it: without
    the secret you can't get one.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import secrets
import threading
import time
import urllib.parse
from pathlib import Path

ALLOWED_REDIRECT_HOSTS = {"claude.ai", "claude.com", "www.claude.ai", "www.claude.com"}
LOCAL_REDIRECT_HOSTS = {"localhost", "127.0.0.1"}  # Claude Code / CLI clients use loopback callbacks
CODE_TTL = 300
ACCESS_TTL = 30 * 24 * 3600
REFRESH_TTL = 180 * 24 * 3600


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _redirect_ok(uri: str) -> bool:
    try:
        u = urllib.parse.urlparse(uri)
    except ValueError:
        return False
    if u.scheme == "https" and u.hostname in ALLOWED_REDIRECT_HOSTS:
        return True
    return u.scheme == "http" and u.hostname in LOCAL_REDIRECT_HOSTS


class OAuthShim:
    def __init__(self, secret: str, state_path: Path | None):
        self.secret = secret
        self.state_path = state_path
        self.lock = threading.Lock()
        self.clients: dict[str, list[str]] = {}       # client_id -> redirect_uris
        self.codes: dict[str, dict] = {}               # code -> {client_id, redirect_uri, challenge, exp}
        self.tokens: dict[str, dict] = {}              # sha256(token) -> {kind, client_id, exp}
        if state_path and state_path.exists():
            try:
                st = json.loads(state_path.read_text(encoding="utf-8"))
                self.clients = st.get("clients", {})
                self.tokens = {k: v for k, v in st.get("tokens", {}).items() if v["exp"] > time.time()}
            except (ValueError, KeyError):
                pass

    def _save(self):
        if not self.state_path:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"clients": self.clients, "tokens": self.tokens}), encoding="utf-8")
        tmp.replace(self.state_path)

    # ------------------------------------------------------------ metadata
    def protected_resource(self, base: str) -> dict:
        return {"resource": f"{base}/mcp/{self.secret}", "authorization_servers": [base],
                "bearer_methods_supported": ["header"]}

    def authorization_server(self, base: str) -> dict:
        return {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["jobs"],
        }

    # ------------------------------------------------------------ DCR
    def register(self, body: dict) -> tuple[int, dict]:
        uris = body.get("redirect_uris")
        if not isinstance(uris, list) or not uris or not all(isinstance(u, str) and _redirect_ok(u) for u in uris):
            return 400, {"error": "invalid_redirect_uri",
                         "error_description": "redirect_uris must be Claude callback URLs"}
        client_id = "c_" + secrets.token_urlsafe(16)
        with self.lock:
            if len(self.clients) > 200:  # bound growth from unauthenticated registration
                self.clients.pop(next(iter(self.clients)))
            self.clients[client_id] = uris
            self._save()
        return 201, {"client_id": client_id, "redirect_uris": uris, "client_id_issued_at": int(time.time()),
                     "token_endpoint_auth_method": "none", "grant_types": ["authorization_code", "refresh_token"],
                     "response_types": ["code"], "client_name": body.get("client_name", "client")}

    # ------------------------------------------------------------ authorize
    def _resource_has_secret(self, resource: str | None) -> bool:
        if not resource:
            return False
        path = urllib.parse.urlparse(resource).path.rstrip("/")
        return path.startswith("/mcp/") and hmac.compare_digest(path[5:].encode(), self.secret.encode())

    def authorize(self, q: dict, typed_secret: str | None = None) -> tuple[int, dict, str]:
        """Returns (status, headers, html_body)."""
        client_id = q.get("client_id", "")
        redirect_uri = q.get("redirect_uri", "")
        with self.lock:
            registered = self.clients.get(client_id)
        if registered is None or redirect_uri not in registered:
            return 400, {}, "<h1>Unknown client or redirect URI</h1>"
        if q.get("response_type") != "code" or q.get("code_challenge_method") != "S256" or not q.get("code_challenge"):
            return self._redirect_error(redirect_uri, q, "invalid_request")

        ok = self._resource_has_secret(q.get("resource"))
        if not ok and typed_secret is not None:
            ok = hmac.compare_digest(typed_secret.strip().encode(), self.secret.encode())
        if not ok:
            return 200, {"Content-Type": "text/html; charset=utf-8"}, self._form(q, typed_secret is not None)

        code = secrets.token_urlsafe(32)
        with self.lock:
            now = time.time()
            self.codes = {k: v for k, v in self.codes.items() if v["exp"] > now}
            self.codes[code] = {"client_id": client_id, "redirect_uri": redirect_uri,
                                "challenge": q["code_challenge"], "exp": now + CODE_TTL}
        params = {"code": code}
        if q.get("state"):
            params["state"] = q["state"]
        return 302, {"Location": self._with_query(redirect_uri, params)}, ""

    def _redirect_error(self, redirect_uri, q, err):
        params = {"error": err}
        if q.get("state"):
            params["state"] = q["state"]
        return 302, {"Location": self._with_query(redirect_uri, params)}, ""

    @staticmethod
    def _with_query(uri: str, params: dict) -> str:
        return uri + ("&" if "?" in uri else "?") + urllib.parse.urlencode(params)

    @staticmethod
    def _form(q: dict, failed: bool) -> str:
        hidden = "".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
                         for k, v in q.items() if k != "secret")
        err = '<p style="color:#b00">That secret didn\'t match.</p>' if failed else ""
        return f"""<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Job Worker</title><body style="font:16px system-ui;max-width:28rem;margin:3rem auto;padding:0 1rem">
<h2>Connect Job Worker</h2><p>Paste the connector secret (the last part of the connector URL).</p>{err}
<form method="post" action="/authorize">{hidden}
<input name="secret" type="password" autocomplete="off" style="width:100%;font-size:16px;padding:.5rem">
<p><button style="font-size:16px;padding:.5rem 1rem">Connect</button></p></form></body>"""

    # ------------------------------------------------------------ token
    def token(self, form: dict) -> tuple[int, dict]:
        grant = form.get("grant_type")
        now = time.time()
        if grant == "authorization_code":
            with self.lock:
                entry = self.codes.pop(form.get("code", ""), None)
            if (entry is None or entry["exp"] < now or entry["client_id"] != form.get("client_id")
                    or entry["redirect_uri"] != form.get("redirect_uri")):
                return 400, {"error": "invalid_grant"}
            verifier = form.get("code_verifier", "")
            calc = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            if not hmac.compare_digest(calc, entry["challenge"]):
                return 400, {"error": "invalid_grant", "error_description": "PKCE verification failed"}
            return 200, self._issue(entry["client_id"])
        if grant == "refresh_token":
            h = _h(form.get("refresh_token", ""))
            with self.lock:
                t = self.tokens.pop(h, None)
                self._save()
            if not t or t["kind"] != "refresh" or t["exp"] < now or t["client_id"] != form.get("client_id", t["client_id"]):
                return 400, {"error": "invalid_grant"}
            return 200, self._issue(t["client_id"])
        return 400, {"error": "unsupported_grant_type"}

    def _issue(self, client_id: str) -> dict:
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            self.tokens = {k: v for k, v in self.tokens.items() if v["exp"] > now}
            self.tokens[_h(access)] = {"kind": "access", "client_id": client_id, "exp": now + ACCESS_TTL}
            self.tokens[_h(refresh)] = {"kind": "refresh", "client_id": client_id, "exp": now + REFRESH_TTL}
            self._save()
        return {"access_token": access, "token_type": "Bearer", "expires_in": ACCESS_TTL,
                "refresh_token": refresh, "scope": "jobs"}

    def valid_access_token(self, token: str) -> bool:
        with self.lock:
            t = self.tokens.get(_h(token))
        return bool(t and t["kind"] == "access" and t["exp"] > time.time())
