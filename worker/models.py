"""Read-only model file server for the iPad (streamed Inkling builds made by tools/expert_repack.py).

Routes (behind the worker's bearer token, like every route):

    GET /models                    {"models": [{"name", "files", "bytes"}]}
    GET /models/<name>             {"name", "files": [{"name", "size"}], "bytes"}
    GET /models/<name>/<file>      the file; honours a single `Range: bytes=a-b` (206)

Only regular files directly inside `<models_dir>/<name>/` are served. Names are one path
component of `[A-Za-z0-9][A-Za-z0-9._-]*`; symlinks, hidden files, subdirectories and
`*.partial` files (an unfinished repack) are never served or listed.
"""
from __future__ import annotations

import re
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _valid(name: str) -> bool:
    return bool(NAME.match(name)) and ".." not in name and not name.endswith(".partial")


def _model_dir(root: Path, name: str) -> Path | None:
    if not _valid(name):
        return None
    d = root / name
    if d.is_symlink() or not d.is_dir():
        return None
    try:
        if d.resolve().parent != root.resolve():
            return None
    except OSError:
        return None
    return d


def _files(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.iterdir()):
        if _valid(p.name) and not p.is_symlink() and p.is_file():
            out.append({"name": p.name, "size": p.stat().st_size})
    return out


def resolve_file(root: Path, name: str, file: str) -> Path | None:
    """The servable file `<root>/<name>/<file>`, or None."""
    d = _model_dir(root, name)
    if d is None or not _valid(file):
        return None
    p = d / file
    if p.is_symlink() or not p.is_file():
        return None
    try:
        if p.resolve().parent != d.resolve():
            return None
    except OSError:
        return None
    return p


def make_models_router(models_dir: Path):
    """A router for jobworker.make_handler: GET /models[...]; None for other routes."""
    root = Path(models_dir)

    def route(method, parts, query, read_json, read_raw):
        if not parts or parts[0] != "models":
            return None
        if method != "GET":
            return 405, {"error": "read-only"}
        if not root.is_dir():
            return 404, {"error": "no models directory configured"}
        if parts == ["models"]:
            models = []
            for d in sorted(root.iterdir()):
                md = _model_dir(root, d.name)
                if md is not None:
                    files = _files(md)
                    models.append({"name": d.name, "files": len(files),
                                   "bytes": sum(f["size"] for f in files)})
            return 200, {"models": models}
        if len(parts) == 2:
            d = _model_dir(root, parts[1])
            if d is None:
                return 404, {"error": "no such model"}
            files = _files(d)
            return 200, {"name": parts[1], "files": files, "bytes": sum(f["size"] for f in files)}
        if len(parts) == 3:
            p = resolve_file(root, parts[1], parts[2])
            if p is None:
                return 404, {"error": "no such file"}
            return 200, p
        return 404, {"error": "not found"}

    return route


def chain(*routers):
    """The first router that claims a request wins."""
    routers = [r for r in routers if r is not None]

    def route(method, parts, query, read_json, read_raw):
        for r in routers:
            out = r(method, parts, query, read_json, read_raw)
            if out is not None:
                return out
        return None

    return route


RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, size: int):
    """(start, end_inclusive) for a single satisfiable range, None for no/ignored header,
    or "invalid" for an unsatisfiable one (416)."""
    if not header:
        return None
    m = RANGE.match(header.strip())
    if not m:
        return None  # multi-range or other units: serve the whole file
    a, b = m.groups()
    if a == "" and b == "":
        return "invalid"
    if a == "":  # suffix range: last b bytes
        n = int(b)
        if n == 0:
            return "invalid"
        return max(0, size - n), size - 1
    start = int(a)
    end = int(b) if b else size - 1
    if start >= size or end < start:
        return "invalid"
    return start, min(end, size - 1)
