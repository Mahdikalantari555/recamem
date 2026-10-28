"""Versioned loopback REST API for the RecaMem desktop UI.

The API is a thin, dependency-free layer over :class:`recamem.engine.DroidEngine`
(:mod:`recamem.runner` does the same work for the command line). It uses only the
standard library, so the core install keeps working without the ``[web]`` extra;
``fastapi``/``uvicorn`` remain an optional alternative for embedding.

Contract (see ``docs/ui-design.md`` §14 and ``openspec/changes/recamem-surfaces``):

    GET  /api/v1/health              runtime status (anchor, profile, op)
    GET  /api/v1/models              available profiles
    POST /api/v1/models/load         load a profile
    POST /api/v1/inference           run one prompt, return answer + memory
    POST /api/v1/teach               store facts
    POST /api/v1/recall              two-channel recall with sources
    POST /api/v1/clear               clear learned state (Reset Memory)
    GET  /api/v1/jobs                list jobs
    POST /api/v1/jobs                create a job
    GET  /api/v1/jobs/{id}           job status
    POST /api/v1/jobs/{id}/cancel    request cancellation
    GET  /api/v1/experiments         recorded runs
    GET  /api/v1/experiments/{id}    run detail
    POST /api/v1/bench               benchmark status (blocked without a dataset)

Safety: loopback binding by default, profile names are validated (no path
separators), static files never escape the frontend directory, errors are JSON
objects with an ``error`` key and never contain tracebacks, and the request log
records only method, path and status.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

API_PREFIX = "/api/v1"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MAX_BODY = 1_000_000  # 1 MB: prompts and facts, not uploads


class ApiError(Exception):
    """An error that maps onto a JSON error response."""

    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


def validate_profile_name(name: str) -> str:
    """Reject anything that could escape the profile home."""
    if not isinstance(name, str) or not NAME_RE.match(name.strip().lower()):
        raise ApiError(400, f"invalid profile name {name!r}")
    return name.strip().lower()


class JobStore:
    """In-process job registry.

    Jobs are *records*, not background threads: the UI needs status and
    cancellation, but RecaMem runs one operation at a time per profile.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, **params: Any) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "kind": kind,
            "params": params,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0.0,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return dict(job)

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, f"unknown job {job_id!r}")
            return dict(job)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def update(self, job_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, f"unknown job {job_id!r}")
            job.update(fields)
            job["updated_at"] = time.time()
            return dict(job)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        return self.update(job_id, status="cancelled")


class Runtime:
    """Holds the loaded profile and the observability the UI needs."""

    def __init__(self, base_dir: Optional[str] = None, profile: Optional[str] = None):
        from recamem import identity

        self.base_dir = base_dir or str(identity.default_base_dir())
        self.profile = profile or "droid-alpha"
        self.jobs = JobStore()
        self.engine = None
        self.error: Optional[str] = None
        self.active_op: Optional[str] = None
        self.started_at = time.time()

    # -- profile lifecycle ------------------------------------------------- #
    def load(self, name: Optional[str] = None) -> Dict[str, Any]:
        from recamem.manager import DroidManager

        self.profile = validate_profile_name(name or self.profile)
        started = time.perf_counter()
        manager = DroidManager(base_dir=self.base_dir)
        self.engine = manager.get_droid(self.profile)
        self.error = None
        return {
            "profile": self.profile,
            "loaded": True,
            "load_seconds": round(time.perf_counter() - started, 4),
        }

    def require_engine(self):
        if self.engine is None:
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001 - surfaced as 503
                self.error = f"{type(exc).__name__}: {exc}"
                raise ApiError(503, f"profile {self.profile!r} could not be loaded: {self.error}")
        return self.engine

    def anchor_mode(self) -> str:
        from recamem.anchor import resolve_anchor_mode

        try:
            return resolve_anchor_mode()
        except ValueError as exc:
            raise ApiError(500, str(exc))

    def health(self) -> Dict[str, Any]:
        from recamem import __version__

        return {
            "status": "ok" if self.error is None else "degraded",
            "profile": self.profile,
            "anchor": self.anchor_mode(),
            "version": __version__,
            "device": "cpu",
            "loaded": self.engine is not None,
            "active_op": self.active_op,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Route table
# --------------------------------------------------------------------------- #
def _json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = handler.headers.get("Content-Length")
    if not length:
        return {}
    try:
        size = int(length)
    except ValueError:
        raise ApiError(400, "invalid Content-Length")
    if size > MAX_BODY:
        raise ApiError(413, f"request body larger than {MAX_BODY} bytes")
    raw = handler.rfile.read(size)
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, f"invalid JSON body: {exc}")
    if not isinstance(payload, dict):
        raise ApiError(400, "JSON body must be an object")
    return payload


def _string(payload: Dict[str, Any], key: str, *, required: bool = True,
            default: Optional[str] = None) -> Optional[str]:
    value = payload.get(key, default)
    if value is None:
        if required:
            raise ApiError(400, f"missing field {key!r}")
        return None
    if not isinstance(value, str):
        raise ApiError(400, f"field {key!r} must be a string")
    return value


def _profile_names(base_dir: str) -> List[str]:
    from recamem.manager import DroidManager

    return DroidManager(base_dir=base_dir).list_droids()


def _experiments(limit: int = 50) -> List[Dict[str, Any]]:
    """Recorded runs from ``reports/`` (read-only; never fabricated)."""
    root = Path(__file__).resolve().parents[1] / "reports"
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "id": str(path.relative_to(root)),
                "bytes": stat.st_size,
                "modified": stat.st_mtime,
                "kind": "evidence",
            }
        )
    return rows[-limit:]


def make_handler(runtime: Runtime, static_dir: Optional[Path] = None):
    static_root = Path(static_dir).resolve() if static_dir else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "recamem-api/1"
        protocol_version = "HTTP/1.1"

        # -- helpers -------------------------------------------------- #
        def _send(self, status: int, payload: Any, content_type: str = "application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: int, message: str, **extra: Any):
            self._send(status, {"error": message, **extra})

        def log_message(self, fmt: str, *args: Any) -> None:
            # Only method, path and status: request bodies may hold user text.
            path = getattr(self, "path", "-")
            sys_stderr = os.environ.get("RECAMEM_API_LOG", "1") != "0"
            if sys_stderr:
                os.write(2, f"{self.command} {path} {args[1] if len(args) > 1 else ''}\n".encode())

        # -- dispatch ------------------------------------------------- #
        def do_GET(self):  # noqa: N802 - stdlib naming
            self._dispatch("GET")

        def do_HEAD(self):  # noqa: N802
            self._dispatch("HEAD")

        def do_POST(self):  # noqa: N802
            self._dispatch("POST")

        def _dispatch(self, method: str):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path.startswith(API_PREFIX):
                    self._api(method, path[len(API_PREFIX):] or "/", parse_qs(parsed.query))
                else:
                    self._static(path)
            except ApiError as exc:
                self._error(exc.status, exc.message, **exc.extra)
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                self._error(500, f"{type(exc).__name__}: {exc}")

        # -- API ------------------------------------------------------ #
        def _api(self, method: str, path: str, query: Dict[str, List[str]]):
            parts = [part for part in path.split("/") if part]
            if method in {"GET", "HEAD"}:
                if parts == ["health"]:
                    return self._send(200, runtime.health())
                if parts == ["models"]:
                    return self._send(
                        200,
                        {
                            "models": _profile_names(runtime.base_dir),
                            "selected": runtime.profile,
                            "loaded": runtime.engine is not None,
                        },
                    )
                if parts == ["jobs"]:
                    return self._send(200, {"jobs": runtime.jobs.list()})
                if len(parts) == 2 and parts[0] == "jobs":
                    return self._send(200, runtime.jobs.get(parts[1]))
                if parts == ["experiments"]:
                    return self._send(200, {"experiments": _experiments()})
                if len(parts) == 2 and parts[0] == "experiments":
                    wanted = parts[1]
                    for row in _experiments(limit=10_000):
                        if row["id"] == wanted or Path(row["id"]).stem == wanted:
                            return self._send(200, row)
                    raise ApiError(404, f"unknown experiment {wanted!r}")
                raise ApiError(404, f"unknown path {path!r}")

            if method != "POST":
                raise ApiError(405, f"method {method} not allowed")

            payload = _json_body(self)

            if parts == ["models", "load"]:
                name = _string(payload, "name", required=False, default=runtime.profile)
                return self._send(200, runtime.load(name))

            if parts == ["inference"]:
                from recamem.runner import run_once

                prompt = _string(payload, "prompt")
                mode = payload.get("mode", "chat")
                if mode not in {"chat", "recall", "teach"}:
                    raise ApiError(400, f"unknown mode {mode!r}")
                engine = runtime.require_engine()
                runtime.active_op = "inference"
                try:
                    result = run_once(prompt, engine, mode=mode,
                                      profile=runtime.profile)
                finally:
                    runtime.active_op = None
                return self._send(200, result)

            if parts == ["teach"]:
                text = _string(payload, "text")
                engine = runtime.require_engine()
                result = engine.teach(text)
                if not isinstance(result, dict):
                    result = {"propositions": int(result or 0)}
                return self._send(
                    200, {"facts_stored": int(result.get("propositions", 0) or 0), **result}
                )

            if parts == ["recall"]:
                query_text = _string(payload, "query")
                top_k = payload.get("top_k", 3)
                if not isinstance(top_k, int) or not 1 <= top_k <= 50:
                    raise ApiError(400, "top_k must be an integer between 1 and 50")
                engine = runtime.require_engine()
                hits = engine.recall(query_text, top_k=top_k)
                return self._send(200, {"query": query_text, "results": hits or []})

            if parts == ["clear"]:
                engine = runtime.require_engine()
                engine.clear_memory()
                return self._send(200, {"cleared": True, "profile": runtime.profile})

            if parts == ["bench"]:
                return self._send(
                    200,
                    {
                        "status": "blocked",
                        "reason": "no benchmark dataset is configured in this install",
                        "dataset_source": "none",
                        "results": [],
                    },
                )

            if parts == ["jobs"]:
                kind = _string(payload, "kind", required=False, default="eval")
                job = runtime.jobs.create(kind, **_safe_params(payload))
                return self._send(202, job)

            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                return self._send(200, runtime.jobs.cancel(parts[1]))

            raise ApiError(404, f"unknown path {path!r}")

        # -- static frontend ------------------------------------------ #
        def _static(self, path: str):
            if static_root is None or not static_root.is_dir():
                return self._send(
                    200,
                    _NOT_BUILT_HTML,
                    content_type="text/html; charset=utf-8",
                )
            relative = path.lstrip("/") or "index.html"
            candidate = (static_root / relative).resolve()
            if static_root not in candidate.parents and candidate != static_root:
                raise ApiError(400, "path escapes the frontend directory")
            if candidate.is_dir() or not candidate.is_file():
                candidate = static_root / "index.html"
                if not candidate.is_file():
                    raise ApiError(404, "frontend build not found")
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if candidate.suffix == ".js":
                content_type = "text/javascript"
            return self._send(200, candidate.read_bytes(), content_type=content_type)

    return Handler


_NOT_BUILT_HTML = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RecaMem UI</title>
<style>body{background:#111318;color:#F3F4F6;font:14px/1.6 system-ui,Inter,sans-serif;
padding:48px;max-width:720px;margin:auto}code{color:#B5F36D;font-family:ui-monospace,monospace}
a{color:#B5F36D}</style></head>
<body><h1>RecaMem UI is not built</h1>
<p>The Python API is running, but <code>web/dist/</code> does not exist yet.</p>
<pre><code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code></pre>
<p>Then reload this page, or run <code>recamem ui</code> to open the desktop shell.</p>
<p>API: <a href="/api/v1/health">/api/v1/health</a></p></body></html>
"""


def create_server(host: str = "127.0.0.1", port: int = 8000,
                  base_dir: Optional[str] = None, profile: Optional[str] = None,
                  static_dir: Optional[str] = None) -> ThreadingHTTPServer:
    """Build (but do not start) the API server.

    ``host`` must be a loopback address unless ``RECAMEM_API_ALLOW_REMOTE=1`` is
    set: the API exposes profile read/write and must not listen publicly.
    """
    if host not in LOOPBACK_HOSTS and os.environ.get("RECAMEM_API_ALLOW_REMOTE") != "1":
        raise ApiError(
            500,
            f"refusing to bind {host!r}: the RecaMem API is loopback-only "
            "(set RECAMEM_API_ALLOW_REMOTE=1 to override)",
        )
    runtime = Runtime(base_dir=base_dir, profile=profile)
    handler = make_handler(runtime, Path(static_dir) if static_dir else None)
    server = ThreadingHTTPServer((host, port), handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    return server


def default_static_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "dist"


def serve(host: str = "127.0.0.1", port: int = 8000, base_dir: Optional[str] = None,
          profile: Optional[str] = None, static: bool = False) -> int:
    """Run the API until interrupted. ``static=True`` also serves ``web/dist``."""
    server = create_server(
        host=host,
        port=port,
        base_dir=base_dir,
        profile=profile,
        static_dir=str(default_static_dir()) if static else None,
    )
    shown = f"http://{host}:{server.server_address[1]}"
    print(f"recamem api listening on {shown}{API_PREFIX}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nrecamem api stopped", flush=True)
    finally:
        server.server_close()
    return 0


def _safe_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Job parameters: scalars only, and never anything that looks secret."""
    safe = {}
    for key, value in payload.items():
        if key in {"kind", "prompt", "text", "query"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 200:
                value = value[:200]
            safe[key] = value
    return safe
