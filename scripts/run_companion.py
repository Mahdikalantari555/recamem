#!/usr/bin/env python3
"""Headless companion smoke: no Node, no Rust, no weights, no network.

Starts the stdlib API with the committed Svelte build and checks that the
additive endpoints answer with honest, measured payloads.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recamem import api  # noqa: E402


def _get(url: str):
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.status, response.read()


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    dist = api.companion_static_dir()
    if not (dist / "index.html").is_file():
        print(json.dumps({"error": "companion build missing", "path": str(dist)}))
        return 1
    home = tempfile.mkdtemp(prefix="recamem-companion-")
    os.environ["RECAMEM_ANCHOR"] = "mock"
    os.environ["RECAMEM_HOME"] = home
    server = api.create_server(host="127.0.0.1", port=0, base_dir=home,
                               profile="smoke", static_dir=str(dist))
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks = {}
    try:
        status, body = _get(f"{base}/")
        checks["index"] = status == 200 and b"RecaMem Companion" in body
        status, payload = _get(f"{base}/api/v1/health")
        checks["health"] = status == 200 and json.loads(payload)["status"] == "ok"
        status, payload = _get(f"{base}/api/v1/companion/state")
        state_payload = json.loads(payload)
        checks["state"] = (status == 200 and "state" in state_payload
                           and "pet" in state_payload
                           and state_payload["state"]["anchor"] == "mock")
        status, payload = _get(f"{base}/api/v1/checkpoints/best")
        checks["best_none"] = json.loads(payload)["status"] == "none"
        status, payload = _post(f"{base}/api/v1/companion/act",
                                {"action": "feed", "text": "The calibration constant is 0.98."})
        checks["feed"] = (status == 200 and payload["event"]["engine_call"].startswith("teach("))
        status, payload = _get(f"{base}/api/v1/companion/state")
        checks["pet_after_feed"] = json.loads(payload)["state"]["facts_active"] >= 1
        status, payload = _post(f"{base}/api/v1/companion/act", {"action": "pet"})
        checks["pet_without_target_rejected"] = status == 400
    except urllib.error.HTTPError as exc:
        # A 400 on the last check is an expected part of the smoke.
        if exc.code != 400 or "pet_without_target_rejected" in checks:
            raise
        checks["pet_without_target_rejected"] = True
    finally:
        server.shutdown()
        server.server_close()
    print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2))
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
