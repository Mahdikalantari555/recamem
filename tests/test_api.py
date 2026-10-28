"""Versioned loopback REST API (surfaces 2.1, 2.2, 9.2)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from recamem import api


@pytest.fixture()
def server(tmp_path):
    srv = api.create_server(port=0, base_dir=str(tmp_path / "home"), profile="demo")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _url(server, path):
    return f"http://127.0.0.1:{server.server_address[1]}{path}"


def _get(server, path):
    with urllib.request.urlopen(_url(server, path), timeout=10) as response:
        return response.status, json.loads(response.read())


def _post(server, path, payload):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        _url(server, path), data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def _error(server, path, payload=None):
    try:
        if payload is None:
            urllib.request.urlopen(_url(server, path), timeout=10)
        else:
            urllib.request.urlopen(
                urllib.request.Request(
                    _url(server, path),
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=10,
            )
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    raise AssertionError("expected an HTTP error")


def test_health_reports_anchor_and_profile(server):
    status, payload = _get(server, "/api/v1/health")
    assert status == 200
    assert payload["anchor"] == "mock"
    assert payload["profile"] == "demo"
    assert payload["status"] in {"ok", "degraded"}
    assert "version" in payload


def test_teach_then_recall_returns_sources(server):
    status, payload = _post(
        server, "/api/v1/teach", {"text": "The calibration constant is 0.98."}
    )
    assert status == 200
    assert payload["facts_stored"] >= 1

    status, payload = _post(
        server, "/api/v1/recall", {"query": "What is the calibration constant?", "top_k": 2}
    )
    assert status == 200
    assert payload["results"]
    assert payload["results"][0]["text"]


def test_inference_returns_answer_and_memory(server):
    _post(server, "/api/v1/teach", {"text": "The project lead is Dr. Mina Rostami."})
    status, payload = _post(server, "/api/v1/inference", {"prompt": "Who is the project lead?"})
    assert status == 200
    assert payload["response"]
    assert payload["prompt"] == "Who is the project lead?"
    assert payload["anchor"] == "mock"


def test_models_lists_profiles_and_load_selects_one(server):
    status, payload = _get(server, "/api/v1/models")
    assert status == 200
    assert payload["selected"] == "demo"
    status, payload = _post(server, "/api/v1/models/load", {"name": "second"})
    assert status == 200 and payload["profile"] == "second"


def test_clear_resets_learned_state(server):
    _post(server, "/api/v1/teach", {"text": "Station 4 is offline."})
    status, payload = _post(server, "/api/v1/clear", {})
    assert status == 200 and payload["cleared"] is True
    _status, recalled = _post(server, "/api/v1/recall", {"query": "Station 4"})
    assert recalled["results"] == []


def test_jobs_lifecycle(server):
    status, job = _post(server, "/api/v1/jobs", {"kind": "eval", "seed": 1})
    assert status == 202 and job["status"] == "queued"
    _status, fetched = _get(server, f"/api/v1/jobs/{job['id']}")
    assert fetched["kind"] == "eval"
    _status, cancelled = _post(server, f"/api/v1/jobs/{job['id']}/cancel", {})
    assert cancelled["status"] == "cancelled"
    _status, listing = _get(server, "/api/v1/jobs")
    assert [row["id"] for row in listing["jobs"]] == [job["id"]]


def test_experiments_endpoint_is_a_listing(server):
    status, payload = _get(server, "/api/v1/experiments")
    assert status == 200
    assert isinstance(payload["experiments"], list)


def test_bench_reports_blocked_instead_of_inventing_metrics(server):
    status, payload = _post(server, "/api/v1/bench", {})
    assert status == 200
    assert payload["status"] == "blocked"
    assert payload["results"] == []


def test_unknown_path_returns_json_404(server):
    code, payload = _error(server, "/api/v1/nope")
    assert code == 404
    assert "error" in payload


def test_error_bodies_never_contain_tracebacks(server):
    code, payload = _error(server, "/api/v1/inference", {"mode": "chat"})
    assert code == 400
    assert "missing field" in payload["error"]
    assert "Traceback" not in json.dumps(payload)


def test_invalid_profile_name_is_rejected(server):
    code, payload = _error(server, "/api/v1/models/load", {"name": "../../etc"})
    assert code == 400
    assert "invalid profile name" in payload["error"]


def test_non_loopback_binding_is_refused(monkeypatch):
    monkeypatch.delenv("RECAMEM_API_ALLOW_REMOTE", raising=False)
    with pytest.raises(api.ApiError) as excinfo:
        api.create_server(host="0.0.0.0", port=0)
    assert "loopback-only" in str(excinfo.value)


def test_static_serving_blocks_path_traversal(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>ok</html>")
    srv = api.create_server(port=0, base_dir=str(tmp_path / "home"), static_dir=str(dist))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        code, payload = _error(srv, "/../../etc/passwd")
        assert code == 400 and "escapes" in payload["error"]
        with urllib.request.urlopen(_url(srv, "/"), timeout=10) as response:
            assert b"ok" in response.read()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_web_fallback_page_explains_the_build_step(tmp_path):
    srv = api.create_server(port=0, base_dir=str(tmp_path / "home"))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(_url(srv, "/"), timeout=10) as response:
            body = response.read().decode()
        assert "not built" in body
        assert "npm run build" in body
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
