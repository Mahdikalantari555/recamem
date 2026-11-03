"""Additive companion endpoints answer with measured, honest payloads."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from recamem import api


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAMEM_ANCHOR", "mock")
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path))
    instance = api.create_server(host="127.0.0.1", port=0, base_dir=str(tmp_path),
                                 profile="api-test",
                                 static_dir=str(api.companion_static_dir()))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_address[1]}"
    instance.shutdown()
    instance.server_close()


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as response:
        return response.status, json.loads(response.read())


def _post(url, path, payload):
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_companion_state_and_pet_are_measured(server):
    status, payload = _get(server, "/api/v1/companion/state")
    assert status == 200
    assert payload["state"]["anchor"] == "mock"
    assert payload["state"]["facts_active"] == 0
    assert payload["pet"]["energy"]["value"] is None
    assert payload["state"]["signals"]


def test_feed_updates_state_and_pet_needs_a_target(server):
    status, payload = _post(server, "/api/v1/companion/act",
                            {"action": "feed", "text": "The calibration constant is 0.98."})
    assert status == 200 and payload["event"]["engine_call"].startswith("teach(")
    status, state = _get(server, "/api/v1/companion/state")
    assert state["state"]["facts_active"] >= 1
    status, error = _post(server, "/api/v1/companion/act", {"action": "pet"})
    assert status == 400 and "fact_id" in error["error"]


def test_best_checkpoint_is_none_not_a_fake_model(server):
    status, payload = _get(server, "/api/v1/checkpoints/best")
    assert status == 200 and payload["status"] == "none"


def test_tokens_are_rejected(server):
    status, payload = _post(server, "/api/v1/companion/act",
                            {"action": "feed", "text": "x", "HF_TOKEN": "secret"})
    assert status == 400 and "server-side" in payload["error"]


def test_existing_routes_are_unchanged(server):
    status, health = _get(server, "/api/v1/health")
    assert status == 200 and health["status"] == "ok" and health["anchor"] == "mock"
    status, models = _get(server, "/api/v1/models")
    assert status == 200 and "models" in models


def test_index_is_served_from_the_committed_build(server):
    with urllib.request.urlopen(server + "/", timeout=30) as response:
        body = response.read().decode()
    assert "RecaMem Companion" in body


def test_state_exposes_the_recalled_fact_target(server):
    status, payload = _get(server, "/api/v1/companion/state")
    assert status == 200 and "last_fact_id" in payload
    assert payload["last_fact_id"] is None


def test_checkpoint_selection_is_honest_about_no_weight_loader(server, tmp_path):
    import hashlib

    directory = tmp_path / "checkpoints" / "run-a"
    directory.mkdir(parents=True)
    weights = b"trained-weights"
    (directory / "weights.bin").write_bytes(weights)
    manifest = {
        "files": [{"path": "weights.bin", "sha256": hashlib.sha256(weights).hexdigest()}],
        "config": {"d_model": 64}, "seed": 1337, "dataset": "fixture",
        "commit": "0" * 40, "command": "python scripts/train_lm.py",
        "gates": {"G2": "passed"}, "held_out_bpb": 1.0,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    status, best = _get(server, "/api/v1/checkpoints/best")
    assert status == 200 and best["status"] == "ok"
    status, selected = _post(server, "/api/v1/checkpoints/select", {"name": "run-a"})
    assert status == 200 and selected["selected"] == "run-a"
    assert selected["applied_to_generation"] is False
    assert "no trained-weight loader" in selected["reason"]
    status, payload = _get(server, "/api/v1/companion/state")
    assert payload["state"]["selected_checkpoint"] == "run-a"
