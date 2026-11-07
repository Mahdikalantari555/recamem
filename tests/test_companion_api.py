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


def test_job_status_exposes_elapsed_log_tail_and_checkpoint_path(server):
    status, job = _post(server, "/api/v1/jobs", {"kind": "eval", "seed": 42})
    assert status == 202
    status, fetched = _get(server, f"/api/v1/jobs/{job['id']}")
    assert status == 200
    assert "elapsed" in fetched and isinstance(fetched["elapsed"], (int, float))
    assert fetched["elapsed"] >= 0.0
    assert "log_tail" in fetched and isinstance(fetched["log_tail"], list)
    assert "progress" in fetched and fetched["progress"] == 0.0
    assert "checkpoint_path" in fetched


def test_hub_push_and_pull_validation_and_token_rejection(server):
    # Tokens in payload rejected
    status, err1 = _post(server, "/api/v1/hub/push", {"repo_id": "test/repo", "HF_TOKEN": "abc"})
    assert status == 400 and "tokens must stay server-side" in err1["error"]

    status, err2 = _post(server, "/api/v1/hub/pull", {"repo_id": "test/repo", "token": "xyz"})
    assert status == 400 and "tokens must stay server-side" in err2["error"]

    # Missing confirmation rejected
    status, err3 = _post(server, "/api/v1/hub/push", {"repo_id": "test/repo"})
    assert status == 400 and "require explicit confirmation" in err3["error"]

    status, err4 = _post(server, "/api/v1/hub/pull", {"repo_id": "test/repo", "confirm": False})
    assert status == 400 and "require explicit confirmation" in err4["error"]


def test_companion_search_action_and_chat_remembering(server):
    # Chat naturally through play action
    status, payload = _post(server, "/api/v1/companion/act", {
        "action": "play",
        "text": "The orbital altitude of Terra satellite is 705 kilometers.",
    })
    assert status == 200
    assert "absorbed this into my memory" in payload["result"]["response"]

    # Search action works through companion
    status, res = _post(server, "/api/v1/companion/act", {
        "action": "search",
        "text": "remote sensing",
    })
    assert status == 200
    assert "result" in res
    assert "searched" in res["result"] or "available" in res["result"]




def test_cpu_smoke_training_and_native_generation(server):
    import time
    status, job = _post(server, '/api/v1/jobs', {
        'kind': 'train', 'dataset': 'cpu_smoke', 'core': 'C7', 'lm_tokens': 32,
        'threads': 1, 'd_model': 8, 'n_layers': 1, 'n_heads': 2, 'd_ff': 16,
        'vocab_size': 256, 'seq_len': 16, 'batch_size': 1, 'grad_accum': 1,
        'checkpoint_every': 100, 'seed': 1337,
    })
    assert status == 202, job
    for _ in range(150):
        _, current = _get(server, '/api/v1/jobs/' + job['id'])
        if current['status'] in ('done', 'failed', 'cancelled'):
            break
        time.sleep(.1)
    assert current['status'] == 'done', current
    assert current['result']['report']['dataset_meta']['kind'] == 'smoke_fixture'
    status, answer = _post(server, '/api/v1/native/generate', {
        'job_id': job['id'], 'prompt': 'A seed', 'max_tokens': 2})
    assert status == 200 and answer['memory_used'] is False
    assert isinstance(answer['response'], str)
    status, _ = _post(server, '/api/v1/native/generate', {
        'job_id': job['id'], 'prompt': 'A seed', 'max_tokens': 1000})
    assert status == 400


def test_learning_status_and_configuration_are_honest(server):
    status, result = _get(server, '/api/v1/learning')
    assert status == 200 and result['status'] == 'blocked'
    assert result['training_active'] is False
    status, result = _post(server, '/api/v1/learning/configure', {'enabled': False, 'threads': 2})
    assert status == 200 and result['status'] == 'paused'
    status, result = _get(server, '/api/v1/learning')
    assert result['threads'] == 2 and not result['enabled']
    status, result = _post(server, '/api/v1/learning/configure', {'enabled': True, 'threads': True})
    assert status == 400


def test_native_memory_generation_requires_real_job_and_returns_evidence(server):
    import time

    status, result = _post(server, '/api/v1/native/memory-generate', {'job_id': 'missing', 'prompt': 'hi'})
    assert status == 404
    status, job = _post(server, '/api/v1/jobs', {
        'kind': 'train', 'dataset': 'cpu_smoke', 'core': 'C4', 'lm_tokens': 32,
        'd_model': 8, 'n_layers': 1, 'n_heads': 2, 'd_ff': 16, 'vocab_size': 256,
        'seq_len': 16, 'batch_size': 1, 'grad_accum': 1, 'threads': 1, 'seed': 1337,
    })
    assert status == 202
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        _, current = _get(server, f"/api/v1/jobs/{job['id']}")
        if current['status'] in ('done', 'failed', 'cancelled'):
            break
        time.sleep(0.1)
    assert current['status'] == 'done', current
    status, _ = _post(server, '/api/v1/teach', {'text': 'The calibration constant is 0.98.'})
    assert status == 200
    status, result = _post(server, '/api/v1/native/memory-generate',
                           {'job_id': job['id'], 'prompt': 'What is the calibration constant?', 'max_tokens': 3})
    assert status == 200, result
    assert result['memory_used'] is True and result['model_weights_updated'] is False
    assert result['evidence'][0]['text'] == 'The calibration constant is 0.98.'
    assert result['evidence'][0]['source'] == 'user_text'
    status, result = _post(server, '/api/v1/native/memory-generate',
                           {'job_id': job['id'], 'prompt': 'hello', 'max_tokens': True})
    assert status == 400
