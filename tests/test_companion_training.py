"""Training jobs are real subprocesses with real output and provenance."""
from __future__ import annotations

import json
import time
from pathlib import Path

from recamem import companion

ROOT = Path(__file__).resolve().parents[1]


def _wait(events_fn, predicate, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        events = events_fn()
        if predicate(events):
            return events
        time.sleep(0.2)
    return events_fn()


def test_training_job_runs_and_writes_provenance(tmp_path):
    updates = {}
    runner = companion.TrainingJobRunner(
        ROOT, lambda job_id, **fields: updates.update(fields), home=tmp_path)
    runner.start("job-1", {"lm_tokens": 1000, "d_model": 64, "n_layers": 2,
                          "d_ff": 128, "vocab_size": 1024, "seq_len": 64,
                          "batch_size": 2, "grad_accum": 1,
                          "checkpoint_every": 2, "seed": 1337})
    events = _wait(lambda: runner.events("job-1"),
                   lambda rows: any(row["type"] == "finished" for row in rows))
    assert any(row["type"] == "log" and "step=" in row["line"] for row in events), \
        "no real training log lines were captured"
    assert any(row["type"] == "notice" and "fixture" in row["line"] for row in events)
    assert updates.get("status") == "done", updates
    out_dir = Path(str(updates["result"]["out_dir"]))
    assert (out_dir / "report.json").is_file()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["dataset"] == "fixture"
    assert isinstance(manifest["held_out_bpb"], (int, float))
    assert manifest["commit"] and manifest["command"]
    assert "G2" in manifest["gates"]
    assert "HF_TOKEN" not in manifest["command"]


def test_unsupported_training_params_are_rejected(tmp_path):
    runner = companion.TrainingJobRunner(ROOT, lambda *_a, **_k: None, home=tmp_path)
    try:
        runner.start("job-2", {"lm_tokens": 1000, "drop_all_tables": True})
    except companion.CompanionActionError as exc:
        assert "drop_all_tables" in str(exc)
    else:
        raise AssertionError("unsupported parameters must be rejected")


def test_cancel_terminates_a_running_job(tmp_path):
    runner = companion.TrainingJobRunner(
        ROOT, lambda job_id, **fields: None, home=tmp_path)
    runner.start("job-3", {"lm_tokens": 2000000, "d_model": 64, "n_layers": 6,
                          "vocab_size": 1024, "seq_len": 64, "batch_size": 2})
    time.sleep(1.0)
    assert runner.cancel("job-3") is True
    events = _wait(lambda: runner.events("job-3"),
                   lambda rows: any(row["type"] == "cancelled" for row in rows), timeout=20)
    assert any(row["type"] == "cancelled" for row in events)
