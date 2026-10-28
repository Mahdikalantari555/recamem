"""The `recamem` command: new subcommands, exit codes, parseable output."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args, home, extra_env=None):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "RECAMEM_HOME": str(home),
        "RECAMEM_ANCHOR": "mock",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "recamem.cli", *args],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )


def test_doctor_reports_mock_and_exits_nonzero(tmp_path):
    result = run_cli(["doctor", "--mock"], tmp_path)
    assert result.returncode == 1
    assert "anchor=mock" in result.stdout
    assert "item=anchor ready=no" in result.stdout
    assert "network" in result.stdout and "offline" in result.stdout


def test_migrate_is_repeatable_and_reports_the_record(tmp_path):
    source = tmp_path / "legacy" / "droids" / "agent-a"
    source.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"name": "agent-a"}))
    first = run_cli(["migrate", str(tmp_path / "legacy" / "droids")], tmp_path / "home")
    assert first.returncode == 0, first.stderr
    assert "profile=agent-a status=migrated" in first.stdout
    assert f"record={tmp_path / 'home' / 'agents' / 'migration.json'}" in first.stdout
    second = run_cli(["migrate", str(tmp_path / "legacy" / "droids")], tmp_path / "home")
    assert "profile=agent-a status=already_migrated" in second.stdout


def test_ingest_then_recall_round_trip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "lab.md").write_text("# Lab\n\nThe calibration constant is 0.98.\n")
    ingest = run_cli(["ingest", str(docs), "--name", "agent-rs", "--save"], tmp_path / "home")
    assert ingest.returncode == 0, ingest.stderr
    assert "facts_stored=1" in ingest.stdout
    assert not any(line.startswith(("recall_at", "precision_at", "accuracy=")) for line in ingest.stdout.splitlines())

    recall = run_cli(["recall", "What is the calibration constant?", "--name", "agent-rs"], tmp_path / "home")
    assert recall.returncode == 0, recall.stderr
    assert "results=1" in recall.stdout
    assert "source=" in recall.stdout
    assert "calibration constant" in recall.stdout


def test_ingest_unchanged_is_a_no_op(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("The gateway is on site C.\n")
    run_cli(["ingest", str(docs), "--name", "agent-b", "--save"], tmp_path / "home")
    second = run_cli(["ingest", str(docs), "--name", "agent-b"], tmp_path / "home")
    assert "facts_stored=0" in second.stdout
    assert "files_unchanged=1" in second.stdout


def test_rtu_missing_checkpoint_exits_one(tmp_path):
    result = run_cli(["rtu", "--checkpoint", str(tmp_path / "absent.safetensors")], tmp_path / "home")
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_invalid_arguments_exit_two(tmp_path):
    result = run_cli(["recall"], tmp_path / "home")
    assert result.returncode == 2


def test_help_lists_every_command(tmp_path):
    result = run_cli(["--help"], tmp_path)
    assert result.returncode == 0
    for command in ("ingest", "recall", "doctor", "migrate", "rtu", "chat", "serve"):
        assert command in result.stdout
