"""`recamem run` / `ui` / `serve` wiring (surfaces 1.3, 5.3, 6.1, 10.1)."""

import json
import subprocess
import sys


def run_cli(args, home, extra_env=None):
    import os

    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": ".",
        "RECAMEM_HOME": str(home),
        "RECAMEM_ANCHOR": "mock",
        "RECAMEM_API_LOG": "0",
    }
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "recamem.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_run_prompt_prints_one_json_object(tmp_path):
    result = run_cli(["run", "--prompt", "hello", "--name", "demo"], tmp_path / "home")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["prompt"] == "hello"
    assert payload["anchor"] == "mock"
    assert "response" in payload


def test_run_reads_stdin_lines(tmp_path):
    import os

    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": ".",
        "RECAMEM_HOME": str(tmp_path / "home"),
        "RECAMEM_ANCHOR": "mock",
    }
    result = subprocess.run(
        [sys.executable, "-m", "recamem.cli", "run", "--name", "demo"],
        input="a\nb\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 2


def test_ui_dry_run_reports_the_resolved_shell(tmp_path):
    result = run_cli(["ui", "--dry-run"], tmp_path / "home")
    assert result.returncode == 0
    assert "shell=" in result.stdout
    assert "url=http://127.0.0.1:8000/" in result.stdout


def test_ui_without_a_build_explains_the_build_step(tmp_path):
    result = run_cli(["ui", "--dry-run"], tmp_path / "home")
    assert "dist=" in result.stdout
    # The real launch path refuses to pretend when nothing is built.
    assert "exists=" in result.stdout


def test_serve_requires_a_transport(tmp_path):
    result = run_cli(["serve"], tmp_path / "home")
    assert result.returncode == 1
    assert "Must specify --mcp, --rest or --web" in result.stderr


def test_help_lists_the_new_surfaces(tmp_path):
    result = run_cli(["--help"], tmp_path / "home")
    assert result.returncode == 0
    for name in ("run", "ui", "serve", "doctor"):
        assert name in result.stdout
