"""Headless runner: JSON in, JSON out, core dependencies only (surfaces 1.5)."""

import io
import json
import sys

import pytest

from recamem.runner import RunnerError, main, run_loop, run_once


@pytest.fixture()
def engine(tmp_path):
    from recamem.manager import DroidManager

    manager = DroidManager(base_dir=str(tmp_path / "home"))
    return manager.get_droid("demo")


def test_run_once_reports_answer_memory_and_anchor(engine):
    engine.teach("The calibration constant is 0.98.")
    payload = run_once("What is the calibration constant?", engine, profile="demo")
    assert payload["prompt"].startswith("What is")
    assert payload["profile"] == "demo"
    assert payload["anchor"] == "mock"
    assert payload["response"]
    assert payload["memory"], "recall evidence must be attached to every answer"
    assert set(payload["memory"][0]) >= {"text", "source"}
    assert json.loads(json.dumps(payload))  # JSON-serialisable


def test_run_once_teach_mode_stores_facts(engine):
    payload = run_once("The gateway forwards readings every minute.", engine, mode="teach")
    assert payload["facts_stored"] >= 1
    assert payload["memory"] == []


def test_run_once_rejects_empty_prompt(engine):
    with pytest.raises(RunnerError):
        run_once("   ", engine)
    with pytest.raises(RunnerError):
        run_once("hi", engine, mode="nope")


def test_run_loop_emits_one_json_line_per_prompt(engine):
    engine.teach("Station 4 recorded a moisture reading of 0.4.")
    stdin = io.StringIO("hello\n\nstation 4?\n")
    stdout = io.StringIO()
    count = run_loop(engine, stdin, stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert count == 2
    assert [line["prompt"] for line in lines] == ["hello", "station 4?"]


def test_runner_does_not_need_streamlit_or_node(engine, monkeypatch):
    run_once("hello", engine)
    assert "streamlit" not in sys.modules


def test_main_prompt_mode_prints_one_json_object(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path / "home"))
    assert main(["--prompt", "hello", "--name", "demo"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    payload = json.loads(out[0])
    assert payload["prompt"] == "hello"
    assert "response" in payload


def test_main_stdin_loop_exits_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("a\nb\n"))
    assert main(["--name", "demo"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_main_without_input_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path / "home"))

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    assert main(["--name", "demo"]) == 1
    assert "nothing to do" in capsys.readouterr().err
