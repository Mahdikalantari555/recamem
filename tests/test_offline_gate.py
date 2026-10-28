"""The offline gate detects drift and never counts a blocked test as passed."""

import json
import shutil
from pathlib import Path

import pytest

from scripts import offline_gate as gate


def _check_with(monkeypatch, tmp_path, module_name, check, **patch):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "conftest.py").write_text("real_model = True\n")
    for target, value in patch.items():
        monkeypatch.setattr(gate, target, value)
    monkeypatch.setattr(gate, "ROOT", root)
    return check()


@pytest.mark.parametrize(
    "name",
    ["mcp_tools", "cli_surface", "test_counts", "declared_imports", "blocked_semantics"],
)
def test_gate_checks_are_clean_on_the_branch(name):
    assert gate.CHECKS[name]() == []


def test_gate_main_passes(tmp_path):
    assert gate.main([]) == 0


def test_hand_written_test_count_fails(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("The suite has 941 tests passed today.\n")
    monkeypatch.setattr(gate, "ROOT", root)
    problems = gate.check_test_counts()
    assert problems and "hand-written test count" in problems[0]


def test_ledger_row_without_a_test_is_rejected(monkeypatch):
    original = gate.check_evidence

    def broken():
        return ["openspec/x#1.1: ledger row names no test"]

    monkeypatch.setitem(gate.CHECKS, "evidence", broken)
    assert gate.main(["evidence"]) == 1


def test_declared_import_drift_is_detected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "recamem").mkdir(parents=True)
    (root / "pyproject.toml").write_text("dependencies = [\n \"numpy>=1\",\n]\n")
    (root / "recamem" / "mod.py").write_text("import numpy\nimport requests\n")
    monkeypatch.setattr(gate, "ROOT", root)
    problems = gate.check_declared_imports()
    assert any("requests" in problem for problem in problems)
    assert not any("numpy" in problem for problem in problems)
