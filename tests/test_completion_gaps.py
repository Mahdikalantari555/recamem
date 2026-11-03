"""Contract tests for the completion gap inventory guard (stdlib only)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "scripts" / "check_completion_gaps.py"


def _guard():
    spec = importlib.util.spec_from_file_location("check_completion_gaps", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(rid: str, status: str = "closed", **overrides):
    row = {
        "id": rid,
        "source": f"source#{rid}",
        "owner": "finish-implementation-and-close-gaps#1.1",
        "tier": "offline",
        "status": status,
        "verify_command": "PYTHONPATH=. pytest tests/test_completion_gaps.py -q",
        "artifact": "reports/evidence/fixture.json",
        "tests": ["tests/test_completion_gaps.py::test_guard_accepts_complete_inventory"],
        "evidence": ["reports/evidence/fixture.json"],
        "reason": "fixture row",
    }
    row.update(overrides)
    return row


def _fixture_root(tmp_path: Path) -> Path:
    """A minimal repository-shaped root with every expected source satisfied."""
    guard = _guard()
    (tmp_path / "reports" / "evidence").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "openspec" / "changes").mkdir(parents=True)
    (tmp_path / "openspec" / "changes" / "archive" / "2026-10-28-post-merge-completion").mkdir(parents=True)
    (tmp_path / "reports" / "completion-evidence.json").write_text('{"entries": []}')
    (tmp_path / "docs" / "release-matrix.md").write_text("# matrix\n")
    (tmp_path / "docs" / "surfaces.md").write_text("# surfaces\n")
    for change in guard.ACTIVE_CHANGES:
        (tmp_path / "openspec" / "changes" / change).mkdir(parents=True)
        (tmp_path / "openspec" / "changes" / change / "tasks.md").write_text("# tasks\n")
    (tmp_path / "openspec" / "changes" / "archive" / "2026-10-28-post-merge-completion"
     / "tasks.md").write_text("# tasks\n")
    (tmp_path / "reports" / "evidence" / "fixture.json").write_text(
        json.dumps({"outcome": "passed", "tier": "offline"})
    )
    test_file = tmp_path / "tests" / "test_completion_gaps.py"
    test_file.write_text(ROOT.joinpath("tests/test_completion_gaps.py").read_text())
    return tmp_path


def test_guard_accepts_complete_inventory(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    doc = {"schema_version": 1, "rows": [_row("G-1")]}
    assert guard.validate(doc, root, require_closed=True) == []


def test_guard_rejects_closed_row_without_passing_evidence(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    (root / "reports" / "evidence" / "fixture.json").write_text(
        json.dumps({"outcome": "blocked", "skip_reason": "no weights"})
    )
    doc = {"schema_version": 1, "rows": [_row("G-1")]}
    errors = guard.validate(doc, root, require_closed=True)
    assert any("evidence not passing" in error for error in errors)


def test_guard_rejects_closed_row_without_artifact_or_test(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    doc = {"schema_version": 1, "rows": [
        _row("G-1", artifact="reports/evidence/missing.json", tests=[]),
    ]}
    errors = guard.validate(doc, root, require_closed=True)
    assert any("artifact missing" in error for error in errors)
    assert any("needs at least one test" in error for error in errors)


def test_guard_rejects_duplicate_ids(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    doc = {"schema_version": 1, "rows": [_row("G-1"), _row("G-1")]}
    errors = guard.validate(doc, root, require_closed=False)
    assert any("duplicate id" in error for error in errors)


def test_guard_reports_open_rows_in_require_closed_mode(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    doc = {"schema_version": 1, "rows": [
        _row("G-1"),
        _row("G-2", status="blocked", reason="needs real weights", tier="real-model"),
    ]}
    assert guard.validate(doc, root, require_closed=False) == []
    errors = guard.validate(doc, root, require_closed=True)
    assert any("1 gap(s) are not closed" in error for error in errors)


def test_guard_rejects_unknown_status_and_tier(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    doc = {"schema_version": 1, "rows": [
        _row("G-1", status="done", tier="made-up"),
    ]}
    errors = guard.validate(doc, root, require_closed=False)
    assert any("invalid status" in error for error in errors)
    assert any("invalid tier" in error for error in errors)


def test_guard_detects_uncovered_sources(tmp_path):
    guard = _guard()
    root = _fixture_root(tmp_path)
    (root / "reports" / "completion-evidence.json").write_text(json.dumps({
        "entries": [{"id": "some/change/tasks.md#9.9", "status": "blocked"}]
    }))
    doc = {"schema_version": 1, "rows": [_row("G-1")]}
    errors = guard.validate(doc, root, require_closed=False)
    assert any("inventory does not cover" in error for error in errors)


def test_real_inventory_is_structurally_valid():
    """The shipped inventory must parse and cover its sources; completion is not claimed."""
    guard = _guard()
    doc = json.loads((ROOT / "reports" / "completion-gaps.json").read_text())
    assert guard.validate(doc, ROOT, require_closed=False) == []


def test_schema_only_cli_exits_zero_on_the_real_inventory():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(GUARD_PATH), "--schema-only"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completion is NOT claimed" in result.stdout
