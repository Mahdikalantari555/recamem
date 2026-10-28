"""Console-dependent lifecycle actions, exercised through the same engine APIs.

Streamlit is an optional extra (it is not installed in the offline core
environment), so the widget layer is checked structurally and every action the
console triggers is executed here through `DroidEngine`/`DroidManager`. Each
flow ends with a save + reload, proving the action survives a restart.
"""

import json
import re
import time
from pathlib import Path

import pytest

from src.model.droid_manager import DroidManager

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "recamem" / "console_legacy.py"


@pytest.fixture()
def manager(tmp_path):
    mgr = DroidManager(base_dir=str(tmp_path / "droids"))
    yield mgr
    mgr.shutdown()


def _reload(manager, name="lifecycle"):
    """Force a cold load from disk instead of reusing the cached instance."""
    manager.cached_instances.pop(name, None)
    return manager.get_droid(name)


# --------------------------------------------------------------------------- #
# Console wiring
# --------------------------------------------------------------------------- #
def test_app_calls_every_lifecycle_action():
    source = APP_PY.read_text(encoding="utf-8")
    for call in (
        "active_droid.teach(",
        "active_droid.knowledge.update_fact(",
        "active_droid.knowledge.supersede(",
        "active_droid.knowledge.mark_superseded(",
        "active_droid.approve(",
        "active_droid.reject(",
        "active_droid.knowledge.retrieval_health(",
        "snapshot_mgr.rollback(",
        "snapshot_mgr.branch_from(",
        "start_continuous_stream",
        "pause_stream",
        "resume_stream",
    ):
        assert call in source, f"app.py never calls {call}"


def test_app_rollback_reports_the_pre_rollback_snapshot():
    source = APP_PY.read_text(encoding="utf-8")
    assert re.search(r"report\[.pre_rollback_version.\]", source)
    # The console must not hand-roll its own restore path any more.
    assert not re.search(r"snapshot_mgr\.restore_snapshot\(\s*active_droid", source)


# --------------------------------------------------------------------------- #
# End-to-end lifecycle through the manager
# --------------------------------------------------------------------------- #
def test_teach_edit_supersede_and_feedback_survive_restart(manager):
    engine = manager.create_droid("lifecycle")
    engine.teach("The mixer is on site B.")
    engine.teach("The crane is inspected weekly.")
    ids = [row[0] for row in engine.knowledge.conn.execute(
        "SELECT id FROM facts ORDER BY id")]
    assert len(ids) == 2

    # Edit (console: "Save Edit").
    assert engine.knowledge.update_fact(ids[0], "The mixer is on site C.") == 1

    # Supersede with a real successor (console: replacement text).
    emb = engine.anchor.embed("The crane is inspected monthly.")
    if emb.dim() > 1:
        emb = emb.squeeze(0)
    successor = engine.knowledge.insert_fact(
        "The crane is inspected monthly.", "memory_console_supersede",
        time.time(), engine.step_count, 1.0, emb,
    )
    assert engine.knowledge.supersede(ids[1], successor) == 1
    assert engine.knowledge.get_fact(ids[1])["superseded_by"] == successor

    # Feedback (console: Approve/Reject buttons).
    engine.retrieval_policy.epsilon = 0.0
    engine.recall("mixer", top_k=2)
    approved = engine.approve(ids[0])
    assert approved["status"] == "approved"
    assert approved["strategy"] in engine.retrieval_policy.STRATEGIES
    assert approved["health"]["available"] is False  # no labelled ranking yet
    rejected = engine.reject(successor)
    assert rejected["status"] == "rejected"

    manager.save_droid("lifecycle")

    reloaded = _reload(manager)
    assert reloaded.knowledge.get_fact(ids[0])["text"] == "The mixer is on site C."
    assert reloaded.knowledge.get_fact(ids[1])["superseded_by"] == successor
    assert reloaded.knowledge.get_fact(successor)["needs_review"] == 1
    actions = [row["action"] for row in reloaded.knowledge.get_feedback_log()]
    assert "approve" in actions and "reject" in actions
    assert reloaded.knowledge.retrieval_health()["available"] is False


def test_stream_control_uses_the_engine_api(manager):
    first = manager.create_droid("first")
    second = manager.create_droid("second")
    assert first.start_continuous_stream()["status"] == "started"
    # A Streamlit rerun calls start again; the worker is reused, not doubled.
    assert first.start_continuous_stream()["status"] == "already_running"
    assert first.pause_stream()["status"] == "paused"
    assert first.resume_stream()["status"] == "resumed"

    manager.activate_droid("second")
    assert first.get_stream_status()["alive"] is False
    assert manager.stop_continuous_stream("first")["status"] == "not_running"


def test_snapshot_and_rollback_survive_restart(manager):
    engine = manager.create_droid("lifecycle")
    engine.teach("The mixer is on site B.")
    manager.create_version("lifecycle", "v1")
    engine.teach("The crane is on site C.")
    manager.save_droid("lifecycle")
    assert len(engine.knowledge.get_audit_trail()) == 2

    reloaded = _reload(manager)
    report = manager.rollback_version("lifecycle", "v1")
    assert report["restored"] is True
    assert reloaded.knowledge.active_count() == 1
    assert reloaded.knowledge.get_audit_trail()[0]["text"].startswith("The mixer")

    # Rollback of rollback, after another restart.
    manager.save_droid("lifecycle")
    again = _reload(manager)
    back = manager.rollback_version("lifecycle", report["pre_rollback_version"])
    assert back["restored"] is True
    assert again.knowledge.active_count() == 2


def test_sleep_and_snapshot_actions_coexist(manager):
    engine = manager.create_droid("lifecycle")
    for i in range(4):
        engine.teach(f"Site fact {i} about the pump")
    engine.consolidate_memory(max_prune_ratio=0.25)
    engine.create_snapshot("after-consolidation")
    profile = engine.snapshot_manager.list_snapshots(engine)
    assert any(meta["version_id"] == "after-consolidation" for meta in profile)
