"""Persistent snapshots, staged rollback and power-law forgetting (4.4, 4.5, 4.7).

The snapshots here are file-backed: the database is copied with the SQLite
backup API (so uncheckpointed WAL frames travel with it), tensors and policy
state are separate artifacts, and the manifest is written last.
"""

import json
import sqlite3
import time
from pathlib import Path

import pytest
import torch

from src.model.droid import DroidEngine
from src.model.knowledge_store import KnowledgeStore
from src.model.memory_lifecycle import (
    SNAPSHOT_CONFIG,
    SNAPSHOT_DB,
    SNAPSHOT_MANIFEST,
    SNAPSHOT_META,
    SNAPSHOT_POLICY,
    SNAPSHOT_TENSORS,
    RetrievalPolicy,
    SnapshotManager,
    SnapshotValidationError,
)


def _engine(tmp_path, name="snap"):
    return DroidEngine(name=name, db_path=str(tmp_path / f"{name}.db"))


def _facts(engine):
    return [row[0] for row in engine.knowledge.conn.execute(
        "SELECT id FROM facts ORDER BY id")]


# --------------------------------------------------------------------------- #
# 4.4 Complete snapshots
# --------------------------------------------------------------------------- #
def test_snapshot_writes_every_artifact_and_manifest_last(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "v1")

    directory = Path(snapshot["path"])
    for name in (SNAPSHOT_DB, SNAPSHOT_TENSORS, SNAPSHOT_META,
                 SNAPSHOT_POLICY, SNAPSHOT_CONFIG, SNAPSHOT_MANIFEST):
        assert (directory / name).is_file(), name

    manifest = json.loads((directory / SNAPSHOT_MANIFEST).read_text())
    assert manifest["format"] == 2
    assert manifest["version_id"] == "v1"
    assert set(manifest["artifacts"]) == {
        SNAPSHOT_DB, SNAPSHOT_TENSORS, SNAPSHOT_META, SNAPSHOT_POLICY, SNAPSHOT_CONFIG
    }
    for info in manifest["artifacts"].values():
        assert len(info["sha256"]) == 64
        assert info["size"] > 0
    assert manifest["database"]["integrity"] == "ok"
    assert "facts" in manifest["database"]["tables"]

    policy = json.loads((directory / SNAPSHOT_POLICY).read_text())
    assert policy["classifier_version"] == RetrievalPolicy.CLASSIFIER_VERSION
    config = json.loads((directory / SNAPSHOT_CONFIG).read_text())
    assert config["adapter"] == engine.adapter_name
    engine.knowledge.close()


def test_snapshot_includes_uncheckpointed_wal_frames(tmp_path):
    engine = _engine(tmp_path)
    assert engine.knowledge.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    engine.teach("The mixer is on site B")
    engine.teach("The crane is on site C")
    live_ids = _facts(engine)
    assert len(live_ids) == 2

    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "wal")

    with sqlite3.connect(str(Path(snapshot["path"]) / SNAPSHOT_DB)) as conn:
        rows = [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id")]
    assert rows == live_ids
    engine.knowledge.close()


def test_restore_brings_back_database_tensors_and_policy(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    original_ids = _facts(engine)
    engine.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    engine.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    before = {k: v.clone() for k, v in engine.memory.state_dict().items()
              if isinstance(v, torch.Tensor)}
    policy_before = engine.retrieval_policy.select_strategy("specific_fact")

    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "v1")

    # Diverge every captured dimension.
    engine.teach("The crane is on site C")
    engine.teach("The pump is on site D")
    engine.retrieval_policy.update("specific_fact", "semantic_primary", reward=1.0)
    engine.retrieval_policy.update("specific_fact", "semantic_primary", reward=1.0)
    assert len(_facts(engine)) > len(original_ids)

    assert manager.restore_snapshot(engine, "v1") is True, manager.last_error
    assert _facts(engine) == original_ids
    assert engine.retrieval_policy.select_strategy("specific_fact") == policy_before
    for key, value in before.items():
        torch.testing.assert_close(engine.memory.state_dict()[key], value)
    # Store caches were invalidated with the database swap.
    assert engine.knowledge.active_count() == 1
    engine.knowledge.close()


def test_snapshot_survives_restart_and_lists_from_disk(tmp_path):
    engine = _engine(tmp_path / "a")
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "persisted")
    engine.knowledge.close()

    fresh = SnapshotManager(str(tmp_path / "snapshots"))
    listed = fresh.list_snapshots()
    assert [meta["version_id"] for meta in listed] == ["persisted"]
    assert listed[0]["database"]["fact_rows"] == 1

    reopened = _engine(tmp_path / "a")
    assert fresh.restore_snapshot(reopened, "persisted") is True, fresh.last_error
    assert len(_facts(reopened)) == 1
    reopened.knowledge.close()


def test_snapshots_are_not_aliased(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    first = manager.create_snapshot(engine, "one")

    engine.teach("The crane is on site C")
    second = manager.create_snapshot(engine, "two")
    assert first["path"] != second["path"]
    first_db = Path(first["path"]) / SNAPSHOT_DB
    second_db = Path(second["path"]) / SNAPSHOT_DB
    assert first_db.resolve() != second_db.resolve()

    # Mutating the live engine cannot reach either snapshot.
    engine.teach("The pump is on site D")
    with sqlite3.connect(str(first_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    with sqlite3.connect(str(second_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2
    engine.knowledge.close()


def test_branch_copies_artifacts_and_is_independent(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "base")
    branch = manager.branch_from("base", "branch", engine)

    assert branch["metadata"]["parent_version"] == "base"
    assert Path(branch["path"]).resolve() != Path(
        manager.snapshots["base"]["metadata"]["path"]).resolve()
    branch_manifest = json.loads((Path(branch["path"]) / SNAPSHOT_MANIFEST).read_text())
    assert branch_manifest["version_id"] == "branch"

    (Path(branch["path"]) / SNAPSHOT_POLICY).write_text("{}")
    base_policy = json.loads(
        (Path(manager.snapshots["base"]["metadata"]["path"]) / SNAPSHOT_POLICY).read_text()
    )
    assert base_policy["classifier_version"] == RetrievalPolicy.CLASSIFIER_VERSION
    engine.knowledge.close()


def test_snapshot_ids_cannot_escape_the_snapshot_root(tmp_path):
    engine = _engine(tmp_path)
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "../../escape")
    assert Path(snapshot["path"]).parent == (tmp_path / "snapshots").resolve()
    engine.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.5 Staged restore and reversible rollback
# --------------------------------------------------------------------------- #
def test_corrupt_database_artifact_is_rejected_without_state_change(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "v1")
    engine.teach("The crane is on site C")
    live = _facts(engine)

    db_path = Path(snapshot["path"]) / SNAPSHOT_DB
    db_path.write_bytes(b"not a sqlite database")
    assert manager.restore_snapshot(engine, "v1") is False
    assert "checksum" in (manager.last_error or "")
    assert _facts(engine) == live
    engine.knowledge.close()


def test_missing_artifact_is_rejected(tmp_path):
    engine = _engine(tmp_path)
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "v1")
    (Path(snapshot["path"]) / SNAPSHOT_TENSORS).unlink()
    assert manager.restore_snapshot(engine, "v1") is False
    assert "missing" in (manager.last_error or "")
    engine.knowledge.close()


def test_incompatible_tensor_shape_is_rejected(tmp_path):
    engine = _engine(tmp_path)
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "v1")

    class WrongDimMemory:
        """Static state with a different embedding width."""

        def __init__(self):
            self.trace = torch.zeros(64)

        def state_dict(self):
            return {"trace": self.trace.clone()}

        def load_state_dict(self, state):
            incoming = state.get("trace")
            if incoming is None or incoming.shape != self.trace.shape:
                raise RuntimeError(
                    "size mismatch for trace: "
                    f"{tuple(incoming.shape) if incoming is not None else 'missing'} vs "
                    f"{tuple(self.trace.shape)}"
                )
            self.trace = incoming.clone()

    import types

    other = types.SimpleNamespace(
        memory=WrongDimMemory(),
        retrieval_policy=None,
        step_count=0,
        current_version=None,
    )
    assert manager.restore_snapshot(other, "v1") is False
    assert "size mismatch" in (manager.last_error or "")
    torch.testing.assert_close(other.memory.trace, torch.zeros(64))
    engine.knowledge.close()


def test_grown_capacity_buffer_is_still_restorable(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "v1")
    engine.teach("The crane is on site C")
    engine.teach("The pump is on site D")
    assert manager.restore_snapshot(engine, "v1") is True, manager.last_error
    assert len(_facts(engine)) == 1
    engine.knowledge.close()


def test_rollback_takes_a_pre_rollback_snapshot_and_is_reversible(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    manager.create_snapshot(engine, "v1")
    engine.teach("The crane is on site C")
    after_v1 = _facts(engine)

    report = manager.rollback(engine, "v1")
    assert report["restored"] is True
    pre = report["pre_rollback_version"]
    assert pre and pre != "v1"
    assert len(_facts(engine)) == 1

    # Rollback of rollback: back to the newer state, from disk.
    back = manager.rollback(engine, pre)
    assert back["restored"] is True
    assert _facts(engine) == after_v1
    engine.knowledge.close()


def test_engine_rollback_uses_shared_staged_path(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    engine.create_snapshot("v1")
    engine.teach("The crane is on site C")
    before = _facts(engine)

    report = engine.rollback("v1")
    assert report["restored"] is True
    assert len(_facts(engine)) == 1
    assert engine.current_version.startswith("v1_branch_")

    restored_back = engine.rollback(report["pre_rollback_version"])
    assert restored_back["restored"] is True
    assert _facts(engine) == before
    engine.knowledge.close()


def test_failed_rollback_leaves_target_untouched(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "v1")
    engine.teach("The crane is on site C")
    live = _facts(engine)
    (Path(snapshot["path"]) / SNAPSHOT_DB).write_bytes(b"broken")

    report = manager.rollback(engine, "v1")
    assert report["restored"] is False
    assert report["error"]
    assert _facts(engine) == live
    # The pre-rollback snapshot still exists, so the failed attempt is auditable.
    assert report["pre_rollback_version"] in manager.snapshots
    engine.knowledge.close()


def test_rollback_of_unknown_version_reports_without_writing(tmp_path):
    engine = _engine(tmp_path)
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    report = manager.rollback(engine, "nope")
    assert report == {
        "restored": False,
        "version_id": "nope",
        "pre_rollback_version": None,
        "error": "snapshot 'nope' not found",
    }
    engine.knowledge.close()


def test_validation_error_is_a_named_exception(tmp_path):
    engine = _engine(tmp_path)
    manager = SnapshotManager(str(tmp_path / "snapshots"))
    snapshot = manager.create_snapshot(engine, "v1")
    (Path(snapshot["path"]) / SNAPSHOT_META).write_text("{not json")
    with pytest.raises(SnapshotValidationError):
        manager._stage(engine, snapshot)
    engine.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.7 Power-law accessibility and bounded reinforcement
# --------------------------------------------------------------------------- #
def test_accessibility_matches_the_power_law(store_factory=None):
    store = KnowledgeStore(":memory:", dim=4)
    fact = store.insert_fact("alpha", "teach", 0.0, 1, 1.0, torch.tensor([1.0, 0, 0, 0]))
    last = 1_000_000.0
    store.conn.execute(
        "UPDATE facts SET last_retrieved=?, access_count=2, decay_beta=1.0 WHERE id=?",
        (last, fact),
    )
    store.conn.commit()

    beta = store.decay_exponent(fact)
    assert beta == pytest.approx(1.0)
    for age in (0.0, 86_400.0, 10 * 86_400.0):
        expected = 2.0 * (1.0 + (1.0 / 86_400.0) * age) ** (-beta)
        assert store.accessibility(fact, now=last + age) == pytest.approx(expected)
    store.close()


def test_reinforcement_decreases_beta_within_bounds():
    store = KnowledgeStore(":memory:", dim=4)
    fact = store.insert_fact("alpha", "teach", 0.0, 1, 1.0, torch.tensor([1.0, 0, 0, 0]))
    before = store.decay_exponent(fact)
    result = store.reinforce(fact, now=10.0)
    assert result["beta_after"] == pytest.approx(before * (1.0 - 0.2))
    assert 0.1 <= result["delta"] <= 0.3
    slower = store.accessibility(fact, now=10.0 + 5 * 86_400.0)

    for _ in range(50):
        store.reinforce(fact, now=10.0)
    floor = store.decay_exponent(fact)
    assert floor == pytest.approx(store.DECAY_BETA_MIN)
    assert floor > 0.0
    assert slower > 0.0
    for _ in range(100):
        store.reinforce(fact, now=10.0)
    assert store.decay_exponent(fact) == pytest.approx(store.DECAY_BETA_MIN)
    store.close()


def test_never_retrieved_fact_follows_base_decay():
    store = KnowledgeStore(":memory:", dim=4)
    fact = store.insert_fact("alpha", "teach", 0.0, 1, 1.0, torch.tensor([1.0, 0, 0, 0]))
    store.conn.execute(
        "UPDATE facts SET last_retrieved=0.0, access_count=1, decay_beta=1.0 WHERE id=?",
        (fact,),
    )
    store.conn.commit()
    assert store.accessibility(fact, now=86_400.0) == pytest.approx(0.5)
    assert store.accessibility(fact, now=0.0) == pytest.approx(1.0)
    store.close()


def test_decay_state_survives_restart(tmp_path):
    path = tmp_path / "decay.db"
    store = KnowledgeStore(str(path), dim=4)
    fact = store.insert_fact("alpha", "teach", 0.0, 1, 1.0, torch.tensor([1.0, 0, 0, 0]))
    store.reinforce(fact, now=50.0)
    store.reinforce(fact, now=60.0)
    probe = 60.0 + 3 * 86_400.0
    reference = store.accessibility(fact, now=probe)
    beta = store.decay_exponent(fact)
    store.close()

    reopened = KnowledgeStore(str(path), dim=4)
    assert reopened.decay_exponent(fact) == pytest.approx(beta)
    assert reopened.accessibility(fact, now=probe) == pytest.approx(reference)
    reopened.close()
