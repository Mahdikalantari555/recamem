"""Profile integrity, merge lifecycle, consolidation and stream ownership.

Covers post-merge tasks 2.7-2.9, 4.6, 4.8 and 4.11: versioned non-tensor
metadata, transactional profile loads, reversible migrations, merge refusals
without writes, measured consolidation and single-owner workers.
"""

import json
import threading
import time
from pathlib import Path

import pytest
import torch

from src.model.droid import DroidEngine
from src.model.droid_manager import DroidManager
from src.model.memory_lifecycle import ContinuousStream
from src.model.rtu import InvalidProfileError


def _engine(tmp_path, name="profile", **kwargs):
    return DroidEngine(name=name, db_path=str(tmp_path / f"{name}.db"), **kwargs)


# --------------------------------------------------------------------------- #
# 2.7 Versioned non-tensor metadata
# --------------------------------------------------------------------------- #
def test_capacity_eviction_and_policy_survive_restart(tmp_path):
    engine = _engine(tmp_path, K_max=3)
    for i in range(5):
        engine.teach(f"Fact number {i} about the site")
    engine.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    engine.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    evictions = int(getattr(engine.memory, "eviction_count", 0))
    registry = len(getattr(engine.memory, "_key_registry", []))
    schedule = engine.lr_schedule.state_dict()
    engine.save_profile(str(tmp_path / "profile"))

    meta = json.loads((tmp_path / "profile" / "adapter_meta.json").read_text())
    assert meta["adapter_name"] == engine.adapter_name
    assert "eviction_count" in meta
    assert "K_max" in meta

    reloaded = _engine(tmp_path, name="reloaded")
    reloaded.load_profile(str(tmp_path / "profile"))
    assert reloaded.K_max == 3
    assert int(getattr(reloaded.memory, "eviction_count", 0)) == evictions
    assert len(getattr(reloaded.memory, "_key_registry", [])) == registry
    assert reloaded.lr_schedule.state_dict() == schedule
    assert reloaded.retrieval_policy.select_strategy("specific_fact") == "episodic_only"

    # Behavioural parity: the next update behaves like the uninterrupted engine.
    reference = engine
    reference.teach("A further fact about the crane")
    reloaded.teach("A further fact about the crane")
    assert reloaded.knowledge.active_count() == reference.knowledge.active_count()
    for eng in (engine, reloaded, reference):
        eng.knowledge.close()


def test_legacy_profile_without_sidecar_still_loads(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    engine.save_profile(str(tmp_path / "profile"))
    (tmp_path / "profile" / "adapter_meta.json").unlink()

    reloaded = _engine(tmp_path, name="reloaded")
    reloaded.load_profile(str(tmp_path / "profile"))
    assert reloaded.knowledge.active_count() == 1
    reloaded.knowledge.close()
    engine.knowledge.close()


# --------------------------------------------------------------------------- #
# 2.8 Transactional profile load
# --------------------------------------------------------------------------- #
def _live_signature(engine):
    return {
        "facts": engine.knowledge.active_count(),
        "step": engine.step_count,
        "name": engine.name,
        "adapter": engine.adapter_name,
        "policy": engine.retrieval_policy.state_dict()["probs"],
        "trace": float(torch.linalg.vector_norm(
            engine.memory.state_dict().get("h_trace", torch.zeros(1))
        )),
    }


@pytest.mark.parametrize("corruption", ["manifest", "database", "metadata", "mas"])
def test_failed_load_preserves_the_active_profile(tmp_path, corruption):
    source = _engine(tmp_path, name="source")
    source.teach("The mixer is on site B")
    source.save_profile(str(tmp_path / "source"))

    target = _engine(tmp_path, name="target")
    target.teach("The crane is on site C")
    before = _live_signature(target)

    profile = tmp_path / "source"
    if corruption == "manifest":
        (profile / "config.json").write_text("{oops")
    elif corruption == "database":
        (profile / "knowledge.db").write_bytes(b"not a database")
    elif corruption == "metadata":
        (profile / "adapter_meta.json").write_text("[1, 2, 3]")
    else:
        config = json.loads((profile / "config.json").read_text())
        config["mas_state"] = {"file": "missing.safetensors"}
        (profile / "config.json").write_text(json.dumps(config))

    with pytest.raises(InvalidProfileError):
        target.load_profile(str(profile))
    assert _live_signature(target) == before
    assert target.knowledge is not None
    assert target.knowledge.active_count() == 1
    # The target's knowledge store is still queryable after the failure.
    assert target.knowledge.recall(
        target.anchor.embed("crane"), query_text="crane", top_k=1
    )
    target.knowledge.close()
    source.knowledge.close()


def test_interrupted_save_is_not_advertised(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    profile = tmp_path / "profile"
    engine.save_profile(str(profile))
    # Simulate an interrupted save: the manifest is gone, artifacts remain.
    (profile / "config.json").unlink()

    reloaded = _engine(tmp_path, name="reloaded")
    with pytest.raises(InvalidProfileError):
        reloaded.load_profile(str(profile))
    reloaded.knowledge.close()
    engine.knowledge.close()


# --------------------------------------------------------------------------- #
# 2.9 Reversible migration metadata
# --------------------------------------------------------------------------- #
def test_legacy_manifest_records_a_versioned_migration(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    engine.save_profile(str(tmp_path / "profile"))
    config_path = tmp_path / "profile" / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("plastic_adapter", None)
    config.pop("adapter_kwargs", None)
    config_path.write_text(json.dumps(config))

    reloaded = _engine(tmp_path, name="reloaded")
    reloaded.load_profile(str(tmp_path / "profile"))
    assert reloaded.adapter_name == "associative_rtu"
    assert any(m["migration"] == "adapter_identity_default" for m in reloaded.migrations)

    # Re-saving keeps the migrated identity explicit and the record intact.
    reloaded.save_profile(str(tmp_path / "resaved"))
    resaved = json.loads((tmp_path / "resaved" / "config.json").read_text())
    assert resaved["plastic_adapter"] == "associative_rtu"
    assert any(m["migration"] == "adapter_identity_default"
               for m in resaved.get("migrations", []))
    reloaded.knowledge.close()
    engine.knowledge.close()


def test_migrated_copy_keeps_independent_artifacts(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    engine.save_profile(str(tmp_path / "source"))
    engine.save_profile(str(tmp_path / "copy"))

    assert (tmp_path / "source" / "knowledge.db").resolve() != (
        tmp_path / "copy" / "knowledge.db"
    ).resolve()
    copy_engine = _engine(tmp_path, name="copy_engine")
    copy_engine.load_profile(str(tmp_path / "copy"))
    copy_engine.teach("The crane is on site C")
    copy_engine.save_profile(str(tmp_path / "copy"))
    source_again = _engine(tmp_path, name="source_again")
    source_again.load_profile(str(tmp_path / "source"))
    assert source_again.knowledge.active_count() == 1
    for eng in (engine, copy_engine, source_again):
        eng.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.6 Merge lifecycle
# --------------------------------------------------------------------------- #
def test_merge_require_manual_refuses_without_writes(tmp_path):
    left = _engine(tmp_path, name="left")
    right = _engine(tmp_path, name="right")
    left.teach("The pump is broken")
    right.teach("The pump is working")
    left_before = left._profile_fingerprint()
    right_before = right._profile_fingerprint()

    report = left.merge_memory(right)
    assert report["status"] in ("manual_required", "merged")
    if report["conflicts"]:
        assert report["status"] == "manual_required"
        assert report["writes"] == 0
        assert left._profile_fingerprint() == left_before
    assert right._profile_fingerprint() == right_before
    left.knowledge.close()
    right.knowledge.close()


def test_merge_keep_newest_writes_and_leaves_the_source_alone(tmp_path):
    left = _engine(tmp_path, name="left")
    right = _engine(tmp_path, name="right")
    left.teach("The pump is broken")
    right.teach("The crane is on site C")
    right_before = right._profile_fingerprint()

    report = left.memory and left.merge_memory(right, policy="keep_newest")
    assert report["status"] == "merged"
    assert report["writes"] >= 1
    assert report["source_unchanged"] is True
    assert right._profile_fingerprint() == right_before
    assert left.knowledge.active_count() >= 2
    left.knowledge.close()
    right.knowledge.close()


def test_merge_rejects_incompatible_adapter_or_dimension(tmp_path):
    left = _engine(tmp_path, name="left")
    right = _engine(tmp_path, name="right")
    left.teach("The pump is broken")
    right.teach("The crane is on site C")
    left_before = left._profile_fingerprint()

    right.adapter_name = "sparse_delta"
    report = left.merge_memory(right, policy="keep_newest")
    assert report["status"] == "rejected"
    assert "adapter" in report["error"]
    assert left._profile_fingerprint() == left_before

    right.adapter_name = left.adapter_name
    right.dim = 64
    report = left.merge_memory(right, policy="keep_newest")
    assert report["status"] == "rejected"
    assert "dimension" in report["error"]
    assert left._profile_fingerprint() == left_before
    left.knowledge.close()
    right.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.8 Consolidation: measured objective, bounded audit-safe pruning
# --------------------------------------------------------------------------- #
def test_consolidation_measures_objective_and_keeps_audit(tmp_path):
    engine = _engine(tmp_path)
    for i in range(10):
        engine.teach(f"Site fact {i} mentions the pump and the crane")
    audit_before = len(engine.knowledge.get_audit_trail())
    frozen_before = engine._frozen_fingerprint()

    report = engine.consolidate_memory(max_prune_ratio=0.5)
    assert report["status"] == "consolidated"
    assert report["pruning"]["bounded"] is True
    assert report["pruned"] <= report["pruning"]["cap"]
    assert report["pruning"]["audit_retained"] is True
    assert len(engine.knowledge.get_audit_trail()) == audit_before
    assert report["frozen_unchanged"] is True
    assert engine._frozen_fingerprint() == frozen_before
    objective = report["objective"]
    assert objective["before"]["samples"] > 0
    assert objective["before"]["orthogonality_loss"] is not None
    assert objective["after"]["orthogonality_loss"] <= objective["before"]["orthogonality_loss"]
    assert objective["improved"] is True
    engine.knowledge.close()


def test_consolidation_prunes_only_superseded_rows(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("The mixer is on site B")
    # Age the fact well past the recency horizon so its utility drops below the
    # pruning threshold; a freshly retrieved fact is never pruned.
    engine.knowledge.conn.execute(
        "UPDATE facts SET timestamp=?, last_retrieved=?, access_count=1, novelty=0.0",
        (time.time() - 365 * 86400, time.time() - 365 * 86400),
    )
    engine.knowledge.conn.commit()
    report = engine.consolidate_memory(max_prune_ratio=1.0)
    assert report["status"] == "consolidated"
    rows = engine.knowledge.conn.execute(
        "SELECT superseded, needs_review FROM facts"
    ).fetchall()
    assert rows and all(row[0] == 1 for row in rows)
    assert engine.knowledge.active_count() == 0
    # Audit trail keeps the original text and its review flag.
    trail = engine.knowledge.get_audit_trail()
    assert trail and trail[0]["needs_review"] == 1
    engine.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.11 Stream ownership and bounded shutdown
# --------------------------------------------------------------------------- #
def test_stream_start_is_idempotent_and_stop_is_bounded(tmp_path):
    engine = _engine(tmp_path)
    first = engine.start_continuous_stream()
    second = engine.start_continuous_stream()
    assert first["status"] == "started"
    assert second["status"] == "already_running"
    threads = [t for t in threading.enumerate() if t.name == "continuous-stream"]
    assert len(threads) == 1

    report = engine.stop_continuous_stream(timeout=2.0)
    assert report["status"] == "stopped"
    assert report["joined"] is True
    assert report["stream"]["alive"] is False
    assert engine.stop_continuous_stream()["status"] == "not_running"
    engine.knowledge.close()


def test_stream_pause_resume_and_shutdown(tmp_path):
    engine = _engine(tmp_path)
    engine.start_continuous_stream()
    assert engine.pause_stream()["status"] == "paused"
    assert engine.get_stream_status()["paused"] is True
    assert engine.resume_stream()["status"] == "resumed"
    assert engine.get_stream_status()["paused"] is False

    report = engine.shutdown()
    assert report["stream_alive"] is False
    assert engine.get_stream_status()["running"] is False
    engine.shutdown()  # repeated shutdown is safe
    engine.knowledge.close()


def test_stream_ingest_is_thread_safe_with_teach(tmp_path):
    engine = _engine(tmp_path)
    engine.start_continuous_stream(max_fpm=1000)
    errors = []

    def teach():
        try:
            for i in range(5):
                engine.teach(f"Concurrent fact {i} about the crane")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread = threading.Thread(target=teach)
    thread.start()
    for i in range(5):
        engine.continuous_learn(f"Streamed fact {i} about the pump")
    thread.join(timeout=5)
    assert not errors
    deadline = time.time() + 5
    while time.time() < deadline and engine.get_stream_status()["facts_processed"] < 5:
        time.sleep(0.01)
    engine.stop_continuous_stream()
    assert engine.knowledge.active_count() >= 5
    engine.knowledge.close()


def test_manager_switches_and_deletes_stop_streams(tmp_path):
    manager = DroidManager(base_dir=str(tmp_path / "droids"))
    left = manager.create_droid("left")
    right = manager.create_droid("right")
    left.start_continuous_stream()
    right.start_continuous_stream()

    manager.activate_droid("left")
    assert right.get_stream_status()["alive"] is False
    assert manager.stop_continuous_stream("left")["status"] == "stopped"

    left.start_continuous_stream()
    assert manager.delete_droid("left") is True
    assert left.get_stream_status()["alive"] is False
    manager.shutdown()
    right.knowledge.close()
    left.knowledge.close()
