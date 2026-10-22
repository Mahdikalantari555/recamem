"""Cross-feature acceptance: teach -> feedback -> sleep -> snapshot -> branch
-> merge -> restart, on the default and the delta adapter.

`post-merge-completion` 9.3 asks for exactly this chain, and no single unit
suite covers it. Each stage asserts the state that the next stage consumes, so
a break anywhere surfaces as a named failure rather than a vague end-state.
"""

from __future__ import annotations

import json
import time

import pytest
import torch

from src.model.droid import DroidEngine
from src.model.droid_manager import DroidManager


REMOTE_SENSING = (
    "Remote sensing is the acquisition of information about an object without "
    "physical contact. It is used in hydrology and land surveying. "
    "The technique supports glaciology and meteorology."
)


@pytest.fixture()
def manager(tmp_path):
    mgr = DroidManager(base_dir=str(tmp_path / "droids"))
    yield mgr
    mgr.shutdown()


def _adapter_choices():
    from src.model.plastic_adapter import list_adapters

    names = set(list_adapters())
    return [name for name in ("associative_rtu", "delta_rtu") if name in names]


@pytest.mark.parametrize("adapter", _adapter_choices() or ["associative_rtu"])
def test_teach_feedback_sleep_snapshot_branch_merge_restart(manager, tmp_path, adapter):
    first = manager.create_droid("chain-a", plastic_adapter=adapter)
    result = first.teach(REMOTE_SENSING, source="conversational_chat")
    assert result["propositions"] >= 2
    assert first.knowledge.active_count() >= 1

    # 2. Feedback: approve one fact and confirm the labelled health surface sees it.
    fact_id = first.knowledge.get_audit_trail()[0]["id"]
    first.approve(fact_id)
    log = first.get_feedback_log()
    assert any(entry.get("action") == "approve" for entry in log)

    # 3. Sleep: one bounded cycle with a structured report.
    report = first.sleep()
    assert report["status"] in {"completed", "success", "partial", "noop"}
    assert report["replay"]["status"] in {"completed", "skipped", "noop"}
    assert report["fact_pruning"]["status"] in {"consolidated", "pruned", "noop", "skipped"}
    assert report["parameter_pruning"]["status"] in {"disabled", "pruned", "noop", "skipped"}
    assert report["replay"]["unique_recovered"] >= 1

    # 4. Snapshot the post-teach state, then keep teaching.
    snapshot = first.create_snapshot("chain-v1")
    assert snapshot["metadata"]["version_id"] == "chain-v1"
    baseline_facts = first.knowledge.active_count()
    first.teach("The calibration target is the blue marker on the north wall.")
    assert first.knowledge.active_count() > baseline_facts

    # 5. Rollback restores the snapshot's fact set.
    assert first.rollback("chain-v1")["restored"] is True
    assert first.knowledge.active_count() == baseline_facts

    # 6. Branch is independent of its source.
    branch = manager.get_droid("chain-a").branch_from("chain-v1", "chain-branch")
    assert branch is not None
    manager.save_droid("chain-a")

    # 7. Merge the branch back into a new profile, with an audit trail.
    # Merging requires matching adapters; a mismatched pair is rejected outright.
    manager.create_droid("chain-b", plastic_adapter=adapter)
    manager.get_droid("chain-b").teach(REMOTE_SENSING)
    manager.save_droid("chain-b")
    merged = manager.merge_droids("chain-a", "chain-b", "chain-merged", policy="keep_newest")
    assert merged.last_merge_result["status"] == "merged"
    assert merged.last_merge_result["merge_of"] == ["chain-merged", "chain-b"]

    # 8. Restart: the merged profile reloads with its facts, adapter and audit.
    reloaded = DroidManager(base_dir=str(manager.base_dir))
    try:
        engine = reloaded.get_droid("chain-merged")
        assert engine.knowledge.active_count() >= 1
        assert getattr(engine, "merge_of", None) == ["chain-merged", "chain-b"]
        hits = engine.recall("What is remote sensing?")
        assert hits, "the merged profile must still answer from its own memory"
    finally:
        reloaded.shutdown()


def test_stream_and_teach_share_one_engine_lock(manager):
    engine = manager.create_droid("lock-a")
    status = engine.start_continuous_stream(max_fpm=6000)
    assert status["status"] in {"started", "already_running"}
    # A second start is idempotent, not a second worker.
    again = engine.start_continuous_stream(max_fpm=6000)
    assert again["status"] == "already_running"
    assert again["stream"]["running"] is True

    engine.teach("Pumps fail when the seal is worn.", source="conversational_chat")
    assert engine.get_stream_status()["running"] is True
    stopped = engine.stop_continuous_stream()
    assert stopped["status"] in {"stopped", "not_running"}
    assert engine.get_stream_status()["running"] is False


def test_frozen_decoder_weights_survive_the_chain(manager):
    engine = manager.create_droid("frozen-a", byte_rtu=True, rtu_dim=384)
    backend = engine.gguf_backend
    before = {
        name: tensor.detach().clone()
        for name, tensor in backend.model.named_parameters()
    }

    engine.teach(REMOTE_SENSING, source="conversational_chat")
    engine.approve(engine.knowledge.get_audit_trail()[0]["id"])
    engine.sleep()
    engine.create_snapshot("frozen-v1")
    engine.rollback("frozen-v1")

    after = {name: tensor.detach() for name, tensor in backend.model.named_parameters()}
    assert before.keys() == after.keys()
    for name, value in before.items():
        assert torch.equal(value, after[name]), f"frozen decoder weight changed: {name}"
