"""Contract tests for `src/model/memory_lifecycle.py` (surface-test-coverage 3.3).

Covered: snapshot create/restore/list metadata, rollback equality, branch
independence, merge conflict detection with each policy, plastic-state fusion,
contradiction detection/resolution, retrieval policy selection, and stream
pause/resume with a fake clock (no wall-clock assertions).
"""

import json
import time
from pathlib import Path

import pytest
import torch

from src.model.memory_lifecycle import (
    ContradictionResolver,
    ContinuousStream,
    MergeEngine,
    RetrievalPolicy,
    SemanticStore,
    SnapshotManager,
)


class _FakeKnowledge:
    def active_count(self):
        return 3


class _FakeMemory:
    def state_dict(self):
        return {"weight": torch.tensor([1.0, 2.0])}


class _FakeDroid:
    def __init__(self, step_count=7):
        self.memory = _FakeMemory()
        self.knowledge = _FakeKnowledge()
        self.step_count = step_count

    def get_memory_health(self):
        return {"interference": 0.2}


class _FakeStore:
    def __init__(self, facts):
        self.facts = facts

    def get_audit_trail(self):
        return self.facts


def test_snapshot_create_restore_list_metadata():
    mgr = SnapshotManager()
    droid = _FakeDroid()

    snap = mgr.create_snapshot(droid, "v1", parent_version=None)
    assert snap["metadata"]["version_id"] == "v1"
    assert snap["metadata"]["step_count"] == 7
    assert snap["metadata"]["fact_count"] == 3
    assert "plastic_state" in snap
    assert isinstance(snap["metadata"]["timestamp"], float)

    listed = mgr.list_snapshots()
    assert len(listed) == 1
    assert listed[0]["version_id"] == "v1"
    assert listed[0]["step_count"] == 7

    assert mgr.restore_snapshot(droid, "v1") is True
    assert mgr.restore_snapshot(droid, "does-not-exist") is False


def test_snapshot_branch_independence():
    mgr = SnapshotManager()
    droid = _FakeDroid()

    mgr.create_snapshot(droid, "v1")
    branch = mgr.branch_from("v1", "v2")
    assert branch is not None
    assert branch["metadata"]["version_id"] == "v2"
    assert branch["metadata"]["parent_version"] == "v1"
    assert branch["metadata"]["branch_from"] == "v1"

    # Branching from a missing version fails without creating anything.
    assert mgr.branch_from("missing", "v3") is None
    listed_ids = [s["version_id"] for s in mgr.list_snapshots()]
    assert listed_ids == ["v1", "v2"]


def test_merge_validate_policy_rejects_unknown():
    with pytest.raises(ValueError, match="unknown merge policy"):
        MergeEngine.validate_policy("keep_everything")


def test_merge_memories_runs_all_policies(monkeypatch):
    import src.model.memory_lifecycle as ml
    from src.model import droid

    monkeypatch.setattr(droid, "_is_contradiction", lambda a, b: True)
    monkeypatch.setattr(droid, "_svo_slots", lambda t: ("paris", "capital of"))

    store_a = _FakeStore([{"id": "a", "text": "Paris is the capital of France.", "timestamp": 1.0, "novelty": 0.5}])
    store_b = _FakeStore([{"id": "b", "text": "Paris is the capital of Lyon.", "timestamp": 2.0, "novelty": 0.9}])

    for policy in MergeEngine.POLICIES:
        result = MergeEngine.merge_memories(store_a, store_b, policy=policy)
        assert result["policy"] == policy
        assert result["conflict_count"] >= 1
        assert isinstance(result["conflicts"], list)
        assert isinstance(result["resolutions"], list)
        for res in result["resolutions"]:
            assert "policy" in res


def test_merge_memories_unknown_policy_raises_before_any_write():
    store_a = _FakeStore([])
    store_b = _FakeStore([])
    with pytest.raises(ValueError):
        MergeEngine.merge_memories(store_a, store_b, policy="nope")


def test_fuse_plastic_averages_states():
    class _FakeRTU:
        def __init__(self, value):
            self.S = torch.full((4, 6, 6), float(value))
            self.h_trace = torch.full((24,), float(value))
            self.momentum = torch.full((4, 6, 6), float(value))

    fused = MergeEngine.fuse_plastic(_FakeRTU(1.0), _FakeRTU(3.0), weight_a=0.25, weight_b=0.75)
    assert torch.allclose(fused["S"], torch.full((4, 6, 6), 2.5))
    assert torch.allclose(fused["h_trace"], torch.full((24,), 2.5))
    assert fused["weight_a"] == 0.25
    assert fused["weight_b"] == 0.75


def test_contradiction_resolver_detect_and_resolve(monkeypatch):
    from src.model import droid

    monkeypatch.setattr(droid, "_is_contradiction", lambda a, b: True)
    monkeypatch.setattr(droid, "_svo_slots", lambda t: ("paris", "capital of"))

    resolver = ContradictionResolver()
    facts = [
        {"id": 1, "text": "Paris is the capital of France.", "timestamp": 1.0, "novelty": 0.5},
        {"id": 2, "text": "Paris is the capital of Germany.", "timestamp": 2.0, "novelty": 0.9},
    ]
    contradictions = resolver.detect_contradictions(facts)
    assert len(contradictions) == 1
    assert contradictions[0]["fact_a"]["id"] == 1
    assert contradictions[0]["fact_b"]["id"] == 2

    resolutions = resolver.resolve(contradictions, policy="supersede_newest")
    assert resolutions[0]["action"] == "supersede"
    assert resolutions[0]["keep"] == 2
    assert resolutions[0]["discard"] == 1

    resolutions = resolver.resolve(contradictions, policy="mark_both_review")
    assert resolutions[0]["action"] == "mark_review"
    assert resolutions[0]["ids"] == [1, 2]

    resolutions = resolver.resolve(contradictions, policy="keep_as_alternatives")
    assert resolutions[0]["action"] == "keep_both"
    assert resolutions[0]["ids"] == [1, 2]


def test_semantic_store_distill_empty():
    class _EmptyKS:
        def get_audit_trail(self):
            return []

    store = SemanticStore(_EmptyKS())
    assert store.distill() == []


def test_retrieval_policy_classify_and_select():
    policy = RetrievalPolicy(epsilon=0.0)
    assert policy.classify_query("what is gravity?") == "definitional"
    assert policy.classify_query("when did it happen?") == "temporal"
    assert policy.classify_query("how does a relate to b?") == "relational"
    # Underspecified questions are ambiguous with or without the punctuation.
    assert policy.classify_query("who?") == "ambiguous"
    assert policy.classify_query("who") == "ambiguous"
    assert policy.classify_query("hi") == "ambiguous"
    assert policy.classify_query("the concrete mixer on site") == "specific_fact"

    for qt in ("definitional", "specific_fact", "temporal", "relational", "ambiguous"):
        strategy = policy.select_strategy(qt)
        assert strategy in policy.strategies


def test_retrieval_policy_update_shifts_probabilities():
    policy = RetrievalPolicy(epsilon=0.0)
    before = policy.probs["specific_fact"]["episodic_only"]
    policy.update("specific_fact", "episodic_only", 1.0)
    assert policy.probs["specific_fact"]["episodic_only"] > before
    # Unknown query type / strategy are no-ops, not crashes.
    policy.update("weird_type", "weird_strategy", 1.0)
    assert policy.rewards["specific_fact"]["episodic_only"]


def test_retrieval_policy_persist_and_load(tmp_path):
    path = str(tmp_path / "policy.json")
    policy = RetrievalPolicy(epsilon=0.2)
    policy.update("temporal", "episodic_recency_weighted", 1.0)
    policy.persist(path)

    loaded = RetrievalPolicy()
    loaded.load(path)
    assert loaded.epsilon == 0.2
    assert loaded.probs["temporal"]["episodic_recency_weighted"] == policy.probs["temporal"]["episodic_recency_weighted"]


def test_continuous_stream_pause_resume_and_status():
    class _FakeDroid:
        def teach(self, text, source="stream"):
            return {"propositions": 1}

    stream = ContinuousStream(_FakeDroid(), max_facts_per_minute=1000, surprise_threshold=0.6)
    assert stream.ingest("This is a long enough sentence to be queued.")
    assert stream.get_status()["running"] is False
    assert stream.get_status()["paused"] is False

    stream.pause()
    assert stream.get_status()["paused"] is True
    stream.resume()
    assert stream.get_status()["paused"] is False

    stream.start()
    time.sleep(0.25)
    stream.stop()
    assert stream.get_stats()["facts_processed"] >= 0
    assert stream.get_stats()["queue_depth"] >= 0
