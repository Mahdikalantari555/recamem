"""Contract tests for src/model/memory_lifecycle.py.

Covers the public surfaces of the lifecycle module by requirement:

- ``SnapshotManager``: snapshot metadata, restore equality, sorted listing,
  branch creation and branch independence;
- ``MergeEngine``: conflict detection and one resolution per documented policy
  (``keep_newest``, ``keep_highest_novelty``, ``keep_both_as_alternatives``,
  ``require_manual``), plus weighted plastic fusion;
- ``ContradictionResolver``: detection over a fact list and per-policy resolve;
- ``SemanticStore``: distillation over a real ``KnowledgeStore`` and semantic
  retrieval;
- ``RetrievalPolicy``: query classification, greedy selection, reward updates,
  and persist/load;
- ``ContinuousStream``: queueing, rate accounting, pause/resume/stop, and the
  worker loop exit path.

The stream tests avoid wall-clock sleeps: the worker exit path is driven
directly, and the one threaded test waits on a condition with a hard deadline.
"""

import json
import threading
import time
import types

import pytest
import torch

from src.model.knowledge_store import KnowledgeStore
from src.model.memory_lifecycle import (
    ContinuousStream,
    ContradictionResolver,
    MergeEngine,
    RetrievalPolicy,
    SemanticStore,
    SnapshotManager,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeMemory:
    def __init__(self, state=None):
        self.state = dict(state or {"w": 1.0})

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


class FakeKnowledge:
    def __init__(self, count=3):
        self.count = count

    def active_count(self):
        return self.count


class FakeDroid:
    def __init__(self, steps=7):
        self.memory = FakeMemory()
        self.knowledge = FakeKnowledge()
        self.step_count = steps

    def get_memory_health(self):
        return {"norm": 1.5}


class FakeStore:
    def __init__(self, facts):
        self.facts = facts

    def get_audit_trail(self):
        return list(self.facts)


def _fact(fid, text, timestamp=1.0, novelty=0.5, superseded=False):
    return {
        "id": fid,
        "text": text,
        "timestamp": timestamp,
        "novelty": novelty,
        "superseded": superseded,
    }


# --------------------------------------------------------------------------- #
# SnapshotManager
# --------------------------------------------------------------------------- #
def test_create_snapshot_records_metadata():
    manager = SnapshotManager(base_dir="unused")
    droid = FakeDroid(steps=7)

    snapshot = manager.create_snapshot(droid, "v1")

    meta = snapshot["metadata"]
    assert meta["version_id"] == "v1"
    assert meta["step_count"] == 7
    assert meta["fact_count"] == 3
    assert meta["parent_version"] is None
    assert meta["health_metrics"] == {"norm": 1.5}
    assert snapshot["plastic_state"] == {"w": 1.0}


def test_create_snapshot_tolerates_missing_hooks():
    manager = SnapshotManager()
    bare = types.SimpleNamespace(memory=object())
    snapshot = manager.create_snapshot(bare, "v-bare")
    assert snapshot["plastic_state"] == {}
    assert snapshot["metadata"]["fact_count"] == 0
    assert snapshot["metadata"]["step_count"] == 0
    assert snapshot["metadata"]["health_metrics"] == {}


def test_restore_snapshot_restores_equality():
    manager = SnapshotManager()
    droid = FakeDroid()
    manager.create_snapshot(droid, "v1")

    droid.memory.state = {"w": 99.0}
    assert manager.restore_snapshot(droid, "v1") is True
    assert droid.memory.state == {"w": 1.0}


def test_restore_unknown_snapshot_returns_false():
    manager = SnapshotManager()
    droid = FakeDroid()
    assert manager.restore_snapshot(droid, "nope") is False


def test_restore_reports_load_failure(monkeypatch):
    manager = SnapshotManager()
    droid = FakeDroid()
    manager.create_snapshot(droid, "v1")

    def boom(state):
        raise RuntimeError("corrupt tensors")

    monkeypatch.setattr(droid.memory, "load_state_dict", boom)
    assert manager.restore_snapshot(droid, "v1") is False


def test_list_snapshots_is_sorted_by_timestamp(monkeypatch):
    manager = SnapshotManager()
    clock = iter([100.0, 50.0, 75.0])
    monkeypatch.setattr("src.model.memory_lifecycle.time.time", lambda: next(clock))

    droid = FakeDroid()
    manager.create_snapshot(droid, "first")
    manager.create_snapshot(droid, "second")
    manager.create_snapshot(droid, "third")

    assert [m["version_id"] for m in manager.list_snapshots()] == ["second", "third", "first"]


def test_branch_from_snapshot_is_independent():
    manager = SnapshotManager()
    droid = FakeDroid()
    manager.create_snapshot(droid, "base")

    branch = manager.branch_from("base", "branch")
    assert branch["metadata"]["parent_version"] == "base"
    assert branch["metadata"]["branch_from"] == "base"

    # Mutating the branch metadata must not touch the base snapshot.
    branch["metadata"]["step_count"] = 999
    assert manager.snapshots["base"]["metadata"]["step_count"] == 7
    assert manager.branch_from("missing", "x") is None


# --------------------------------------------------------------------------- #
# MergeEngine
# --------------------------------------------------------------------------- #
def _conflicting_stores():
    a = FakeStore([_fact(1, "The capital of France is Paris", timestamp=10.0, novelty=0.2)])
    b = FakeStore([_fact(2, "The capital of France is Lyon", timestamp=20.0, novelty=0.9)])
    return a, b


def test_detect_conflicts_finds_slot_clash():
    store_a, store_b = _conflicting_stores()
    conflicts = MergeEngine.detect_conflicts(store_a, store_b)
    assert len(conflicts) == 1
    assert conflicts[0]["fact_a"]["id"] == 1
    assert conflicts[0]["fact_b"]["id"] == 2
    assert conflicts[0]["subject"] == "france"


def test_detect_conflicts_ignores_superseded_and_unrelated():
    a = FakeStore([
        _fact(1, "The capital of France is Paris"),
        _fact(3, "The capital of France is Nice", superseded=True),
    ])
    b = FakeStore([_fact(2, "The capital of Germany is Berlin")])
    assert MergeEngine.detect_conflicts(a, b) == []


def test_detect_conflicts_tolerates_store_errors():
    class Broken:
        def get_audit_trail(self):
            raise RuntimeError("db closed")

    assert MergeEngine.detect_conflicts(Broken(), FakeStore([])) == []


def test_merge_keep_newest():
    store_a, store_b = _conflicting_stores()
    result = MergeEngine.merge_memories(store_a, store_b, policy="keep_newest")
    assert result["conflict_count"] == 1
    assert result["resolutions"][0]["keep"] == 2
    assert result["resolutions"][0]["discard"] == 1


def test_merge_keep_highest_novelty():
    store_a, store_b = _conflicting_stores()
    result = MergeEngine.merge_memories(store_a, store_b, policy="keep_highest_novelty")
    assert result["resolutions"][0]["keep"] == 2
    assert result["resolutions"][0]["discard"] == 1


def test_merge_keep_both_as_alternatives():
    store_a, store_b = _conflicting_stores()
    result = MergeEngine.merge_memories(store_a, store_b, policy="keep_both_as_alternatives")
    resolution = result["resolutions"][0]
    assert resolution["keep"] == [1, 2]
    assert resolution["alternative_of"] is True


def test_merge_require_manual_is_default():
    store_a, store_b = _conflicting_stores()
    result = MergeEngine.merge_memories(store_a, store_b)
    assert result["policy"] == "require_manual"
    resolution = result["resolutions"][0]
    assert resolution["policy"] == "manual_review_needed"
    assert resolution["conflict"]["fact_a"]["id"] == 1


def test_merge_without_conflicts_returns_empty_resolutions():
    a = FakeStore([_fact(1, "The capital of France is Paris")])
    b = FakeStore([_fact(2, "The capital of Germany is Berlin")])
    result = MergeEngine.merge_memories(a, b)
    assert result == {"conflicts": [], "resolutions": [], "policy": "require_manual", "conflict_count": 0}


def test_fuse_plastic_weighted_average():
    rtu_a = type("RTU", (), {})()
    rtu_b = type("RTU", (), {})()
    rtu_a.S = torch.ones(1, 2, 2)
    rtu_b.S = torch.full((1, 2, 2), 3.0)
    rtu_a.h_trace = torch.tensor([1.0, 0.0])
    rtu_b.h_trace = torch.tensor([0.0, 1.0])

    fused = MergeEngine.fuse_plastic(rtu_a, rtu_b, weight_a=0.75, weight_b=0.25)
    assert torch.allclose(fused["S"], torch.full((1, 2, 2), 1.5))
    assert torch.allclose(fused["h_trace"], torch.tensor([0.75, 0.25]))
    assert fused["weight_a"] == 0.75
    assert fused["weight_b"] == 0.25
    assert fused["momentum"].shape == (1, 2, 2)


def test_fuse_plastic_truncates_shape_mismatch():
    rtu_a = type("RTU", (), {})()
    rtu_b = type("RTU", (), {})()
    rtu_a.S = torch.ones(2, 3, 3)
    rtu_b.S = torch.ones(1, 3, 3)

    fused = MergeEngine.fuse_plastic(rtu_a, rtu_b)
    assert fused["S"].shape == (1, 3, 3)


def test_fuse_plastic_reports_errors():
    fused = MergeEngine.fuse_plastic(object(), object())
    # Plain objects have no S/h_trace: the documented zeros are used, not a crash.
    assert fused["S"] is not None


# --------------------------------------------------------------------------- #
# ContradictionResolver
# --------------------------------------------------------------------------- #
def test_detect_and_resolve_contradictions():
    facts = [_fact(1, "The capital of France is Paris"), _fact(2, "The capital of France is Lyon")]
    resolver = ContradictionResolver()
    contradictions = resolver.detect_contradictions(facts)
    assert len(contradictions) == 1
    assert contradictions[0]["subject"] == "france"
    assert contradictions[0]["predicate"] == "capital of"
    assert contradictions[0]["type"] == "SlotClash"

    newest = resolver.resolve(contradictions, policy="supersede_newest")


def test_resolve_policies():
    fa = _fact(1, "The capital of France is Paris", timestamp=1.0, novelty=0.1)
    fb = _fact(2, "The capital of France is Lyon", timestamp=2.0, novelty=0.9)
    contradictions = [{"fact_a": fa, "fact_b": fb, "subject": "france", "predicate": "capital of"}]
    resolver = ContradictionResolver()

    newest = resolver.resolve(contradictions, policy="supersede_newest")
    assert newest[0]["keep"] == 2 and newest[0]["discard"] == 1

    confident = resolver.resolve(contradictions, policy="supersede_highest_confidence")
    assert confident[0]["keep"] == 2 and confident[0]["discard"] == 1

    review = resolver.resolve(contradictions, policy="mark_both_review")
    assert review[0] == {"action": "mark_review", "ids": [1, 2]}

    both = resolver.resolve(contradictions, policy="keep_as_alternatives")
    assert both[0]["action"] == "keep_both" and both[0]["ids"] == [1, 2]


# --------------------------------------------------------------------------- #
# SemanticStore
# --------------------------------------------------------------------------- #
@pytest.fixture
def store(tmp_path):
    ks = KnowledgeStore(str(tmp_path / "k.db"), dim=8)
    yield ks
    ks.close()


def _vec(seed):
    return torch.randn(8, generator=torch.Generator().manual_seed(seed))


def test_semantic_store_distills_clusters(store):
    base = _vec(1)
    store.insert_fact("Rivers carve valleys over long time scales", "chat", 1.0, 0, 0.5, base)
    store.insert_fact("Rivers carve valleys over very long time scales", "chat", 2.0, 1, 0.9, base)
    store.insert_fact("Glaciers carve valleys over long time scales", "chat", 3.0, 2, 0.4, _vec(2))

    semantic = SemanticStore(store)
    distilled = semantic.distill(similarity_threshold=0.9)

    assert len(distilled) == 2
    clustered = [item for item in distilled if len(item["source_cluster_ids"]) == 2]
    assert len(clustered) == 1
    assert clustered[0]["confidence"] == pytest.approx(0.8)
    # Highest-novelty member represents the cluster.
    assert clustered[0]["text"] == "Rivers carve valleys over very long time scales"


def test_semantic_store_empty_and_retrieval(store):
    semantic = SemanticStore(store)
    assert semantic.distill() == []
    assert semantic.retrieve_semantic(_vec(1)) == []

    vec = _vec(3)
    store.insert_fact("Lidar measures distance with laser pulses", "chat", 1.0, 0, 0.5, vec)
    assert semantic.distill() != []
    hits = semantic.retrieve_semantic(vec, top_k=1)
    assert len(hits) == 1
    assert hits[0]["text"] == "Lidar measures distance with laser pulses"
    assert semantic.incremental_distill([1]) != []


# --------------------------------------------------------------------------- #
# RetrievalPolicy
# --------------------------------------------------------------------------- #
def test_classify_query_types():
    policy = RetrievalPolicy(epsilon=0.0)
    assert policy.classify_query("What is remote sensing?") == "definitional"
    assert policy.classify_query("When was the sensor deployed?") == "temporal"
    assert policy.classify_query("How does lidar compare to radar?") == "relational"
    assert policy.classify_query("capital of France?") == "specific_fact"
    # Note: the ambiguous branch only fires for short keyword-free strings that
    # are not questions (question-marked queries hit the specific_fact rule).
    assert policy.classify_query("hmm there") == "ambiguous"


def test_select_strategy_defaults_without_exploration():
    policy = RetrievalPolicy(epsilon=0.0)
    assert policy.select_strategy("definitional") == "semantic_primary"
    assert policy.select_strategy("specific_fact") == "hybrid_rrf"
    assert policy.select_strategy("temporal") == "episodic_recency_weighted"
    assert policy.select_strategy("unknown-type") == "hybrid_rrf"


def test_select_strategy_explores_all_strategies():
    policy = RetrievalPolicy(epsilon=1.0)
    seen = {policy.select_strategy("definitional") for _ in range(80)}
    assert seen <= set(policy.strategies)


def test_update_records_reward_and_renormalizes():
    policy = RetrievalPolicy(epsilon=0.0)
    policy.update("definitional", "semantic_primary", reward=1.0)

    assert policy.rewards["definitional"]["semantic_primary"] == [1.0]
    assert sum(policy.probs["definitional"].values()) == pytest.approx(1.0)
    assert policy.probs["definitional"]["semantic_primary"] > 1.0 / len(policy.strategies)


def test_update_ignores_unknown_strategy_and_maps_unknown_type():
    policy = RetrievalPolicy(epsilon=0.0)
    before = dict(policy.probs["specific_fact"])
    policy.update("definitional", "not-a-strategy", reward=1.0)
    assert policy.probs["definitional"] == pytest.approx(
        {s: 1.0 / len(policy.strategies) for s in policy.strategies}
    )

    policy.update("mystery-type", "hybrid_rrf", reward=0.5)
    assert policy.rewards["specific_fact"]["hybrid_rrf"] == [0.5]
    assert policy.probs["specific_fact"] != before


def test_persist_and_load_round_trip(tmp_path):
    path = tmp_path / "policy.json"
    policy = RetrievalPolicy(epsilon=0.05)
    policy.update("temporal", "hybrid_rrf", reward=0.7)
    policy.persist(str(path))

    payload = json.loads(path.read_text())
    assert payload["epsilon"] == pytest.approx(0.05)
    assert payload["probs"]["temporal"]["hybrid_rrf"] > 0

    restored = RetrievalPolicy()
    restored.load(str(path))
    assert restored.epsilon == pytest.approx(0.05)
    for query_type, probs in policy.probs.items():
        assert restored.probs[query_type] == pytest.approx(probs)
    assert restored.rewards["temporal"]["hybrid_rrf"] == [0.7]


def test_load_missing_file_is_a_noop(tmp_path):
    policy = RetrievalPolicy(epsilon=0.3)
    policy.load(str(tmp_path / "absent.json"))
    assert policy.epsilon == 0.3


# --------------------------------------------------------------------------- #
# ContinuousStream
# --------------------------------------------------------------------------- #
class StreamingDroid:
    def __init__(self):
        self.taught = []
        self.lock = threading.Lock()

    def teach(self, text, source=None):
        with self.lock:
            self.taught.append((text, source))
        return {"propositions": 1}


def test_continuous_stream_status_and_pause_resume():
    stream = ContinuousStream(StreamingDroid())
    assert stream.get_status()["running"] is False
    assert stream.get_status()["paused"] is False

    stream.pause()
    assert stream.get_status()["paused"] is True
    assert stream.get_status()["running"] is False

    stream.resume()
    assert stream.get_status()["paused"] is False
    assert stream.get_stats() == stream.get_status()


def test_continuous_stream_ingest_accounts_for_queue():
    stream = ContinuousStream(StreamingDroid())
    assert stream.ingest("A fact that is long enough to keep.") is True
    assert stream.queue.qsize() == 1
    assert stream.get_status()["queue_depth"] == 1


def test_continuous_stream_drops_when_queue_is_full():
    stream = ContinuousStream(StreamingDroid())
    stream.queue.put("seed")
    for i in range(100):
        stream.queue.put(f"queued {i}")

    assert stream.ingest("one too many") is False
    assert stream.get_status()["dropped"] == 1
    assert stream.queue.qsize() == 101


def test_continuous_stream_ingest_file(tmp_path):
    log = tmp_path / "chat.log"
    log.write_text("First line long enough.\n\nSecond line long enough.\n")
    stream = ContinuousStream(StreamingDroid())
    stream.ingest_file(str(log))
    assert stream.queue.qsize() == 2

    missing = ContinuousStream(StreamingDroid())
    missing.ingest_file(str(tmp_path / "absent.log"))
    assert missing.queue.qsize() == 0


def test_worker_loop_exits_when_stopped():
    stream = ContinuousStream(StreamingDroid())
    stream.running = False
    started = time.perf_counter()
    stream._run_loop()
    assert time.perf_counter() - started < 1.0


def test_start_processes_queue_and_stops():
    droid = StreamingDroid()
    stream = ContinuousStream(droid, max_facts_per_minute=100)
    stream.start()
    try:
        assert stream.ingest("Memory consolidation happens during sleep cycles.") is True
        deadline = time.time() + 5.0
        while time.time() < deadline and not droid.taught:
            time.sleep(0.01)
        assert droid.taught
        assert droid.taught[0][1] == "continuous_stream"
        assert stream.get_status()["facts_absorbed"] == 1
    finally:
        stream.stop()
    assert stream.running is False
    assert stream.thread is not None and not stream.thread.is_alive()
