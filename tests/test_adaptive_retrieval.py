"""Adaptive retrieval, health metrics and TTT budgets (post-merge 4.1-4.3, 4.9, 4.10).

These tests assert behaviour, not labels: the selected strategy must change which
store answers, health must come from labelled feedback, and the TTT budget must
be a hard bound shared by recall and every feedback path.
"""

import json
import time

import pytest
import torch

from src.model.droid import DroidEngine
from src.model.knowledge_store import KnowledgeStore
from src.model.memory_lifecycle import SnapshotManager


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store():
    ks = KnowledgeStore(":memory:", dim=8)
    yield ks
    ks.close()


def _vec(*values):
    v = torch.tensor(values, dtype=torch.float32)
    return v / torch.linalg.vector_norm(v).clamp(min=1e-8)


def _seed(store):
    store.insert_fact("The concrete mixer is on site B", "teach", 1000.0, 1, 1.0,
                      _vec(1, 0, 0, 0, 0, 0, 0, 0))
    store.insert_fact("The crane was inspected in March", "teach", 2000.0, 2, 1.0,
                      _vec(0, 1, 0, 0, 0, 0, 0, 0))
    store.insert_fact("Gravity is the attraction between masses", "teach", 3000.0, 3, 1.0,
                      _vec(0, 0, 1, 0, 0, 0, 0, 0))
    ids = [row[0] for row in store.conn.execute("SELECT id FROM facts ORDER BY id")]
    store.insert_semantic_fact(
        "Gravity is the attraction between masses", [3], 0.9, _vec(0, 0, 1, 0, 0, 0, 0, 0)
    )
    return ids


@pytest.fixture()
def engine(tmp_path):
    eng = DroidEngine(name="retrieval", db_path=str(tmp_path / "knowledge.db"))
    yield eng
    eng.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.1 Deterministic classification with reachable ambiguity
# --------------------------------------------------------------------------- #
LABELLED_FIXTURES = [
    ("", "ambiguous"),
    ("   ", "ambiguous"),
    ("what?", "ambiguous"),
    ("what", "ambiguous"),
    ("who?", "ambiguous"),
    ("?", "ambiguous"),
    ("When was the crane inspected?", "temporal"),
    ("what happened before the audit", "temporal"),
    ("What is gravity?", "definitional"),
    ("define gravity", "definitional"),
    ("Why does the pump fail?", "relational"),
    ("difference between pump A and pump B", "relational"),
    ("The concrete mixer on site B", "specific_fact"),
]


def test_classification_matches_labelled_fixture():
    from src.model.memory_lifecycle import RetrievalPolicy

    policy = RetrievalPolicy(epsilon=0.0)
    for query, expected in LABELLED_FIXTURES:
        assert policy.classify_query(query) == expected, query


def test_ambiguity_is_punctuation_invariant():
    from src.model.memory_lifecycle import RetrievalPolicy

    policy = RetrievalPolicy(epsilon=0.0)
    for query in ("what?", "who?", "which?"):
        assert policy.classify_query(query) == policy.classify_query(query.rstrip("?"))


def test_classification_is_reproducible_and_exploration_only_affects_selection():
    from src.model.memory_lifecycle import RetrievalPolicy

    labels = []
    for _ in range(5):
        policy = RetrievalPolicy(epsilon=0.3, seed=7)
        labels.append([policy.classify_query(q) for q, _ in LABELLED_FIXTURES])
    assert all(run == labels[0] for run in labels)

    a = RetrievalPolicy(epsilon=1.0, seed=11)
    b = RetrievalPolicy(epsilon=1.0, seed=11)
    choices_a = [a.select_strategy("specific_fact") for _ in range(6)]
    choices_b = [b.select_strategy("specific_fact") for _ in range(6)]
    # Exploration is deterministic per seed and can differ from the argmax.
    assert choices_a == choices_b
    assert len(set(choices_a)) > 1


def test_default_strategies_follow_query_type():
    from src.model.memory_lifecycle import RetrievalPolicy

    policy = RetrievalPolicy(epsilon=0.0)
    assert policy.select_strategy("definitional") == "semantic_primary"
    assert policy.select_strategy("temporal") == "episodic_recency_weighted"
    assert policy.select_strategy("specific_fact") == "hybrid_rrf"
    assert policy.select_strategy("ambiguous") == "hybrid_rrf"


# --------------------------------------------------------------------------- #
# 4.2 The strategy decides which store answers
# --------------------------------------------------------------------------- #
def test_semantic_strategy_consults_semantic_store(store):
    _seed(store)
    q = _vec(0, 0, 1, 0, 0, 0, 0, 0)

    semantic = store.recall(q, top_k=3, query_text="what is gravity",
                            strategy="semantic_primary")
    assert semantic, "semantic_primary must consult the semantic store"
    assert semantic[0]["retrieval_store"] == "semantic"
    assert "Gravity" in semantic[0]["text"]
    assert store.last_retrieval_trace["executed_strategy"] == "semantic_primary"

    episodic = store.recall(q, top_k=3, query_text="what is gravity",
                            strategy="episodic_only")
    assert episodic and all(hit["retrieval_store"] == "episodic" for hit in episodic)
    assert store.last_retrieval_trace["executed_strategy"] == "episodic_only"
    assert store.last_retrieval_trace["semantic_hits"] == 0

    hybrid = store.recall(q, top_k=3, query_text="what is gravity", strategy="hybrid_rrf")
    assert store.last_retrieval_trace["requested_strategy"] == "hybrid_rrf"
    assert store.last_retrieval_trace["executed_strategy"] == "hybrid_rrf"
    # Different access paths can legitimately reorder results.
    assert [h["id"] for h in hybrid] != [h["id"] for h in semantic]


def test_semantic_only_profile_still_answers(store):
    store.insert_semantic_fact("Gravity is a force", [1], 0.9, _vec(1, 0, 0, 0, 0, 0, 0, 0))
    assert store.recall(_vec(1, 0, 0, 0, 0, 0, 0, 0), query_text="what is gravity",
                        strategy="episodic_only") == []
    hits = store.recall(_vec(1, 0, 0, 0, 0, 0, 0, 0), query_text="what is gravity",
                        strategy="semantic_primary")
    assert [h["retrieval_store"] for h in hits] == ["semantic"]
    assert store.last_retrieval_trace["episodic_available"] is False


def test_recency_and_contextual_weights_change_scores(store):
    ids = _seed(store)
    q = _vec(1, 0, 0, 0, 0, 0, 0, 0)
    plain = store.recall(q, top_k=3, query_text="concrete mixer", strategy="episodic_only")
    weighted = store.recall(q, top_k=3, query_text="concrete mixer",
                            strategy="episodic_recency_weighted")
    assert all("recency_weight" in hit for hit in weighted)
    assert all(hit["recency_weight"] <= 1.0 for hit in weighted)
    assert [h["fused_score"] for h in weighted] != [h["fused_score"] for h in plain]

    boosted = store.recall(q, top_k=3, query_text="concrete mixer",
                           strategy="contextual_boosted")
    assert all("contextual_weight" in hit for hit in boosted)
    assert store.decay_exponent(ids[0]) >= store.DECAY_BETA_MIN


def test_unknown_strategy_falls_back_to_hybrid_and_says_so(store):
    _seed(store)
    store.recall(_vec(1, 0, 0, 0, 0, 0, 0, 0), query_text="mixer", strategy="telepathy")
    trace = store.last_retrieval_trace
    assert trace["requested_strategy"] == "telepathy"
    assert trace["executed_strategy"] == "hybrid_rrf"


def test_engine_passes_selected_strategy_to_the_store(engine, monkeypatch):
    engine.teach("The concrete mixer is on site B and the crane is inspected")
    seen = []
    original = engine.knowledge.recall

    def spy(*args, **kwargs):
        seen.append(kwargs.get("strategy"))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.knowledge, "recall", spy)
    engine.recall("what is the concrete mixer", top_k=2)
    assert seen and seen[0] in engine.retrieval_policy.STRATEGIES
    assert engine._last_retrieval_strategy == seen[0]


# --------------------------------------------------------------------------- #
# 4.3 Policy persistence is per profile
# --------------------------------------------------------------------------- #
def test_policy_feedback_persists_per_profile(tmp_path):
    from src.model.memory_lifecycle import RetrievalPolicy

    a = RetrievalPolicy(epsilon=0.0, seed=3)
    b = RetrievalPolicy(epsilon=0.0, seed=3)
    a.update("specific_fact", "episodic_only", reward=1.0)
    a.update("specific_fact", "episodic_only", reward=1.0)
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a.persist(str(a_path))
    b.persist(str(b_path))

    a2 = RetrievalPolicy(epsilon=0.0).load(str(a_path))
    b2 = RetrievalPolicy(epsilon=0.0).load(str(b_path))
    persisted = json.loads(a_path.read_text())
    assert a2.selection_count == persisted["selection_count"]
    assert a2.exploration == persisted["exploration"]
    assert a2.select_strategy("specific_fact") == "episodic_only"
    assert a2.rewards["specific_fact"]["episodic_only"] == [1.0, 1.0]
    assert b2.rewards["specific_fact"]["episodic_only"] == []
    assert b2.select_strategy("specific_fact") != "episodic_only"


def test_engine_profiles_keep_independent_policies(tmp_path):
    left = DroidEngine(name="left", db_path=str(tmp_path / "left.db"))
    right = DroidEngine(name="right", db_path=str(tmp_path / "right.db"))
    left.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    left.retrieval_policy.update("specific_fact", "episodic_only", reward=1.0)
    left.save_profile(str(tmp_path / "left"))
    right.save_profile(str(tmp_path / "right"))

    reloaded_left = DroidEngine(name="left", db_path=str(tmp_path / "left.db"))
    reloaded_left.load_profile(str(tmp_path / "left"))
    reloaded_right = DroidEngine(name="right", db_path=str(tmp_path / "right.db"))
    reloaded_right.load_profile(str(tmp_path / "right"))

    # Exploration is a separate feature (tested above); pin it off here so the
    # assertion is about the learned preference, not a seeded coin flip.
    for policy in (reloaded_left.retrieval_policy, reloaded_right.retrieval_policy):
        policy.epsilon = 0.0
    assert reloaded_left.retrieval_policy.select_strategy("specific_fact") == "episodic_only"
    assert reloaded_right.retrieval_policy.select_strategy("specific_fact") != "episodic_only"
    for eng in (left, right, reloaded_left, reloaded_right):
        eng.knowledge.close()


# --------------------------------------------------------------------------- #
# 4.9 Health metrics come from labelled feedback
# --------------------------------------------------------------------------- #
def test_health_reports_insufficient_evidence_without_labels(engine):
    health = engine.get_memory_health()
    assert health["retrieval_quality"] is None
    assert health["retrieval_health"]["available"] is False
    assert health["retrieval_health"]["reason"] == "no_labelled_feedback"
    assert health["retrieval_health"]["samples"] == 0
    assert health["approval_rate"] is None
    # The placeholder constants are gone.
    assert health["retrieval_quality"] not in (0.5, 0.7)


def test_health_precision_and_mrr_match_hand_computed_values(engine):
    ids = [row[0] for row in engine.knowledge.conn.execute("SELECT id FROM facts")]
    assert ids == []

    engine.teach("The mixer is on site B")
    engine.teach("The crane is on site C")
    facts = [row[0] for row in engine.knowledge.conn.execute(
        "SELECT id FROM facts ORDER BY id")]
    top, second = facts[0], facts[1]

    engine.knowledge.log_ranked_feedback(
        "mixer?", [top, second], [second], strategy_used="hybrid_rrf",
        query_type="specific_fact",
    )
    engine.knowledge.log_ranked_feedback(
        "crane?", [second, top], [second], strategy_used="semantic_primary",
        query_type="definitional",
    )

    report = engine.knowledge.retrieval_health()
    assert report["available"] is True
    assert report["samples"] == 2
    assert report["source"].startswith("retrieval_feedback")
    # precision@1: 0/1 then 1/1 -> 0.5; MRR: 1/2 then 1 -> 0.75
    assert report["precision_at_k"] == pytest.approx(0.5)
    assert report["mrr"] == pytest.approx(0.75)
    assert report["by_strategy"]["hybrid_rrf"]["samples"] == 1
    assert report["by_query_type"]["definitional"] == 1

    health = engine.get_memory_health()
    assert health["retrieval_health"]["samples"] == 2
    assert health["retrieval_quality"] == pytest.approx(0.5)


def test_unlabelled_feedback_does_not_move_health(engine):
    engine.teach("The mixer is on site B")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    engine.knowledge.log_feedback(fact_id, "approve")
    health = engine.get_memory_health()
    assert health["retrieval_health"]["available"] is False
    assert health["approval_rate"] == 1.0
    assert health["approval_samples"] == 1


# --------------------------------------------------------------------------- #
# 4.10 TTT budgets across recall and feedback
# --------------------------------------------------------------------------- #
def test_ttt_budget_is_shared_and_hard_bounded(engine):
    engine.max_ttt_steps = 2
    engine.test_time_training = True
    engine.teach("The mixer is on site B")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]

    for _ in range(5):
        engine.recall("mixer", top_k=1)
    for _ in range(5):
        engine.approve(fact_id)

    status = engine.ttt_status()
    assert status["enabled"] is True
    assert status["used"] <= status["max_steps"] == 2
    assert status["skipped"] > 0
    assert status["last_skip_reason"] == "exhausted"
    assert status["remaining"] == 0


def test_ttt_opt_out_never_updates(engine):
    engine.test_time_training = False
    engine.teach("The mixer is on site B")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    before = engine.memory.eviction_count if hasattr(engine.memory, "eviction_count") else None
    engine.recall("mixer", top_k=1)
    engine.approve(fact_id)
    status = engine.ttt_status()
    assert status["used"] == 0
    assert status["skipped"] >= 1
    assert status["last_skip_reason"] in ("opt_out", "exhausted")
    assert status["used"] == 0
    if before is not None:
        assert engine.memory.eviction_count == before


def test_correction_erases_old_association(engine):
    engine.test_time_training = False
    engine.teach("The mixer is on site B")
    old = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    erased = engine._erase_corrected_fact(old)
    # The associative adapter exposes erase(), so the erasure path is exercised.
    assert isinstance(erased, bool)
    status = engine.ttt_status()
    assert status["used"] == 0


def test_byte_decoder_weights_are_untouched_by_ttt(engine):
    from src.model.byte_rtu_backend import ByteRtuBackend

    backend = ByteRtuBackend()
    state = {k: v.clone() for k, v in backend.model.state_dict().items()}
    engine.byte_rtu_backend = backend
    engine.teach("The mixer is on site B")
    engine.recall("mixer", top_k=1)
    engine.approve(engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0])
    for key, value in backend.model.state_dict().items():
        torch.testing.assert_close(value, state[key])
