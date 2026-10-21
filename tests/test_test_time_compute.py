"""Bounded test-time compute: diversity, verifier scoring, budgets, no writes."""
import threading

import pytest

from src.model.hybrid_retrieval import ExactDenseIndex
from src.model.knowledge_store import KnowledgeStore
from src.model.test_time_compute import (
    CallableVerifier,
    KeywordVerifier,
    TTCBudget,
    TestTimeCompute,
    Verifier,
)


class CountingGenerator:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt, max_tokens=32, temperature=0.8, seed=0):
        self.prompts.append(prompt)
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return text


class CountingVerifier(Verifier):
    name = "counting"

    def __init__(self, scores=None):
        self.calls = 0
        self.scores = list(scores or [])

    def score(self, prompt, candidate):
        self.calls += 1
        if self.scores:
            return self.scores[self.calls % len(self.scores)]
        return len(candidate) / 100.0


# ---------------------------------------------------------------------------
# Budget discipline
# ---------------------------------------------------------------------------
def test_candidate_budget_is_strict():
    generator = CountingGenerator([f"answer {i}" for i in range(20)])
    ttc = TestTimeCompute(generator, CountingVerifier(),
                          budget=TTCBudget(max_candidates=4, max_depth=2,
                                           max_tokens=100, max_seconds=5.0))
    result = ttc.search("What is remote sensing?")
    assert result["status"] == "selected"
    assert len(result["candidates"]) <= 4
    assert generator.calls <= 4
    assert result["consumed"]["candidates"] == len(result["candidates"])


def test_depth_budget_is_enforced():
    generator = CountingGenerator([f"reasoning step {i}" for i in range(20)])
    ttc = TestTimeCompute(generator, CountingVerifier(),
                          budget=TTCBudget(max_candidates=9, max_depth=1,
                                           max_tokens=100, max_seconds=5.0))
    result = ttc.search("q")
    assert max(candidate["depth"] for candidate in result["candidates"]) == 1


def test_token_budget_stops_generation():
    generator = CountingGenerator(["a very long answer with many repeated words " * 10])
    ttc = TestTimeCompute(
        generator, CountingVerifier(),
        budget=TTCBudget(max_candidates=10, max_depth=1, max_tokens=20,
                         max_seconds=5.0),
        token_counter=lambda text: len(text.split()),
    )
    result = ttc.search("q")
    assert result["consumed"]["tokens"] <= 20 + 80  # one over-budget request max
    assert generator.calls >= 1


def test_wall_clock_budget_stops_generation():
    def slow(prompt, max_tokens=32, temperature=0.8, seed=0):
        import time
        time.sleep(0.03)
        return f"candidate {time.time()}"

    ttc = TestTimeCompute(slow, CountingVerifier(),
                          budget=TTCBudget(max_candidates=50, max_depth=1,
                                           max_tokens=500, max_seconds=0.05))
    result = ttc.search("q")
    assert result["consumed"]["seconds"] < 1.0
    assert len(result["candidates"]) < 50


def test_cancellation_is_observed():
    cancel = threading.Event()
    cancel.set()
    generator = CountingGenerator(["unused"])
    ttc = TestTimeCompute(generator, CountingVerifier(),
                          budget=TTCBudget(max_candidates=4))
    result = ttc.search("q", cancel_event=cancel)
    assert result["status"] == "cancelled"
    assert generator.calls == 0


# ---------------------------------------------------------------------------
# Diversity, verifier and selection
# ---------------------------------------------------------------------------
def test_duplicate_candidates_do_not_consume_verifier_work():
    generator = CountingGenerator(["same answer"] * 5 + ["different answer"])
    verifier = CountingVerifier()
    ttc = TestTimeCompute(generator, verifier,
                          budget=TTCBudget(max_candidates=6, max_depth=1))
    result = ttc.search("q")
    normalized = {" ".join(c["text"].lower().split()) for c in result["candidates"]}
    assert len(normalized) == len(result["candidates"])
    assert verifier.calls == len(result["candidates"])


def test_selection_uses_verifier_scores_with_deterministic_ties():
    generator = CountingGenerator(["first candidate", "second candidate",
                                   "third candidate"])
    ttc = TestTimeCompute(generator, CountingVerifier(),
                          budget=TTCBudget(max_candidates=3, max_depth=1))
    result = ttc.search("q")
    assert result["status"] == "selected"
    selected = result["selected"]
    assert selected["selected"] is True
    assert "score" in selected and "verifier" in selected
    assert result["trace"]["selected_id"] == selected["id"]
    assert result["trace"]["selected_score"] == selected["score"]

    constant = TestTimeCompute(generator, CallableVerifier(lambda p, c: 1.0),
                               budget=TTCBudget(max_candidates=3, max_depth=1))
    first = constant.search("q")
    second = constant.search("q")
    assert first["selected"]["id"] == 0
    assert second["selected"]["id"] == first["selected"]["id"]


def test_all_rejected_candidates_yield_no_selection():
    generator = CountingGenerator(["low quality"])
    verifier = CallableVerifier(lambda prompt, candidate: -1.0)
    ttc = TestTimeCompute(generator, verifier,
                          budget=TTCBudget(max_candidates=2, max_depth=1,
                                           min_verifier_score=0.5))
    result = ttc.search("q")
    assert result["status"] == "no_candidate"
    assert result["stop_reason"] == "all_candidates_rejected"
    assert result["selected"] is None
    assert all(candidate["rejected"] for candidate in result["candidates"])


def test_no_candidate_result_when_generator_returns_nothing():
    ttc = TestTimeCompute(lambda prompt, **kwargs: "", CountingVerifier(),
                          budget=TTCBudget(max_candidates=2, max_depth=1))
    result = ttc.search("q")
    assert result["status"] == "no_candidate"
    assert result["selected"] is None


def test_keyword_verifier_prefers_expected_content():
    verifier = KeywordVerifier(["berlin", "capital"])
    assert verifier.score("q", "Berlin is the capital of Germany") > verifier.score(
        "q", "Paris is a city"
    )
    evidence = verifier.evidence("q", "Berlin is the capital")
    assert evidence["verifier"] == "keyword-overlap"


def test_budget_validation_rejects_invalid_limits():
    with pytest.raises(ValueError):
        TTCBudget(max_candidates=0)
    with pytest.raises(ValueError):
        TTCBudget(max_depth=0)
    with pytest.raises(ValueError):
        TTCBudget(max_tokens=0)
    with pytest.raises(ValueError):
        TTCBudget(max_seconds=0)


# ---------------------------------------------------------------------------
# Persistence safety
# ---------------------------------------------------------------------------
def _store(tmp_path):
    return KnowledgeStore(str(tmp_path / "kb.db"), dim=32)


def test_unselected_candidates_are_never_persisted(tmp_path):
    store = _store(tmp_path)
    generator = CountingGenerator(["candidate alpha", "candidate beta", "candidate gamma"])
    ttc = TestTimeCompute(
        generator, CallableVerifier(lambda p, c: float(len(c)), name="length"),
        budget=TTCBudget(max_candidates=3, max_depth=1),
    )
    result = ttc.search("q")
    assert result["selected"] is not None

    with pytest.raises(PermissionError):
        TestTimeCompute.persist_selected(store, result, approved=False)
    assert store.active_count() == 0

    written = TestTimeCompute.persist_selected(store, result, approved=True)
    assert written["status"] == "persisted"
    assert store.active_count() == 1
    assert store.get_fact(written["fact_id"])["text"] == result["selected"]["text"]
    for candidate in result["candidates"]:
        if candidate["id"] != result["selected"]["id"]:
            assert candidate["text"] not in store.get_active_texts()


def test_rejected_selection_is_not_persisted(tmp_path):
    store = _store(tmp_path)
    ttc = TestTimeCompute(
        CountingGenerator(["bad answer"]),
        CallableVerifier(lambda p, c: -1.0),
        budget=TTCBudget(max_candidates=1, max_depth=1, min_verifier_score=0.0),
    )
    result = ttc.search("q")
    outcome = TestTimeCompute.persist_selected(store, result, approved=True)
    assert outcome["status"] == "skipped"
    assert store.active_count() == 0


def test_search_does_not_touch_a_knowledge_store(tmp_path):
    store = _store(tmp_path)
    store.insert_fact("existing fact", "test", 0.0, 0, 0.5,
                      ExactDenseIndex(32).vectors.new_zeros(32))
    before = store.get_active_texts()
    ttc = TestTimeCompute(CountingGenerator(["generated"]),
                          CallableVerifier(lambda p, c: 1.0),
                          budget=TTCBudget(max_candidates=2, max_depth=1))
    ttc.search("q")
    assert store.get_active_texts() == before
