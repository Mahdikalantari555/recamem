"""Companion state is measured, never invented."""
from __future__ import annotations

from recamem import companion


class FakeKnowledge:
    def total_count(self): return 12
    def active_count(self): return 10
    def retrieval_health(self):
        return {"available": False, "precision_at_k": None, "reason": "no_labelled_feedback"}


class FakeEngine:
    knowledge = FakeKnowledge()
    logs = []

    def get_memory_health(self): return {"saturation": 0.1}
    def ttt_status(self): return {"allowed": False, "budget": 0}
    def get_feedback_log(self):
        return [{"action": "approve"}, {"action": "reject"}, {"action": "approve"}]


def _state():
    return companion.measured_state(FakeEngine(), anchor="mock", backend="memory-only",
                                    profile="demo")


def test_state_reports_only_engine_signals():
    state = _state()
    assert state["facts_total"] == 12 and state["facts_active"] == 10
    assert state["facts_retired"] == 2
    assert state["feedback_counts"] == {"approve": 2, "reject": 1, "total": 3}
    assert state["retrieval_hit_rate"] is None
    assert state["last_sleep_at"] is None
    assert state["anchor"] == "mock"


def test_unavailable_signal_is_not_replaced_by_a_counter():
    class Broken(FakeEngine):
        knowledge = None
        def get_memory_health(self): raise RuntimeError("unavailable")
        def ttt_status(self): raise RuntimeError("unavailable")
        def get_feedback_log(self): raise RuntimeError("unavailable")
    state = companion.measured_state(Broken(), anchor="mock", backend="memory-only", profile="demo")
    assert state["facts_active"] is None and state["facts_total"] is None
    assert state["memory_health"] == {} and state["feedback_counts"]["total"] == 0


def test_pet_attributes_are_null_when_source_is_missing():
    pet = companion.derive_pet(_state())
    assert pet["energy"]["value"] is None
    assert pet["knowledge"]["value"] == 0.5
    assert pet["bond"]["value"] == 0.2
    assert pet["mood"]["sources"] == ["retrieval_health.precision_at_k", "feedback_counts.approve"]
