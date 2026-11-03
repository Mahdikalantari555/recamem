"""Every game action maps to a real engine call."""
from __future__ import annotations

import pytest

from recamem import companion


class SpyEngine:
    def __init__(self):
        self.calls = []
        self.facts = 0
        self.feedback = []
        self.logs = []
        self.knowledge = self._K(self)

    def approve(self, fact_id):
        self.calls.append(("approve", fact_id)); self.feedback.append({"action": "approve"})
        return {"status": "approved"}
    def reject(self, fact_id):
        self.calls.append(("reject", fact_id)); self.feedback.append({"action": "reject"})
        return {"status": "rejected"}
    def correct(self, fact_id, text):
        self.calls.append(("correct", fact_id)); return {"status": "corrected"}
    def teach(self, text, source="user_text"):
        self.calls.append(("teach", text, source)); self.facts += 1
        return {"propositions": 1}
    def sleep(self):
        self.calls.append(("sleep",)); self.logs.append({"timestamp": 1.0, "message": "sleep done"})
        return {"status": "ok"}
    def chat(self, text):
        self.calls.append(("chat", text)); return "a real answer"
    def get_feedback_log(self): return self.feedback
    def get_memory_health(self): return {}
    def ttt_status(self): return {}
    class _K:
        def __init__(self, engine): self.engine = engine
        def total_count(self): return self.engine.facts
        def active_count(self): return self.engine.facts
        def retrieval_health(self): return {"available": False}


def _setup():
    engine = SpyEngine()
    ctx = {"events": [], "last_fact_id": 42}
    def state_fn():
        return companion.measured_state(engine, anchor="mock", backend="memory-only", profile="demo")
    return engine, ctx, state_fn


def test_pet_and_treat_call_approve():
    engine, ctx, state_fn = _setup()
    for action in ("pet", "treat"):
        outcome = companion.dispatch_action(engine, action, {}, ctx, state_fn)
        assert outcome["event"]["engine_call"] == "approve(42)"
    assert [call[0] for call in engine.calls] == ["approve", "approve"]


def test_feed_scold_correct_sleep_and_play_are_real_calls():
    engine, ctx, state_fn = _setup()
    companion.dispatch_action(engine, "feed", {"text": "A fact."}, ctx, state_fn)
    companion.dispatch_action(engine, "scold", {}, ctx, state_fn)
    companion.dispatch_action(engine, "correct", {"text": "Fixed."}, ctx, state_fn)
    companion.dispatch_action(engine, "sleep", {}, ctx, state_fn)
    play = companion.dispatch_action(engine, "play", {"text": "hello"}, ctx, state_fn)
    assert [call[0] for call in engine.calls] == ["teach", "reject", "correct", "sleep", "chat"]
    assert play["result"]["response"] == "a real answer"


def test_action_without_target_is_rejected_not_rewarded():
    engine, ctx, state_fn = _setup()
    ctx["last_fact_id"] = None
    with pytest.raises(companion.CompanionActionError, match="recalled fact_id"):
        companion.dispatch_action(engine, "pet", {}, ctx, state_fn)
    with pytest.raises(companion.CompanionActionError, match="non-empty text"):
        companion.dispatch_action(engine, "feed", {}, ctx, state_fn)
    assert engine.calls == []
    assert ctx["events"] == []


def test_unknown_action_is_rejected():
    engine, ctx, state_fn = _setup()
    with pytest.raises(companion.CompanionActionError, match="unknown action"):
        companion.dispatch_action(engine, "decorate", {}, ctx, state_fn)


def test_event_log_records_before_and_after():
    engine, ctx, state_fn = _setup()
    outcome = companion.dispatch_action(engine, "feed", {"text": "A fact."}, ctx, state_fn)
    assert outcome["event"]["before"]["facts_active"] == 0
    assert outcome["event"]["before"]["facts_active"] == 0
    assert outcome["event"]["after"]["facts_active"] == 1
    assert outcome["event"]["id"] and ctx["events"][0]["id"] == outcome["event"]["id"]
