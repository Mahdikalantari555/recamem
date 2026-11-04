"""Companion pet profile is clean: zero domain facts, web search ready."""
from __future__ import annotations

import pytest

from recamem import api
from recamem.manager import DroidManager


@pytest.fixture()
def manager(tmp_path):
    mgr = DroidManager(base_dir=str(tmp_path / "agents"))
    yield mgr
    mgr.shutdown() if hasattr(mgr, "shutdown") else None


def test_fresh_companion_pet_has_zero_facts_and_search(manager):
    engine = manager.get_droid("companion-pet")
    assert engine.name == "companion-pet"
    assert engine.knowledge.active_count() == 0
    assert engine.research_enabled is True
    assert engine.search_learner is not None
    assert engine.search_learner.available is True
    assert engine.research_config["provider"] == "duckduckgo"
    assert engine.research_config["max_queries_per_session"] == 8
    # No remote-sensing contamination.
    hits = engine.recall("remote sensing synthetic aperture radar", top_k=5)
    assert hits == [] or all("remote" not in str(h).lower() for h in hits)


def test_companion_pet_persists_clean_research_config(manager, tmp_path):
    engine = manager.get_droid("companion-pet")
    manager.save_droid("companion-pet")
    loaded = DroidManager(base_dir=str(tmp_path / "agents")).get_droid("companion-pet")
    assert loaded.research_enabled is True
    assert loaded.research_config["provider"] == "duckduckgo"
    assert loaded.knowledge.active_count() == 0


def test_companion_pet_absorbs_personal_chat(manager):
    engine = manager.get_droid("companion-pet")
    reply = engine.chat("My name is Alex and I have two cats.")
    assert isinstance(reply, str) and len(reply.strip()) > 0
    assert engine.knowledge.active_count() >= 1
    # A follow-up question should not crash and should return a string.
    followup = engine.chat("What is my name?")
    assert isinstance(followup, str)


def test_api_default_profile_is_companion_pet(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAMEM_ANCHOR", "mock")
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path))
    runtime = api.Runtime(base_dir=str(tmp_path))
    assert runtime.profile == "companion-pet"