"""Memory-health metrics that the ledger owed a concrete test for.

`plastic-memory-lifelong` 7.2 (saturation) and 7.4 (interference) were
implemented inside `DroidEngine.get_memory_health` but no test named them, so
the reconciliation could only call them implemented-and-unverified. These tests
pin the arithmetic, not just the presence of a key.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from src.model.droid import DroidEngine


def _engine(tmp_path, name="health"):
    return DroidEngine(name=name, db_path=str(tmp_path / f"{name}.db"))


def _unit(seed, dim=384):
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dim, generator=generator)
    return vector / vector.norm()


def _insert(engine, text, vector):
    """INSERT a fact at the store's own width (a narrower blob corrupts the cache)."""
    count = engine.knowledge.total_count()
    engine.knowledge.insert_fact(text, "test", float(time.time()), count, 1.0, vector)


def test_saturation_is_active_facts_over_the_capacity_bound(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("Alpha carries the calibration marker.")
    health = engine.get_memory_health()

    assert health["active_facts"] >= 1
    assert health["K_max"] == engine.K_max
    # The fact term alone would be active/K_max; the associative state adds a
    # bounded term and the result is clamped to [0, 1].
    assert 0.0 <= health["saturation"] <= 1.0
    assert health["saturation"] >= health["active_facts"] / max(1, engine.K_max) - 1e-9


def test_saturation_is_clamped_and_grows_with_active_facts(tmp_path):
    engine = _engine(tmp_path)
    before = engine.get_memory_health()["saturation"]
    for index in range(6):
        engine.teach(f"Marker {index} sits at station {index} in the north sector.")
    after = engine.get_memory_health()
    assert after["active_facts"] > 1
    assert after["saturation"] >= before
    assert after["saturation"] <= 1.0


def test_forgetting_rate_counts_facts_below_the_accessibility_threshold(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("Fresh fact about the south ridge.")
    fresh = engine.get_memory_health()["forgetting_rate"]
    assert fresh == pytest.approx(0.0, abs=1e-9)

    # Backdate one fact by ten days: exp(-10) * access_count falls under 0.1.
    row = engine.knowledge.conn.execute(
        "SELECT id FROM facts ORDER BY id LIMIT 1"
    ).fetchone()
    engine.knowledge.conn.execute(
        "UPDATE facts SET timestamp = ?, last_retrieved = ?, access_count = 1 WHERE id = ?",
        (time.time() - 10 * 86400, time.time() - 10 * 86400, row[0]),
    )
    engine.knowledge.conn.commit()
    assert engine.get_memory_health()["forgetting_rate"] > 0.0


def test_forgetting_rate_matches_the_decayed_accessibility_formula(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("Fact one about the estuary.")
    engine.teach("Fact two about the delta.")
    rows = engine.knowledge.conn.execute(
        "SELECT id FROM facts ORDER BY id"
    ).fetchall()
    assert len(rows) == 2

    ten_days = 10 * 86400
    engine.knowledge.conn.execute(
        "UPDATE facts SET access_count = 5, last_retrieved = ?, timestamp = ? WHERE id = ?",
        (time.time() - ten_days, time.time() - ten_days, rows[0][0]),
    )
    engine.knowledge.conn.execute(
        "UPDATE facts SET access_count = 1, last_retrieved = ?, timestamp = ? WHERE id = ?",
        (time.time(), time.time(), rows[1][0]),
    )
    engine.knowledge.conn.commit()

    expected_low = 1 if 5 * math.exp(-10.0) < 0.1 else 0
    rate = engine.get_memory_health()["forgetting_rate"]
    assert rate == pytest.approx(expected_low / 2, abs=1e-6)


def test_interference_is_the_mean_off_diagonal_cosine(tmp_path):
    engine = _engine(tmp_path)
    # Identical directions: every off-diagonal cosine is 1.0.
    direction = _unit(5, engine.knowledge.dim)
    for index in range(3):
        _insert(engine, f"parallel fact {index}", direction)
    assert engine.get_memory_health()["interference"] == pytest.approx(1.0, abs=1e-4)


def test_interference_is_low_for_orthogonal_facts(tmp_path):
    engine = _engine(tmp_path)
    for index in range(4):
        vector = torch.zeros(engine.knowledge.dim)
        vector[index] = 1.0
        _insert(engine, f"orthogonal fact {index}", vector)
    assert engine.get_memory_health()["interference"] == pytest.approx(0.0, abs=1e-6)


def test_health_reports_unavailable_retrieval_quality_without_judgements(tmp_path):
    engine = _engine(tmp_path)
    engine.teach("A fact with no feedback attached.")
    health = engine.get_memory_health()
    assert health["retrieval_quality"] is None
    assert health["retrieval_health"]["available"] is False
    assert health["approval_rate"] is None
    assert health["approval_samples"] == 0
