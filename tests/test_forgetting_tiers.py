"""Forgetting never destroys data: tiers, utility, probes (recamem/forgetting)."""

import math
import time

import pytest

from recamem import forgetting
from recamem.engine import DroidEngine
pytestmark = pytest.mark.phase0



@pytest.fixture()
def engine():
    return DroidEngine(name="forget-test")


def test_utility_matches_the_documented_formula():
    now = 1_000_000.0
    access, last, novelty = 4, now - 3600, 0.5
    tau = forgetting.DEFAULT_TAU_HALF_SECONDS
    expected = access * math.exp(-(now - last) / tau) + 1.0 * novelty
    assert forgetting.utility(access, last, novelty, now=now) == pytest.approx(expected)


def test_capacity_retires_the_lowest_utility_rows_without_deleting(engine):
    for index in range(5):
        engine.teach(f"The sensor {index} measures moisture at site {index}.")
    ids = [row[0] for row in engine.knowledge.conn.execute("SELECT id FROM facts ORDER BY id")]
    # Make the first fact the least useful: it is old and never retrieved.
    with engine.knowledge._write_lock:
        engine.knowledge.conn.execute(
            "UPDATE facts SET access_count=0, last_retrieved=?, novelty=0.0 WHERE id=?",
            (time.time() - 30 * 24 * 3600, ids[0]),
        )
    before_rows = engine.knowledge.total_count()
    report = forgetting.enforce_capacity(engine.knowledge, K_max=3)
    assert engine.knowledge.active_count() <= 3
    assert ids[0] in report["retired"]
    assert engine.knowledge.total_count() == before_rows, "retirement never deletes"
    assert engine.knowledge.tier_of(ids[0]) == "retired"


def test_no_delete_statement_runs_in_a_forgetting_cycle(engine):
    for index in range(3):
        engine.teach(f"The gateway {index} forwards readings to the server.")
    statements = []
    engine.knowledge.conn.set_trace_callback(statements.append)
    forgetting.forgetting_cycle(engine, K_max=2)
    engine.knowledge.conn.set_trace_callback(None)
    assert not any("DELETE FROM facts" in stmt.upper() for stmt in statements)


def test_consolidate_then_forget_keeps_unabsorbed_facts(engine):
    engine.teach("The calibration constant is 0.98.")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    report = forgetting.consolidate_then_forget(
        engine, candidates=[fact_id], probe=lambda text: 0.0
    )
    assert engine.knowledge.tier_of(fact_id) == "active"
    assert report["kept"][0]["status"] == "kept"
    assert "below threshold" in report["kept"][0]["reason"]


def test_absorbed_fact_is_retired_and_remains_auditable(engine):
    engine.teach("The project lead is Dr. Mina Rostami.")
    fact_id = engine.knowledge.conn.execute("SELECT id, text, source FROM facts").fetchone()
    report = forgetting.consolidate_then_forget(
        engine, candidates=[fact_id[0]], probe=lambda text: 0.9
    )
    assert report["retired_count"] == 1
    assert engine.knowledge.tier_of(fact_id[0]) == "retired"
    row = engine.knowledge.conn.execute(
        "SELECT text, source FROM facts WHERE id=?", (fact_id[0],)
    ).fetchone()
    assert row == (fact_id[1], fact_id[2]), "text, source and history stay queryable"


def test_retired_facts_stay_out_of_recall_then_come_back_on_restore(engine):
    engine.teach("The mixing station is on site B.")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    engine.knowledge.retire_fact(fact_id, reason="test")
    assert engine.recall("mixing station", top_k=3) == []
    assert forgetting.restore(engine.knowledge, fact_id) is True
    assert engine.knowledge.tier_of(fact_id) == "active"
    hits = engine.recall("mixing station", top_k=3)
    assert any(int(hit["id"]) == int(fact_id) for hit in hits)


def test_active_count_excludes_retired_rows(engine):
    engine.teach("The gateway forwards readings every minute.")
    fact_id = engine.knowledge.conn.execute("SELECT id FROM facts").fetchone()[0]
    assert engine.knowledge.active_count() == 1
    engine.knowledge.retire_fact(fact_id)
    assert engine.knowledge.active_count() == 0
    assert engine.knowledge.total_count() == 1
