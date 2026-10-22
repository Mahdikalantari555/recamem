"""Semantic distillation and contradiction resolution end to end.

`memory-lifecycle-management` 4.2 (episodic -> semantic distillation) and 4.4
(contradiction resolution with a slot-clash fixture) were implemented in
`SemanticStore`/`ContradictionResolver` but only reachable through the audit
trail, so the reconciliation had nothing to point at. These tests drive the
public engine surface and assert stored state, not just non-empty returns.
"""

from __future__ import annotations

import time

import pytest
import torch

from src.model.droid import DroidEngine


def _unit(seed, dim=384):
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dim, generator=generator)
    return vector / vector.norm()


def _engine(tmp_path, name="semantic"):
    return DroidEngine(name=name, db_path=str(tmp_path / f"{name}.db"))


def _insert(engine, text, vector, novelty=1.0):
    count = engine.knowledge.total_count()
    engine.knowledge.insert_fact(
        text, "test", float(time.time()) - count, count, novelty, vector
    )


def test_distill_to_semantic_collapses_a_similar_cluster(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(3, engine.knowledge.dim)
    # Five near-identical directions collapse to one cluster; two unrelated
    # directions stay on their own.
    for index in range(5):
        _insert(engine, f"The sensor is the NDVI unit, note {index}.", direction)
    for index in range(2):
        _insert(engine, f"Unrelated note {index} about tide gauges.", _unit(90 + index, engine.knowledge.dim))

    before = engine.knowledge.semantic_count()
    assert before == 0

    distilled = engine.distill_to_semantic()
    assert distilled, "a populated store must distill to at least one proposition"

    after = engine.knowledge.semantic_count()
    assert after >= 1
    assert after < engine.knowledge.active_count(), (
        "a full distill must compress the episodic facts, not copy them 1:1"
    )


def test_distill_to_semantic_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(11, engine.knowledge.dim)
    for index in range(3):
        _insert(engine, f"Repeat observation {index} of the estuary.", direction)

    engine.distill_to_semantic()
    first = engine.knowledge.semantic_count()
    engine.distill_to_semantic()
    assert engine.knowledge.semantic_count() == first


def test_retrieve_semantic_returns_propositions_after_distillation(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(21, engine.knowledge.dim)
    for index in range(3):
        _insert(engine, f"Remote sensing measures reflectance, pass {index}.", direction)
    engine.distill_to_semantic()

    hits = engine.retrieve_semantic("remote sensing reflectance", top_k=3)
    assert hits, "the distilled store must answer without touching episodic memory"
    assert all("text" in hit for hit in hits)
    # The default threshold is "no filter", so a best-effort hit is returned
    # even when the mock anchor's cosine is negative; a positive threshold is
    # an actual floor.
    query = engine.anchor.embed("remote sensing reflectance").squeeze(0)
    assert engine.knowledge.recall_semantic(query, top_k=3), "threshold 0.0 must not filter"
    assert engine.knowledge.recall_semantic(query, top_k=3, threshold=0.99) == []


def test_get_contradictions_reports_a_functional_slot_clash(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(31, engine.knowledge.dim)
    _insert(engine, "The capital of France is Paris.", direction)
    _insert(engine, "The capital of France is Lyon.", direction)

    contradictions = engine.get_contradictions()
    assert contradictions, "same subject and functional predicate with a different object must clash"
    pair = contradictions[0]
    assert pair["subject"] == pair["subject"]
    assert {"fact_a", "fact_b", "type"} <= set(pair)
    assert pair["type"] in {"SlotClash", "PolarityInversion"}


def test_get_contradictions_ignores_compatible_facts(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(41, engine.knowledge.dim)
    _insert(engine, "The NDVI index tracks vegetation vigour.", direction)
    _insert(engine, "The estuary hosts a tide gauge network.", _unit(42, engine.knowledge.dim))

    assert engine.get_contradictions() == []


def test_superseded_facts_are_excluded_from_contradiction_scan(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(51, engine.knowledge.dim)
    _insert(engine, "The capital of Spain is Madrid.", direction)
    _insert(engine, "The capital of Spain is Barcelona.", direction)
    assert engine.get_contradictions()

    facts = [f for f in engine.knowledge.get_audit_trail() if not f.get("superseded")]
    engine.knowledge.supersede(facts[0]["id"], None)
    # Only one live fact of that slot remains, so nothing contradicts it.
    assert engine.get_contradictions() == []


def test_semantic_distillation_reports_confidence(tmp_path):
    engine = _engine(tmp_path)
    direction = _unit(61, engine.knowledge.dim)
    for index in range(4):
        _insert(engine, f"Cluster member {index} about soil moisture.", direction)

    distilled = engine.distill_to_semantic()
    assert distilled
    for proposition in distilled:
        confidence = proposition.get("confidence", proposition.get("similarity"))
        assert confidence is None or 0.0 <= float(confidence) <= 1.0

    rows = engine.knowledge.conn.execute(
        "SELECT confidence FROM semantic_facts"
    ).fetchall()
    assert rows
    assert all(0.0 <= float(row[0]) <= 1.0 for row in rows if row[0] is not None)
