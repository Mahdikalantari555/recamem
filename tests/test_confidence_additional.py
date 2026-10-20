"""Contract tests for `src/model/confidence.py` (surface-test-coverage 3.1).

Covered: component bounds, the harmonic-mean fusion formula, the 0.5/0.8 tier
boundaries, and the low-confidence clarification prefix.
"""

import pytest

from src.model.confidence import (
    confidence_tier,
    estimate_confidence,
    fuse_confidence,
    generation_confidence,
    format_low_confidence_response,
    retrieval_confidence,
)


def test_retrieval_confidence_stays_in_unit_interval():
    for sim in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert 0.0 <= retrieval_confidence(sim) <= 1.0


def test_retrieval_confidence_is_monotonic_in_similarity():
    assert retrieval_confidence(0.9) >= retrieval_confidence(0.5)


def test_retrieval_confidence_exact_at_similarity_one():
    assert retrieval_confidence(1.0) == 1.0


def test_retrieval_confidence_surprise_and_health_adjustments():
    base = retrieval_confidence(0.8)
    assert 0.0 <= retrieval_confidence(0.8, surprise=2.0) <= base
    assert 0.0 <= retrieval_confidence(0.8, health_metrics={"interference": 0.9, "retrieval_quality": 0.1}) <= 1.0


def test_generation_confidence_stays_in_unit_interval():
    assert 0.0 <= generation_confidence(token_logprobs=[-0.5, -1.0]) <= 1.0
    assert 0.0 <= generation_confidence(self_consistency=0.7) <= 1.0
    assert 0.0 <= generation_confidence(entropy=0.4) <= 1.0


def test_generation_confidence_exact_from_logprobs():
    # mean logprob 0.0 -> exp(0) = 1.0
    assert generation_confidence(token_logprobs=[0.0, 0.0]) == 1.0


def test_fuse_confidence_matches_weighted_harmonic_mean():
    Cr, Cg = 0.6, 0.3
    expected = 2.0 / (1.0 / Cr + 1.0 / Cg)
    assert fuse_confidence(Cr, Cg) == pytest.approx(expected, rel=1e-6)

    weights = {"retrieval": 0.7, "generation": 0.3}
    expected = (0.7 + 0.3) / (0.7 / Cr + 0.3 / Cg)
    assert fuse_confidence(Cr, Cg, weights=weights) == pytest.approx(expected, rel=1e-6)


def test_fuse_confidence_equal_weights_reduce_to_simple_harmonic_mean():
    Cr, Cg = 0.4, 0.8
    assert fuse_confidence(Cr, Cg, weights={"retrieval": 0.5, "generation": 0.5}) == fuse_confidence(Cr, Cg)


def test_fuse_confidence_penalizes_imbalance():
    assert fuse_confidence(1.0, 0.1) < fuse_confidence(0.55, 0.55)


def test_fuse_confidence_zeros_do_not_crash():
    assert 0.0 <= fuse_confidence(0.0, 0.5) <= 1.0
    assert 0.0 <= fuse_confidence(0.0, 0.0) <= 1.0


def test_confidence_tier_boundaries():
    assert confidence_tier(0.9) == "high"
    assert confidence_tier(0.81) == "high"
    assert confidence_tier(0.8) == "medium"   # high is strictly above 0.8
    assert confidence_tier(0.5) == "medium"
    assert confidence_tier(0.49) == "low"
    assert confidence_tier(0.0) == "low"


def test_estimate_confidence_is_fuse_confidence_alias():
    assert estimate_confidence(0.5, 0.5) == fuse_confidence(0.5, 0.5)


def test_format_low_confidence_carries_uncertainty_prefix():
    facts = [{"text": "The sky is blue.", "similarity": 0.42}]
    msg = format_low_confidence_response("what color is the sky?", facts, 0.33)
    assert "I'm uncertain" in msg
    assert "The sky is blue" in msg
    assert "0.33" in msg


def test_format_low_confidence_with_no_facts():
    msg = format_low_confidence_response("what?", [], 0.2)
    assert "I'm uncertain" in msg
    assert "relevant facts in memory" in msg
