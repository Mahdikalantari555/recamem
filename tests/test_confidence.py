"""Contract tests for src/model/confidence.py.

Invariants under test (see openspec/specs/model/confidence-estimation):
- every component score is clamped to [0, 1];
- fusion is the (optionally weighted) harmonic mean, not an arithmetic mean;
- tiers split at 0.8 (exclusive high), and 0.5 (inclusive medium);
- the low tier carries an explicit uncertainty prefix and cites facts.
"""

import math

import pytest

from src.model.confidence import (
    confidence_tier,
    estimate_confidence,
    format_low_confidence_response,
    fuse_confidence,
    generation_confidence,
    retrieval_confidence,
)


# --------------------------------------------------------------------------- #
# Retrieval confidence
# --------------------------------------------------------------------------- #
def test_retrieval_confidence_is_bounded():
    assert retrieval_confidence(-1.0) == 0.0
    assert retrieval_confidence(2.0) == 1.0
    assert retrieval_confidence(0.8, surprise=50.0, health_metrics={"interference": 5.0}) >= 0.0


def test_retrieval_confidence_uses_similarity_as_base():
    for sim in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert retrieval_confidence(sim) == pytest.approx(sim)


def test_retrieval_confidence_surprise_factor():
    # surprise 0 -> factor 1.0, so the score is unchanged.
    assert retrieval_confidence(0.8, surprise=0.0) == pytest.approx(0.8)
    # surprise 5 clamps the factor at 0.5 -> 0.8 * (0.8 + 0.2 * 0.5)
    assert retrieval_confidence(0.8, surprise=5.0) == pytest.approx(0.8 * 0.9)
    # surprise 2 -> factor 0.8 -> 0.8 * (0.8 + 0.16)
    assert retrieval_confidence(0.8, surprise=2.0) == pytest.approx(0.8 * 0.96)


def test_retrieval_confidence_health_metrics_factor():
    conf = retrieval_confidence(
        0.9,
        health_metrics={"interference": 0.7, "retrieval_quality": 0.8},
    )
    expected = 0.9 * (1.0 - 0.7 * 0.3) * (0.7 + 0.3 * 0.8)
    assert conf == pytest.approx(expected)


def test_retrieval_confidence_health_defaults():
    # An empty metrics dict is falsy: no adjustment is applied.
    assert retrieval_confidence(1.0, health_metrics={}) == pytest.approx(1.0)
    # Missing keys fall back to the documented 0.5 defaults.
    assert retrieval_confidence(1.0, health_metrics={"interference": 0.5}) == pytest.approx(
        1.0 * (1.0 - 0.5 * 0.3) * (0.7 + 0.3 * 0.5)
    )


# --------------------------------------------------------------------------- #
# Generation confidence
# --------------------------------------------------------------------------- #
def test_generation_confidence_default_is_neutral():
    assert generation_confidence() == 0.5


def test_generation_confidence_from_logprobs():
    logprobs = [-0.1, -0.1, -0.1]
    assert generation_confidence(token_logprobs=logprobs) == pytest.approx(math.exp(-0.1))


def test_generation_confidence_zeroes_very_low_logprobs():
    assert generation_confidence(token_logprobs=[-20.0, -20.0]) == 0.0


def test_generation_confidence_self_consistency_blend():
    assert generation_confidence(self_consistency=1.0) == pytest.approx(0.5 * 0.6 + 0.4)
    assert generation_confidence(self_consistency=0.0) == pytest.approx(0.5 * 0.6)


def test_generation_confidence_entropy_damping():
    assert generation_confidence(entropy=1.0) == pytest.approx(0.25)
    assert generation_confidence(entropy=0.0) == pytest.approx(0.5)


def test_generation_confidence_is_bounded():
    assert generation_confidence(token_logprobs=[0.5], self_consistency=1.0) <= 1.0
    assert generation_confidence(token_logprobs=[-5.0], entropy=1.0) >= 0.0
    assert generation_confidence(entropy=10.0) == 0.0


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def test_fuse_confidence_is_harmonic_mean():
    assert fuse_confidence(0.5, 0.5) == pytest.approx(0.5)
    assert fuse_confidence(1.0, 0.5) == pytest.approx(2.0 / (1.0 / 1.0 + 1.0 / 0.5))
    # Harmonic mean penalizes imbalance: it is below the arithmetic mean.
    assert fuse_confidence(0.9, 0.1) < (0.9 + 0.1) / 2


def test_fuse_confidence_weighted():
    fused = fuse_confidence(0.8, 0.2, weights={"retrieval": 0.7, "generation": 0.3})
    assert fused == pytest.approx(1.0 / (0.7 / 0.8 + 0.3 / 0.2))
    # Equal weights reduce to the plain harmonic mean.
    assert fuse_confidence(
        0.6, 0.3, weights={"retrieval": 0.5, "generation": 0.5}
    ) == pytest.approx(2.0 / (1.0 / 0.6 + 1.0 / 0.3))


def test_fuse_confidence_treats_zero_as_epsilon():
    # A zero component must not divide by zero; the fused score collapses instead.
    assert fuse_confidence(1.0, 0.0) < 1e-6
    assert fuse_confidence(0.0, 0.0) < 1e-6


def test_estimate_confidence_aliases_fuse_confidence():
    assert estimate_confidence(0.9, 0.4) == pytest.approx(fuse_confidence(0.9, 0.4))
    assert estimate_confidence(
        0.9, 0.4, weights={"retrieval": 0.25, "generation": 0.75}
    ) == pytest.approx(fuse_confidence(0.9, 0.4, weights={"retrieval": 0.25, "generation": 0.75}))


# --------------------------------------------------------------------------- #
# Tiering and low-confidence formatting
# --------------------------------------------------------------------------- #
def test_confidence_tier_boundaries():
    assert confidence_tier(1.0) == "high"
    assert confidence_tier(0.8000001) == "high"
    assert confidence_tier(0.8) == "medium"
    assert confidence_tier(0.5) == "medium"
    assert confidence_tier(0.4999) == "low"
    assert confidence_tier(0.0) == "low"


def test_low_confidence_response_carries_uncertainty_prefix_and_citations():
    facts = [
        {"text": "Saturn has rings.", "similarity": 0.41},
        {"text": "Mars has two moons.", "similarity": 0.32},
        {"text": "Venus is hot.", "similarity": 0.11},
        {"text": "Pluto is small.", "similarity": 0.05},
    ]
    reply = format_low_confidence_response("tell me about planets", facts, 0.3)
    assert reply.startswith("I'm uncertain")
    assert "confidence 0.30" in reply
    # Citations replace the query echo when at least one fact is present.
    assert "tell me about planets" not in reply
    # Only the top three facts are cited.
    assert "Saturn has rings." in reply
    assert "Mars has two moons." in reply
    assert "Venus is hot." in reply
    assert "Pluto is small." not in reply
    assert "[sim=0.41]" in reply


def test_low_confidence_response_without_facts():
    reply = format_low_confidence_response("what is the airspeed of a swallow", [], 0.1)
    assert "I'm uncertain" in reply
    assert "what is the airspeed of a swallow" in reply
    assert "I don't have relevant facts in memory" in reply
