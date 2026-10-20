"""Contract tests for src/model/decision_head.py (System-1 hyperspherical routing).

Invariants under test (see openspec/specs/system1-decision):
- routing is cosine argmax over the prototype matrix;
- confidence is derived from the normalized softmax entropy and stays in [0, 1];
- a query whose best cosine falls below the threshold is reported as "ambiguous";
- the structured decision object is JSON serializable.
"""

import json
import math

import pytest
import torch

from src.model.decision_head import DecisionHead


DEFAULT_NAMES = ["chat", "recall", "teach", "tool", "fast_recall", "slow_synthesis"]


def test_default_construction_has_normalized_prototypes():
    head = DecisionHead(dim=32)
    assert head.K == len(DEFAULT_NAMES)
    assert head.prototype_names == DEFAULT_NAMES
    assert head.P.shape == (len(DEFAULT_NAMES), 32)
    norms = torch.linalg.vector_norm(head.P, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_decide_returns_argmax_prototype():
    head = DecisionHead(dim=32)
    with torch.no_grad():
        head.P = torch.nn.functional.normalize(torch.randn(6, 32, generator=torch.Generator().manual_seed(3)), dim=1)
    query = head.P[2].clone()
    decision = head.decide(query)
    assert decision["action"] == head.prototype_names[2]
    assert decision["max_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_decide_probabilities_and_confidence_bounds():
    head = DecisionHead(dim=16)
    query = torch.randn(16, generator=torch.Generator().manual_seed(7))
    decision = head.decide(query)

    probs = decision["probabilities"]
    assert len(probs) == head.K
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert 0.0 <= decision["confidence"] <= 1.0
    assert math.isfinite(decision["confidence"])
    assert decision["latency_ms"] >= 0.0


def test_decide_confidence_is_high_for_aligned_query():
    head = DecisionHead(dim=32)
    decision = head.decide(head.P[0].clone())
    # One prototype dominates the temp=0.1 softmax -> low entropy -> high confidence.
    assert decision["confidence"] > 0.5


def test_decide_rejects_out_of_domain_query():
    head = DecisionHead(dim=32)
    # An impossible threshold forces every query below it.
    decision = head.decide(head.P[0].clone(), threshold=1.5)
    assert decision["action"] == "ambiguous"
    assert decision["confidence"] <= 0.5 * decision["max_similarity"] / 1.5 + 1e-9


def test_decide_accepts_unnormalized_and_mismatched_query_shapes():
    head = DecisionHead(dim=8)
    long_query = torch.randn(12, generator=torch.Generator().manual_seed(1))
    short_query = torch.randn(4, generator=torch.Generator().manual_seed(2))
    batched = torch.randn(1, 8, generator=torch.Generator().manual_seed(3))

    for query in (long_query, short_query, batched):
        decision = head.decide(query)
        assert decision["action"] in head.prototype_names
        assert len(decision["similarities"]) == head.K


def test_decide_result_is_json_serializable():
    head = DecisionHead(dim=16)
    decision = head.decide(torch.eye(16)[0])
    payload = json.loads(json.dumps(decision))
    assert payload["action"] == decision["action"]
    assert payload["prototype_names"] == DEFAULT_NAMES
    assert set(payload) >= {
        "action",
        "probabilities",
        "confidence",
        "latency_ms",
        "max_similarity",
        "similarities",
    }


def test_custom_prototypes_are_normalized_and_routed():
    prototypes = {
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta": [0.0, 2.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 0.5, 0.0],
    }
    head = DecisionHead(prototypes=prototypes, dim=4)
    assert head.prototype_names == ["alpha", "beta", "gamma"]
    assert head.K == 3
    norms = torch.linalg.vector_norm(head.P, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    decision = head.decide(torch.tensor([0.0, 3.0, 0.0, 0.0]))
    assert decision["action"] == "beta"


# --------------------------------------------------------------------------- #
# Ridge probe
# --------------------------------------------------------------------------- #
def test_train_probe_solves_ridge_regression():
    torch.manual_seed(0)
    head = DecisionHead(dim=8)
    X = torch.randn(12, 8)
    labels = [0, 1, 2]
    Y = torch.zeros(12, 3)
    for i in range(12):
        Y[i, labels[i % 3]] = 1.0

    W = head.train_probe(X, Y, lambda_reg=1e-3)
    assert W.shape == (8, 3)
    assert head.W_probe is not None
    assert head.lambda_reg == pytest.approx(1e-3)

    # The fit should recover the label for most training rows.
    preds = (X @ W).argmax(dim=1)
    assert (preds == torch.tensor([labels[i % 3] for i in range(12)])).float().mean() > 0.9


def test_train_probe_validates_inputs():
    head = DecisionHead(dim=8)
    with pytest.raises(ValueError):
        head.train_probe(torch.randn(8), torch.randn(1, 3))
    with pytest.raises(ValueError):
        head.train_probe(torch.randn(6, 8), torch.randn(1, 3))
    with pytest.raises(ValueError):
        head.train_probe(torch.randn(4, 8), torch.randn(4, 3))


def test_probe_predict_falls_back_to_prototype_routing():
    head = DecisionHead(dim=8)
    query = torch.randn(8, generator=torch.Generator().manual_seed(5))
    assert head.W_probe is None
    assert head.probe_predict(query)["action"] == head.decide(query)["action"]


def test_probe_predict_uses_learned_head():
    torch.manual_seed(1)
    head = DecisionHead(dim=8)
    X = torch.randn(10, 8)
    Y = torch.zeros(10, 2)
    Y[:5, 0] = 1.0
    Y[5:, 1] = 1.0
    head.train_probe(X, Y)
    head.probe_labels = ["left", "right"]

    decision = head.probe_predict(X[0])
    assert decision["action"] == "left"
    assert decision["prototype_names"] == ["left", "right"]
    assert 0.0 <= decision["confidence"] <= 1.0
    assert sum(decision["probabilities"]) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_state_dict_round_trip():
    source = DecisionHead(dim=8)
    state = source.state_dict()
    assert state["prototype_names"] == DEFAULT_NAMES
    assert state["dim"] == 8

    target = DecisionHead(dim=4, prototype_names=["only"])
    target.load_state_dict(state)
    assert target.dim == 8
    assert target.prototype_names == DEFAULT_NAMES
    assert target.K == len(DEFAULT_NAMES)
    assert torch.equal(target.P, source.P)


def test_to_json_reports_architecture():
    head = DecisionHead(dim=8)
    payload = json.loads(head.to_json())
    assert payload == {
        "prototype_names": DEFAULT_NAMES,
        "dim": 8,
        "K": len(DEFAULT_NAMES),
    }


@pytest.mark.parametrize("threshold", [0.35, 0.0, -0.5])
def test_antipodal_query_has_bounded_confidence(threshold):
    head = DecisionHead(prototypes={"a": [1.0, 0.0], "b": [1.0, 0.0]}, dim=2)
    result = head.decide(torch.tensor([-1.0, 0.0]), threshold=threshold)
    assert result["action"] == "ambiguous"
    assert 0.0 <= result["confidence"] <= 1.0
