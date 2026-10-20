"""Contract tests for `src/model/decision_head.py` (surface-test-coverage 3.2).

Covered: prototype argmax routing, entropy-derived confidence bounds,
out-of-domain rejection, JSON-serializable structured output, and the Ridge
probe training/predict path.
"""

import json

import pytest
import torch

from src.model.decision_head import DecisionHead


def _head(dim: int = 8) -> DecisionHead:
    return DecisionHead(
        prototypes={
            "chat": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "recall": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "teach": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        dim=dim,
    )


def test_default_head_has_six_normalized_prototypes():
    head = DecisionHead(dim=32)
    assert head.K == 6
    assert head.prototype_names == ["chat", "recall", "teach", "tool", "fast_recall", "slow_synthesis"]
    assert head.P.shape == (6, 32)
    norms = torch.linalg.vector_norm(head.P, dim=1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)


def test_decide_returns_the_structured_result_dict():
    head = _head()
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = head.decide(query)

    assert set(result) >= {
        "action",
        "probabilities",
        "confidence",
        "latency_ms",
        "max_similarity",
        "prototype_names",
        "similarities",
    }
    assert result["action"] in head.prototype_names
    assert isinstance(result["latency_ms"], float)
    assert len(result["probabilities"]) == head.K
    assert len(result["similarities"]) == head.K


def test_decide_argmax_picks_closest_prototype():
    head = _head()
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = head.decide(query)
    assert result["action"] == "chat"
    assert result["max_similarity"] == pytest.approx(1.0, abs=1e-6)
    probs = result["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-5
    assert probs[head.prototype_names.index("chat")] > 0.5


def test_decide_confidence_is_bounded():
    head = _head()
    for query in (
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ):
        result = head.decide(query)
        assert 0.0 <= result["confidence"] <= 1.0


def test_decide_below_threshold_returns_ambiguous():
    head = _head()
    # Equal similarity to two prototypes (~0.707 each) stays under a 0.9 threshold.
    query = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = head.decide(query, threshold=0.9)
    assert result["action"] == "ambiguous"


def test_decide_output_is_json_serializable():
    head = _head()
    result = head.decide(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    payload = json.dumps(result)
    assert "\"action\"" in payload
    json.loads(payload)


def test_train_probe_returns_d_k_weights():
    head = _head(dim=8)
    X = torch.randn(8, 8)
    Y = torch.zeros(8, 3)
    Y[0, 0] = 1.0; Y[1, 1] = 1.0; Y[2, 2] = 1.0
    Y[3, 0] = 1.0; Y[4, 1] = 1.0; Y[5, 2] = 1.0
    Y[6, 0] = 1.0; Y[7, 1] = 1.0
    W = head.train_probe(X, Y, lambda_reg=1.0)
    assert head.W_probe is W
    assert W.shape == (8, 3)


def test_train_probe_rejects_bad_inputs():
    head = _head(dim=8)
    X = torch.randn(8, 8)
    Y = torch.zeros(8, 3)
    with pytest.raises(ValueError, match="2D"):
        head.train_probe(torch.randn(8), Y)
    with pytest.raises(ValueError, match="2D"):
        head.train_probe(X, torch.zeros(8))
    with pytest.raises(ValueError, match="N mismatch"):
        head.train_probe(X, torch.zeros(6, 3))
    with pytest.raises(ValueError, match="at least 5 examples"):
        head.train_probe(torch.randn(4, 8), torch.zeros(4, 3))


def test_probe_predict_falls_back_to_prototype_routing_without_probe():
    head = _head()
    result = head.probe_predict(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert result["action"] == "chat"


def test_state_dict_load_state_dict_round_trip():
    head_a = _head(dim=8)
    head_a.train_probe(
        torch.randn(8, 8),
        torch.tensor([[1.0, 0.0, 0.0]] * 8),
    )
    state = head_a.state_dict()
    head_b = DecisionHead(dim=8)
    head_b.load_state_dict(state)
    assert torch.allclose(head_a.P, head_b.P)
    assert head_b.prototype_names == head_a.prototype_names
    assert torch.allclose(head_b.W_probe, head_a.W_probe)
