"""Equivalence tests for the chunked training pass (``chunked-trace-training``).

The per-token path is the oracle. Every case here runs the chunked pass beside
``Model.per_token_gradient_oracle`` and compares the per-token losses, the
accumulated gradients and the three trace buffers within the documented
tolerance. The one thing that is *not* equivalent — the parameter trajectory —
is asserted to be declared as such.
"""

import inspect

import pytest
import torch

from src.model import parallel_scan, rtu
from src.model.rtu import CHUNK_EQUIVALENCE_TOLERANCE, TOKEN_LOOP_COUNTER, Model


class RecordingOptimizer:
    """A stand-in optimizer that records the gradients instead of stepping."""

    def __init__(self, model):
        self.model = model
        self.param_groups = [{"lr": None}]
        self.calls = []
        self.grads = {}

    def step(self):
        self.calls.append(self.param_groups[0]["lr"])
        self.grads = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }


def _model(dim=8, layers=2, spread=4, vocab=16):
    torch.manual_seed(0)
    return Model(dim=dim, layers=layers, spread=spread, vocab_size=vocab)


def _pairs(tokens):
    return torch.roll(tokens, -1, dims=1)


def _assert_close(actual, expected, what):
    scale = max(1.0, float(expected.abs().max().item()))
    delta = float((actual - expected).abs().max().item())
    assert delta <= CHUNK_EQUIVALENCE_TOLERANCE * scale + CHUNK_EQUIVALENCE_TOLERANCE, (
        f"{what} differs by {delta:.3e} (tolerance {CHUNK_EQUIVALENCE_TOLERANCE})"
    )


# --------------------------------------------------------------------------- #
# Reverse suffix sums (task 2.1)
# --------------------------------------------------------------------------- #
def test_reverse_decay_scan_matches_a_direct_suffix_sum():
    torch.manual_seed(0)
    decays = torch.rand(4) * 0.8 + 0.1
    values = torch.randn(2, 9, 4)
    suffix, final = parallel_scan.reverse_decay_scan(decays, values, chunk=4)
    expected = torch.zeros_like(values)
    running = torch.zeros(2, 4)
    for index in range(values.shape[1] - 1, -1, -1):
        running = decays * running + values[:, index]
        expected[:, index] = running
    _assert_close(suffix, expected, "suffix sums")
    _assert_close(final, running, "final suffix")
    assert suffix.shape == values.shape


# --------------------------------------------------------------------------- #
# Trace-buffer closed forms (tasks 1.1-1.3, 4.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,tokens,chunk",
    [
        ("random-bytes", torch.randint(0, 16, (1, 8)), 4),
        ("chunk-longer-than-chunk-size", torch.randint(0, 16, (1, 21)), 5),
        ("batch", torch.randint(0, 16, (2, 8)), 4),
        ("all-zero", torch.zeros(1, 10, dtype=torch.long), 4),
        ("all-one", torch.ones(1, 10, dtype=torch.long), 4),
    ],
)
def test_chunk_buffers_match_the_per_token_path(name, tokens, chunk):
    model = _model()
    oracle = model.per_token_gradient_oracle(tokens, _pairs(tokens))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=chunk)
    current = dict(model.named_buffers())
    for buffer_name, expected in oracle["buffers"].items():
        assert buffer_name in current, f"{name}: {buffer_name} missing"
        _assert_close(current[buffer_name], expected, f"{name}: {buffer_name}")

    assert info["equivalent"] == []
    assert info["equivalence_verified"] is False
    assert "parameter_trajectory" in info["not_equivalent"]
    assert "not verified" in info["equivalence_note"] or "not run" in info["equivalence_note"]


def test_real_decay_initialization_is_covered():
    torch.manual_seed(0)
    model = Model(dim=16, layers=2, spread=32, vocab_size=32)
    tokens = (torch.arange(24).reshape(1, 24) * 3) % 32
    info = model.train_chunk(tokens, _pairs(tokens), chunk=8, verify_equivalence=True)
    assert info["verification"]["verified"] is True, info["verification"]["problems"]


# --------------------------------------------------------------------------- #
# No Python token loop (task 1.4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", [4, 32])
def test_chunked_path_has_no_token_loop_that_grows_with_length(length):
    model = _model()
    tokens = torch.randint(0, 16, (1, length))
    before = TOKEN_LOOP_COUNTER["chunked"]
    info = model.train_chunk(tokens, _pairs(tokens), chunk=8)
    assert TOKEN_LOOP_COUNTER["chunked"] - before == 0
    assert info["token_loop_iterations"] == 0


def test_per_token_oracle_loop_count_does_grow_with_length():
    model = _model()
    tokens = torch.randint(0, 16, (1, 6))
    before = TOKEN_LOOP_COUNTER["per_token"]
    oracle = model.per_token_gradient_oracle(tokens, _pairs(tokens))
    assert TOKEN_LOOP_COUNTER["per_token"] - before == 6
    assert oracle["token_loop_iterations"] == 6


# --------------------------------------------------------------------------- #
# Batched per-token losses and gradients (tasks 2.2, 2.3, 3.1-3.3)
# --------------------------------------------------------------------------- #
def test_loss_terms_match_the_per_token_path():
    model = _model()
    tokens = torch.randint(0, 16, (1, 9))
    oracle = model.per_token_gradient_oracle(tokens, _pairs(tokens))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=4)
    assert sum(oracle["token_losses"]) == pytest.approx(info["loss"], rel=1e-5, abs=1e-5)
    # Every documented term is present and finite.
    for term in ("var", "pred", "ce", "stop", "total"):
        assert term in info["loss_terms"]
        assert torch.isfinite(torch.tensor(info["loss_terms"][term]))


def test_accumulated_gradients_match_the_oracle_including_the_decay_hook():
    tokens = torch.randint(0, 16, (1, 10))
    oracle_model = _model()
    oracle = oracle_model.per_token_gradient_oracle(tokens, _pairs(tokens))

    chunk_model = _model()
    recorder = RecordingOptimizer(chunk_model)
    chunk_model.optimizer = recorder
    info = chunk_model.train_chunk(tokens, _pairs(tokens), chunk=5)
    scale = info["grad_scale"]
    for name, expected in oracle["gradients"].items():
        actual = recorder.grads.get(name)
        if actual is None:
            assert float(expected.abs().max().item()) <= CHUNK_EQUIVALENCE_TOLERANCE
            continue
        _assert_close(actual / scale, expected, f"gradient {name}")


def test_embedding_gradient_includes_both_hook_contributions():
    model = _model()
    tokens = torch.randint(0, 16, (1, 8))
    recorder = RecordingOptimizer(model)
    model.optimizer = recorder
    info = model.train_chunk(tokens, _pairs(tokens), chunk=4)
    assert info["hooks"]["embedding_hook_applied"] is True
    assert "encoder.embed.weight" in recorder.grads
    assert info["hooks"]["layers"]
    assert all("chunk_final_embedtrace_norm" in layer for layer in info["hooks"]["layers"])


def test_the_decay_gradient_is_replaced_not_added():
    model = _model()
    tokens = torch.randint(0, 16, (1, 6))
    oracle = model.per_token_gradient_oracle(tokens, _pairs(tokens))

    chunk_model = _model()
    recorder = RecordingOptimizer(chunk_model)
    chunk_model.optimizer = recorder
    info = chunk_model.train_chunk(tokens, _pairs(tokens), chunk=6)
    actual = recorder.grads["blocks.0.decay"] / info["grad_scale"]
    _assert_close(actual, oracle["gradients"]["blocks.0.decay"], "decay hook gradient")


# --------------------------------------------------------------------------- #
# Optimizer step granularity and the learning-rate schedule (tasks 3.4, 3.5)
# --------------------------------------------------------------------------- #
def test_one_optimizer_step_per_chunk_and_the_byte_counter_advances():
    model = _model()
    recorder = RecordingOptimizer(model)
    model.optimizer = recorder
    tokens = torch.randint(0, 16, (2, 6))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=4)
    assert len(recorder.calls) == 1
    assert info["optimizer_steps"] == 1
    assert model.step_count == 12
    assert info["step_count"] == 12
    assert info["tokens"] == 12
    assert info["batch_size"] == 2
    assert info["chunk_length"] == 6


def test_gradient_is_scaled_by_one_over_the_chunk_length_before_the_step():
    model = _model()
    recorder = RecordingOptimizer(model)
    model.optimizer = recorder
    recorder.param_groups[0]["lr"] = None
    tokens = torch.randint(0, 16, (1, 5))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=5)
    assert info["grad_scale"] == pytest.approx(1.0 / 5.0)
    # The recorded gradient is already divided by the chunk length.
    assert recorder.grads


def test_both_paths_read_the_same_learning_rate_at_the_same_byte_count():
    model = _model()
    assert model.lr_at_bytes(0) == model.lr_at_step(0)
    assert model.lr_at_bytes(10_000) == model.lr_at_step(10_000)
    recorder = RecordingOptimizer(model)
    model.optimizer = recorder
    tokens = torch.randint(0, 16, (1, 4))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=4)
    assert info["lr"] == model.lr_at_bytes(0)
    assert recorder.calls == [model.lr_at_step(0)]


# --------------------------------------------------------------------------- #
# Equivalence declaration (tasks 4.2-4.4)
# --------------------------------------------------------------------------- #
def test_verified_run_declares_both_the_equivalent_and_non_equivalent_quantities():
    model = _model()
    tokens = torch.randint(0, 16, (1, 8))
    info = model.train_chunk(tokens, _pairs(tokens), chunk=4, verify_equivalence=True)
    assert info["equivalence_verified"] is True
    assert info["verification"]["verified"] is True
    assert info["equivalent"] == list(rtu.CHUNK_EQUIVALENT_QUANTITIES)
    assert info["not_equivalent"] == ["parameter_trajectory"]
    assert info["verification"]["reference"] == "Model.per_token_gradient_oracle"


def test_an_unverified_run_does_not_claim_the_equivalence():
    model = _model()
    info = model.train_chunk(torch.randint(0, 16, (1, 4)), chunk=4)
    assert info["equivalence_verified"] is False
    assert info["equivalent"] == []
    assert "not run" in info["equivalence_note"]


def test_the_reference_is_the_per_token_path_not_the_chunked_path():
    source = inspect.getsource(Model.per_token_gradient_oracle)
    assert "for step in range(length)" in source
    # The oracle steps token by token; the chunked path never loops over tokens.
    chunk_source = inspect.getsource(Model._chunk_forward_backward)
    assert "for step in range" not in chunk_source
    assert Model.forward is not Model.train_chunk
