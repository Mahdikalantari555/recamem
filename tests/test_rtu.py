import math
import pytest
import torch
from src.model.rtu import Layer

def test_rtu_decay_bounds():
    dim = 64
    layer = Layer(dim=dim, spread=16)
    decay = torch.sigmoid(layer.decay)
    assert torch.all(decay > 0.0)
    assert torch.all(decay < 1.0)

def test_rtu_state_update_equation():
    dim = 64
    layer = Layer(dim=dim, spread=16)
    layer.states.normal_()
    enc = torch.randn(dim)
    dummy = torch.randn(dim)

    decay = torch.sigmoid(layer.decay)
    expected_state = (decay * layer.states) + enc + dummy
    _, state, returned_decay = layer(enc, enc, dummy)

    assert torch.allclose(state, expected_state, atol=1e-6)
    assert torch.allclose(returned_decay, decay, atol=1e-6)

def test_rtu_decay_extremes():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.fill_(5.0)
    enc = torch.full((dim,), 2.0)
    dummy = torch.zeros(dim)

    # decay -> 1 (param large positive)
    with torch.no_grad():
        layer.decay.fill_(30.0)
    _, state_retention, _ = layer(enc, enc, dummy)
    assert torch.allclose(state_retention, layer.states + enc + dummy, atol=1e-4)

    # decay -> 0 (param large negative)
    with torch.no_grad():
        layer.decay.fill_(-30.0)
    _, state_forget, _ = layer(enc, enc, dummy)
    assert torch.allclose(state_forget, enc + dummy, atol=1e-4)

def test_rtu_reset():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.fill_(1.2)
    layer.decaytrace.fill_(0.8)
    layer.embedtrace.fill_(0.5)

    layer.reset()
    assert torch.all(layer.states == 0.0)
    assert torch.all(layer.decaytrace == 0.0)
    assert torch.all(layer.embedtrace == 0.0)

def test_rtu_gradient_flow():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.normal_()
    enc = torch.randn(dim, requires_grad=True)
    x = enc.clone()
    dummy = torch.zeros(dim, requires_grad=True)

    x_out, state, decay = layer(enc, x, dummy)
    loss = x_out.sum()
    loss.backward()

    assert enc.grad is not None and enc.grad.abs().sum() > 0
    assert layer.decay.grad is not None and layer.decay.grad.abs().sum() > 0
    assert dummy.grad is not None and dummy.grad.abs().sum() > 0

def test_rtu_trace_updates():
    dim = 16
    layer = Layer(dim=dim, vocab_size=256)
    layer.decaytrace.fill_(0.2)
    layer.embedtrace.fill_(0.1)
    layer.states.fill_(0.3)

    decay = torch.sigmoid(layer.decay).detach()
    c = 10
    c_range = (torch.arange(256) == c).float().unsqueeze(1)

    expected_embedtrace = (layer.embedtrace * decay) + c_range
    expected_decaytrace = (decay * layer.decaytrace) + (decay * (1.0 - decay) * layer.states)

    decay_embedtrace = layer.embedtrace * decay
    actual_embedtrace = decay_embedtrace + c_range
    actual_decaytrace = (decay * layer.decaytrace) + (decay * (1.0 - decay) * layer.states)

    assert torch.allclose(actual_embedtrace, expected_embedtrace, atol=1e-6)
    assert torch.allclose(actual_decaytrace, expected_decaytrace, atol=1e-6)

def test_rtu_stop_gradient_detached():
    dim = 16
    layer = Layer(dim=dim)
    enc = torch.randn(dim, requires_grad=True)
    x = enc.clone()
    _, state, _ = layer(enc, x)

    # Detaching persistent buffer assignment
    layer.states.copy_(state.detach())
    assert not layer.states.requires_grad

def test_model_forward_accepts_ce_only():
    """Model.forward takes the ce_only flag and still commits trace buffers.

    Return contract: Model.forward returns (sampled_byte, stop_value); the
    training loss lives on model.last_loss. The MLX reference returns
    (sample, stop) at main_mlx.py:114 and main_mlx.py:172, so the return
    signature is intentionally unchanged -- readers must not destructure the
    stop value as a loss. The expected CE is computed independently from
    compute_losses(..., ce_only=True) on the same tensor and compared against
    model.last_loss with a tolerance.
    """
    from src.model.rtu import Model, compute_losses

    torch.manual_seed(0)
    model = Model(dim=32, layers=2, spread=16)

    before_states = model.blocks[0].states.detach().clone()

    # Independently computed cross-entropy on the same tensors the forward
    # pass will see. `model.step` does not mutate persistent buffers, so the
    # subsequent `model(...)` call sees identical states. `x` and `stop` are
    # unused in ce_only mode; only `output` and `nextb` matter.
    c = torch.tensor(10, dtype=torch.long, device=model.device)
    dummies = [torch.zeros(model.dim, dtype=torch.float32, device=model.device) for _ in range(model.layers)]
    (_, _, _), (output, _) = model.step(c, dummies=dummies, frozen=False)
    expected_ce = compute_losses(torch.zeros(()), output, output.new_zeros(()),
                                 nextb=11, end=False, ce_only=True)[0]

    sample, stop = model(10, nextb=11, end=False, frozen=False, ce_only=True)

    assert isinstance(sample, int) and 0 <= sample < model.vocab_size
    assert 0.0 <= stop <= 1.0
    assert model.last_loss == pytest.approx(expected_ce.item(), rel=1e-6)
    assert model.step_count == 1
    # Trace buffers must still advance under ce_only.
    assert not torch.allclose(model.blocks[0].states, before_states)

def test_model_save_preserves_trace_buffers():
    """Regression guard: trace buffers stay in checkpoints (upstream removed them, we must not)."""
    import os
    import tempfile
    from src.model.rtu import Model

    torch.manual_seed(0)
    model = Model(dim=32, layers=2, spread=16)
    model(10, nextb=11, end=False, frozen=False)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'model.safetensors')
        model.save(path)

        from safetensors.torch import load_file
        data = load_file(path)
        for i in range(model.layers):
            assert f'state.{i}' in data
            assert f'decaytrace.{i}' in data
            assert f'embedtrace.{i}' in data

        model2 = Model(dim=32, layers=2, spread=16)
        model2.load(path)
        assert torch.allclose(model2.blocks[0].decaytrace, model.blocks[0].decaytrace)
        assert torch.allclose(model2.blocks[0].embedtrace, model.blocks[0].embedtrace)
