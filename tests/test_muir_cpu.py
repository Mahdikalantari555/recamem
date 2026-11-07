import pytest
import torch
from src.model.cores.muir import MuirCore
from src.model.cores.param_matcher import estimate_candidate_lm_params
from src.model.cores.registry import build_candidate_lm


@pytest.mark.parametrize('readout', ['static', 'dynamic'])
def test_muir_streaming_and_gradients(readout):
    torch.manual_seed(3)
    core = MuirCore(8, 2, 2, 16, conv_kernel=3, readout=readout).double()
    x = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True)
    state = core.init_state(2, dtype=torch.double)
    state = {k: torch.randn_like(v) * .1 for k, v in state.items()}
    full = core.forward_sequence(x, state)
    parts, st = [], state
    for i in range(x.shape[1]):
        out = core.step(x[:, i], st)
        parts.append(out.hidden_states); st = out.state
    torch.testing.assert_close(full.hidden_states, torch.cat(parts, 1), atol=1e-9, rtol=1e-8)
    for key in st:
        torch.testing.assert_close(full.state[key], st[key], atol=1e-9, rtol=1e-8)
    grad, = torch.autograd.grad(full.hidden_states.square().sum(), x)
    assert torch.isfinite(grad).all()
    changed = x.detach().clone(); changed[:, 4:] += 5
    other = core.forward_sequence(changed, state)
    torch.testing.assert_close(full.hidden_states[:, :4], other.hidden_states[:, :4])
    assert core.state_spec().bytes_per_sequence == sum(t[0, 0].numel() for t in state.values()) * 2 * 4


def test_muir_registry_and_parameter_count():
    cfg = dict(core_id='C7', vocab_size=256, d_model=16, n_layers=2, n_heads=2, d_ff=32)
    model = build_candidate_lm(cfg)
    assert sum(p.numel() for p in model.parameters()) == estimate_candidate_lm_params('C7', 256, 16, 2, 2, 32)
    assert model(torch.randint(0, 256, (1, 4))).shape == (1, 4, 256)


@pytest.mark.parametrize('readout', ['static', 'dynamic'])
@pytest.mark.parametrize('kernel', [1, 4])
def test_muir_chunks_nonzero_state_and_reset(readout, kernel):
    torch.manual_seed(721)
    core = MuirCore(8, 2, 2, 16, conv_kernel=kernel, readout=readout).double()
    x = torch.randn(3, 9, 8, dtype=torch.float64)
    initial = core.init_state(3, dtype=torch.float64)
    initial = {key: torch.randn_like(value) * 0.2 for key, value in initial.items()}
    full = core.forward_sequence(x, initial)
    parts, state = [], initial
    for start, end in [(0, 2), (2, 5), (5, 9)]:
        output = core.forward_sequence(x[:, start:end], state)
        parts.append(output.hidden_states)
        state = output.state
    torch.testing.assert_close(torch.cat(parts, 1), full.hidden_states, rtol=1e-8, atol=1e-9)
    for key in state:
        torch.testing.assert_close(state[key], full.state[key], rtol=1e-8, atol=1e-9)
    assert core._persistent_state is not None
    core.reset_state()
    assert core._persistent_state is None
    reset = core.forward_sequence(x)
    zero = core.forward_sequence(x, core.init_state(3, dtype=torch.float64))
    torch.testing.assert_close(reset.hidden_states, zero.hidden_states, rtol=1e-8, atol=1e-9)
    for key in zero.state:
        torch.testing.assert_close(reset.state[key], zero.state[key], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('a_log', [-8., 8.])
@pytest.mark.parametrize('readout', ['static', 'dynamic'])
def test_muir_extreme_decay_float64_finite_and_causal(a_log, readout):
    torch.manual_seed(19)
    core = MuirCore(8, 1, 2, 16, conv_kernel=3, readout=readout).double()
    with torch.no_grad():
        core.layers[0].a_log.fill_(a_log)
    x = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    result = core.forward_sequence(x)
    assert torch.isfinite(result.hidden_states).all()
    assert all(torch.isfinite(t).all() for t in result.state.values())
    gradients = torch.autograd.grad(result.hidden_states.square().sum(), (x, *core.parameters()), allow_unused=True)
    assert all(grad is None or torch.isfinite(grad).all() for grad in gradients)
    changed = x.detach().clone()
    changed[:, 4:] += 20
    altered = core.forward_sequence(changed)
    torch.testing.assert_close(result.hidden_states[:, :4], altered.hidden_states[:, :4], rtol=1e-8, atol=1e-9)


def test_muir_input_and_state_gradcheck():
    torch.manual_seed(29)
    core = MuirCore(4, 1, 1, 8, conv_kernel=2, readout='dynamic').double()
    x = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    matrix = torch.randn(1, 1, 1, 4, 4, dtype=torch.double, requires_grad=True) * 0.1
    conv = torch.randn(1, 1, 12, 1, dtype=torch.double, requires_grad=True) * 0.1

    def run(a, b, c):
        result = core.forward_sequence(a, {'matrix': b, 'conv': c})
        return result.hidden_states, result.state['matrix'], result.state['conv']

    assert torch.autograd.gradcheck(run, (x, matrix, conv), eps=1e-6, atol=1e-4, rtol=1e-3)
