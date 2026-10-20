"""Contract tests for src/model/plastic_adapter.py.

Invariants under test (see openspec/specs/adaptation/adapter-registry):
- importing the module registers the built-in adapters (``associative_rtu``,
  ``legacy_rtu``, ``rtu``);
- ``get_adapter`` constructs by registry name, accepts the ``d_model`` alias for
  ``dim``, and raises ``KeyError`` naming the available adapters for an unknown
  name;
- ``register_adapter`` rejects a non-conforming class in strict mode;
- both built-in adapters round-trip their state through ``state_dict`` /
  ``load_state_dict``.
"""

import pytest
import torch

from src.model import plastic_adapter
from src.model.plastic_adapter import (
    PlasticAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)
from src.model.rtu import PlasticAssociativeRTU


def _vector(dim: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(dim, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_builtin_adapters_are_registered_on_import():
    available = list_adapters()
    assert "associative_rtu" in available
    assert "legacy_rtu" in available
    assert "rtu" in available


def test_register_and_list_custom_adapter():
    class CustomAdapter(PlasticAdapter):
        def __init__(self, dim=4):
            self.dim = dim

        def update(self, x):
            return x

        def retrieve(self, q):
            return q

        def predict_next(self):
            return torch.zeros(self.dim)

        def compute_surprise(self, x):
            return x, 0.0

        def state_dict(self):
            return {"dim": self.dim}

        def load_state_dict(self, state):
            self.dim = state.get("dim", self.dim)

    name = "custom-test-adapter"
    register_adapter(name, CustomAdapter)
    try:
        assert name in list_adapters()
        instance = get_adapter(name, dim=6)
        assert isinstance(instance, CustomAdapter)
        assert instance.dim == 6
    finally:
        plastic_adapter._ADAPTER_REGISTRY.pop(name, None)


def test_register_adapter_rejects_non_conforming_class_in_strict_mode():
    with pytest.raises(TypeError):
        register_adapter("bogus-adapter", object)
    assert "bogus-adapter" not in list_adapters()


def test_register_adapter_can_skip_strict_check():
    name = "loose-test-adapter"
    register_adapter(name, object, strict=False)
    try:
        assert name in list_adapters()
    finally:
        plastic_adapter._ADAPTER_REGISTRY.pop(name, None)


def test_unknown_adapter_raises_key_error_with_available_names():
    with pytest.raises(KeyError) as excinfo:
        get_adapter("does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "Available" in message


def test_d_model_alias_maps_to_dim():
    adapter = get_adapter("associative_rtu", d_model=16, heads=4)
    assert adapter.dim == 16


# --------------------------------------------------------------------------- #
# LegacyRTUAdapter
# --------------------------------------------------------------------------- #
def test_legacy_adapter_implements_the_interface():
    adapter = get_adapter("legacy_rtu", dim=16)
    assert isinstance(adapter, PlasticAdapter)

    x = _vector(16, seed=1)
    out = adapter.update(x)
    assert out.shape == (16,)
    assert torch.allclose(adapter.retrieve(x), adapter.h_trace)

    pred = adapter.predict_next()
    assert pred.shape == (16,)
    assert torch.linalg.vector_norm(pred).item() == pytest.approx(1.0, abs=1e-5)

    error, surprise = adapter.compute_surprise(x)
    assert error.shape == (16,)
    assert 0.0 <= surprise <= 2.0


def test_legacy_adapter_state_round_trip_restores_parameters():
    source = get_adapter("legacy_rtu", dim=8)
    source.update(_vector(8, seed=2))
    with torch.no_grad():
        source.rtu.decay.add_(0.25)
    state = source.state_dict()

    assert state["_adapter_type"] == "legacy_rtu"
    assert set(state) >= {"states", "decay", "proj_weight", "h_trace"}

    target = get_adapter("legacy_rtu", dim=8)
    assert not torch.allclose(target.rtu.decay, source.rtu.decay)
    target.load_state_dict(state)
    assert torch.allclose(target.rtu.decay, source.rtu.decay)
    assert torch.allclose(target.rtu.proj.weight, source.rtu.proj.weight)
    assert torch.allclose(target.rtu.states, source.rtu.states)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LegacyRTUAdapter.load_state_dict copies the 'states' entry into h_trace and ignores "
        "the 'h_trace' entry it exported, so a trace advanced by update() (which does not write "
        "rtu.states) is lost on reload. Pre-existing defect in src/model/plastic_adapter.py; "
        "reported to the owning change instead of patched here (this change is tests-only)."
    ),
)
def test_legacy_adapter_state_round_trip_preserves_trace():
    source = get_adapter("legacy_rtu", dim=8)
    source.update(_vector(8, seed=2))
    state = source.state_dict()

    target = get_adapter("legacy_rtu", dim=8)
    target.load_state_dict(state)
    assert torch.allclose(target.h_trace, source.h_trace)
    assert torch.allclose(target.rtu.states, source.rtu.states)


def test_legacy_alias_constructs_same_class():
    assert type(get_adapter("rtu", dim=8)) is type(get_adapter("legacy_rtu", dim=8))


# --------------------------------------------------------------------------- #
# PlasticAssociativeRTU
# --------------------------------------------------------------------------- #
def test_associative_adapter_update_and_retrieve():
    adapter = get_adapter("associative_rtu", dim=16, heads=4)
    assert isinstance(adapter, PlasticAssociativeRTU)
    assert adapter.heads == 4
    assert adapter.head_dim == 4

    before = adapter.h_trace.clone()
    out = adapter.update(_vector(16, seed=3))
    assert out.shape == (16,)
    assert not torch.allclose(adapter.h_trace, before)

    retrieved = adapter.retrieve(_vector(16, seed=4))
    assert retrieved.shape == (16,)

    error, surprise = adapter.compute_surprise(_vector(16, seed=5))
    assert error.shape == (16,)
    assert 0.0 <= surprise <= 2.0


def test_associative_adapter_state_round_trip():
    source = get_adapter("associative_rtu", dim=16, heads=4)
    source.update(_vector(16, seed=6), )
    state = source.state_dict()

    assert state["_adapter_type"] == "associative_rtu"
    assert state["dim"] == 16
    assert state["heads"] == 4
    assert state["S"].shape == (4, 4, 4)

    target = get_adapter("associative_rtu", dim=16, heads=4)
    target.load_state_dict(state)
    assert torch.allclose(target.h_trace, source.h_trace)
    assert torch.allclose(target.S, source.S)
    assert torch.allclose(target.momentum, source.momentum)


def test_adapter_dim_must_be_divisible_by_heads():
    with pytest.raises(AssertionError):
        get_adapter("associative_rtu", dim=10, heads=4)
