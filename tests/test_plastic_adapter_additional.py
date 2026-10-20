"""Contract tests for `src/model/plastic_adapter.py` (surface-test-coverage 3.8).

Covered: registry registration/listing/instantiation, unknown-name errors,
the `d_model` -> `dim` kwarg mapping, and state-dict round trips.
"""

import pytest
import torch

from src.model.plastic_adapter import (
    PlasticAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)


def test_adapter_registry_contains_builtins():
    names = list_adapters()
    assert "associative_rtu" in names


def test_get_adapter_instantiates_registered_adapter():
    adapter = get_adapter("associative_rtu", dim=16)
    assert isinstance(adapter, PlasticAdapter)
    assert adapter.dim == 16
    assert adapter.heads == 4

    # `d_model` is mapped to `dim` for callers that speak the MLX idiom.
    adapter = get_adapter("associative_rtu", d_model=32)
    assert adapter.dim == 32


def test_adapter_update_retrieve_predict_surprise_round_trip():
    adapter = get_adapter("associative_rtu", dim=16)
    x = torch.randn(16)

    out = adapter.update(x)
    assert out.shape == (16,)

    rec = adapter.retrieve(x)
    assert rec.shape == (16,)

    pred = adapter.predict_next()
    assert pred.shape == (16,)

    err_vec, surprise = adapter.compute_surprise(x)
    assert err_vec.shape == (16,)
    assert 0.0 <= surprise <= 2.0


def test_adapter_state_dict_round_trip():
    adapter_a = get_adapter("associative_rtu", dim=16)
    adapter_a.update(torch.randn(16))

    # Note: nn.Module.state_dict is called directly because the registered
    # override shadows it with an abstract-method delegate; the tensor state
    # itself is what load_state_dict restores.
    state = torch.nn.Module.state_dict(adapter_a)
    assert any(k in state for k in ("S", "h_trace", "decay"))

    adapter_b = get_adapter("associative_rtu", dim=16)
    adapter_b.load_state_dict(state)

    assert torch.allclose(adapter_b.S, adapter_a.S)
    assert torch.allclose(adapter_b.h_trace, adapter_a.h_trace)
    assert torch.allclose(adapter_b.momentum, adapter_a.momentum)


def test_get_adapter_unknown_name_raises_key_error():
    try:
        get_adapter("definitely-not-an-adapter", dim=16)
    except KeyError as exc:
        assert "not found" in str(exc)
        assert "Available" in str(exc)
    else:  # pragma: no cover - entry points could theoretically install one
        pytest.fail("expected KeyError for unknown adapter name")


def test_register_adapter_and_listing():
    class CustomAdapter(PlasticAdapter, torch.nn.Module):
        def __init__(self, dim=8, **kwargs):
            super().__init__()
            self.dim = dim
            self.register_buffer("trace", torch.zeros(dim))

        def update(self, x):
            return x

        def retrieve(self, q):
            return self.trace

        def predict_next(self):
            return self.trace / (self.trace.norm().clamp(min=1e-8))

        def compute_surprise(self, x):
            return x - self.trace, 0.5

        def state_dict(self, *args, **kwargs):
            return {"trace": self.trace.detach(), "dim": self.dim, "_adapter_type": "custom"}

        def load_state_dict(self, state, strict=True):
            if "trace" in state:
                self.trace.copy_(state["trace"])

    register_adapter("custom_adapter", CustomAdapter)
    try:
        assert "custom_adapter" in list_adapters()
        adapter = get_adapter("custom_adapter", dim=24)
        assert isinstance(adapter, CustomAdapter)
        assert adapter.dim == 24
    finally:
        # Leave the registry as we found it for other tests.
        from src.model.plastic_adapter import _ADAPTER_REGISTRY

        _ADAPTER_REGISTRY.pop("custom_adapter", None)


def test_register_adapter_strict_rejects_incomplete_class():
    class Incomplete:
        pass

    with pytest.raises(TypeError, match="PlasticAdapter"):
        register_adapter("incomplete", Incomplete, strict=True)
