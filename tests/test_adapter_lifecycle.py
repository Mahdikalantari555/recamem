"""Post-merge tasks 2.1–2.4: real adapter and orchestration regression tests."""
import torch
import pytest

from src.model.plastic_adapter import (
    PlasticAdapter, PlasticUpdateError, apply_plastic_update, clone_state, get_adapter,
)
from src.model.droid import DroidEngine
from src.model.memory_lifecycle import SnapshotManager


class MinimalAdapter(PlasticAdapter):
    def __init__(self, dim=384, fail=None):
        super().__init__()
        self.register_buffer("trace", torch.zeros(dim))
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.metadata = {"history": []}
        self.calls = 0
        self.fail = fail

    def update(self, x):
        self.calls += 1
        if self.fail == "before":
            raise TypeError("internal failure before mutation")
        self.trace.add_(x)
        self.metadata["history"].append(self.calls)
        if self.fail == "after":
            raise TypeError("internal failure after mutation")
        return self.trace.clone()

    def retrieve(self, q):
        return self.trace.clone()

    def predict_next(self):
        return self.trace.clone()

    def compute_surprise(self, x):
        return x - self.trace, 1.0

    def state_dict(self, *args, **kwargs):
        return {"trace": self.trace, "weight": self.weight, "metadata": self.metadata}

    def load_state_dict(self, state, **kwargs):
        with torch.no_grad():
            self.trace.copy_(state["trace"])
            self.weight.copy_(state["weight"])
        # Deliberately retain the input to exercise restore's ownership boundary.
        self.metadata = state["metadata"]


class OriginalSignatureAdapter(MinimalAdapter):
    def update_associative_memory(self, x, surprise_factor=1.0):
        return self.update(x)


class ScheduledAdapter(MinimalAdapter):
    def update_scheduled(self, x, *, surprise_factor, effective_lr, base_lr):
        return self.update(x * (effective_lr / base_lr)), effective_lr


class FixedAnchor:
    def embed(self, text):
        return torch.ones(384)


@pytest.fixture
def engine_factory(monkeypatch):
    monkeypatch.setattr("src.model.droid.OnnxMiniLM", FixedAnchor)
    def make(adapter):
        return DroidEngine(plastic_adapter=adapter, test_time_training=False)
    return make


def test_minimal_adapter_is_a_module_with_safe_optional_defaults():
    adapter = MinimalAdapter(dim=4)
    assert isinstance(adapter, torch.nn.Module)
    assert dict(adapter.named_parameters())["weight"] is adapter.weight
    assert dict(adapter.named_buffers())["trace"] is adapter.trace
    before = adapter.create_snapshot()
    assert adapter.consolidate() == {"status": "unsupported", "updated": False}
    assert adapter.get_health() == {"status": "unsupported", "metrics": {}}
    torch.testing.assert_close(adapter.trace, before["trace"])
    adapter.to(dtype=torch.float64)
    assert adapter.trace.dtype == adapter.weight.dtype == torch.float64


def test_snapshot_and_restore_own_tensors_and_nested_metadata():
    adapter = MinimalAdapter(dim=4)
    adapter.update(torch.ones(4))
    snapshot = adapter.create_snapshot()
    adapter.update(torch.ones(4))
    assert snapshot["metadata"]["history"] == [1]
    torch.testing.assert_close(snapshot["trace"], torch.ones(4))
    assert not snapshot["weight"].requires_grad
    adapter.restore_snapshot(snapshot)
    adapter.metadata["history"].append(99)
    adapter.trace.add_(5)
    assert snapshot["metadata"]["history"] == [1]
    torch.testing.assert_close(snapshot["trace"], torch.ones(4))


@pytest.mark.parametrize("name", ["legacy_rtu", "associative_rtu"])
def test_builtin_module_mro_and_snapshot_roundtrip(name):
    adapter = get_adapter(name, dim=16)
    adapter.update(torch.ones(16))
    snapshot = adapter.create_snapshot()
    expected = adapter.retrieve(torch.ones(16)).clone()
    adapter.update(-torch.ones(16))
    adapter.restore_snapshot(snapshot)
    torch.testing.assert_close(adapter.retrieve(torch.ones(16)), expected)
    assert list(adapter.named_parameters())
    assert list(adapter.named_buffers())


@pytest.mark.parametrize("kind", [MinimalAdapter, OriginalSignatureAdapter, ScheduledAdapter])
def test_engine_negotiates_update_once_and_reports_actual_rate(kind, engine_factory):
    adapter = kind()
    engine = engine_factory(adapter)
    engine.step_count = 80
    result = engine.teach("Remote sensing records information about the Earth.")
    assert adapter.calls == 1
    assert torch.count_nonzero(adapter.trace) > 0
    assert result["requested_lr"] == result["effective_lr"]
    if kind is ScheduledAdapter:
        assert result["applied_lr"] == result["requested_lr"]
        assert result["scheduling_supported"] is True
    else:
        assert result["applied_lr"] is None
        assert result["applied_lrs"] == [None]
        assert result["scheduling_supported"] is False
    assert engine.logs[-1]["details"]["applied_lr"] == result["applied_lr"]


@pytest.mark.parametrize("name", ["legacy_rtu", "associative_rtu"])
def test_builtin_scheduling_capabilities_are_honest(name, engine_factory, monkeypatch):
    adapter = get_adapter(name, dim=384)
    method = adapter.update_associative_memory
    calls = []
    def track(*args, **kwargs):
        calls.append(kwargs)
        return method(*args, **kwargs)
    monkeypatch.setattr(adapter, "update_associative_memory", track)
    engine = engine_factory(adapter)
    result = engine.teach("Remote sensing records information about the Earth.")
    assert len(calls) == 1
    assert result["scheduling_supported"] == (name == "associative_rtu")
    assert result["applied_lr"] == (engine.base_lr if name == "associative_rtu" else None)


@pytest.mark.parametrize("fail", ["before", "after"])
@pytest.mark.parametrize("kind", [MinimalAdapter, OriginalSignatureAdapter, ScheduledAdapter])
def test_failure_propagates_without_retry_or_success_log(fail, kind, engine_factory):
    adapter = kind(fail=fail)
    engine = engine_factory(adapter)
    with pytest.raises(PlasticUpdateError, match="state may be partially changed") as error:
        engine.teach("Remote sensing records information about the Earth.")
    assert isinstance(error.value.__cause__, TypeError)
    assert adapter.calls == 1
    assert engine.knowledge.total_count() == 0
    assert not any("Successfully learned" in str(log) for log in engine.logs)
    assert bool(torch.count_nonzero(adapter.trace)) == (fail == "after")


@pytest.mark.parametrize("lr,base", [(-1, .1), (float("nan"), .1), (.1, 0)])
def test_invalid_rate_is_rejected_before_mutation(lr, base):
    adapter = MinimalAdapter(dim=4)
    with pytest.raises(ValueError):
        apply_plastic_update(adapter, torch.ones(4), surprise_factor=1,
                             effective_lr=lr, base_lr=base)
    assert adapter.calls == 0


@pytest.mark.parametrize("use_manager", [True, False])
def test_engine_and_manager_snapshot_branch_own_state(use_manager, engine_factory, tmp_path):
    adapter = MinimalAdapter()
    engine = engine_factory(adapter)
    manager = SnapshotManager(str(tmp_path)) if use_manager else engine
    adapter.update(torch.ones(384))
    source = (manager.create_snapshot(engine, "source") if use_manager
              else manager.create_snapshot("source"))
    branch = manager.branch_from("source", "branch")
    adapter.update(torch.ones(384))
    torch.testing.assert_close(source["plastic_state"]["trace"], torch.ones(384))
    branch["plastic_state"]["trace"].add_(3)
    branch["plastic_state"]["metadata"]["history"].append(999)
    assert source["plastic_state"]["metadata"]["history"] == [1]
    torch.testing.assert_close(source["plastic_state"]["trace"], torch.ones(384))
    if use_manager:
        assert manager.restore_snapshot(engine, "source")
    else:
        assert manager.restore_snapshot("source")
    adapter.update(torch.ones(384))
    torch.testing.assert_close(source["plastic_state"]["trace"], torch.ones(384))
    assert source["plastic_state"]["metadata"]["history"] == [1]


def test_restoring_pristine_associative_snapshot_clears_later_keys():
    adapter = get_adapter("associative_rtu", dim=16, K_max=2)
    snapshot = adapter.create_snapshot()
    adapter.update(torch.ones(16))
    assert len(adapter._key_registry) == 1
    adapter.restore_snapshot(snapshot)
    assert adapter._key_registry == []
    assert adapter.eviction_count == 0
    torch.testing.assert_close(adapter.S, snapshot["S"])
    adapter.update(-torch.ones(16))
    assert snapshot["_key_registry_norms"].numel() == 0
