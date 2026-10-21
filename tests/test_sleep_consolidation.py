"""Autonomous sleep, MAS pruning and the decreasing-backprop schedule.

Covers ``memory/sleep-consolidation`` and ``memory/memory-aware-synapses``
tasks 1.1-1.5, 2.1-2.5, 3.1-3.4 and 4.1-4.4.
"""
import threading
import time

import pytest
import torch

from src.model.droid import DroidEngine
from src.model.droid_manager import DroidManager
from src.model.sleep_consolidation import (
    AutoSleepWorker,
    DecreasingBackprop,
    MemoryAwareSynapses,
    ScheduleValidationError,
    SleepConsolidation,
    relu_down,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _engine(tmp_path, name="sleeper", **kwargs):
    droid = DroidEngine(name=name, db_path=str(tmp_path / f"{name}.db"), **kwargs)
    return droid


def _store_correlated(droid, count=100, dim=384, seed=0):
    """Insert correlated embeddings directly: an internal memory fixture."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(dim, generator=generator)
    base = base / base.norm()
    ids = []
    for i in range(count):
        noise = torch.randn(dim, generator=generator)
        vec = base + 0.5 * noise / noise.norm()
        vec = vec / vec.norm()
        fact_id = droid.knowledge.insert_fact(
            f"correlated pattern {i}", "fixture", time.time(), i, 0.5, vec
        )
        ids.append(fact_id)
    droid.knowledge._vec_cache = None
    return ids


class _ScriptedQuality:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if self.values else None


# ---------------------------------------------------------------------------
# DecreasingBackprop (tasks 3.1-3.4)
# ---------------------------------------------------------------------------
def test_schedule_formula_and_monotonicity():
    schedule = DecreasingBackprop(base_lr=0.1, decay_factor=0.5)
    assert schedule.lr_at(0) == pytest.approx(0.1)
    assert schedule.lr_at(1) == pytest.approx(0.05)
    assert schedule.lr_at(3) == pytest.approx(0.0125)
    rates = [schedule.lr_at(i) for i in range(10)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert schedule.diagnostics() == {
        "base_lr": 0.1, "decay_factor": 0.5, "i": 0, "lr_i": 0.1,
        "experimental_relu_down": False,
    }


@pytest.mark.parametrize("kwargs", [
    {"base_lr": 0.0}, {"base_lr": -1.0}, {"base_lr": float("nan")},
    {"decay_factor": 0.0}, {"decay_factor": 1.5}, {"index": -1},
    {"index": 1.5},
])
def test_schedule_rejects_invalid_configuration(kwargs):
    with pytest.raises(ScheduleValidationError):
        DecreasingBackprop(**{"base_lr": 0.1, **kwargs})


def test_schedule_advances_once_per_commit_and_persists():
    schedule = DecreasingBackprop(base_lr=0.2, decay_factor=0.9)
    schedule.advance()
    schedule.advance()
    assert schedule.index == 2
    assert schedule.current_lr == pytest.approx(0.2 * 0.9 ** 2)
    restored = DecreasingBackprop().load_state_dict(schedule.state_dict())
    assert restored.index == 2
    assert restored.current_lr == pytest.approx(schedule.current_lr)


def test_schedule_integrates_with_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    schedule = DecreasingBackprop(base_lr=0.1, decay_factor=0.5, index=2)
    result = schedule.apply_to_optimizer(optimizer)
    assert result["applied_lr"] == pytest.approx(0.025)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_relu_down_is_experimental_only():
    schedule = DecreasingBackprop(base_lr=0.1)
    assert schedule.experimental_relu_down is False
    x = torch.tensor([2.0, -2.0])
    assert torch.allclose(relu_down(x, 0.5), torch.tensor([2.0, -1.0]))
    with pytest.raises(ValueError):
        relu_down(x, 2.0)


# ---------------------------------------------------------------------------
# MAS importance and pruning (tasks 2.1-2.5)
# ---------------------------------------------------------------------------
@pytest.fixture
def small_adapter():
    from src.model.delta_memory import DeltaPlasticAdapter
    torch.manual_seed(0)
    return DeltaPlasticAdapter(dim=32, heads=4)


def test_mas_importance_is_finite_non_negative_and_label_free(small_adapter):
    mas = MemoryAwareSynapses(small_adapter)
    inputs = torch.randn(16, 32)
    stats = mas.estimate_importance(inputs, batch_size=8)
    assert stats["batches"] == 2
    assert stats["backward_passes"] == 2
    assert set(stats["eligible_parameters"]) == set(mas.importance)
    assert "S" not in mas.importance  # buffers are not eligible
    for score in mas.importance.values():
        assert torch.isfinite(score).all()
        assert torch.all(score >= 0)


def test_mas_performs_one_backward_pass_per_batch(small_adapter):
    mas = MemoryAwareSynapses(small_adapter)
    calls = {"count": 0}

    def hook(grad):
        calls["count"] += 1
        return grad

    # A parameter that the differentiable output actually uses: exactly one
    # gradient (hence one backward pass) per sampled batch.
    handle = small_adapter.W_K.weight.register_hook(hook)
    try:
        stats = mas.estimate_importance(torch.randn(20, 32), batch_size=5)
    finally:
        handle.remove()
    assert stats["batches"] == 4
    assert calls["count"] == 4


def test_mas_prune_is_exact_deterministic_and_reversible(small_adapter):
    mas = MemoryAwareSynapses(small_adapter, ratio=0.1)
    mas.estimate_importance(torch.randn(8, 32))
    total = sum(p.numel() for p in mas.eligible_parameters().values())
    report = mas.prune()
    assert report["selected"] == int(total * 0.1)
    assert report["total"] == total
    assert report["rounding"] == "floor"
    assert sum(int((~mask).sum()) for mask in mas.masks.values()) == report["selected"]
    assert sum(int((p == 0).sum()) for p in mas.eligible_parameters().values()) >= report["selected"]

    # Re-running the same prune on the same scores yields the same masks.
    first = {name: mask.clone() for name, mask in mas.masks.items()}
    mas.importance = {name: value.clone() for name, value in mas.importance.items()}
    mas.prune()
    for name, mask in mas.masks.items():
        assert torch.equal(first[name], mask)

    originals = {name: value.clone() for name, value in mas.originals.items()}
    restored = mas.restore()
    assert restored["status"] == "restored"
    params = mas.eligible_parameters()
    for name, value in originals.items():
        assert torch.equal(params[name].detach(), value)


def test_mas_prune_reports_retrieval_quality_before_and_after(small_adapter):
    """Task 2.5: pruning quality is measured, reported and never hidden."""
    scores = iter([0.82, 0.74])

    def quality():
        return next(scores)

    mas = MemoryAwareSynapses(small_adapter, ratio=0.1, quality_fn=quality)
    mas.estimate_importance(torch.randn(8, 32))
    report = mas.prune()

    assert report["quality_before"] == pytest.approx(0.82)
    assert report["quality_after"] == pytest.approx(0.74)
    assert report["quality_delta"] == pytest.approx(-0.08)
    # A regression is surfaced, not silently swallowed.
    assert "regressed" in report["warning"]


def test_mas_prune_reports_absent_quality_instead_of_guessing(small_adapter):
    mas = MemoryAwareSynapses(small_adapter, ratio=0.1)
    mas.estimate_importance(torch.randn(8, 32))
    report = mas.prune()
    assert report["quality_before"] is None
    assert report["quality_after"] is None
    assert report["quality_delta"] is None
    assert report["warning"] is None


def test_mas_does_not_mutate_buffers_or_ineligible_parameters(small_adapter):
    mas = MemoryAwareSynapses(small_adapter)
    mas.estimate_importance(torch.randn(4, 32))
    buffers_before = {name: value.clone() for name, value in mas.non_eligible_buffers().items()}
    excluded = {name: p.detach().clone() for name, p in small_adapter.named_parameters()
                if name not in mas.eligible_parameters()}
    mas.prune()
    for name, value in mas.non_eligible_buffers().items():
        assert torch.equal(value, buffers_before[name])
    for name, value in excluded.items():
        assert torch.equal(dict(small_adapter.named_parameters())[name].detach(), value)


def test_mas_allowlist_limits_pruning_scope(small_adapter):
    mas = MemoryAwareSynapses(small_adapter, eligible=["W_K.weight"])
    mas.estimate_importance(torch.randn(4, 32))
    others = {name: p.detach().clone() for name, p in small_adapter.named_parameters()
              if name != "W_K.weight"}
    mas.prune()
    for name, value in others.items():
        assert torch.equal(dict(small_adapter.named_parameters())[name].detach(), value)


def test_mas_state_round_trip(tmp_path, small_adapter):
    mas = MemoryAwareSynapses(small_adapter, ratio=0.1,
                              quality_fn=_ScriptedQuality([0.9, 0.85]))
    mas.estimate_importance(torch.randn(4, 32))
    report = mas.prune()
    assert report["quality_before"] == 0.9
    assert report["quality_after"] == 0.85
    assert report["quality_delta"] == pytest.approx(-0.05)
    assert "regressed" in report["warning"]

    path = tmp_path / "mas.safetensors"
    meta = tmp_path / "mas_meta.json"
    mas.save_state(str(path), str(meta))

    fresh_adapter = type(small_adapter)(dim=32, heads=4)
    fresh = MemoryAwareSynapses(fresh_adapter)
    loaded = fresh.load_state(str(path), str(meta))
    assert loaded["masks"] == len(mas.masks)
    for name, mask in mas.masks.items():
        assert torch.equal(fresh.masks[name], mask)
        assert torch.allclose(fresh.importance[name], mas.importance[name])
    expected = {name: value.clone() for name, value in mas.originals.items()}
    fresh.restore()
    for name, value in expected.items():
        assert torch.allclose(dict(fresh_adapter.named_parameters())[name], value)


def test_mas_requires_importance_before_prune(small_adapter):
    with pytest.raises(RuntimeError):
        MemoryAwareSynapses(small_adapter).prune()


# ---------------------------------------------------------------------------
# Autonomous retrieval (tasks 1.1-1.5)
# ---------------------------------------------------------------------------
def test_autonomous_retrieval_recovers_95_of_100_correlated_patterns(tmp_path):
    droid = _engine(tmp_path)
    expected = _store_correlated(droid, count=100)
    consolidator = SleepConsolidation(
        droid, max_replay_steps=160, max_seconds=20.0, duplicate_cap=2
    )
    # No expected-pattern list is passed: coverage is scored afterwards.
    report = consolidator.autonomous_retrieval()
    assert report["status"] == "completed"
    assert report["eligible_patterns"] == 100
    assert report["unique_recovered"] >= 95
    assert report["coverage"] >= 0.95
    assert set(report["retrieved_ids"]).issubset(set(expected))
    assert report["replay_writes"] == report["steps"]


def test_self_inhibition_surfaces_other_attractors(tmp_path):
    droid = _engine(tmp_path)
    _store_correlated(droid, count=10, seed=4)
    consolidator = SleepConsolidation(droid, max_replay_steps=6, max_seconds=5.0)
    report = consolidator.autonomous_retrieval()
    assert len(set(report["retrieved_ids"])) > 1
    assert report["steps"] == 6


def test_autonomous_retrieval_is_bounded_and_cancellable(tmp_path):
    droid = _engine(tmp_path)
    _store_correlated(droid, count=40)
    cancel = threading.Event()
    cancel.set()
    report = SleepConsolidation(droid).autonomous_retrieval(cancel_event=cancel)
    assert report["status"] == "cancelled"
    assert report["steps"] == 0

    bounded = SleepConsolidation(droid, max_replay_steps=5).autonomous_retrieval()
    assert bounded["steps"] <= 5


def test_autonomous_retrieval_on_empty_memory(tmp_path):
    droid = _engine(tmp_path)
    report = SleepConsolidation(droid).autonomous_retrieval()
    assert report["status"] == "no_patterns"
    assert report["steps"] == 0


def test_duplicate_cap_is_enforced(tmp_path):
    droid = _engine(tmp_path)
    _store_correlated(droid, count=3, seed=9)
    report = SleepConsolidation(
        droid, max_replay_steps=50, duplicate_cap=1
    ).autonomous_retrieval()
    assert report["duplicates"] == 0
    counts = {fact_id: report["retrieved_ids"].count(fact_id)
              for fact_id in set(report["retrieved_ids"])}
    assert all(count == 1 for count in counts.values())


# ---------------------------------------------------------------------------
# DroidEngine.sleep (tasks 1.3-1.4, 4.1-4.4)
# ---------------------------------------------------------------------------
def test_sleep_on_empty_droid_is_a_successful_noop(tmp_path):
    droid = _engine(tmp_path)
    report = droid.sleep()
    assert report["status"] == "no_op_empty"
    assert report["replay"]["steps"] == 0
    assert report["fact_pruning"]["pruned"] == 0
    assert report["parameter_pruning"]["selected"] == 0
    assert report["errors"] == []


def test_sleep_runs_one_cycle_and_separates_counts(tmp_path):
    droid = _engine(tmp_path)
    droid.teach("Remote sensing acquires information about the Earth.")
    droid.teach("Lidar measures distance using laser pulses.")
    report = droid.sleep(max_seconds=5.0)
    assert report["status"] == "completed"
    assert report["replay"]["steps"] >= 1
    assert "pruned" in report["fact_pruning"]
    assert report["parameter_pruning"]["status"] == "disabled"
    assert report["parameter_pruning"]["selected"] == 0
    assert report["learning_rate"]["i"] == 1
    assert report["cycles"] == 1
    assert droid.sleep_cycles == 1


def test_cancelled_sleep_does_not_advance_schedule(tmp_path):
    droid = _engine(tmp_path)
    droid.teach("Synthetic aperture radar images the surface.")
    cancel = threading.Event()
    cancel.set()
    report = droid.sleep(cancel_event=cancel)
    assert report["status"] == "cancelled"
    assert report["learning_rate"]["i"] == 0
    # A cancelled cycle must not be remembered as a committed cycle.
    assert droid.sleep_cycles == 0


def test_sleep_with_mas_pruning_reports_parameter_counts(tmp_path, monkeypatch):
    droid = _engine(tmp_path)
    droid.teach("The satellite orbits the Earth twice per day.")
    droid.configure_sleep(mas_enabled=True, parameter_prune_ratio=0.1)
    buffers_before = {name: value.clone() for name, value in droid.memory.named_buffers()}
    total = sum(p.numel() for _, p in droid.memory.named_parameters())
    # Isolate the MAS phase: replay and fact-consolidation legitimately write
    # associative buffers, while MAS must not.
    monkeypatch.setattr(droid, "consolidate_memory", lambda **kwargs: {
        "status": "consolidated", "pruned": 0,
        "total_before": droid.knowledge.active_count(),
        "total_after": droid.knowledge.active_count(),
    })
    report = droid.sleep(max_seconds=5.0, max_replay_steps=0)
    pruning = report["parameter_pruning"]
    assert pruning["status"] == "pruned"
    assert pruning["selected"] == int(total * 0.1)
    assert pruning["total"] == total
    for name, value in droid.memory.named_buffers():
        assert torch.equal(value, buffers_before[name])
    # Reversible originals are retained for an explicit restore.
    assert droid.mas.originals
    droid.mas.restore()


def test_sleep_command_dispatches_without_storing_a_fact(tmp_path):
    droid = _engine(tmp_path)
    droid.teach("Berlin is the capital of Germany.")
    before = droid.knowledge.active_count()
    text = droid.chat("/sleep")
    assert "Sleep cycle" in text
    assert droid.knowledge.active_count() == before
    assert droid.chat("/sleep").startswith("Sleep cycle")


def test_sleep_reports_survive_profile_round_trip(tmp_path):
    droid = _engine(tmp_path)
    droid.teach("A fact worth consolidating.")
    droid.configure_sleep(enabled=True, idle_seconds=12.5, max_replay_steps=7)
    droid.sleep(max_seconds=5.0)
    profile = tmp_path / "profile"
    droid.save_profile(str(profile))

    loaded = DroidEngine(name="sleeper")
    loaded.load_profile(str(profile))
    assert loaded.sleep_config["enabled"] is True
    assert loaded.sleep_config["idle_seconds"] == 12.5
    assert loaded.sleep_config["max_replay_steps"] == 7
    assert loaded.lr_schedule.index == droid.lr_schedule.index


def test_mas_state_survives_profile_round_trip(tmp_path):
    droid = _engine(tmp_path)
    droid.teach("Consolidated fact for MAS persistence.")
    droid.configure_sleep(mas_enabled=True)
    droid.sleep(max_seconds=5.0)
    profile = tmp_path / "mas-profile"
    droid.save_profile(str(profile))
    assert (profile / "mas.safetensors").exists()

    loaded = DroidEngine(name="sleeper")
    loaded.load_profile(str(profile))
    assert loaded.mas is not None
    for name, mask in droid.mas.masks.items():
        assert torch.equal(loaded.mas.masks[name], mask)
    loaded.mas.restore()


# ---------------------------------------------------------------------------
# Auto-sleep worker lifecycle (task 4.2-4.4)
# ---------------------------------------------------------------------------
def test_auto_sleep_is_disabled_by_default(tmp_path):
    manager = DroidManager(base_dir=str(tmp_path / "droids"))
    result = manager.start_auto_sleep("worker")
    assert result["status"] == "disabled"
    assert manager.auto_sleep_status("worker")["alive"] is False


def test_single_worker_and_bounded_stop(tmp_path):
    manager = DroidManager(base_dir=str(tmp_path / "droids"))
    droid = manager.get_droid("worker")
    droid.configure_sleep(enabled=True, idle_seconds=0.0, interval_seconds=0.02)
    first = manager.start_auto_sleep("worker")
    second = manager.start_auto_sleep("worker")
    assert first["status"] == "started"
    assert second["status"] == "already_running"
    time.sleep(0.15)
    stopped = manager.stop_auto_sleep("worker", timeout=5.0)
    assert stopped["status"] == "stopped"
    assert stopped["alive"] is False
    assert manager.auto_sleep_status("worker")["alive"] is False


def test_worker_restarts_and_shutdown_stops_all(tmp_path):
    manager = DroidManager(base_dir=str(tmp_path / "droids"))
    droid = manager.get_droid("worker")
    droid.configure_sleep(enabled=True, idle_seconds=0.0, interval_seconds=0.02)
    assert manager.start_auto_sleep("worker")["status"] == "started"
    assert manager.stop_auto_sleep("worker")["status"] == "stopped"
    assert manager.start_auto_sleep("worker")["status"] == "started"
    manager.shutdown()
    assert manager.auto_sleep_status("worker")["alive"] is False


def test_delete_droid_cancels_worker(tmp_path):
    manager = DroidManager(base_dir=str(tmp_path / "droids"))
    droid = manager.get_droid("doomed")
    droid.configure_sleep(enabled=True, idle_seconds=0.0, interval_seconds=0.02)
    manager.start_auto_sleep("doomed")
    assert manager.delete_droid("doomed") is True
    assert manager.auto_sleep_status("doomed")["alive"] is False


def test_worker_reports_cycles(tmp_path):
    droid = _engine(tmp_path, name="cycle")
    droid.teach("A fact that keeps the cycle non-empty.")
    worker = AutoSleepWorker(droid, {"enabled": True, "idle_seconds": 0.0,
                                     "interval_seconds": 0.01, "max_seconds": 2.0})
    assert worker.start() is True
    time.sleep(0.25)
    assert worker.stop(timeout=5.0) is True
    assert worker.cycles >= 1
    assert worker.last_report is not None
