"""Two-tier plasticity: schedule, no-grad FAST writes, gated SLOW updates."""

import json

import pytest
import torch

from recamem.engine import DroidEngine
from recamem.plasticity import (
    PLASTIC_SCHEMA_VERSION,
    PlasticStateStore,
    SlowTier,
    UnsupportedPlasticSchema,
    effective_lr,
    fast_write,
    upgrade_state,
)
from recamem.plastic_adapter import get_adapter
pytestmark = pytest.mark.phase0



def _delta_engine():
    return DroidEngine(name="fast", plastic_adapter="delta_rtu")


def test_effective_lr_matches_the_schedule():
    assert effective_lr(0.05, 0) == pytest.approx(0.05)
    assert effective_lr(0.05, 80) == pytest.approx(0.05 / (5 ** 0.5))


def test_fast_write_changes_state_by_the_schedule_ratio():
    engine = _delta_engine()
    x = torch.ones(384) / (384 ** 0.5)
    early = fast_write(engine, x, base_lr=0.05, step_count=0)
    state_early = engine.memory.states.detach().clone()
    fresh = _delta_engine()
    late = fast_write(fresh, x, base_lr=0.05, step_count=80)
    state_late = fresh.memory.states.detach().clone()
    assert early["applied_lr"] > late["applied_lr"]
    ratio = (state_early.norm() / state_late.norm()).item()
    assert ratio == pytest.approx(early["applied_lr"] / late["applied_lr"], rel=0.05)


def test_fast_write_produces_no_gradient_tracking():
    engine = _delta_engine()
    record = fast_write(engine, torch.ones(384) / 20.0)
    assert record["grad_tracking"] is False
    for tensor in (engine.memory.states,):
        assert tensor.requires_grad is False


def test_engine_records_requested_and_applied_rates():
    engine = _delta_engine()
    result = engine.teach("The calibration constant is 0.98.")
    assert "requested_lr" in result and "applied_lr" in result
    assert result["applied_lrs"]


def test_slow_tier_commit_is_gated_by_the_probe():
    tier = SlowTier(dim=8, tolerance=0.05)
    items = torch.eye(8) + 0.1
    accepted = tier.gated_update(items, lambda: {"generation": 1.0})
    assert accepted["accepted"] is True
    assert accepted["after"] == {"generation": 1.0}

    calls = {"n": 0}

    def regressing_probe():
        calls["n"] += 1
        return {"generation": 1.0 if calls["n"] == 1 else 0.4}

    before_state = tier.snapshot()
    rejected = tier.gated_update(items, regressing_probe, lr=0.5)
    assert rejected["accepted"] is False
    assert rejected["rejected_domain"] == "generation"
    assert rejected["reason"]
    assert torch.equal(tier.state_dict()["meta.weight"], before_state["meta.weight"])


def test_memory_aware_penalty_moves_important_parameters_less():
    tier = SlowTier(dim=4)
    anchor = tier.meta.weight.detach().clone()
    with torch.no_grad():
        tier.meta.weight.add_(0.1)
    tier.synapses.anchor_state = {"meta.weight": anchor}
    tier.synapses.importance = {"meta.weight": torch.ones_like(tier.meta.weight)}
    important = tier.synapses.penalty([("meta.weight", tier.meta.weight)]).item()
    tier.synapses.importance = {"meta.weight": torch.zeros_like(tier.meta.weight)}
    unimportant = tier.synapses.penalty([("meta.weight", tier.meta.weight)]).item()
    assert important > unimportant == 0.0


def test_newer_schema_is_refused_and_older_is_upgraded(tmp_path):
    with pytest.raises(UnsupportedPlasticSchema) as excinfo:
        upgrade_state({"schema_version": PLASTIC_SCHEMA_VERSION + 1})
    assert "newer" in str(excinfo.value)

    upgraded = upgrade_state({"schema_version": 1, "weights": [1, 2, 3]})
    assert upgraded["schema_version"] == PLASTIC_SCHEMA_VERSION
    assert upgraded["weights"] == [1, 2, 3]

    store = PlasticStateStore()
    path = store.save(tmp_path / "p", upgraded)
    loaded = store.load(tmp_path / "p")
    assert loaded == upgraded
    assert json.loads(path.read_text())["schema_version"] == PLASTIC_SCHEMA_VERSION
