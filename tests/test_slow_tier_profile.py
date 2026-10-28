"""SLOW tier meta-parameters round-trip through a profile (plasticity 3.6)."""

import json

import torch

from recamem.engine import DroidEngine


def test_slow_tier_meta_parameters_survive_save_and_load(tmp_path):
    engine = DroidEngine(name="slow")
    tier = engine.enable_slow_tier(tolerance=0.02)
    tier.gated_update(torch.eye(384), lambda: {"generation": 1.0})

    engine.teach("The calibration constant is 0.98.")
    engine.save_profile(str(tmp_path / "p"))
    payload = json.loads((tmp_path / "p" / "slow_tier.json").read_text())
    assert payload["tolerance"] == 0.02
    assert "meta.weight" in payload["weights"]

    restored = DroidEngine(name="slow")
    assert restored.load_profile(str(tmp_path / "p")) is True
    assert restored.slow_tier is not None
    assert restored.slow_tier.tolerance == 0.02
    for key, value in tier.state_dict().items():
        assert torch.allclose(restored.slow_tier.state_dict()[key], value)
