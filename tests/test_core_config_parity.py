"""Parity between the shipped core manifest and its tournament evidence.

`configs/recamem_selected_core.json` is a generated artifact. The tournament
ledger is the source of truth for the winning stack; the runtime status may stay
`provisional_core` until the 17M target shape is actually trained
(`src/model/recamem_core.py:validate_selected_core_manifest`).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recamem_selected_core.json"
LEDGER = ROOT / "reports" / "evidence" / "recamem-core-tournament.json"
REPORT = ROOT / "reports" / "recamem-core-selection.md"


def _load():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    selected = ledger["measurements"]["selected_manifest"]["value"]
    return config, selected, REPORT.read_text(encoding="utf-8")


def test_selected_architecture_matches_tournament_winner():
    config, selected, _ = _load()
    assert config["selected_architecture"] == selected["selected_architecture"]


def test_hyperparameters_and_calibration_match_tournament_winner():
    config, selected, _ = _load()
    assert config["hyperparameters"] == selected["hyperparameters"]
    assert config["calibration"] == selected["calibration"]


def test_runtime_status_is_provisional_while_17m_is_untrained():
    config, selected, report = _load()
    assert config["target_17m_hyperparameters"]["trained"] is False
    assert config["tournament_provenance"]["target_17m_trained"] is False
    assert config["selection_status"] == "provisional_core"
    # The ledger is the raw tournament output and uses a different status notion:
    # "verified_winner" for the 350K custom tier. Keep the divergence documented.
    assert selected["selection_status"] == "verified_winner"
    assert "**Selection Status**: `provisional_core`" in report
    assert "350K" in report or "349,920" in report


def test_provenance_is_complete_and_points_at_existing_artifacts():
    config, _, _ = _load()
    prov = config["tournament_provenance"]
    assert prov["dataset_source"] == "fixture"
    assert prov["stages_completed"] == [0, 1, 2, 3, 4]
    assert prov["unmeasured_metrics_count"] == 0
    assert prov["all_stage_gates_evaluated"] is True
    for key in ("evidence_ledger_path", "selection_report_path"):
        assert (ROOT / prov[key]).is_file(), f"{key} does not resolve: {prov[key]}"
    assert (ROOT / config["tokenizer_artifact_path"]).is_file()


def test_config_is_described_as_a_generated_artifact():
    for name in ("core-candidates.md", "architecture-tournament.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "generated" in text.lower(), f"docs/{name} must document the generated config"
        assert "recamem_selected_core.json" in text
