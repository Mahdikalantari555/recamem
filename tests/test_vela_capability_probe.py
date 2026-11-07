"""The Vela harness must prove hidden injection before claiming a plastic effect.

`evaluation/vela-benchmark-poc` requires the harness to probe the backend and
report `blocked`, naming the missing capability, when hidden injection is not
available. Prompt-only C-versus-D numbers must never be turned into a
plastic-memory claim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_vela_benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("vela_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vela_benchmark"] = module
    spec.loader.exec_module(module)
    return module


vela = _load()


class _PromptOnly:
    """A backend that advertises nothing: prompt-only by the safe default."""

    def generate(self, prompt, **kwargs):  # pragma: no cover - not used here
        return "IRRIGATE"


class _Hidden:
    supports_hidden_injection = True
    supports_kv_priming = True

    def generate(self, prompt, **kwargs):  # pragma: no cover - not used here
        return "IRRIGATE"


class _ExplicitlyFalse:
    supports_hidden_injection = False


def test_prompt_only_backend_is_blocked_with_the_missing_capability():
    probe = vela.probe_hidden_injection(_PromptOnly())

    assert probe["outcome"] == "blocked"
    assert probe["missing_capability"] == "hidden_injection"
    assert probe["hidden_injection"] is False
    assert probe["backend"] == "_PromptOnly"


def test_explicit_false_declaration_is_respected():
    probe = vela.probe_hidden_injection(_ExplicitlyFalse())

    assert probe["outcome"] == "blocked"
    assert probe["missing_capability"] == "hidden_injection"


def test_hidden_injection_backend_passes_the_probe():
    probe = vela.probe_hidden_injection(_Hidden())

    assert probe["outcome"] == "passed"
    assert probe["missing_capability"] is None
    assert probe["hidden_injection"] is True


def test_no_model_is_reported_blocked_and_claims_nothing(tmp_path):
    report = tmp_path / "probe.json"

    exit_code = vela.main(["--probe-only", "--model", "", "--out", str(report)])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["outcome"] == "blocked"
    assert payload["missing_artifact"]
    assert payload["conditions"] == {}
    assert "no plastic-memory effect is claimed" in payload["claim"]


def test_missing_model_file_names_the_artifact(tmp_path):
    report = tmp_path / "probe.json"
    missing = tmp_path / "absent.gguf"

    exit_code = vela.main(["--probe-only", "--model", str(missing), "--out", str(report)])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["outcome"] == "blocked"
    assert str(missing) in payload["reason"]
    assert payload["missing_artifact"] == str(missing)


def test_default_model_is_not_a_machine_local_path():
    """A machine-local provenance path must never be the default artifact."""
    assert "/mnt/" not in vela.MODEL_PATH
    assert vela.MODEL_PATH == __import__("os").environ.get("RECAMEM_GGUF", "")
