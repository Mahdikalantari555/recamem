"""The documented remote-sensing end-to-end script, run as a test.

`post-merge-completion` 9.1 asks for "remote-sensing E2E" on the combined
implementation commit; `hybrid-knowledge-tms` T5.4 asks the same. The script
`scripts/verify_remote_sensing_e2e.py` is the shipped artifact, so the test
runs it as a subprocess in an isolated directory and asserts its own exit code
and gate lines instead of re-implementing the checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_remote_sensing_e2e.py"


def _run(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_remote_sensing_e2e_script_passes(tmp_path):
    completed = _run(tmp_path)
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-4000:]
    assert "ALL END-TO-END VERIFICATION CHECKS PASSED" in output
    # The script's own gates, named so a silent regression is visible here.
    for gate in (
        "Passed definition recall check",
        "Passed contrast recall check",
        "Passed field application recall check",
        "Zero Catastrophic Collapse",
        "Reloaded profile successfully restored exact memory state",
    ):
        assert gate in output, f"missing gate {gate!r} in:\n{output[-4000:]}"


def test_remote_sensing_e2e_writes_only_inside_the_working_directory(tmp_path):
    """The script owns its profile directory; it must not write into the repo."""
    before = {p for p in ROOT.iterdir()}
    completed = _run(tmp_path)
    assert completed.returncode == 0
    after = {p for p in ROOT.iterdir()}

    assert after == before, f"the E2E run created repository artifacts: {after - before}"
    assert (tmp_path / "droids").is_dir(), "the profile belongs to the run directory"
