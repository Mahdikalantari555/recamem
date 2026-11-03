"""Checkpoint eligibility is provenance-backed, not recency-based."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from recamem import companion


def _make(root: Path, name: str, bpb: float, *, gates=None, extra=None, tamper=False):
    directory = root / name
    directory.mkdir(parents=True)
    weights = directory / "weights.bin"
    weights.write_bytes(b"weights-" + name.encode())
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "files": [{"path": "weights.bin", "sha256": sha}],
        "config": {"d_model": 64}, "seed": 1337, "dataset": "fixture",
        "commit": "0" * 40, "command": "python scripts/train_lm.py",
        "gates": gates or {"G0": "not_evaluated", "G2": "passed"},
        "held_out_bpb": bpb,
    }
    if extra:
        manifest.update(extra)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if tamper:
        weights.write_bytes(b"tampered")
    return directory


def test_eligible_checkpoint_and_best_selection(tmp_path):
    _make(tmp_path, "run-a", 2.0)
    _make(tmp_path, "run-b", 1.5)
    rows = companion.list_checkpoints(tmp_path)
    assert all(row["eligible"] for row in rows)
    best = companion.best_checkpoint(rows)
    assert best["status"] == "ok" and best["checkpoint"]["name"] == "run-b"
    assert best["eligible_count"] == 2


def test_ineligible_without_provenance_or_gates(tmp_path):
    _make(tmp_path, "no-bpb", 1.0, extra={"held_out_bpb": None})
    _make(tmp_path, "no-g2", 1.0, gates={"G2": "not_evaluated"})
    (tmp_path / "no-manifest").mkdir()
    rows = companion.list_checkpoints(tmp_path)
    assert companion.best_checkpoint(rows)["status"] == "none"
    assert all(not row["eligible"] for row in rows)


def test_tampered_manifest_is_rejected(tmp_path):
    _make(tmp_path, "tampered", 1.0, tamper=True)
    row = companion.inspect_checkpoint(tmp_path / "tampered")
    assert not row["eligible"] and "checksum" in row["reason"]


def test_checkpoint_root_rejects_escape(tmp_path):
    # The API resolves candidate names under the root; this pins the helper too.
    root = tmp_path / "checkpoints"
    root.mkdir()
    row = companion.inspect_checkpoint(root / "..")
    assert not row["eligible"]
