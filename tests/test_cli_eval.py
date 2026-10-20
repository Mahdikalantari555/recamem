"""Tests for `python -m src.cli eval` (product-surface-completion 5.1/5.2).

The command must measure something real, name the mode it ran, print stable
`key=value` lines and return honest exit codes: 0 success, 1 evaluation
failure, 2 invalid input.
"""

import json
import shutil
import tempfile

import pytest

from src.cli import main

TEACH_TEXT = (
    "Remote sensing is the acquisition of information about an object or phenomenon "
    "without making physical contact with the object. "
    "Lidar measures distance with laser pulses. "
    "Hydrology studies the movement of water over land."
)


@pytest.fixture
def droid_base(tmp_path):
    from src.model.droid_manager import DroidManager

    base = tmp_path / "droids"
    mgr = DroidManager(base_dir=str(base))
    droid = mgr.create_droid("droid-rs")
    droid.teach(TEACH_TEXT, source="test")
    mgr.save_droid("droid-rs")
    yield str(base)
    for name in list(mgr.cached_instances):
        try:
            mgr.cached_instances[name].knowledge.close()
        except Exception:
            pass


def _parse_kv(stdout: str) -> dict:
    pairs = {}
    for line in stdout.strip().splitlines():
        assert "=" in line, f"non key=value line: {line!r}"
        key, value = line.split("=", 1)
        assert key and value
        pairs[key] = value
    return pairs


def test_self_recall_eval_reports_mode_and_metrics(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "eval", "--name", "droid-rs", "--sample", "3"])
    out = capsys.readouterr()

    assert rc == 0, out.err
    fields = _parse_kv(out.out)
    assert fields["mode"] == "self_recall"
    assert fields["profile"] == "droid-rs"
    assert int(fields["facts"]) > 0
    assert int(fields["samples"]) == min(3, int(fields["facts"]))
    assert int(fields["top_k"]) == 5
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        value = float(fields[metric])
        assert 0.0 <= value <= 1.0
    # Every metric line is plain ASCII with no ANSI escapes.
    assert "\x1b" not in out.out


def test_labeled_eval_uses_query_file(droid_base, tmp_path, capsys):
    from src.model.droid_manager import DroidManager

    mgr = DroidManager(base_dir=droid_base)
    facts = mgr.get_droid("droid-rs").knowledge.get_active_facts()
    target = facts[0]
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query": target["text"], "fact_ids": [target["id"]]}) + "\n")

    rc = main([
        "--base-dir", droid_base, "eval", "--name", "droid-rs",
        "--queries", str(queries), "--top-k", "3",
    ])
    out = capsys.readouterr()

    assert rc == 0, out.err
    fields = _parse_kv(out.out)
    assert fields["mode"] == "labeled"
    assert fields["samples"] == "1"
    assert fields["top_k"] == "3"
    assert float(fields["recall_at_k"]) == pytest.approx(1.0)


def test_missing_profile_is_invalid_input(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "eval", "--name", "does-not-exist"])
    out = capsys.readouterr()

    assert rc == 2
    assert out.out == ""
    assert out.err.startswith("error:")


def test_malformed_query_file_is_invalid_input(droid_base, tmp_path, capsys):
    queries = tmp_path / "broken.jsonl"
    queries.write_text('{"query": "no fact ids"}\n')

    rc = main(["--base-dir", droid_base, "eval", "--name", "droid-rs", "--queries", str(queries)])
    out = capsys.readouterr()

    assert rc == 2
    assert out.out == ""
    assert out.err.startswith("error:")
    assert "broken.jsonl" in out.err


def test_bad_top_k_is_invalid_input(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "eval", "--name", "droid-rs", "--top-k", "0"])
    out = capsys.readouterr()

    assert rc == 2
    assert out.out == ""
    assert "top-k" in out.err


def test_empty_store_is_invalid_input(tmp_path, capsys):
    from src.model.droid_manager import DroidManager

    base = tmp_path / "droids"
    DroidManager(base_dir=str(base)).create_droid("empty-droid")

    rc = main(["--base-dir", str(base), "eval", "--name", "empty-droid"])
    out = capsys.readouterr()

    assert rc == 2
    assert "empty knowledge store" in out.err
    assert out.out == ""


def test_evaluation_failure_returns_one(droid_base, capsys, monkeypatch):
    from src.model.droid import DroidEngine

    def boom(self, *args, **kwargs):
        raise RuntimeError("index corrupt")

    monkeypatch.setattr(DroidEngine, "recall", boom)
    rc = main(["--base-dir", droid_base, "eval", "--name", "droid-rs"])
    out = capsys.readouterr()

    assert rc == 1
    assert out.err.startswith("error:")
    assert "index corrupt" in out.err
    # A failed run prints no metric lines.
    assert out.out == ""
