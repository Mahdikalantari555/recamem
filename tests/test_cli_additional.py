"""Contract tests for `src/cli.py` (surface-test-coverage 3.9).

Covered: subcommand dispatch, `eval` output shape and exit codes, unknown
subcommand handling, the stdin/file teach path with a fake engine, and the
decide fallback when no anchor embedding is available.
"""

import json

import pytest

import src.cli as cli
from src.cli import main

TEACH_TEXT = (
    "Remote sensing is the acquisition of information about an object or "
    "phenomenon without making physical contact with the object. "
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


def _parse_kv(stdout):
    pairs = {}
    for line in stdout.strip().splitlines():
        assert "=" in line, f"non key=value line: {line!r}"
        key, value = line.split("=", 1)
        assert key and value
        pairs[key] = value
    return pairs


def test_eval_self_recall_reports_mode_and_metrics(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "eval", "--name", "droid-rs", "--sample", "3"])
    out = capsys.readouterr()

    assert rc == 0, out.err
    fields = _parse_kv(out.out)
    assert fields["mode"] == "self_recall"
    assert fields["profile"] == "droid-rs"
    assert int(fields["facts"]) > 0
    assert int(fields["samples"]) == min(3, int(fields["facts"]))
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        assert 0.0 <= float(fields[metric]) <= 1.0


def test_eval_missing_profile_is_invalid_input(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "eval", "--name", "does-not-exist"])
    out = capsys.readouterr()

    assert rc == 2
    assert out.out == ""
    assert out.err.startswith("error:")


def test_serve_requires_mcp_or_rest(droid_base, capsys):
    rc = main(["--base-dir", droid_base, "serve"])
    out = capsys.readouterr()

    assert rc == 1
    assert "Must specify --mcp or --rest" in out.err


def test_unknown_subcommand_exits_non_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code not in (None, 0)


def test_package_export_import_list_round_trip(droid_base, tmp_path, capsys):
    archive = tmp_path / "droid-rs.droid"

    rc = main(["--base-dir", droid_base, "package", "export", "droid-rs", "--out", str(archive)])
    assert rc == 0, capsys.readouterr().err
    assert archive.exists()

    rc = main(["package", "list", str(archive)])
    assert rc == 0, capsys.readouterr().err

    rc = main(["package", "import", str(archive), "--name", "imported-rs"])
    assert rc == 0, capsys.readouterr().err


def test_chat_single_query_returns(droid_base, capsys):
    # `query` is a positional argument of the `chat` subcommand.
    rc = main(["--base-dir", droid_base, "chat", "--name", "droid-rs", "what is lidar?"])
    assert not rc, capsys.readouterr().err  # cmd_chat returns None -> exit code 0


def test_teach_from_file_with_fake_engine(monkeypatch, tmp_path, capsys):
    class FakeDroid:
        def teach(self, text, source="cli"):
            return {
                "propositions": 2,
                "episodic_absorbed": 2,
                "memory_norm": 0.5,
                "avg_surprise": 0.1,
                "total_knowledge": 2,
            }

    class FakeMgr:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def get_droid(self, name):
            return FakeDroid()

        def save_droid(self, name):
            pass

    monkeypatch.setattr("src.model.droid_manager.DroidManager", FakeMgr)

    text_file = tmp_path / "input.md"
    text_file.write_text("The ozone layer absorbs most of the sun's ultraviolet radiation.")

    out = capsys.readouterr()  # clear any startup logging
    rc = main(["teach", "--name", "droid-x", "--file", str(text_file), "--source", "cli"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "Absorbed 2 facts" in out.out


def test_decide_falls_back_to_prototype_routing(monkeypatch, capsys):
    class FakeMgr:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def get_droid(self, name):
            raise RuntimeError("no anchor available")

    monkeypatch.setattr("src.model.droid_manager.DroidManager", FakeMgr)

    err = capsys.readouterr()  # clear any startup logging
    rc = main(["decide", "which tool should I use?", "--options", "recall,chat", "--json"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "Selected action:" in out.out
    payload = json.loads(out.out[out.out.index("{"):])
    assert payload["action"] in ("recall", "chat", "ambiguous")
    assert "confidence" in payload and 0.0 <= payload["confidence"] <= 1.0
