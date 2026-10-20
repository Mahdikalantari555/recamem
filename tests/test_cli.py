"""Contract tests for src/cli.py.

The CLI is a process boundary, so these tests never spawn a subprocess and never
touch a real profile: ``DroidManager`` is replaced at the module boundary and
``sys.argv``/``sys.stdin`` are injected. What is pinned:

- the subcommand -> handler dispatch table, and argparse's non-zero exit for an
  unknown or missing subcommand;
- ``teach`` reads stdin or ``--file``, reports absorbed/novelty/total counts and
  returns an exit code (1 when there is no input);
- ``package export|import|list`` drive ``DroidPackage`` and an unknown package
  subcommand fails;
- ``decide`` and ``generate`` print their documented output shape;
- ``serve`` without a transport and a failed model download both return 1.

The eval metric format is owned by the change that implements ``cmd_eval``; here
only its routing and exit code are asserted.
"""

import io
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from src import cli


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeAnchor:
    def embed(self, text):
        generator = torch.Generator().manual_seed(len(str(text)))
        return torch.randn(384, generator=generator)


class FakeKnowledge:
    def __init__(self, facts):
        self._facts = facts

    def active_count(self):
        return len(self._facts)

    def total_count(self):
        return len(self._facts)

    def get_active_facts(self):
        return list(self._facts)


FACTS = [
    {"id": 1, "text": "Radar uses radio waves."},
    {"id": 2, "text": "Lidar uses lasers."},
]


class FakeDroid:
    def __init__(self, name="droid-alpha"):
        self.name = name
        self.dim = 384
        self.step_count = 4
        self.anchor = FakeAnchor()
        self.knowledge = FakeKnowledge(FACTS)
        self.memory = SimpleNamespace(states=torch.tensor([3.0, 4.0]))
        self.logs = [{"event": "teach"}]
        self.taught = []
        self.queries = []
        self.decisions = []

    def teach(self, text, source=None):
        self.taught.append((text, source))
        return {
            "propositions": 2,
            "episodic_absorbed": 2,
            "memory_norm": 1.25,
            "avg_surprise": 0.75,
            "total_knowledge": 6,
        }

    def recall(self, query, top_k=4, threshold=0.0):
        self.queries.append(query)
        return [
            {"id": 1, "text": "Radar uses radio waves.", "similarity": 0.91},
            {"id": 2, "text": "Lidar uses lasers.", "similarity": 0.72},
        ][:top_k]

    def chat(self, query):
        self.queries.append(query)
        return f"chat answer for {query}"

    def decide_path(self, query):
        self.decisions.append(query)
        return {"action": "fast_recall", "confidence": 0.87}

    def generate_with_memory(self, query, strategy="auto"):
        return {"text": f"generated with {strategy} for {query}", "strategy": strategy}


class FakeManager:
    def __init__(self, base_dir="droids"):
        self.base_dir = base_dir
        self.droid = FakeDroid()
        self.saved = []
        self.requested = []

    def get_droid(self, name):
        self.requested.append(name)
        self.droid.name = name
        return self.droid

    def save_droid(self, name):
        self.saved.append(name)
        return True


@pytest.fixture
def fake_manager(monkeypatch):
    """Replace DroidManager at its defining module (cmd_* import it lazily)."""
    holder = {}

    def factory(base_dir="droids"):
        manager = FakeManager(base_dir=base_dir)
        holder["manager"] = manager
        return manager

    monkeypatch.setattr("src.model.droid_manager.DroidManager", factory)
    return holder


def _args(**kwargs):
    defaults = dict(
        base_dir="droids",
        name="droid-alpha",
        file=None,
        source=None,
        queries=None,
        top_k=5,
        sample=50,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv, expected_command",
    [
        (["chat"], "chat"),
        (["teach"], "teach"),
        (["decide", "what is radar?"], "decide"),
        (["serve"], "serve"),
        (["package", "export", "droid-alpha", "--out", "x.droid"], "package"),
        (["eval", "--name", "demo"], "eval"),
        (["download-model", "org/model", "model.gguf"], "download-model"),
        (["generate", "what is radar?"], "generate"),
    ],
)
def test_main_dispatches_each_subcommand(monkeypatch, argv, expected_command):
    calls = []

    def recorder(name):
        def handler(args):
            calls.append((name, args))
            return 0

        return handler

    for name in ("cmd_chat", "cmd_teach", "cmd_decide", "cmd_serve", "cmd_package",
                 "cmd_eval", "cmd_download_model", "cmd_generate"):
        monkeypatch.setattr(cli, name, recorder(name.replace("cmd_", "").replace("_", "-")))

    monkeypatch.setattr(sys, "argv", ["tmt-droid", *argv])
    assert cli.main() == 0

    assert len(calls) == 1
    handler_name, args = calls[0]
    assert args.command == expected_command
    assert handler_name == expected_command


def test_main_unknown_subcommand_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tmt-droid", "frobnicate"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_main_missing_subcommand_exits_nonzero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tmt-droid"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2


def test_package_requires_nested_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tmt-droid", "package"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# teach
# --------------------------------------------------------------------------- #
def test_teach_reads_stdin_and_reports_counts(monkeypatch, fake_manager, capsys):
    monkeypatch.setattr(sys, "argv", ["tmt-droid", "teach", "--name", "demo"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("Radar uses radio waves for detection."))

    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "Absorbed 2 facts (episodic: 2)" in out
    assert "Memory norm: 1.25, avg surprise: 0.750" in out
    assert "Total knowledge: 6" in out

    manager = fake_manager["manager"]
    assert manager.requested == ["demo"]
    assert manager.saved == ["demo"]
    assert manager.droid.taught == [("Radar uses radio waves for detection.", "cli")]


def test_teach_reads_file_and_explicit_source(monkeypatch, fake_manager, tmp_path, capsys):
    passage = tmp_path / "notes.md"
    passage.write_text("Lidar measures distance with laser pulses.")
    args = _args(file=str(passage), source="markdown")

    assert cli.cmd_teach(args) == 0
    assert "Absorbed 2 facts" in capsys.readouterr().out
    assert fake_manager["manager"].droid.taught == [(passage.read_text(), "markdown")]


def test_teach_empty_input_returns_one(monkeypatch, fake_manager, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))

    assert cli.cmd_teach(_args()) == 1
    assert "No input text provided." in capsys.readouterr().err
    assert fake_manager["manager"].droid.taught == []
    assert fake_manager["manager"].saved == []


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #
def _profile(directory, name="droid-alpha"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"name": name, "dim": 384}))
    (directory / "memory.safetensors").write_bytes(b"weights")
    return directory


def test_package_export_import_list_round_trip(tmp_path, capsys):
    base = _profile(tmp_path / "droids" / "droid-alpha")
    archive = tmp_path / "alpha.droid"

    assert cli.cmd_package(_args(package_cmd="export", name="droid-alpha", out=str(archive), base_dir=str(tmp_path / "droids"))) == 0
    out = capsys.readouterr().out
    assert f"Exported droid-alpha to {archive}" in out
    assert "Files: ['config.json', 'memory.safetensors']" in out

    assert cli.cmd_package(_args(package_cmd="list", archive=str(archive))) == 0
    out = capsys.readouterr().out
    assert f"Archive: {archive}" in out
    assert "manifest.json" in out

    import_base = tmp_path / "imported"
    assert cli.cmd_package(
        _args(package_cmd="import", archive=str(archive), name="copy", base_dir=str(import_base))
    ) == 0
    out = capsys.readouterr().out
    assert "Verified: True" in out
    assert (import_base / "copy" / "config.json").read_text() == (base / "config.json").read_text()


def test_package_export_missing_profile_returns_one(tmp_path, capsys):
    code = cli.cmd_package(
        _args(package_cmd="export", name="ghost", out=str(tmp_path / "g.droid"), base_dir=str(tmp_path / "droids"))
    )
    assert code == 1
    assert "Droid profile not found" in capsys.readouterr().err


def test_package_unknown_subcommand_returns_one(tmp_path, capsys):
    assert cli.cmd_package(_args(package_cmd="frobnicate", archive="x")) == 1
    assert "Unknown package subcommand" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# decide / generate / chat
# --------------------------------------------------------------------------- #
def test_decide_prints_decision_summary(fake_manager, capsys):
    args = _args(query="what is radar?", options=None, json=False)

    assert cli.cmd_decide(args) == 0

    out = capsys.readouterr().out
    assert "Query: what is radar?" in out
    assert "Selected action:" in out
    assert "Confidence:" in out
    assert "Latency:" in out
    assert "Probabilities:" in out
    assert "chat:" in out


def test_decide_json_flag_appends_parseable_payload(fake_manager, capsys):
    args = _args(query="what is radar?", options=None, json=True)
    assert cli.cmd_decide(args) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    start = next(idx for idx, line in enumerate(lines) if line.strip() == "{")
    payload = json.loads("\n".join(lines[start:]))
    # A random anchor embedding usually lands below the routing threshold, so the
    # action is either a prototype name or the documented "ambiguous" rejection.
    assert payload["action"] in payload["prototype_names"] + ["ambiguous"]
    assert len(payload["probabilities"]) == len(payload["prototype_names"])


def test_decide_with_options_builds_official_options(fake_manager, capsys):
    args = _args(query="pick a tool", options="radar, lidar", json=False)
    assert cli.cmd_decide(args) == 0

    out = capsys.readouterr().out
    # Options become the prototype set, so only those two names are scored.
    assert "radar:" in out and "lidar:" in out
    assert "chat:" not in out
    assert len([line for line in out.splitlines() if line.startswith("  ")]) == 2


def test_decide_falls_back_to_random_embedding(monkeypatch, capsys):
    class BrokenManager:
        def __init__(self, base_dir="droids"):
            pass

        def get_droid(self, name):
            raise RuntimeError("anchor unavailable")

    monkeypatch.setattr("src.model.droid_manager.DroidManager", BrokenManager)
    args = _args(query="what is radar?", options=None, json=False)

    assert cli.cmd_decide(args) == 0
    captured = capsys.readouterr()
    assert "could not get anchor embedding" in captured.err
    assert "Selected action:" in captured.out


def test_generate_prints_text(fake_manager, capsys):
    args = _args(query="what is radar?", strategy="prompt", verbose=True)

    assert cli.cmd_generate(args) is None
    captured = capsys.readouterr()
    assert "generated with prompt for what is radar?" in captured.out
    assert '"strategy": "prompt"' in captured.err


def test_chat_single_query_fast_mode_prints_recall(fake_manager, capsys):
    args = _args(query="what is radar?", mode="fast")

    assert cli.cmd_chat(args) is None
    out = capsys.readouterr().out
    assert out.strip() == "Radar uses radio waves. Lidar uses lasers."
    assert fake_manager["manager"].droid.queries == ["what is radar?"]


def test_chat_slash_fast_override(fake_manager, capsys):
    args = _args(query="/fast what is radar?", mode="auto")
    cli.cmd_chat(args)

    out = capsys.readouterr().out
    assert "Radar uses radio waves." in out
    assert fake_manager["manager"].droid.queries == ["what is radar?"]


def test_chat_auto_mode_reports_path_and_uses_slow_generation(monkeypatch, fake_manager, capsys):
    monkeypatch.setattr(
        FakeDroid,
        "decide_path",
        lambda self, query: {"action": "slow_synthesis", "confidence": 0.42},
    )

    cli.cmd_chat(_args(query="explain radar", mode="auto"))

    out = capsys.readouterr().out
    assert "[slow path, conf=0.42]" in out
    assert "generated with auto for explain radar" in out


def test_chat_recall_miss_falls_back_to_chat(monkeypatch, fake_manager, capsys):
    monkeypatch.setattr(FakeDroid, "recall", lambda self, query, top_k=4: [])

    cli.cmd_chat(_args(query="unknown thing", mode="fast"))
    assert "chat answer for unknown thing" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# serve / download-model
# --------------------------------------------------------------------------- #
def test_serve_without_transport_returns_one(capsys):
    assert cli.cmd_serve(_args(mcp=False, rest=False, port=8000)) == 1
    assert "Must specify --mcp or --rest" in capsys.readouterr().err


def test_download_model_success(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.model.gguf_backend.GgufBackend.download_model",
        lambda self, repo_id, filename, local_dir: f"{local_dir}/{filename}",
    )
    args = _args(repo_id="org/model", filename="m.gguf", local_dir="models")

    assert cli.cmd_download_model(args) == 0
    assert "Downloaded model to models/m.gguf" in capsys.readouterr().out


def test_download_model_failure_returns_one(monkeypatch, capsys):
    def boom(self, repo_id, filename, local_dir):
        raise RuntimeError("hub down")

    monkeypatch.setattr("src.model.gguf_backend.GgufBackend.download_model", boom)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    args = _args(repo_id="org/model", filename="m.gguf", local_dir="models")

    assert cli.cmd_download_model(args) == 1
    err = capsys.readouterr().err
    assert "Download failed: hub down" in err
    assert "Fallback also failed" in err


# --------------------------------------------------------------------------- #
# eval (implemented by product-surface-completion; the metric contract is
# asserted there, so these cover routing, validation and the output shape)
# --------------------------------------------------------------------------- #
def _profile_dir(base, name="droid-alpha"):
    profile = base / name
    profile.mkdir(parents=True)
    (profile / "config.json").write_text(json.dumps({"name": name, "dim": 384}))
    return profile


def test_eval_requires_a_profile_name(capsys):
    assert cli.cmd_eval(_args(name="   ")) == 2
    assert "error: --name is required" in capsys.readouterr().err


def test_eval_missing_profile_exits_two(tmp_path, capsys):
    code = cli.cmd_eval(_args(name="ghost", base_dir=str(tmp_path)))
    assert code == 2
    assert "error: droid profile not found" in capsys.readouterr().err


def test_eval_rejects_invalid_top_k_and_sample(tmp_path, capsys):
    _profile_dir(tmp_path)
    base = str(tmp_path)

    assert cli.cmd_eval(_args(base_dir=base, top_k=0)) == 2
    assert "--top-k must be >= 1" in capsys.readouterr().err

    assert cli.cmd_eval(_args(base_dir=base, sample=0)) == 2
    assert "--sample must be >= 1" in capsys.readouterr().err


def test_eval_self_recall_reports_key_value_metrics(fake_manager, tmp_path, capsys):
    _profile_dir(tmp_path)
    args = _args(base_dir=str(tmp_path), top_k=2, sample=2)

    assert cli.cmd_eval(args) == 0

    out = dict(line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert out["mode"] == "self_recall"
    assert out["profile"] == "droid-alpha"
    assert out["facts"] == "2"
    assert out["samples"] == "2"
    assert out["top_k"] == "2"
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        assert 0.0 <= float(out[metric]) <= 1.0
