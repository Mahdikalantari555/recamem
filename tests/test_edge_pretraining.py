"""Corpus preparation, resumability, reporting and acceptance for a scale run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.model import benchmarks, edge_pretraining
from src.model.edge_pretraining import (
    CorpusContaminationError,
    RunState,
    build_model,
    evaluate_acceptance,
    parameter_count,
    prepare_corpus,
    read_checkpoint,
    run_scale,
    write_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG = {"dim": 16, "layers": 2, "spread": 4, "vocab_size": 256}


def _write_corpus(directory: Path, names):
    """Write one document per name, with content derived from the name alone.

    The content never depends on the position in ``names``, so preparing the
    same set of files in a different creation order is the same corpus. Each
    document gets its own pseudo-random byte stream so no 64-byte window is
    shared between documents (which the harness would rightly call
    contamination).
    """
    import random

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,\n"
    for name in names:
        path = directory / name
        rng = random.Random(name)
        body = f"document {name}\n" + "".join(rng.choice(alphabet) for _ in range(900))
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def _corpus(tmp_path, names=("a.md", "b.md", "c.md", "d.md", "e.md")):
    return _write_corpus(tmp_path / "corpus", names)


# --------------------------------------------------------------------------- #
# 5.1 Deterministic preparation
# --------------------------------------------------------------------------- #
def test_preparation_is_byte_identical_and_independent_of_filesystem_order(tmp_path):
    first = _corpus(tmp_path / "one")
    second = _corpus(tmp_path / "two", names=("e.md", "c.md", "a.md", "d.md", "b.md"))

    prepared_a = prepare_corpus(first, seed=7)
    prepared_b = prepare_corpus(second, seed=7)
    prepared_a_again = prepare_corpus(first, seed=7)

    assert prepared_a["train_bytes"] == prepared_a_again["train_bytes"]
    assert prepared_a["train_bytes"] == prepared_b["train_bytes"]
    assert prepared_a["train_ids"] == prepared_b["train_ids"]
    assert prepared_a["order"] == prepared_b["order"]
    assert prepared_a["corpus_id"] == prepared_b["corpus_id"]
    assert prepared_a["train_sha256"] == prepared_b["train_sha256"]


def test_document_order_follows_the_seed_and_not_the_identifiers(tmp_path):
    files = _corpus(tmp_path)
    first = prepare_corpus(files, seed=0)
    second = prepare_corpus(files, seed=1)
    assert first["order"] != second["order"]
    assert sorted(first["order"]) == sorted(second["order"])


# --------------------------------------------------------------------------- #
# 5.2 Document-level split
# --------------------------------------------------------------------------- #
def test_split_is_document_level_and_reproducible_from_the_seed(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(12)))
    corpus = prepare_corpus(files, seed=3, held_out_fraction=0.25)

    assert corpus["held_out_ids"], "the split must hold documents out"
    assert not set(corpus["train_ids"]) & set(corpus["held_out_ids"])
    assert set(corpus["train_ids"]) | set(corpus["held_out_ids"]) <= set(corpus["order"])

    again = prepare_corpus(files, seed=3, held_out_fraction=0.25)
    assert again["train_ids"] == corpus["train_ids"]
    assert again["held_out_ids"] == corpus["held_out_ids"]

    other_seed = prepare_corpus(files, seed=4, held_out_fraction=0.25)
    assert set(other_seed["held_out_ids"]) != set(corpus["held_out_ids"])


# --------------------------------------------------------------------------- #
# 5.3 Exclusions and contamination
# --------------------------------------------------------------------------- #
def test_held_out_probe_and_reserved_documents_are_excluded(tmp_path):
    files = _corpus(tmp_path)
    probe = tmp_path / "probe.jsonl"
    probe.write_text('{"id": "probe-1"}\n', encoding="utf-8")
    reserved = tmp_path / "reserved"
    reserved.mkdir()
    (reserved / "bench.md").write_text("reserved benchmark corpus\n", encoding="utf-8")

    corpus = prepare_corpus(files, seed=5, probe_files=[probe], reserved_sources=[reserved])

    assert "probe.jsonl" in corpus["excluded_sets"]["grammar_probe_file"]
    assert "bench.md" in corpus["excluded_sets"]["reserved_benchmark_corpus"]
    for doc_id in corpus["held_out_ids"]:
        assert doc_id not in corpus["train_ids"]
    assert probe.read_bytes() not in corpus["train_bytes"]
    assert b"reserved benchmark corpus" not in corpus["train_bytes"]
    assert corpus["contaminated"] is False
    ledger = edge_pretraining.excluded_report(corpus)
    assert ledger["excluded_from_training"] is True
    assert ledger["contamination_check"] == "passed"


def test_contamination_check_fails_and_names_the_overlapping_document(tmp_path):
    files = _corpus(tmp_path)
    corpus = prepare_corpus(files, seed=5, held_out_fraction=0.4)
    leaked_id = corpus["held_out_ids"][0]

    offenders = edge_pretraining.check_contamination(
        corpus["train_ids"] + [leaked_id],
        corpus["train_bytes"],
        corpus["excluded_documents"],
    )
    assert any(item["document_id"] == leaked_id for item in offenders)

    # A held-out document whose bytes were copied into a training document is
    # the contamination the check exists for: identical content, different ids.
    copied = tmp_path / "copied"
    copied.mkdir()
    shared = "the same paragraph kept in two places. " * 20
    (copied / "train.md").write_text(shared, encoding="utf-8")
    (copied / "held-out.md").write_text(shared, encoding="utf-8")
    with pytest.raises(CorpusContaminationError) as failure:
        prepare_corpus(
            [copied / "train.md", copied / "held-out.md"],
            seed=0, held_out_fraction=0.5, root=copied,
        )
    offenders = failure.value.offenders
    held_out_id = [item["document_id"] for item in offenders][0]
    assert held_out_id in ("train.md", "held-out.md")
    assert any(item["document_id"] in str(failure.value) for item in offenders)


def test_a_leaked_document_is_reported_by_a_run_instead_of_scored(tmp_path):
    files = _corpus(tmp_path)
    corpus = prepare_corpus(files, seed=2, held_out_fraction=0.5, strict=False)
    corpus["contaminated"] = True
    corpus["contamination"] = [{"document_id": corpus["held_out_ids"][0],
                                "reason": "document bytes are in the training set"}]
    report = edge_pretraining.scale_report(
        command="pytest",
        config={**TINY_CONFIG, "byte_target": 1, "chunked": True, "chunk_size": 4},
        corpus=corpus,
        run={"bytes_consumed": 0, "wall_clock_seconds": 0.0,
             "initial_loss": None, "final_loss": None, "chunks": 0},
        model=build_model(TINY_CONFIG),
        perplexity=benchmarks.skipped("the contamination check failed"),
        outcome_override="failed",
    )
    assert report["outcome"] == "failed"
    assert report["corpus"]["contamination_check"] == "failed"
    assert corpus["held_out_ids"][0] in report["corpus"]["contaminated_documents"]


# --------------------------------------------------------------------------- #
# 6.1 Resume from the recorded position
# --------------------------------------------------------------------------- #
def test_a_resumed_run_does_not_reprocess_bytes_before_the_checkpoint(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(6)))
    corpus = prepare_corpus(files, seed=0, held_out_fraction=0.2)
    checkpoint = tmp_path / "run.safetensors"

    first_model = build_model(TINY_CONFIG)
    state = RunState(seed=0, config=TINY_CONFIG)
    first = edge_pretraining.train_stream(
        first_model, corpus, state,
        chunk_size=8, chunked=True, byte_target=10_000,
        checkpoint_interval=8, checkpoint_path=str(checkpoint),
        max_chunks=5,
    )
    assert first["checkpoints"], "the interval must have produced a checkpoint"
    recorded = read_checkpoint(checkpoint)
    assert recorded.corpus_position > 0
    assert recorded.step_count == recorded.corpus_position
    assert recorded.step_count == first_model.step_count

    second_model = build_model(TINY_CONFIG)
    resumed_state = read_checkpoint(checkpoint, model=second_model)
    resumed = edge_pretraining.train_stream(
        second_model, corpus, resumed_state,
        chunk_size=8, chunked=True, byte_target=10_000,
        max_chunks=2,
    )
    assert resumed["start_position"] == recorded.corpus_position
    assert resumed["positions"], "the resumed run must train"
    assert min(resumed["positions"]) >= recorded.corpus_position
    assert resumed["positions"][0] == recorded.corpus_position


# --------------------------------------------------------------------------- #
# 6.2 Atomic checkpoint contents
# --------------------------------------------------------------------------- #
def test_checkpoint_carries_weights_optimizer_state_and_position(tmp_path):
    import torch

    model = build_model(TINY_CONFIG)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    model.train_chunk(tokens, torch.roll(tokens, -1, dims=1), chunk=4)
    state = RunState(corpus_position=8, step_count=model.step_count, seed=3,
                     config=TINY_CONFIG)
    checkpoint = tmp_path / "ckpt.safetensors"
    written = write_checkpoint(checkpoint, model, state)

    assert Path(written["state"]).is_file()
    loaded = RunState.load(written["state"])
    assert loaded.corpus_position == 8
    assert loaded.step_count == model.step_count
    assert loaded.seed == 3
    assert loaded.config == TINY_CONFIG

    restored = build_model(TINY_CONFIG)
    read_back = read_checkpoint(checkpoint, model=restored)
    assert read_back.corpus_position == 8
    assert restored.step_count == model.step_count
    assert restored.optimizer.state, "the optimizer state must be restored"
    for (name, original), (other, copy) in zip(
        model.named_parameters(), restored.named_parameters()
    ):
        assert name == other
        assert torch.allclose(original.detach(), copy.detach())


def test_an_interrupted_write_leaves_the_previous_checkpoint_readable(tmp_path):
    model = build_model(TINY_CONFIG)
    import torch

    tokens = torch.tensor([[1, 2, 3, 4]])
    model.train_chunk(tokens, torch.roll(tokens, -1, dims=1), chunk=4)
    state = RunState(corpus_position=4, step_count=4, seed=0, config=TINY_CONFIG)
    checkpoint = tmp_path / "ckpt.safetensors"
    write_checkpoint(checkpoint, model, state)
    before = checkpoint.read_bytes()

    class Boom(RuntimeError):
        pass

    def broken_save(_path):
        raise Boom("disk full")

    original_save = model.save
    try:
        model.save = broken_save
        with pytest.raises(Boom):
            write_checkpoint(checkpoint, model, RunState(corpus_position=9, step_count=9))
    finally:
        model.save = original_save

    assert checkpoint.read_bytes() == before
    restored = build_model(TINY_CONFIG)
    read_back = read_checkpoint(checkpoint, model=restored)
    assert read_back.corpus_position == 4
    # The failed write must not have advanced the state file either.
    assert restored.step_count == 4


# --------------------------------------------------------------------------- #
# 7. Acceptance criteria
# --------------------------------------------------------------------------- #
def _acceptance(**overrides):
    values = dict(
        bytes_consumed=100,
        byte_target=100,
        parameter_count_actual=parameter_count(TINY_CONFIG),
        parameter_count_expected=parameter_count(TINY_CONFIG),
        initial_loss=10.0,
        final_loss=5.0,
        perplexity=benchmarks.measurement(2.5, excluded_from_training=True),
        hardware_class=edge_pretraining.HARDWARE_CLASS,
        completed=True,
        contaminated=False,
    )
    values.update(overrides)
    return evaluate_acceptance(**values)


def test_a_passing_run_meets_every_criterion_individually():
    criteria = _acceptance()
    assert all(item["met"] for item in criteria)
    assert {item["criterion"] for item in criteria} == {
        "byte_target", "parameter_count", "loss_decreased",
        "held_out_perplexity", "hardware_class", "completed", "contamination",
    }


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"bytes_consumed": 10}, "byte_target"),
        ({"parameter_count_actual": 1}, "parameter_count"),
        ({"final_loss": 20.0}, "loss_decreased"),
        ({"perplexity": benchmarks.skipped("contaminated")}, "held_out_perplexity"),
        ({"perplexity": benchmarks.measurement(float("inf"), excluded_from_training=True)},
         "held_out_perplexity"),
        ({"perplexity": benchmarks.measurement(2.0)}, "held_out_perplexity"),
        ({"hardware_class": "gpu"}, "hardware_class"),
        ({"completed": False}, "completed"),
        ({"contaminated": True}, "contamination"),
    ],
)
def test_a_missing_criterion_is_named_and_not_reported_as_passed(overrides, expected):
    criteria = _acceptance(**overrides)
    unmet = [item["criterion"] for item in criteria if not item["met"]]
    assert expected in unmet
    assert len(unmet) == 1


def test_a_run_report_that_misses_a_criterion_is_not_passed(tmp_path):
    files = _corpus(tmp_path, names=("a.md", "b.md", "c.md"))
    report = run_scale(
        files,
        command="pytest",
        config=TINY_CONFIG,
        seed=0,
        chunk_size=4,
        chunked=True,
        byte_target=10 ** 9,
        max_chunks=3,
        report_path=str(tmp_path / "report.json"),
    )
    assert report["outcome"] == "failed"
    assert "byte_target" in report["unmet_criteria"]
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["outcome"] == "failed"
    assert "byte_target" in written["unmet_criteria"]


# --------------------------------------------------------------------------- #
# 7.2-7.5 The report
# --------------------------------------------------------------------------- #
def test_a_short_run_writes_the_required_report_fields(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(6)))
    report_path = tmp_path / "report.json"
    report = run_scale(
        files,
        command="pytest short run",
        config=TINY_CONFIG,
        seed=1,
        chunk_size=8,
        chunked=True,
        byte_target=60,
        max_chunks=6,
        checkpoint_interval=16,
        checkpoint_path=str(tmp_path / "run.safetensors"),
        report_path=str(report_path),
        verify_equivalence=True,
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))
    measurements = written["measurements"]
    for name in ("bytes_consumed", "wall_clock_seconds", "final_training_loss",
                 "held_out_perplexity", "parameter_count", "hardware_class"):
        assert name in measurements, name
    assert written["commit"]
    assert written["command"] == "pytest short run"
    assert written["config"]["seed"] == 1
    assert written["hardware"]
    assert written["corpus"]["identity"]
    assert written["corpus"]["split_rule"]
    assert written["corpus"]["excluded_from_training"] is True
    assert written["corpus"]["held_out_split"]
    assert "grammar_probe_file" in written["corpus"]
    assert measurements["bytes_consumed"]["value"] >= 1
    assert measurements["wall_clock_seconds"]["value"] >= 0.0
    assert measurements["parameter_count"]["value"] == parameter_count(TINY_CONFIG)
    assert benchmarks.is_measured(measurements["final_training_loss"])
    assert benchmarks.is_measured(measurements["initial_training_loss"])
    assert benchmarks.is_measured(measurements["held_out_perplexity"])
    assert report["outcome"] in ("passed", "failed")


def test_the_report_states_the_equivalence_boundary(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(6)))
    report = run_scale(
        files, config=TINY_CONFIG, seed=0, chunk_size=8, chunked=True,
        byte_target=40, max_chunks=3, verify_equivalence=True,
    )
    boundary = report["equivalence"]
    assert boundary["claim"] is True
    assert boundary["equivalent"] == list(
        __import__("src.model.rtu", fromlist=["CHUNK_EQUIVALENT_QUANTITIES"])
        .CHUNK_EQUIVALENT_QUANTITIES
    )
    assert boundary["not_equivalent"] == ["parameter_trajectory"]
    assert "parameter trajectory" in json.dumps(boundary).lower().replace("_", " ")


def test_an_unverified_chunked_run_names_the_missing_verification(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(4)))
    report = run_scale(
        files, config=TINY_CONFIG, seed=0, chunk_size=8, chunked=True,
        byte_target=40, max_chunks=2, verify_equivalence=False,
    )
    boundary = report["equivalence"]
    assert boundary["claim"] is False
    assert boundary["equivalent"] == []
    assert "missing_verification" in boundary
    assert "not claimed" in boundary["note"]
    # And it is not presented as passed unless every criterion holds.
    assert report["outcome"] != "passed" or report["acceptance_met"] is True


def test_a_per_token_run_makes_no_chunked_claim(tmp_path):
    files = _corpus(tmp_path, names=("a.md", "b.md"))
    report = run_scale(
        files, config=TINY_CONFIG, seed=0, chunk_size=8, chunked=False,
        byte_target=8, max_chunks=1,
    )
    assert report["equivalence"]["path"] == "per_token"
    assert report["equivalence"]["claim"] is False


def test_an_interrupted_run_still_reports_blocked_bytes_and_wall_clock(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(6)))
    report_path = tmp_path / "blocked.json"
    report = run_scale(
        files, config=TINY_CONFIG, seed=0, chunk_size=8, chunked=True,
        byte_target=10 ** 6, max_chunks=8, interrupt_after_bytes=16,
        report_path=str(report_path),
    )
    assert report["outcome"] == "blocked"
    written = json.loads(report_path.read_text())
    assert written["outcome"] == "blocked"
    assert written["measurements"]["bytes_consumed"]["value"] >= 16
    assert written["measurements"]["wall_clock_seconds"]["value"] >= 0.0
    assert "completed" in written["unmet_criteria"]


# --------------------------------------------------------------------------- #
# 7.1 The command line
# --------------------------------------------------------------------------- #
def test_the_scale_script_help_accepts_every_documented_flag():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_edge_scale.py"), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--corpus", "--dim", "--layers", "--spread", "--vocab-size",
                 "--chunk-size", "--chunked", "--seed", "--checkpoint-interval",
                 "--report", "--byte-target", "--verify-equivalence", "--resume"):
        assert flag in result.stdout, flag


def test_a_short_scale_run_exits_zero_and_writes_a_report(tmp_path):
    files = _corpus(tmp_path, names=tuple(f"doc{i}.md" for i in range(6)))
    report_path = tmp_path / "cli-report.json"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "train_edge_scale.py"),
            "--corpus", *[str(path) for path in files],
            "--dim", "16", "--layers", "2", "--spread", "4", "--vocab-size", "256",
            "--chunk-size", "8", "--chunked", "--seed", "0",
            "--byte-target", "16", "--max-chunks", "2",
            "--report", str(report_path),
        ],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "outcome=" in result.stdout
    written = json.loads(report_path.read_text())
    assert written["measurements"]["bytes_consumed"]["value"] >= 1
    assert written["measurements"]["parameter_count"]["value"] == parameter_count(TINY_CONFIG)
    assert written["measurements"]["held_out_perplexity"]["status"] in ("passed", "skipped")
