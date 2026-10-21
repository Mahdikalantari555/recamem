"""Contract tests for the contrastive grammar probes (``src/model/probes.py``).

The tests use a tiny fixed-preference model so the expected answer is known
without training, plus the shipped probe file to pin coverage, determinism and
the frozen-scoring guarantee.
"""

import json
import math

import torch

from src.model import probes
from src.model.probes import (
    PHENOMENA,
    Probe,
    ProbeSet,
    load_probes,
    parameter_hash,
    probe_file_hash,
    score_continuation,
    score_probe,
    score_probes,
)
from src.model.rtu import Model


class FixedPreferenceModel(torch.nn.Module):
    """Always prefers one byte, so a probe's expected outcome is known."""

    def __init__(self, preferred_byte: int, vocab: int = 256):
        super().__init__()
        self.preferred_byte = preferred_byte
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer("state", torch.zeros(4))
        self.reset_calls = 0
        self.frozen_calls = 0

    def reset(self):
        self.reset_calls += 1

    def step(self, token, frozen=False):
        assert frozen, "probe scoring must run on the frozen path"
        self.frozen_calls += 1
        logits = torch.full((256,), -20.0)
        logits[self.preferred_byte] = 20.0
        return (None, None, None), (logits, torch.zeros(1))


def _line(**overrides):
    record = {
        "id": "p-1",
        "phenomenon": "subject_verb_agreement",
        "context": "The dog ",
        "correct": "Aaa barks.",
        "incorrect": ["Bbb bark."],
    }
    record.update(overrides)
    return json.dumps(record) + "\n"


def _write(tmp_path, lines, name="probes.jsonl"):
    path = tmp_path / name
    path.write_text("".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Format and loader
# --------------------------------------------------------------------------- #
def test_valid_file_loads_and_exposes_phenomenon_and_continuations(tmp_path):
    path = _write(tmp_path, [_line()])
    loaded = load_probes(path)
    assert loaded.errors == []
    assert len(loaded) == 1
    probe = loaded.probes[0]
    assert probe.phenomenon == "subject_verb_agreement"
    assert probe.correct == "Aaa barks."
    assert probe.incorrect == ("Bbb bark.",)
    assert probe.continuations() == ("Aaa barks.", "Bbb bark.")


def test_broken_line_is_rejected_with_its_line_number_and_the_rest_loads(tmp_path):
    path = _write(
        tmp_path,
        [
            _line(id="p-1"),
            "{not json}\n",
            _line(id="p-3", context="", ),
            _line(id="p-4"),
        ],
    )
    loaded = load_probes(path)
    assert [probe.id for probe in loaded.probes] == ["p-1", "p-4"]
    assert [error["line"] for error in loaded.errors] == [2, 3]
    assert "invalid JSON" in loaded.errors[0]["reason"]
    assert "context" in loaded.errors[1]["reason"]


def test_documented_phenomena_are_accepted_and_an_unknown_one_is_rejected(tmp_path):
    lines = [
        _line(id=f"p-{index}", phenomenon=name)
        for index, name in enumerate(PHENOMENA)
    ]
    loaded = load_probes(_write(tmp_path, lines))
    assert {probe.phenomenon for probe in loaded.probes} == set(PHENOMENA)
    assert loaded.errors == []

    bad = _write(tmp_path, [_line(phenomenon="subjunctive_mood")], name="bad.jsonl")
    rejected = load_probes(bad)
    assert rejected.probes == []
    assert any("subjunctive_mood" in error["reason"] for error in rejected.errors)


def test_incorrect_must_be_a_non_empty_list(tmp_path):
    path = _write(
        tmp_path,
        [
            _line(id="p-1", incorrect="Bbb bark."),
            _line(id="p-2", incorrect=[]),
            _line(id="p-3", correct="Bbb bark.", incorrect=["Bbb bark."]),
        ],
    )
    loaded = load_probes(path)
    assert loaded.probes == []
    assert [error["line"] for error in loaded.errors] == [1, 2, 3]
    assert "list" in loaded.errors[0]["reason"]
    assert "non-empty" in loaded.errors[1]["reason"]
    assert "also appears" in loaded.errors[2]["reason"]


def test_shipped_probe_file_covers_every_phenomenon_with_eight_probes():
    loaded = load_probes(probes.PROBE_FILE)
    assert loaded.errors == []
    assert len(loaded) >= 32
    grouped = loaded.by_phenomenon()
    for name in PHENOMENA:
        assert len(grouped[name]) >= 8, f"{name} has only {len(grouped[name])} probes"
    assert all(probe.incorrect for probe in loaded)
    assert probe_file_hash(probes.PROBE_FILE) == probe_file_hash(probes.PROBE_FILE)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_a_model_with_a_known_preference_scores_the_probe_it_prefers():
    probe = Probe(
        id="p-1",
        phenomenon="subject_verb_agreement",
        context="The dog ",
        correct="Aaa barks.",
        incorrect=("Bbb bark.",),
    )
    preferring_correct = score_probe(FixedPreferenceModel(ord("A")), probe)
    assert preferring_correct["correct"] is True
    assert preferring_correct["margin"] > 0
    assert preferring_correct["correct_score"] > preferring_correct["best_incorrect_score"]

    reversed_model = score_probe(FixedPreferenceModel(ord("B")), probe)
    assert reversed_model["correct"] is False
    assert reversed_model["margin"] < 0


def test_score_set_reports_count_correct_accuracy_and_margins():
    probe_set = ProbeSet(probes=[
        Probe("a", "subject_verb_agreement", "ctx ", "Aaa.", ("Bbb.",)),
        Probe("b", "active_voice", "ctx ", "Bbb.", ("Aaa.",)),
    ])
    report = score_probes(FixedPreferenceModel(ord("A")), probe_set)
    assert report["status"] == "passed"
    assert report["probes"] == 2
    assert report["correct"] == 1
    assert math.isclose(report["accuracy"], 0.5)
    assert len(report["margins"]) == 2
    assert all("margin" in entry for entry in report["margins"])


def test_per_phenomenon_zero_is_visible_beside_the_aggregate():
    probe_set = ProbeSet(probes=[
        Probe("a", "subject_verb_agreement", "ctx ", "Aaa.", ("Bbb.",)),
        Probe("b", "active_voice", "ctx ", "Bbb.", ("Aaa.",)),
    ])
    report = score_probes(FixedPreferenceModel(ord("A")), probe_set)
    assert report["per_phenomenon"]["active_voice"]["accuracy"] == 0.0
    assert report["per_phenomenon"]["subject_verb_agreement"]["accuracy"] == 1.0
    assert report["accuracy"] == 0.5


def test_scoring_is_frozen_and_hashes_parameters_and_buffers():
    model = Model(dim=8, layers=1, spread=4, vocab_size=256)
    model.reset()
    model.blocks[0].states.fill_(0.5)
    before_model_hash = parameter_hash(model)
    probe_set = ProbeSet(probes=[
        Probe("a", "subject_verb_agreement", "The dog ", "barks.", ("bark.",)),
    ])
    report = score_probes(model, probe_set)
    assert report["parameter_hash_before"] == report["parameter_hash_after"]
    assert report["frozen"] is True
    assert parameter_hash(model) == before_model_hash
    # A frozen score never takes an optimizer step.
    assert model.step_count == 0


def test_a_score_is_tied_to_its_checkpoint_and_probe_file(tmp_path):
    model = Model(dim=8, layers=1, spread=4, vocab_size=256)
    checkpoint = tmp_path / "rtu.safetensors"
    model.save(str(checkpoint))
    probe_path = _write(tmp_path, [_line()])
    probe_set = load_probes(probe_path)

    first = score_probes(model, probe_set, checkpoint_path=checkpoint, probe_path=probe_path)
    second = score_probes(model, probe_set, checkpoint_path=checkpoint, probe_path=probe_path)
    assert first["checkpoint_hash"] == second["checkpoint_hash"]
    assert first["probe_file_hash"] == second["probe_file_hash"]
    assert first["accuracy"] == second["accuracy"]
    assert first["per_phenomenon"] == second["per_phenomenon"]


def test_empty_probe_set_is_skipped_without_an_accuracy():
    report = score_probes(FixedPreferenceModel(ord("A")), ProbeSet())
    assert report["status"] == "skipped"
    assert "accuracy" not in report
    assert "value" not in report


def test_continuation_scoring_rewards_the_preferred_byte():
    model = FixedPreferenceModel(ord("A"))
    good = score_continuation(model, "prefix ", "Aaaaa")
    bad = score_continuation(model, "prefix ", "Bbbbb")
    assert good > bad
