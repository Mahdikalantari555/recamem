"""Benchmark runners: labelled fixtures, blocked datasets, provenance."""

import json

import pytest

from recamem.bench import runners
from recamem.engine import DroidEngine


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network in the offline tier: every fetch reports blocked."""
    monkeypatch.setattr(
        runners, "fetch_dataset", lambda name, timeout=10.0: {"status": "blocked", "reason": "offline"}
    )


@pytest.mark.parametrize("name", ["locomo", "longmemeval", "dmr"])
def test_fixture_run_is_labelled(name):
    report = runners.run(name)
    assert report["status"] == "not_measured"  # parsed, but nothing was measured
    assert report["dataset_source"] == "fixture"
    assert report["questions"] >= 1
    assert report["metrics"] == runners.BENCHMARKS[name]["metrics"]
    assert "not a LoCoMo" not in json.dumps(report) or report["status"] == "fixture"


def test_fixture_result_is_never_presented_as_a_score():
    report = runners.run("locomo", engine=None)
    lines = runners.describe(report)
    assert "dataset_source=fixture" in lines
    assert not any(line.startswith("recall_at_k=") for line in lines)


def test_blocked_benchmark_prints_no_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(runners, "fixture_path", lambda name: tmp_path / "absent.json")
    report = runners.run("dmr")
    assert report["status"] == "blocked"
    assert report["reason"] == "offline"
    lines = runners.describe(report)
    assert "status=blocked" in lines
    assert "recall_at_k=" not in " ".join(lines)


def test_subset_is_deterministic_for_a_seed():
    questions = [{"question": f"q{i}", "answer": str(i)} for i in range(20)]
    first = runners.deterministic_subset(questions, seed=7, size=5)
    second = runners.deterministic_subset(questions, seed=7, size=5)
    assert first == second
    assert len(first) == 5
    other = runners.deterministic_subset(questions, seed=8, size=5)
    assert [q["question"] for q in other] != [q["question"] for q in first]


def test_provenance_has_every_required_field():
    prov = runners.provenance("recamem bench", seed=0, subset_size=3)
    assert set(prov) == {"commit", "command", "seed", "subset_size", "dependencies", "hardware"}
    assert set(prov["dependencies"]) == {"torch", "numpy", "pypdf"}
    assert "platform" in prov["hardware"]


def test_fixture_run_measures_with_an_engine():
    engine = DroidEngine(name="bench")
    report = runners.run("dmr", engine, seed=0, size=1)
    assert report["status"] in {"fixture", "ok"}
    assert "measured" in report
    assert set(report["measured"]) == set(report["metrics"])


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError):
        runners.run("not-a-benchmark")
