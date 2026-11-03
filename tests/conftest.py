"""Shared fixtures.

The real MiniLM anchor fails fast and mock embeddings are opt-in, so the suite
selects the mock explicitly for the whole session through this fixture instead
of relying on a silent fallback. Tests marked ``real_model`` opt back out and
report blocked when their weights are absent.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def explicit_mock_anchor_session():
    """Opt the suite into the labelled mock anchor before any module fixture."""
    previous = os.environ.get("RECAMEM_ANCHOR")
    os.environ["RECAMEM_ANCHOR"] = "mock"
    yield "mock"
    if previous is None:
        os.environ.pop("RECAMEM_ANCHOR", None)
    else:
        os.environ["RECAMEM_ANCHOR"] = previous


@pytest.fixture(autouse=True)
def real_model_uses_real_anchor(monkeypatch, request):
    """Real-model tests must not inherit the session's mock selection."""
    if request.node.get_closest_marker("real_model"):
        monkeypatch.delenv("RECAMEM_ANCHOR", raising=False)
    yield


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory):
    """A deterministic, self-contained uint16 corpus for training smoke tests.

    The real TinyStories memmaps are intentionally not committed; a training
    smoke test that depends on them fails on a clean checkout. This fixture is
    labelled ``dataset_source=fixture`` and must never be used to report a
    language-learning result.
    """
    import json

    import numpy as np

    directory = tmp_path_factory.mktemp("tiny_corpus")
    rng = np.random.default_rng(1337)
    train = rng.integers(0, 1024, size=8192, dtype=np.uint16)
    val = rng.integers(0, 1024, size=2048, dtype=np.uint16)
    train.tofile(directory / "train.bin")
    val.tofile(directory / "val.bin")
    (directory / "meta.json").write_text(json.dumps({
        "vocab_size": 1024,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "dataset_source": "fixture",
    }), encoding="utf-8")
    return directory
