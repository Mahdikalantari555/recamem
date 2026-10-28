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
