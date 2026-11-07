"""No shipped page may invent data.

`verification/surface-coverage` requires every navigation entry to expose real
engine data or be removed, and a surface test must fail if a page renders
fabricated metrics. The frontend has no JavaScript test runner, so this guard
reads the sources: every entry is enabled, every page calls the versioned API
client, and none of the former "not available yet" placeholders survive.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
APP = WEB / "App.tsx"

# Page component path -> the API method it must call to get its data.
PAGE_ENDPOINTS = {
    "features/memory/MemoryPage.tsx": ("recall", "teach"),
    "features/benchmarks/BenchmarksPage.tsx": ("bench", "experiments"),
    "features/training/TrainingPage.tsx": ("jobs", "cancelJob"),
    "features/architecture/ArchitecturePage.tsx": ("health", "models"),
    "features/settings/SettingsPage.tsx": ("health",),
}


def _nav_entries() -> list[dict[str, str]]:
    text = APP.read_text(encoding="utf-8")
    entries = []
    for match in re.finditer(
        r'\{ id: "(?P<id>[^"]+)", label: "(?P<label>[^"]+)", '
        r'hint: "(?P<hint>[^"]+)", available: (?P<available>true|false) \}',
        text,
    ):
        entries.append(match.groupdict())
    return entries


def test_navigation_entries_are_all_enabled():
    entries = _nav_entries()

    assert entries, "the navigation list must be parseable"
    disabled = [entry["id"] for entry in entries if entry["available"] == "false"]
    assert not disabled, (
        "these entries are still labelled unavailable; implement them against a real "
        f"endpoint or remove them: {disabled}"
    )


@pytest.mark.parametrize("page,methods", sorted(PAGE_ENDPOINTS.items()))
def test_each_page_reads_real_backend_data(page, methods):
    source = (WEB / page).read_text(encoding="utf-8")

    assert 'from "../../lib/api/client"' in source, "a page must use the API client"
    for method in methods:
        assert f"api.{method}(" in source, f"{page} must call api.{method}()"


@pytest.mark.parametrize("page", sorted(PAGE_ENDPOINTS))
def test_no_placeholder_copy_survives(page):
    source = (WEB / page).read_text(encoding="utf-8").lower()

    for phrase in (
        "not available yet",
        "backend wiring pending",
        "not yet wired",
        "status: unavailable.",
        "follows once the engine exposes",
    ):
        assert phrase not in source, f"{page} still ships placeholder copy: {phrase!r}"


def test_no_page_hardcodes_a_measurement():
    """Displayed numbers must come from state, not from a literal in the JSX."""
    metric = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|x|%)\b")
    for page in PAGE_ENDPOINTS:
        source = (WEB / page).read_text(encoding="utf-8")
        offenders = [line.strip() for line in source.splitlines()
                     if metric.search(line) and "Math.round" not in line
                     and "toFixed" not in line]
        assert not offenders, f"{page} hardcodes a measurement: {offenders[:3]}"
