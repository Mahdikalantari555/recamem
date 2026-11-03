"""Repository hygiene regressions for the completion change (stdlib only).

Each test fails on the defect it pins: an unparseable notebook, stale document
links, stray heredoc text, machine-specific interpreter paths, masked CI
verification commands and a packaging list that drifts from the importable tree.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACHINE_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+/(?:miniforge|mambaforge|anaconda|venv|\.local)/")
VERIFICATION_COMMANDS = ("pytest", "scripts/verify_", "offline_gate.py", "check_release_evidence.py",
                         "check_completion_gaps.py", "validate_openspec.py")


def _markdown_files():
    yield from sorted(ROOT.glob("*.md"))
    yield from sorted((ROOT / "docs").glob("*.md"))
    yield from sorted((ROOT / "plans").glob("*.md"))
    for change in sorted((ROOT / "openspec" / "changes").iterdir()):
        if change.is_dir() and change.name != "archive":
            yield from sorted(change.glob("*.md"))


def test_notebook_is_valid_nbformat_json():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc.get("nbformat") == 4, f"{path}: nbformat must be 4"
        assert isinstance(doc.get("cells"), list) and doc["cells"], f"{path}: no cells"


def test_colab_notebook_covers_install_configure_train_and_push():
    path = ROOT / "notebooks" / "recamem_colab.ipynb"
    doc = json.loads(path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in doc["cells"])
    assert "pip install" in sources
    assert "HF_TOKEN" in sources
    assert "colab_train.py" in sources
    assert "reports" not in sources or "report.json" in sources
    assert "recamem push" in sources
    assert "$REPO" not in sources, "Colab shell cannot read a Python variable; use {REPO}"


def test_markdown_has_no_stray_heredoc_markers():
    for path in _markdown_files():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines, f"{path}: empty markdown"
        assert lines[-1] != "EOF", f"{path}: trailing heredoc marker"
        assert not lines[-1].startswith("echo "), f"{path}: trailing shell command"


def test_plan_readme_links_resolve():
    text = (ROOT / "plans" / "README.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"\b(\d{2}_[a-z0-9_]+\.md)\b", text))
    assert refs, "plans/README.md should name its plan files"
    for ref in sorted(refs):
        assert (ROOT / "plans" / ref).is_file(), f"plans/README.md references missing {ref}"


def test_agent_facing_docs_have_no_machine_specific_interpreter_paths():
    scanned = list(_markdown_files())
    assert scanned
    for path in scanned:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            assert not MACHINE_PATH.search(line), f"{path}:{lineno}: machine-specific path: {line.strip()}"


def test_ci_does_not_mask_verification_commands():
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "|| true" in line and any(cmd in line for cmd in VERIFICATION_COMMANDS):
                raise AssertionError(f"{path}:{lineno}: masked verification command")
        # A non-gating step is allowed only when it is explicitly labelled informational.
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "continue-on-error: true" in line:
                window = "\n".join(lines[max(0, index - 12):index]).lower()
                assert "informational" in window, f"{path}:{index + 1}: unlabelled non-gating step"


def test_pyproject_packages_cover_importable_packages():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^packages\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert block, "pyproject.toml must declare packages"
    declared = re.findall(r'"([^"]+)"', block.group(1))
    assert len(declared) == len(set(declared)), f"duplicate package entries: {declared}"
    for base in ("recamem", "src"):
        for init in sorted((ROOT / base).rglob("__init__.py")):
            package = ".".join(init.parent.relative_to(ROOT).parts)
            assert package in declared, f"missing package declaration: {package}"


def test_offline_gate_checks_completion_inventory():
    text = (ROOT / "scripts" / "offline_gate.py").read_text(encoding="utf-8")
    assert "check_completion_inventory" in text
    assert "completion_inventory" in text


def test_testing_doc_does_not_claim_psutil_is_optional():
    text = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    assert "or `psutil`." not in text
    assert "psutil" in text and "declared core dependency" in text


def test_vela_report_has_honest_provenance():
    doc = json.loads((ROOT / "reports" / "vela-poc-final.json").read_text(encoding="utf-8"))
    assert doc["artifact_sha256"] is None
    assert "fabricated" in doc["artifact_sha256_note"]
    assert "origin_host" in doc and "commit_note" in doc
    results = (ROOT / "docs" / "vela-poc-results.md").read_text(encoding="utf-8")
    assert "unavailable" in results.lower()


def test_plans_readme_states_phase_a_decision_is_open():
    text = (ROOT / "plans" / "README.md").read_text(encoding="utf-8")
    assert "not** resolved" in text or "not resolved" in text.lower()
    assert "completion-gaps.json" in text


def test_real_model_runbook_exists_and_is_honest():
    path = ROOT / "docs" / "real-model-runbook.md"
    text = path.read_text(encoding="utf-8")
    assert "sha256" in text and "HF_TOKEN" in text
    assert "blocked" in text


def test_release_matrix_marks_open_targets():
    text = (ROOT / "docs" / "release-matrix.md").read_text(encoding="utf-8")
    assert "## Open targets (2026-11-03)" in text
    assert "completion-gaps.json" in text
