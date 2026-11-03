#!/usr/bin/env python3
"""Validate the completion gap inventory without confusing structure with closure.

Default (``--require-closed``): reject any row that is not closed. This is the
release-facing mode and it is expected to fail while work remains.
``--schema-only``: validate the schema and source coverage and report the open
count, but accept an incomplete inventory. CI uses this mode as a *gating
structure check*, not as a claim that the project is finished.

Like ``scripts/check_release_evidence.py``, this checks references and evidence
shape, not the scientific validity of a measurement. A ``closed`` row needs a
present artifact, a resolvable pytest node id, and a passing evidence record.

Exit codes: 0 valid/complete, 1 invalid or incomplete, 2 unreadable input.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "reports" / "completion-gaps.json"
STATUSES = {"open", "blocked", "closed"}
TIERS = {"offline", "real-model", "grounding", "concurrency-migration", "benchmark"}
REQUIRED_FIELDS = (
    "id", "source", "owner", "tier", "status", "verify_command",
    "artifact", "tests", "evidence", "reason",
)
ACTIVE_CHANGES = (
    "recamem-language-learning",
    "recamem-core-architecture-tournament",
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def expected_sources(root: Path = ROOT) -> List[str]:
    """Source ids that an honest inventory must cover.

    Derived from the release ledger's unresolved rows, the published performance
    misses, the unchecked tasks of the pre-existing active changes, the open
    surface checklist and the post-merge completion tasks. The newer
    ``recamem-svelte-companion`` change is a separate deliverable and is tracked
    by its own tasks file, not by this inventory.
    """
    expected: Set[str] = set()

    ledger = _read_json(root / "reports" / "completion-evidence.json")
    for row in ledger.get("entries", []):
        if row.get("status") in {"missing", "partial", "blocked"}:
            expected.add(str(row["id"]))

    matrix = (root / "docs" / "release-matrix.md").read_text(encoding="utf-8")
    for line in matrix.splitlines():
        if line.startswith("|") and "**miss" in line.lower():
            label = line.split("|")[1].strip()
            if label:
                expected.add(f"docs/release-matrix.md#{_slug(label)}")

    for change in ACTIVE_CHANGES:
        tasks = root / "openspec" / "changes" / change / "tasks.md"
        if not tasks.is_file():
            continue
        for line in tasks.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^- \[ \] (\S+) ", line)
            if match:
                expected.add(f"openspec/changes/{change}/tasks.md#{match.group(1)}")

    surfaces = (root / "docs" / "surfaces.md").read_text(encoding="utf-8")
    for index, line in enumerate(surfaces.splitlines(), start=1):
        if line.startswith("- [ ]"):
            expected.add(f"docs/surfaces.md#line-{index}")

    post_merge = (
        root / "openspec" / "changes" / "archive" / "2026-10-28-post-merge-completion"
        / "tasks.md"
    )
    if post_merge.is_file():
        for line in post_merge.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^- \[ \] (\S+) ", line)
            if match:
                expected.add(
                    "openspec/changes/archive/2026-10-28-post-merge-completion/"
                    f"tasks.md#{match.group(1)}"
                )
    return sorted(expected)


def _test_names(root: Path, node_id: str) -> bool:
    """A pytest node id is resolvable when its file and test name exist."""
    rel, _, name = node_id.partition("::")
    if not name:
        return False
    name = name.split("[")[0].split(".")[0]
    path = root / rel
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    found = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return name in found


def _evidence_passes(root: Path, rel: str) -> bool:
    path = root / rel
    if not path.is_file():
        return False
    try:
        record = _read_json(path)
    except (ValueError, OSError):
        return False
    return record.get("outcome") == "passed"


def validate(doc: Dict[str, Any], root: Path, *, require_closed: bool) -> List[str]:
    errors: List[str] = []
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["reports/completion-gaps.json: rows must be a non-empty array"]

    seen: Set[str] = set()
    covered: Set[str] = set()
    open_rows = 0
    for index, row in enumerate(rows):
        label = f"row {index}"
        if not isinstance(row, dict):
            errors.append(f"{label}: must be an object")
            continue
        missing_fields = [f for f in REQUIRED_FIELDS if f not in row]
        if missing_fields:
            errors.append(f"{label}: missing fields {sorted(missing_fields)}")
            continue
        label = f"row {row.get('id', index)!r}"
        rid = str(row["id"])
        if rid in seen:
            errors.append(f"{label}: duplicate id")
        seen.add(rid)
        if row["status"] not in STATUSES:
            errors.append(f"{label}: invalid status {row['status']!r}")
        if row["tier"] not in TIERS:
            errors.append(f"{label}: invalid tier {row['tier']!r}")
        for key in ("tests", "evidence"):
            if not isinstance(row[key], list) or not all(isinstance(x, str) for x in row[key]):
                errors.append(f"{label}: {key} must be an array of strings")
        if row["status"] != "closed":
            open_rows += 1
        covered.add(str(row["source"]))
        covered.update(str(x) for x in row.get("covers", []))

        if row["status"] == "closed":
            artifact = root / str(row["artifact"])
            if not artifact.is_file():
                errors.append(f"{label}: artifact missing: {row['artifact']}")
            if not row["tests"]:
                errors.append(f"{label}: a closed row needs at least one test")
            for node_id in row["tests"]:
                if not _test_names(root, str(node_id)):
                    errors.append(f"{label}: test not resolvable: {node_id}")
            if not row["evidence"]:
                errors.append(f"{label}: a closed row needs an evidence record")
            for rel in row["evidence"]:
                if not _evidence_passes(root, str(rel)):
                    errors.append(f"{label}: evidence not passing: {rel}")

    missing = sorted(set(expected_sources(root)) - covered)
    if missing:
        errors.append("inventory does not cover: " + ", ".join(missing[:8])
                      + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""))
    if require_closed and open_rows:
        errors.append(f"{open_rows} gap(s) are not closed")
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-closed", action="store_true",
                      help="exit 1 while any row is not closed (default)")
    mode.add_argument("--schema-only", action="store_true",
                      help="validate structure/coverage only; accept open rows")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root: Path = args.root
    path: Path = args.inventory or (root / "reports" / "completion-gaps.json")
    try:
        doc = _read_json(path)
    except (OSError, ValueError) as exc:
        print(f"unreadable inventory {path}: {exc}", file=sys.stderr)
        return 2
    require_closed = not args.schema_only
    errors = validate(doc, root, require_closed=require_closed)
    rows = doc.get("rows", [])
    closed = sum(1 for r in rows if isinstance(r, dict) and r.get("status") == "closed")
    print(f"{len(rows)} gaps: {closed} closed, {len(rows) - closed} open/blocked")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if require_closed:
        print("completion inventory closed; NOT a substitute for the release gate")
    else:
        print("schema/coverage valid; completion is NOT claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
