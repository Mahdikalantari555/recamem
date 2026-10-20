"""Validate completion evidence without confusing inventory coverage with release readiness.

Default: validate the ledger schema and inventory coverage (unresolved work is allowed).
--require-complete: reject unresolved entries and missing acceptance-tier evidence too.
This checks evidence structure/references, not the scientific validity of a measurement.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CHANGE = Path("openspec/changes/post-merge-completion")
STATUSES = {"verified", "missing", "partial", "blocked", "superseded"}
TIERS = {"offline", "real-model", "grounding", "concurrency-migration", "benchmark"}


def inventory_items(inventory):
    result = {}
    for group in ("archived_tasks", "current_requirements", "dependency_tasks"):
        for item in inventory[group]:
            label = item.get("task_id", item.get("requirement"))
            key = item["source"] + "#" + label
            if key in result:
                raise ValueError(f"duplicate inventory id: {key}")
            result[key] = item
    return result


def local_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path


def test_exists(root, nodeid):
    if not isinstance(nodeid, str) or "::" not in nodeid:
        return False
    parts = nodeid.split("::")
    path = local_file(root, parts[0])
    if path is None:
        return False
    try:
        body = ast.parse(path.read_text()).body
        for part in parts[1:]:
            name = part.split("[", 1)[0]
            match = next((n for n in body if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and n.name == name), None)
            if match is None:
                return False
            body = match.body
        return isinstance(match, (ast.FunctionDef, ast.AsyncFunctionDef))
    except (SyntaxError, UnicodeError):
        return False


def validate(root, inventory, ledger, require_complete=False):
    errors = []
    expected = inventory_items(inventory)
    if ledger.get("schema_version") != 1:
        errors.append("unsupported ledger schema_version")
    if ledger.get("baseline_commit") != inventory.get("baseline_commit"):
        errors.append("baseline_commit does not match inventory")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return errors + ["entries must be a list"]
    found = set()
    covered_tiers = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("entry must be an object")
            continue
        key = entry.get("id")
        if not isinstance(key, str) or key not in expected or key in found:
            errors.append(f"unknown or duplicate entry: {key!r}")
            continue
        found.add(key)
        def fail(message):
            errors.append(f"{key}: {message}")
        status = entry.get("status")
        if status not in STATUSES:
            fail("invalid status")
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            fail("reason is required")
        owner = entry.get("owner", "")
        if not isinstance(owner, str) or "#" not in owner:
            fail("owner must reference tasks.md#task-id")
        else:
            source, task = owner.rsplit("#", 1)
            path = local_file(root, source)
            if path is None or not re.search(
                r"^- \[[ xX]\] " + re.escape(task) + r"\s", path.read_text(), re.M
            ):
                fail("owner task does not exist")
        for field in ("implementation", "tests", "evidence"):
            values = entry.get(field)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                fail(f"{field} must be a list of strings")
        if any(not isinstance(entry.get(f), list) or any(not isinstance(v, str) for v in entry[f])
               for f in ("implementation", "tests", "evidence")):
            continue
        for value in entry["implementation"]:
            if local_file(root, value) is None:
                fail(f"missing/unsafe implementation path: {value}")
        for value in entry["tests"]:
            if not test_exists(root, value):
                fail(f"test node does not exist: {value}")
        if status == "verified" and not all(entry[f] for f in ("implementation", "tests", "evidence")):
            fail("verified status requires implementation, tests and evidence")
        if status == "superseded":
            replacement = entry.get("replacement", "")
            if not isinstance(replacement, str) or "#" not in replacement:
                fail("superseded entry requires a replacement requirement")
            else:
                source, requirement = replacement.rsplit("#", 1)
                path = local_file(root, source)
                if path is None or f"### Requirement: {requirement}" not in path.read_text().splitlines():
                    fail("replacement requirement does not exist")
            if not entry["evidence"]:
                fail("superseded entry requires approval evidence")
        tiers = set()
        for value in entry["evidence"]:
            path = local_file(root, value)
            if path is None:
                fail(f"missing/unsafe evidence: {value}")
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                fail(f"evidence must be a JSON record: {value}")
                continue
            if not isinstance(record, dict):
                fail(f"evidence must be an object: {value}")
                continue
            valid = True
            for field in ("commit", "command", "environment", "configuration", "outcome", "tier"):
                if not record.get(field):
                    fail(f"evidence missing {field}: {value}")
                    valid = False
            if not re.fullmatch(r"[0-9a-f]{40}", str(record.get("commit", ""))):
                fail(f"evidence must name a full commit hash: {value}")
                valid = False
            if record.get("tier") not in TIERS:
                fail(f"unknown evidence tier: {value}")
                valid = False
            if record.get("outcome") not in {"passed", "failed", "blocked", "skipped", "approved"}:
                fail(f"unknown evidence outcome: {value}")
                valid = False
            if record.get("outcome") in {"blocked", "skipped"} and not record.get("skip_reason"):
                fail(f"skipped/blocked evidence requires skip_reason: {value}")
                valid = False
            acceptable = "approved" if status == "superseded" else "passed"
            if status in {"verified", "superseded"} and record.get("outcome") != acceptable:
                fail(f"{status} claim has non-{acceptable} evidence: {value}")
                valid = False
            if valid and status == "verified" and record.get("outcome") == "passed":
                tiers.add(record["tier"])
        required = entry.get("required_tiers")
        if (not isinstance(required, list) or not required or
                any(not isinstance(t, str) or t not in TIERS for t in required)):
            fail("required_tiers must be a nonempty list of known tiers")
        elif status == "verified" and not set(required).issubset(tiers):
            fail("verified claim lacks evidence for its required tiers")
        covered_tiers.update(tiers)
        if require_complete and status not in {"verified", "superseded"}:
            fail(f"release incomplete ({status})")
    for key in sorted(expected.keys() - found):
        errors.append(f"missing inventory entry: {key}")
    if require_complete:
        for tier in sorted(TIERS - covered_tiers):
            errors.append(f"release acceptance tier has no passing evidence: {tier}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "reports/completion-evidence.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / CHANGE / "inventory.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        inventory = json.loads(args.inventory.read_text())
        ledger = json.loads(args.ledger.read_text())
        errors = validate(ROOT, inventory, ledger, args.require_complete)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"error: invalid evidence input: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    mode = "release complete" if args.require_complete else "schema/coverage valid; NOT a release-completion claim"
    print(f"{len(ledger['entries'])} entries: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
