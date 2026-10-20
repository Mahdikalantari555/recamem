"""The evidence gate must reject incomplete inventories and unsupported success claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import check_release_evidence as gate


@pytest.fixture
def example(tmp_path):
    (tmp_path / "tasks.md").write_text("- [ ] 1.1 Complete the behavior\n")
    (tmp_path / "impl.py").write_text("VALUE = 1\n")
    (tmp_path / "test_impl.py").write_text("def test_contract():\n    assert True\n")
    (tmp_path / "spec.md").write_text("### Requirement: Replacement behavior\n")
    record = {
        "commit": "a" * 40, "command": "pytest test_impl.py",
        "environment": {"python": "test fixture"},
        "configuration": {"seed": 0}, "tier": "offline", "outcome": "passed",
    }
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    inventory = {"baseline_commit": "baseline", "archived_tasks": [
        {"source": "tasks.md", "task_id": "1.1"}
    ], "current_requirements": [], "dependency_tasks": []}
    entry = {"id": "tasks.md#1.1", "status": "verified", "owner": "tasks.md#1.1",
             "reason": "Test fixture, not repository certification", "implementation": ["impl.py"],
             "tests": ["test_impl.py::test_contract"], "evidence": ["evidence.json"],
             "required_tiers": ["offline"]}
    ledger = {"schema_version": 1, "baseline_commit": "baseline", "entries": [entry]}
    return tmp_path, inventory, ledger, record


def test_valid_ledger_schema_does_not_imply_all_release_tiers_pass(example):
    root, inventory, ledger, _ = example
    assert gate.validate(root, inventory, ledger) == []
    errors = gate.validate(root, inventory, ledger, require_complete=True)
    assert any("real-model" in e for e in errors)
    assert any("grounding" in e for e in errors)


@pytest.mark.parametrize("field,value,expected", [
    ("status", "complete", "invalid status"),
    ("implementation", [], "requires implementation"),
    ("implementation", ["absent.py"], "implementation path"),
    ("implementation", ["../escape.py"], "implementation path"),
    ("tests", ["test_impl.py::test_does_not_exist"], "test node"),
    ("tests", ["test_impl.py"], "test node"),
    ("tests", "bad", "list of strings"),
    ("evidence", ["missing.json"], "missing/unsafe evidence"),
    ("owner", "tasks.md#99.99", "owner task"),
    ("owner", "tasks.md", "owner must"),
    ("reason", "", "reason is required"),
    ("required_tiers", ["real-model"], "required tiers"),
    ("required_tiers", [], "nonempty list"),
    ("required_tiers", ["imaginary"], "known tiers"),
])
def test_invalid_entry_rejected(example, field, value, expected):
    root, inventory, ledger, _ = example
    ledger["entries"][0][field] = value
    assert any(expected in e for e in gate.validate(root, inventory, ledger))


def test_missing_duplicate_and_unknown_inventory_ids_rejected(example):
    root, inventory, ledger, _ = example
    missing = deepcopy(ledger)
    missing["entries"] = []
    assert any("missing inventory entry" in e for e in gate.validate(root, inventory, missing))
    duplicate = deepcopy(ledger)
    duplicate["entries"] *= 2
    assert any("duplicate" in e for e in gate.validate(root, inventory, duplicate))
    ledger["entries"][0]["id"] = "unknown"
    assert any("unknown" in e for e in gate.validate(root, inventory, ledger))


@pytest.mark.parametrize("outcome", ["skipped", "blocked", "failed"])
def test_nonpassing_result_cannot_certify_requirement(example, outcome):
    root, inventory, ledger, record = example
    record.update(outcome=outcome, skip_reason="artifact unavailable")
    (root / "evidence.json").write_text(json.dumps(record))
    assert any("non-passed" in e for e in gate.validate(root, inventory, ledger))


@pytest.mark.parametrize("field", ["commit", "command", "environment", "configuration", "tier"])
def test_evidence_requires_reproduction_metadata(example, field):
    root, inventory, ledger, record = example
    del record[field]
    (root / "evidence.json").write_text(json.dumps(record))
    assert any(f"missing {field}" in e for e in gate.validate(root, inventory, ledger))


def test_supersession_requires_existing_replacement_and_approval(example):
    root, inventory, ledger, record = example
    entry = ledger["entries"][0]
    entry.update(status="superseded", replacement="spec.md#Replacement behavior")
    assert any("non-approved" in e for e in gate.validate(root, inventory, ledger))
    record["outcome"] = "approved"
    (root / "evidence.json").write_text(json.dumps(record))
    assert gate.validate(root, inventory, ledger) == []
    entry["replacement"] = "spec.md#Nonexistent"
    assert any("replacement requirement" in e for e in gate.validate(root, inventory, ledger))


def test_unresolved_entries_are_valid_inventory_but_block_release(example):
    root, inventory, ledger, _ = example
    entry = ledger["entries"][0]
    entry.update(status="blocked", evidence=[], tests=[], implementation=[])
    assert gate.validate(root, inventory, ledger) == []
    assert any("release incomplete" in e for e in gate.validate(root, inventory, ledger, True))


def test_bad_json_and_nonobject_evidence_do_not_crash(example):
    root, inventory, ledger, _ = example
    for value in ("not json", "[]", "null"):
        (root / "evidence.json").write_text(value)
        assert gate.validate(root, inventory, ledger)


def test_symlink_outside_root_is_rejected(example, tmp_path_factory):
    root, inventory, ledger, _ = example
    outside = tmp_path_factory.mktemp("outside") / "evidence.json"
    outside.write_text("{}")
    link = root / "outside.json"
    link.symlink_to(outside)
    ledger["entries"][0]["evidence"] = ["outside.json"]
    assert any("unsafe evidence" in e for e in gate.validate(root, inventory, ledger))


def test_cli_exit_codes_and_completion_warning(example, monkeypatch, capsys):
    root, inventory, ledger, _ = example
    monkeypatch.setattr(gate, "ROOT", root)
    inv = root / "inventory.json"
    led = root / "ledger.json"
    inv.write_text(json.dumps(inventory))
    led.write_text(json.dumps(ledger))
    args = ["--inventory", str(inv), "--ledger", str(led)]
    assert gate.main(args) == 0
    assert "NOT a release-completion claim" in capsys.readouterr().out
    assert gate.main(args + ["--require-complete"]) == 1
    led.write_text("broken")
    assert gate.main(args) == 2


def test_repository_inventory_is_fully_represented_without_claiming_completion():
    inventory = json.loads((gate.ROOT / gate.CHANGE / "inventory.json").read_text())
    ledger = json.loads((gate.ROOT / "reports/completion-evidence.json").read_text())
    assert len(gate.inventory_items(inventory)) == 489
    assert gate.validate(gate.ROOT, inventory, ledger) == []
    assert gate.validate(gate.ROOT, inventory, ledger, require_complete=True)
