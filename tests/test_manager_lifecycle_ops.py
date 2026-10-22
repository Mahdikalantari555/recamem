"""Profile-manager version operations: merge, export and import.

Historical task 7.4 promised manager-level merge/export/import and the
`memory/merge-engine` spec requires an auditable merge trail (`merge_of`,
conflict count, resolutions, new version). Both are asserted here end to end
against a real on-disk profile directory.
"""

from __future__ import annotations

import json

import pytest

from src.model.droid_manager import DroidManager


def _teach(engine, text: str):
    engine.teach(text)
    return engine


@pytest.fixture()
def manager(tmp_path):
    mgr = DroidManager(base_dir=str(tmp_path / "droids"))
    yield mgr
    mgr.shutdown()


def test_manager_merges_exports_and_imports_a_version(manager, tmp_path):
    first = _teach(manager.create_droid("alpha"), "The crane is on site A")
    second = _teach(manager.create_droid("beta"), "The crane is on site A")
    manager.save_droid("alpha")
    manager.save_droid("beta")

    merged = manager.merge_droids("alpha", "beta", "gamma", policy="keep_newest")

    result = merged.last_merge_result
    assert result["status"] == "merged"
    # Auditable: both sources and the policy are recorded, not just a version id.
    assert result["merge_of"] == ["gamma", "beta"]
    assert result["merge_audit"]["policy"] == "keep_newest"
    assert result["merge_audit"]["conflict_count"] == len(result["conflicts"])

    manifest = json.loads((manager.base_dir / "gamma" / "config.json").read_text())
    assert manifest["merge_of"] == ["gamma", "beta"]
    assert manifest["merge_history"]
    assert manifest["merge_history"][0]["merge_of"] == ["gamma", "beta"]

    version = manager.create_version("gamma", "v1")
    assert version, "create_version must return snapshot metadata"

    out = tmp_path / "export" / "gamma.droid"
    exported = manager.export_version("gamma", "v1", str(out))
    assert out.exists()
    assert exported

    restored = manager.import_version("delta", str(out))
    assert "delta" in manager.list_droids()
    assert restored.knowledge.active_count() >= 1


def test_merge_refuses_to_overwrite_an_existing_profile(manager):
    _teach(manager.create_droid("alpha"), "The crane is on site A")
    _teach(manager.create_droid("beta"), "The pump failed")
    manager.save_droid("alpha")
    manager.save_droid("beta")

    _teach(manager.create_droid("gamma"), "Gravity attracts masses")
    manager.save_droid("gamma")

    with pytest.raises(ValueError, match="already exists"):
        manager.merge_droids("alpha", "beta", "gamma")

    with pytest.raises(ValueError, match="new name"):
        manager.merge_droids("alpha", "beta", "beta")


def test_manual_merge_refusal_leaves_no_new_profile(manager):
    _teach(manager.create_droid("alpha"), "The crane is on site A")
    _teach(manager.create_droid("beta"), "The crane is not on site A")
    manager.save_droid("alpha")
    manager.save_droid("beta")

    merged = manager.merge_droids("alpha", "beta", "gamma", policy="require_manual")

    assert merged.last_merge_result["status"] == "manual_required"
    assert merged.last_merge_result["writes"] == 0
    assert "gamma" not in manager.list_droids()


def test_merge_history_survives_restart(manager, tmp_path):
    _teach(manager.create_droid("alpha"), "The crane is on site A")
    _teach(manager.create_droid("beta"), "The pump failed")
    manager.save_droid("alpha")
    manager.save_droid("beta")
    manager.merge_droids("alpha", "beta", "gamma", policy="keep_newest")

    reloaded = DroidManager(base_dir=str(manager.base_dir))
    engine = reloaded.get_droid("gamma")
    try:
        assert getattr(engine, "merge_of", None) == ["gamma", "beta"]
        assert getattr(engine, "merge_history", [])
    finally:
        reloaded.shutdown()
