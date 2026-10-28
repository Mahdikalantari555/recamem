"""Console correctness contracts (product-surface-completion 5.5/5.6) and the
store/model helpers the new console panels call (5.7–5.14).

Streamlit widgets are validated manually; what can be wrong in logic — the
successor id, the merge policy value, the digest check — lives in these calls
and is tested here. `app.py` itself is checked structurally so a bare `except:`
or a sentinel successor cannot come back unnoticed.
"""

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch

from src.model.droid_package import DroidPackage
from src.model.knowledge_store import KnowledgeStore
from src.model.memory_lifecycle import MergeEngine, SnapshotManager

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "recamem" / "console_legacy.py"


@pytest.fixture
def store():
    directory = tempfile.mkdtemp()
    ks = KnowledgeStore(str(Path(directory) / "k.db"), dim=8)
    yield ks
    ks.close()
    shutil.rmtree(directory, ignore_errors=True)


def _vec(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(8, generator=generator)


# ---------------------------------------------------------------------------
# app.py structure
# ---------------------------------------------------------------------------

def test_app_has_no_bare_except():
    for lineno, line in enumerate(APP_PY.read_text(encoding="utf-8").splitlines(), start=1):
        assert not re.match(r"^\s*except\s*:\s*$", line), (
            f"bare except at app.py:{lineno}: name the exception and report it"
        )


def test_app_does_not_supersede_to_sentinel_zero():
    source = APP_PY.read_text(encoding="utf-8")
    assert not re.search(r"supersede\([^)]*,\s*0\s*\)", source)


def test_app_wires_the_lifecycle_and_portability_apis():
    source = APP_PY.read_text(encoding="utf-8")
    for symbol in (
        "SnapshotManager",
        "MergeEngine",
        "ContinuousStream",
        "DroidPackage.export_droid",
        "DroidPackage.import_droid",
        "DroidPackage.list_contents",
        "download_model",
        "decide_path",
        "confidence_tier",
    ):
        assert symbol in source, f"app.py never references {symbol}"


def test_stream_worker_is_owned_by_the_engine():
    source = APP_PY.read_text(encoding="utf-8")
    assert "continuous_stream_" in source
    # The console delegates to the engine; it must not construct its own worker.
    assert "active_droid.start_continuous_stream(" in source
    assert "active_droid.stop_continuous_stream(" in source
    assert "ContinuousStream(active_droid)" not in source


def test_destructive_console_actions_are_confirmation_gated():
    source = APP_PY.read_text(encoding="utf-8")
    for key in ("confirm_rollback", "confirm_merge", "confirm_import", "confirm_gguf_download"):
        assert key in source


# ---------------------------------------------------------------------------
# Supersede semantics
# ---------------------------------------------------------------------------

def test_supersede_records_real_successor_id(store):
    old_id = store.insert_fact("The capital of West Germany is Bonn", "chat", 1.0, 0, 0.5, _vec(1))
    new_id = store.insert_fact("The capital of West Germany is Berlin", "chat", 2.0, 1, 0.5, _vec(2))

    assert store.supersede(old_id, new_id) == 1
    fact = store.get_fact(old_id)
    assert fact["superseded"] is True
    assert fact["superseded_by"] == new_id


def test_supersede_rejects_sentinel_successor(store):
    old_id = store.insert_fact("Fact to supersede", "chat", 1.0, 0, 0.5, _vec(3))
    with pytest.raises(ValueError):
        store.supersede(old_id, 0)
    # The rejected call changed nothing.
    assert store.get_fact(old_id)["superseded"] is False


def test_supersede_rejects_unknown_successor(store):
    old_id = store.insert_fact("Fact with dangling successor", "chat", 1.0, 0, 0.5, _vec(4))
    with pytest.raises(ValueError):
        store.supersede(old_id, 9999)
    assert store.get_fact(old_id)["superseded"] is False


def test_mark_superseded_leaves_successor_null(store):
    old_id = store.insert_fact("Fact retired without replacement", "chat", 1.0, 0, 0.5, _vec(5))

    assert store.mark_superseded(old_id) == 1
    fact = store.get_fact(old_id)
    assert fact["superseded"] is True
    assert fact["superseded_by"] is None
    assert fact["valid_until"] is not None
    # Idempotent: a second call is a no-op.
    assert store.mark_superseded(old_id) == 0


def test_superseded_by_is_always_a_real_id_or_null(store):
    first = store.insert_fact("First", "chat", 1.0, 0, 0.5, _vec(6))
    second = store.insert_fact("Second", "chat", 2.0, 1, 0.5, _vec(7))
    third = store.insert_fact("Third", "chat", 3.0, 2, 0.5, _vec(8))
    store.supersede(first, second)
    store.mark_superseded(second)

    known_ids = {fact["id"] for fact in store.get_audit_trail()}
    for fact in store.get_audit_trail():
        successor = fact["superseded_by"]
        assert successor is None or successor in known_ids
        assert successor != 0


# ---------------------------------------------------------------------------
# Merge policy selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", MergeEngine.POLICIES)
def test_documented_policies_are_accepted(policy):
    assert MergeEngine.validate_policy(policy) == policy


def test_unknown_policy_is_rejected_before_detection(monkeypatch):
    calls = []

    def fake_detect(store_a, store_b):
        calls.append((store_a, store_b))
        return []

    monkeypatch.setattr(MergeEngine, "detect_conflicts", staticmethod(fake_detect))
    with pytest.raises(ValueError):
        MergeEngine.merge_memories(None, None, policy="bogus_policy")
    assert calls == []


def test_merge_engine_passes_selected_policy_through(store):
    other = KnowledgeStore(":memory:", dim=8)
    try:
        result = MergeEngine.merge_memories(store, other, policy="keep_newest")
        assert result["policy"] == "keep_newest"
        assert result["conflict_count"] == 0
    finally:
        other.close()


# ---------------------------------------------------------------------------
# Snapshots (used by the console rollback/branch panel)
# ---------------------------------------------------------------------------

def test_branch_does_not_modify_source_snapshot():
    class FakeMemory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("states", torch.zeros(4))

    class FakeDroid:
        def __init__(self):
            self.memory = FakeMemory()
            self.step_count = 3
            self.knowledge = type("K", (), {"active_count": staticmethod(lambda: 2)})()

    manager = SnapshotManager(base_dir="droids")
    droid = FakeDroid()
    manager.create_snapshot(droid, "v1.0")
    branched = manager.branch_from("v1.0", "experiment-a")

    assert branched["metadata"]["parent_version"] == "v1.0"
    assert [s["version_id"] for s in manager.list_snapshots()] == ["v1.0", "experiment-a"]
    assert manager.list_snapshots()[0]["parent_version"] is None


# ---------------------------------------------------------------------------
# Package digests (console export/import panel)
# ---------------------------------------------------------------------------

def _make_profile(root: Path) -> Path:
    profile = root / "droid-pkg"
    profile.mkdir(parents=True)
    (profile / "config.json").write_text(json.dumps({"name": "droid-pkg", "dim": 8}))
    (profile / "knowledge.db").write_text("fake sqlite payload")
    (profile / "train_log.json").write_text(json.dumps([]))
    return profile


def test_export_is_written_outside_profile_and_lists_digests(tmp_path):
    profile = _make_profile(tmp_path)
    archive = tmp_path / "exports" / "droid-pkg.droid"
    DroidPackage.export_droid(str(profile), str(archive))

    assert archive.exists()
    assert not (profile / "droid-pkg.droid").exists()

    contents = DroidPackage.list_contents(str(archive))
    manifest = contents["manifest"]
    assert "knowledge.db" in contents["files"]
    for member in manifest["files"]:
        assert len(manifest["checksums"][member]) == 64


def test_import_aborts_on_digest_mismatch(tmp_path):
    profile = _make_profile(tmp_path)
    archive = tmp_path / "exports" / "droid-pkg.droid"
    DroidPackage.export_droid(str(profile), str(archive))

    # Rewrite one member without updating the manifest checksums.
    with zipfile.ZipFile(archive, "r") as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    members["knowledge.db"] = b"tampered payload"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    target = tmp_path / "droids" / "restored"
    with pytest.raises(ValueError) as excinfo:
        DroidPackage.import_droid(str(archive), str(target), verify=True)

    message = str(excinfo.value)
    assert "knowledge.db" in message
    assert "expected" in message and "got" in message
    assert not target.exists(), "a failed verification must not leave a partial profile"


def test_verified_import_restores_members(tmp_path):
    profile = _make_profile(tmp_path)
    archive = tmp_path / "exports" / "droid-pkg.droid"
    DroidPackage.export_droid(str(profile), str(archive))

    target = tmp_path / "droids" / "restored"
    result = DroidPackage.import_droid(str(archive), str(target), verify=True)

    assert result["verified"] is True
    assert (target / "config.json").exists()
    assert (target / "knowledge.db").read_text() == "fake sqlite payload"
