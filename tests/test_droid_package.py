"""Contract tests for src/model/droid_package.py.

Invariants under test (see openspec/specs/portable-brain):
- export bundles the recognised profile members plus a manifest whose
  ``checksums``/``files`` entries are SHA-256 digests of the archived bytes;
- import verifies each manifest checksum, writes every member, and refuses
  archives whose members were tampered with or are missing;
- zip-slip members (``../`` or absolute paths) are rejected before extraction.
"""

import hashlib
import json
import zipfile

import pytest

from src.model.droid_package import DroidPackage

MANIFEST_KEYS = {"version", "created_at", "files", "checksums"}


def _make_profile(directory, name="droid-alpha", dim=384):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"name": name, "dim": dim}))
    (directory / "memory.safetensors").write_bytes(b"fake-safetensors-payload")
    (directory / "knowledge.db").write_bytes(b"sqlite-bytes")
    (directory / "train_log.json").write_text(json.dumps([{"step": 1}]))
    return directory


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_writes_members_and_manifest(tmp_path):
    profile = _make_profile(tmp_path / "profiles" / "droid-alpha")
    out = tmp_path / "pkg" / "alpha.droid"

    result = DroidPackage.export_droid(str(profile), str(out))

    assert out.exists()
    assert result["out_path"] == str(out)
    assert sorted(result["files"]) == [
        "config.json",
        "knowledge.db",
        "memory.safetensors",
        "train_log.json",
    ]

    manifest = result["manifest"]
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["dim"] == 384
    assert manifest["droid_name"] == "droid-alpha"
    assert set(manifest["checksums"]) == set(result["files"])
    for member in result["files"]:
        assert manifest["checksums"][member] == _sha256(profile / member)
        assert manifest["files"][member]["sha256"] == manifest["checksums"][member]
        assert manifest["files"][member]["size"] == (profile / member).stat().st_size

    with zipfile.ZipFile(out) as zf:
        assert "manifest.json" in zf.namelist()
        written = json.loads(zf.read("manifest.json"))
        assert written["checksums"] == manifest["checksums"]


def test_export_includes_extra_files(tmp_path):
    profile = _make_profile(tmp_path / "droid")
    (profile / "custom.txt").write_text("extra member")

    result = DroidPackage.export_droid(str(profile), str(tmp_path / "x.droid"), extra_files=["custom.txt"])
    assert "custom.txt" in result["files"]
    assert result["manifest"]["checksums"]["custom.txt"] == _sha256(profile / "custom.txt")

    with zipfile.ZipFile(tmp_path / "x.droid") as zf:
        assert "custom.txt" in zf.namelist()


def test_export_skips_missing_optional_members(tmp_path):
    profile = tmp_path / "partial"
    profile.mkdir()
    (profile / "config.json").write_text(json.dumps({"name": "partial"}))

    result = DroidPackage.export_droid(str(profile), str(tmp_path / "partial.droid"))
    assert result["files"] == ["config.json"]
    # No config dim -> the documented 384 fallback is recorded.
    assert result["manifest"]["dim"] == 384


def test_export_requires_at_least_one_member(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        DroidPackage.export_droid(str(empty), str(tmp_path / "empty.droid"))


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
def test_import_round_trip_verifies_and_extracts(tmp_path):
    profile = _make_profile(tmp_path / "source")
    archive = tmp_path / "round.droid"
    DroidPackage.export_droid(str(profile), str(archive))

    target = tmp_path / "restored"
    result = DroidPackage.import_droid(str(archive), str(target))

    assert result["verified"] is True
    assert result["target_dir"] == str(target)
    assert result["manifest"]["droid_name"] == "droid-alpha"
    for member in ("config.json", "memory.safetensors", "knowledge.db", "train_log.json", "manifest.json"):
        assert (target / member).exists()
    for member in ("config.json", "memory.safetensors", "knowledge.db", "train_log.json"):
        assert (target / member).read_bytes() == (profile / member).read_bytes()


def test_import_requires_existing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        DroidPackage.import_droid(str(tmp_path / "nope.droid"), str(tmp_path / "out"))


def test_import_rejects_missing_manifest(tmp_path):
    archive = tmp_path / "no-manifest.droid"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.json", "{}")

    with pytest.raises(ValueError, match="missing manifest.json"):
        DroidPackage.import_droid(str(archive), str(tmp_path / "out"))


def _rewrite_member(archive, member, payload):
    """Rewrite one member of an existing archive, leaving the manifest stale."""
    with zipfile.ZipFile(archive) as zf:
        items = {name: zf.read(name) for name in zf.namelist()}
    items[member] = payload
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in items.items():
            zf.writestr(name, data)


def test_import_rejects_tampered_member(tmp_path):
    profile = _make_profile(tmp_path / "source")
    archive = tmp_path / "tampered.droid"
    DroidPackage.export_droid(str(profile), str(archive))
    _rewrite_member(archive, "config.json", b'{"name": "evil", "dim": 8}')

    with pytest.raises(ValueError, match="Checksum mismatch"):
        DroidPackage.import_droid(str(archive), str(tmp_path / "out"))
    # Verification runs before the target profile is touched.
    assert not (tmp_path / "out").exists()


def test_import_rejects_member_missing_from_archive(tmp_path):
    profile = _make_profile(tmp_path / "source")
    archive = tmp_path / "dropped.droid"
    DroidPackage.export_droid(str(profile), str(archive))
    with zipfile.ZipFile(archive) as zf:
        items = {name: zf.read(name) for name in zf.namelist()}
    items.pop("knowledge.db")
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in items.items():
            zf.writestr(name, data)

    with pytest.raises(ValueError, match="Missing file in archive"):
        DroidPackage.import_droid(str(archive), str(tmp_path / "out"))


def test_import_can_skip_verification(tmp_path):
    profile = _make_profile(tmp_path / "source")
    archive = tmp_path / "unverified.droid"
    DroidPackage.export_droid(str(profile), str(archive))
    _rewrite_member(archive, "config.json", b'{"name": "changed"}')

    result = DroidPackage.import_droid(str(archive), str(tmp_path / "out"), verify=False)
    assert result["verified"] is False
    assert (tmp_path / "out" / "config.json").read_text() == '{"name": "changed"}'


@pytest.mark.parametrize("member", ["../escape.txt", "nested/../../escape.txt", "/absolute.txt"])
def test_import_rejects_path_traversal_members(tmp_path, member):
    archive = tmp_path / "slip.droid"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, "pwned")
        zf.writestr("manifest.json", json.dumps({"version": "1.0", "checksums": {}}))

    target = tmp_path / "out"
    with pytest.raises(ValueError, match="Security violation"):
        DroidPackage.import_droid(str(archive), str(target))
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "absolute.txt").exists()


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def test_list_contents_does_not_extract(tmp_path):
    profile = _make_profile(tmp_path / "source")
    archive = tmp_path / "listme.droid"
    DroidPackage.export_droid(str(profile), str(archive))

    listing = DroidPackage.list_contents(str(archive))
    assert "manifest.json" in listing["files"]
    assert "config.json" in listing["files"]
    assert listing["manifest"]["droid_name"] == "droid-alpha"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["source", "listme.droid"])
