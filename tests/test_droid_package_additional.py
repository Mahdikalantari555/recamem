"""Contract tests for `src/model/droid_package.py` (surface-test-coverage 3.6).

Covered: export/import round trip in `tmp_path`, manifest digest presence,
tampered-member rejection, and `../` zip-slip rejection.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.model.droid_package import DroidPackage


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_profile(profile_dir: Path) -> dict:
    (profile_dir / "config.json").write_text(json.dumps({"name": "droid-x", "dim": 384}))
    (profile_dir / "train_log.json").write_text(json.dumps([{"step": 1}]))
    (profile_dir / "knowledge.db").write_bytes(b"sqlite-ish bytes")
    (profile_dir / "memory.safetensors").write_bytes(b"safetensors-ish bytes")
    return {name: _sha256_file(profile_dir / name) for name in
            ("config.json", "train_log.json", "knowledge.db", "memory.safetensors")}


def test_export_import_round_trip_in_tmp_path(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    checksums = _make_profile(profile_dir)
    out_path = tmp_path / "droid-x.droid"

    result = DroidPackage.export_droid(str(profile_dir), str(out_path))

    assert out_path.exists()
    assert result["out_path"] == str(out_path)
    assert set(result["files"]) >= {"config.json", "train_log.json", "knowledge.db", "memory.safetensors"}
    manifest = result["manifest"]
    assert manifest["version"] == "1.0"
    assert manifest["dim"] == 384
    assert manifest["droid_name"] == "droid-x"
    for arcname, expected in checksums.items():
        assert manifest["checksums"][arcname] == expected
        assert manifest["files"][arcname]["sha256"] == expected

    target_dir = tmp_path / "imported"
    imported = DroidPackage.import_droid(str(out_path), str(target_dir))
    assert imported["verified"] is True
    for name in ("config.json", "train_log.json", "knowledge.db", "memory.safetensors"):
        assert (target_dir / name).exists()
        assert _sha256_file(target_dir / name) == checksums[name]


def test_list_contents_without_extracting(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _make_profile(profile_dir)
    out_path = tmp_path / "droid-x.droid"
    DroidPackage.export_droid(str(profile_dir), str(out_path))

    listing = DroidPackage.list_contents(str(out_path))
    assert "manifest.json" in listing["files"]
    assert "config.json" in listing["files"]
    assert isinstance(listing["manifest"], dict)
    assert listing["manifest"]["droid_name"] == "droid-x"


def test_import_rejects_tampered_archive(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _make_profile(profile_dir)
    out_path = tmp_path / "droid-x.droid"
    DroidPackage.export_droid(str(profile_dir), str(out_path))

    # Flip bytes of config.json inside the archive.
    with zipfile.ZipFile(out_path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries["config.json"] = entries["config.json"] + b"tampered"
    tampered_path = tmp_path / "tampered.droid"
    with zipfile.ZipFile(tampered_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    target_dir = tmp_path / "imported-tampered"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        DroidPackage.import_droid(str(tampered_path), str(target_dir))


def test_import_rejects_zip_slip(tmp_path):
    out_path = tmp_path / "slip.droid"
    with zipfile.ZipFile(out_path, "w") as zf:
        zf.writestr("manifest.json", '{"version": "1.0", "checksums": {}, "files": {}}')
        zf.writestr("../evil.txt", "pwned")

    target_dir = tmp_path / "imported-slip"
    with pytest.raises(ValueError, match="path traversal|zip-slip"):
        DroidPackage.import_droid(str(out_path), str(target_dir))


def test_import_rejects_archive_without_manifest(tmp_path):
    out_path = tmp_path / "no-manifest.droid"
    with zipfile.ZipFile(out_path, "w") as zf:
        zf.writestr("config.json", "{}")

    target_dir = tmp_path / "imported-nomanifest"
    with pytest.raises(ValueError, match="missing manifest.json"):
        DroidPackage.import_droid(str(out_path), str(target_dir))


def test_export_requires_at_least_one_droid_file(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        DroidPackage.export_droid(str(empty_dir), str(tmp_path / "out.droid"))
