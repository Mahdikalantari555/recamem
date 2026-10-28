"""Identity: one name, one home, a non-destructive migration (recamem/identity)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recamem import identity

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_live_under_the_recamem_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path / "home"))
    assert identity.agents_dir() == tmp_path / "home" / "agents"


def test_legacy_profile_is_read_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAMEM_HOME", str(tmp_path / "home"))
    legacy = tmp_path / "work" / "droids" / "legacy-agent"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    resolved = identity.profile_dir("legacy-agent", cwd=tmp_path / "work")
    assert resolved == legacy.resolve()


def test_default_base_dir_prefers_an_existing_legacy_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("RECAMEM_HOME", raising=False)
    (tmp_path / "droids").mkdir()
    assert identity.default_base_dir(cwd=tmp_path) == (tmp_path / "droids").resolve()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert identity.default_base_dir(cwd=empty) != (empty / "droids")


def test_migration_preserves_the_source_and_is_repeatable(tmp_path):
    source = tmp_path / "droids"
    profile = source / "droid-rs"
    profile.mkdir(parents=True)
    (profile / "config.json").write_text(json.dumps({"name": "droid-rs"}))
    (profile / "knowledge.db").write_bytes(b"sqlite-bytes")
    before = {p.name: identity.sha256_file(p) for p in profile.iterdir()}

    first = identity.migrate_droids(source, tmp_path / "home")
    assert first["counts"]["migrated"] == 1
    after = {p.name: identity.sha256_file(p) for p in profile.iterdir()}
    assert after == before, "the source must be byte-identical after migration"
    copied = tmp_path / "home" / "droid-rs"
    assert copied.is_dir()

    second = identity.migrate_droids(source, tmp_path / "home")
    assert second["counts"]["already_migrated"] == 1
    assert second["counts"]["migrated"] == 0
    assert {p.name: identity.sha256_file(p) for p in copied.iterdir()} == before
    record = json.loads((tmp_path / "home" / "migration.json").read_text())
    assert record["profiles"]["droid-rs"]["checksums"] == before


def test_missing_source_reports_no_source(tmp_path):
    report = identity.migrate_droids(tmp_path / "absent", tmp_path / "home")
    assert report["status"] == "no_source"
    assert "does not exist" in report["reason"]


def test_old_entry_point_still_works_and_warns(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "doctor", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT), "RECAMEM_ANCHOR": "mock"},
    )
    # `doctor` is a recamem command, so the legacy parser reports it as an
    # invalid choice but the deprecation notice must already be on stderr.
    assert "deprecated" in result.stderr
    assert "recamem" in result.stderr


def test_compatibility_shims_alias_the_real_modules():
    import src.model.droid as shim
    import recamem.engine as real

    assert shim.DroidEngine is real.DroidEngine
    assert shim.__name__ == "recamem.engine"


def test_agent_package_keeps_the_old_name():
    from recamem.package import AgentPackage, DroidPackage

    assert AgentPackage is DroidPackage


def test_profile_manifest_records_its_schema_version(tmp_path):
    from recamem.engine import PROFILE_SCHEMA_VERSION, DroidEngine
    from src.model.rtu import InvalidProfileError

    engine = DroidEngine(name="schema")
    engine.teach("The calibration constant is 0.98.")
    engine.save_profile(str(tmp_path / "p"))
    config = json.loads((tmp_path / "p" / "config.json").read_text())
    assert config["schema_version"] == PROFILE_SCHEMA_VERSION

    config["schema_version"] = PROFILE_SCHEMA_VERSION + 5
    (tmp_path / "p" / "config.json").write_text(json.dumps(config))
    with pytest.raises(InvalidProfileError) as excinfo:
        DroidEngine(name="schema").load_profile(str(tmp_path / "p"))
    assert "newer" in str(excinfo.value)


def test_older_manifest_is_upgraded_with_a_record(tmp_path):
    from recamem.engine import DroidEngine

    engine = DroidEngine(name="old")
    engine.teach("The budget is 5 million.")
    engine.save_profile(str(tmp_path / "old"))
    config = json.loads((tmp_path / "old" / "config.json").read_text())
    config.pop("schema_version")
    (tmp_path / "old" / "config.json").write_text(json.dumps(config))

    fresh = DroidEngine(name="old")
    assert fresh.load_profile(str(tmp_path / "old")) is True
    assert any(m["kind"] == "profile_schema" for m in fresh.migrations)
