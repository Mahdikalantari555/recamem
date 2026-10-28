"""Identity, profile home and migration for recamem.

Profiles live under ``~/.recamem/agents/<name>/`` (override with
``RECAMEM_HOME``). A legacy profile under ``droids/<name>/`` stays readable in
place, and ``recamem migrate droids/`` copies it into the recamem home with a
checksum record. Migration never deletes or rewrites the source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MIGRATION_RECORD = "migration.json"


def recamem_home() -> Path:
    """Root of recamem-owned state (``RECAMEM_HOME`` or ``~/.recamem``)."""
    explicit = os.environ.get("RECAMEM_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.home() / ".recamem"


def agents_dir() -> Path:
    """Directory holding recamem agent profiles."""
    return recamem_home() / "agents"


def legacy_agents_dir(cwd: Optional[Path] = None) -> Path:
    """The pre-rename profile directory (``droids/`` in the working tree)."""
    return Path(cwd or Path.cwd()) / "droids"


def legacy_profile_exists(name: str, cwd: Optional[Path] = None) -> bool:
    return (legacy_agents_dir(cwd) / name).is_dir()


def default_base_dir(cwd: Optional[Path] = None) -> Path:
    """Default profile root for a new engine/manager.

    Order: ``RECAMEM_HOME/agents`` when set, else the legacy ``droids/``
    directory when it already exists locally (read in place), else
    ``~/.recamem/agents``.
    """
    if os.environ.get("RECAMEM_HOME"):
        return agents_dir()
    legacy = legacy_agents_dir(cwd)
    if legacy.is_dir():
        return legacy.resolve()
    return agents_dir()


def profile_dir(name: str, base_dir: Optional[os.PathLike | str] = None,
                cwd: Optional[Path] = None) -> Path:
    """Resolve an existing profile directory, preferring an explicit base.

    Without an explicit base the recamem home wins when the profile exists
    there, then a legacy ``droids/<name>`` directory, then the default root.
    """
    if base_dir is not None:
        return Path(base_dir) / name
    candidate = agents_dir() / name
    if candidate.is_dir():
        return candidate
    legacy = legacy_agents_dir(cwd) / name
    if legacy.is_dir():
        return legacy.resolve()
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_checksums(root: Path) -> Dict[str, str]:
    sums: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sums[str(path.relative_to(root))] = sha256_file(path)
    return sums


def migrate_droids(source: Optional[os.PathLike | str] = None,
                   dest: Optional[os.PathLike | str] = None) -> Dict[str, Any]:
    """Copy every legacy profile into the recamem home, non-destructively.

    Returns a report with one entry per profile (``migrated``,
    ``already_migrated`` or ``failed``), the destination root and the path of
    the written ``migration.json`` record. Re-running with an unchanged source
    reports ``already_migrated`` and touches nothing.
    """
    source_dir = Path(source or legacy_agents_dir()).expanduser().resolve()
    dest_dir = Path(dest).expanduser().resolve() if dest else agents_dir()
    report: Dict[str, Any] = {
        "source": str(source_dir),
        "destination": str(dest_dir),
        "profiles": [],
        "source_checksums": {},
    }
    if not source_dir.is_dir():
        report["status"] = "no_source"
        report["reason"] = f"source directory {source_dir} does not exist"
        return report

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_record: Dict[str, Any] = {}
    record_path = dest_dir / MIGRATION_RECORD
    if record_path.exists():
        try:
            existing_record = json.loads(record_path.read_text()).get("profiles", {})
        except (OSError, ValueError, AttributeError):
            existing_record = {}

    for profile in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        entry: Dict[str, Any] = {"name": profile.name, "source": str(profile)}
        checksums = _tree_checksums(profile)
        entry["checksums"] = checksums
        prior = existing_record.get(profile.name, {})
        target = dest_dir / profile.name
        try:
            if prior.get("checksums") == checksums and target.is_dir():
                entry["status"] = "already_migrated"
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(profile, target)
                after = _tree_checksums(target)
                if after != checksums:
                    raise OSError("copied tree checksums differ from the source")
                entry["status"] = "migrated"
        except OSError as exc:
            entry["status"] = "failed"
            entry["reason"] = str(exc)
        report["profiles"].append(entry)
        existing_record[profile.name] = {
            "checksums": checksums,
            "destination": str(target),
            "status": entry["status"],
        }

    report["status"] = "ok"
    report["counts"] = {
        status: sum(1 for p in report["profiles"] if p["status"] == status)
        for status in ("migrated", "already_migrated", "failed")
    }
    # The source is never modified; record the post-copy checksums so a later
    # run can prove that too.
    report["source_checksums"] = {
        p["name"]: p["checksums"] for p in report["profiles"]
    }
    record = {
        "tool": "recamem",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source_dir),
        "profiles": existing_record,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    report["record_path"] = str(record_path)
    return report


def describe_migration(report: Dict[str, Any]) -> List[str]:
    """Human-readable lines for ``recamem migrate`` output."""
    lines = [f"source={report.get('source')}", f"destination={report.get('destination')}"]
    for profile in report.get("profiles", []):
        line = f"profile={profile['name']} status={profile['status']}"
        if profile.get("reason"):
            line += f" reason={profile['reason']}"
        lines.append(line)
    counts = report.get("counts", {})
    for key in ("migrated", "already_migrated", "failed"):
        lines.append(f"{key}={counts.get(key, 0)}")
    if report.get("record_path"):
        lines.append(f"record={report['record_path']}")
    if report.get("reason"):
        lines.append(f"reason={report['reason']}")
    return lines
