
import hashlib
import os
import random
import re
import shutil
import sqlite3
import tempfile
import time
from src.model.plastic_adapter import clone_state
import json
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F

# ponytail: memory lifecycle management - versioning, merge, semantic store, contradiction, adaptive retrieval, continuous stream, TTT

class SnapshotValidationError(RuntimeError):
    """A snapshot artifact failed validation; no live state was activated."""


SNAPSHOT_MANIFEST = "snapshot.json"
SNAPSHOT_DB = "knowledge.db"
SNAPSHOT_TENSORS = "plastic.safetensors"
SNAPSHOT_META = "plastic_meta.json"
SNAPSHOT_POLICY = "retrieval_policy.json"
SNAPSHOT_CONFIG = "lifecycle_config.json"
_SNAPSHOT_REQUIRED = (SNAPSHOT_MANIFEST, SNAPSHOT_TENSORS, SNAPSHOT_META,
                      SNAPSHOT_POLICY, SNAPSHOT_CONFIG)
_REQUIRED_TABLES = ("facts", "facts_vecs", "feedback_log")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(name: str) -> str:
    """Snapshot ids become directory names: reject traversal and separators."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "snapshot"
    return cleaned[:120]


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _split_state(state: Dict[str, Any]):
    tensors, meta = {}, {}
    for key, value in (state or {}).items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().contiguous().clone()
        else:
            meta[key] = clone_state(value)
    return tensors, meta


class SnapshotManager:
    """Complete, staged and reversible memory snapshots.

    A snapshot captures three things and references none of them by alias:

    * the full SQLite knowledge store, copied with the SQLite backup API so a
      WAL that has not yet been checkpointed is included;
    * every plastic tensor plus its non-tensor metadata;
    * the learned retrieval policy and lifecycle configuration/schedule.

    ``snapshot.json`` is written last and carries a checksum per artifact, so a
    half-written directory is never advertised as restorable. Restores stage and
    validate every artifact *before* touching live state, and a rollback takes a
    pre-rollback snapshot so it is itself reversible.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else None
        self.snapshots: Dict[str, Dict] = {}  # version_id -> snapshot payload
        self.last_error: Optional[str] = None
        self._temp_root: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Locations
    # ------------------------------------------------------------------ #
    def _base_for(self, droid) -> Path:
        if self.base_dir is not None:
            return self.base_dir
        db_path = getattr(getattr(droid, "knowledge", None), "db_path", None)
        if db_path and db_path != ":memory:":
            return Path(db_path).expanduser().resolve().parent / "snapshots"
        if self._temp_root is None:
            self._temp_root = Path(tempfile.mkdtemp(prefix="droid-snapshots-"))
        return self._temp_root

    def _candidate_bases(self, droid) -> List[Path]:
        bases = [self._base_for(droid)]
        if self._temp_root is not None and self._temp_root not in bases:
            bases.append(self._temp_root)
        return bases

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    def create_snapshot(self, droid, version_id: str,
                        parent_version: Optional[str] = None,
                        persist: Optional[bool] = None) -> Dict:
        """Capture plastic tensors, the complete database and policy/config."""
        version_id = str(version_id)
        plastic_state: Dict[str, Any] = {}
        try:
            if hasattr(droid.memory, "state_dict"):
                plastic_state = clone_state(droid.memory.state_dict())
        except Exception:
            plastic_state = {}
        tensors, meta_state = _split_state(plastic_state)

        policy_state = None
        policy = getattr(droid, "retrieval_policy", None)
        if policy is not None and hasattr(policy, "state_dict"):
            try:
                policy_state = json.loads(json.dumps(policy.state_dict(), default=str))
            except Exception:
                policy_state = None

        config = self._capture_config(droid)
        health = {}
        if hasattr(droid, "get_memory_health"):
            try:
                health = droid.get_memory_health()
            except Exception:
                health = {}

        store = getattr(droid, "knowledge", None)
        conn = getattr(store, "conn", None)
        fact_count = store.active_count() if hasattr(store, "active_count") else 0
        metadata = {
            "version_id": version_id,
            "timestamp": time.time(),
            "step_count": int(getattr(droid, "step_count", 0)),
            "fact_count": int(fact_count),
            "parent_version": parent_version,
            "health_metrics": health,
            "adapter": getattr(droid, "adapter_name", None),
            "persisted": False,
            "artifacts": {},
        }
        snapshot = {
            "plastic_state": plastic_state,
            "metadata": metadata,
            "policy_state": clone_state(policy_state),
            "config": clone_state(config),
            "version": version_id,
            "version_id": version_id,
            "timestamp": metadata["timestamp"],
            "step_count": metadata["step_count"],
            "fact_count": metadata["fact_count"],
            "persisted": False,
            "artifacts": {},
        }

        want_persist = isinstance(conn, sqlite3.Connection) if persist is None else bool(persist)
        if want_persist:
            directory = self._base_for(droid) / _safe_component(version_id)
            try:
                artifacts, db_stats = self._write_artifacts(
                    directory, store, tensors, meta_state, policy_state, config, metadata
                )
            except Exception as exc:  # never advertise a snapshot we could not write
                raise SnapshotValidationError(
                    f"could not write snapshot {version_id!r}: {exc}"
                ) from exc
            snapshot["artifacts"] = artifacts
            snapshot["persisted"] = True
            snapshot["path"] = str(directory)
            metadata["artifacts"] = artifacts
            metadata["persisted"] = True
            metadata["path"] = str(directory)
            metadata["database"] = db_stats
            snapshot["database"] = db_stats

        return self._remember(snapshot)

    def _capture_config(self, droid) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "adapter": getattr(droid, "adapter_name", None),
            "step_count": int(getattr(droid, "step_count", 0)),
            "current_version": getattr(droid, "current_version", None),
            "dim": getattr(getattr(droid, "memory", None), "dim", None),
            "K_max": getattr(droid, "K_max", None),
            "test_time_training": getattr(droid, "test_time_training", None),
            "ttt": None,
            "sleep_config": None,
            "lr_schedule": None,
        }
        if hasattr(droid, "ttt_status"):
            try:
                config["ttt"] = droid.ttt_status()
            except Exception:
                pass
        for name in ("sleep_config",):
            value = getattr(droid, name, None)
            if isinstance(value, dict):
                try:
                    config[name] = json.loads(json.dumps(value, default=str))
                except Exception:
                    pass
        schedule = getattr(droid, "lr_schedule", None)
        if schedule is not None and hasattr(schedule, "state_dict"):
            try:
                config["lr_schedule"] = clone_state(schedule.state_dict())
            except Exception:
                pass
        elif schedule is not None:
            config["lr_schedule"] = {
                key: clone_state(value) for key, value in vars(schedule).items()
            }
        return config

    def _write_artifacts(self, directory: Path, store, tensors, meta_state,
                         policy_state, config, metadata):
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(directory.parent)))
        try:
            db_stats = self._write_database(staging / SNAPSHOT_DB, store)
            self._write_tensors(staging / SNAPSHOT_TENSORS, tensors)
            _atomic_write_json(staging / SNAPSHOT_META, meta_state)
            _atomic_write_json(staging / SNAPSHOT_POLICY,
                               policy_state if policy_state is not None else {})
            _atomic_write_json(staging / SNAPSHOT_CONFIG, config)
            artifacts = {}
            for name in _SNAPSHOT_REQUIRED[1:] + (SNAPSHOT_DB,):
                path = staging / name
                artifacts[name] = {
                    "path": str(directory / name),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
            manifest = {
                "format": 2,
                "version_id": metadata["version_id"],
                "timestamp": metadata["timestamp"],
                "step_count": metadata["step_count"],
                "fact_count": metadata["fact_count"],
                "parent_version": metadata["parent_version"],
                "adapter": metadata.get("adapter"),
                "database": db_stats,
                "artifacts": artifacts,
            }
            _atomic_write_json(staging / SNAPSHOT_MANIFEST, manifest)
            if directory.exists():
                retired = directory.with_name(directory.name + f".old-{os.getpid()}-{int(time.time()*1000)}")
                os.replace(directory, retired)
                os.replace(staging, directory)
                shutil.rmtree(retired, ignore_errors=True)
            else:
                os.replace(staging, directory)
            return artifacts, db_stats
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _write_database(self, dest: Path, store) -> Dict[str, Any]:
        conn = getattr(store, "conn", None)
        if not isinstance(conn, sqlite3.Connection):
            raise SnapshotValidationError("knowledge store has no SQLite connection")
        tmp = dest.with_name(dest.name + ".tmp")
        target = sqlite3.connect(str(tmp))
        try:
            conn.commit()
            conn.backup(target)          # includes uncheckpointed WAL frames
            target.commit()
        finally:
            target.close()
        os.replace(tmp, dest)
        return self._database_stats(dest)

    @staticmethod
    def _database_stats(path: Path) -> Dict[str, Any]:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            facts = 0
            if "facts" in tables:
                facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        return {
            "integrity": integrity,
            "tables": sorted(tables),
            "fact_rows": int(facts),
            "journal_mode": journal,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }

    @staticmethod
    def _write_tensors(path: Path, tensors: Dict[str, torch.Tensor]) -> None:
        try:
            from safetensors.torch import save_file

            if tensors:
                save_file({k: v for k, v in tensors.items()}, str(path))
            else:
                path.write_bytes(b"")
        except ImportError:  # pragma: no cover - safetensors is a core dependency
            torch.save({k: v for k, v in tensors.items()}, str(path))

    @staticmethod
    def _read_tensors(path: Path) -> Dict[str, torch.Tensor]:
        if path.stat().st_size == 0:
            return {}
        try:
            from safetensors.torch import load_file

            return dict(load_file(str(path)))
        except ImportError:  # pragma: no cover
            return dict(torch.load(str(path), map_location="cpu"))

    def _remember(self, snapshot: Dict) -> Dict:
        self.snapshots[snapshot["metadata"]["version_id"]] = {
            "plastic_state": clone_state(snapshot.get("plastic_state", {})),
            "policy_state": clone_state(snapshot.get("policy_state")),
            "config": clone_state(snapshot.get("config")),
            "metadata": clone_state(snapshot.get("metadata", {})),
            "artifacts": dict(snapshot.get("artifacts", {})),
            "persisted": snapshot.get("persisted", False),
            "path": snapshot.get("path"),
            "version": snapshot.get("version"),
        }
        return snapshot

    # ------------------------------------------------------------------ #
    # Restore
    # ------------------------------------------------------------------ #
    def _resolve_snapshot(self, droid, version_id: str) -> Optional[Dict]:
        if version_id in self.snapshots:
            return self.snapshots[version_id]
        if self.base_dir is None and getattr(self, "_temp_root", None) is None and not getattr(
            droid, "knowledge", None
        ):
            return None
        for base in self._candidate_bases(droid):
            directory = base / _safe_component(version_id)
            manifest_path = directory / SNAPSHOT_MANIFEST
            if manifest_path.exists():
                try:
                    snapshot = self._snapshot_from_disk(directory)
                except Exception as exc:
                    self.last_error = str(exc)
                    return None
                self.snapshots[version_id] = snapshot
                return snapshot
        return None

    def _snapshot_from_disk(self, directory: Path) -> Dict:
        with open(directory / SNAPSHOT_MANIFEST, "r") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != 2:
            raise SnapshotValidationError(
                f"unsupported snapshot format {manifest.get('format')!r}"
            )
        version_id = manifest["version_id"]
        tensors = self._read_tensors(directory / SNAPSHOT_TENSORS)
        with open(directory / SNAPSHOT_META, "r") as handle:
            meta = json.load(handle)
        with open(directory / SNAPSHOT_POLICY, "r") as handle:
            policy_state = json.load(handle)
        with open(directory / SNAPSHOT_CONFIG, "r") as handle:
            config = json.load(handle)
        plastic_state = {**meta, **tensors}
        metadata = {
            "version_id": version_id,
            "timestamp": manifest["timestamp"],
            "step_count": manifest["step_count"],
            "fact_count": manifest["fact_count"],
            "parent_version": manifest.get("parent_version"),
            "adapter": manifest.get("adapter"),
            "persisted": True,
            "path": str(directory),
            "artifacts": manifest.get("artifacts", {}),
            "database": manifest.get("database"),
        }
        return {
            "plastic_state": plastic_state,
            "metadata": metadata,
            "policy_state": policy_state,
            "config": config,
            "version": version_id,
            "version_id": version_id,
            "timestamp": metadata["timestamp"],
            "step_count": metadata["step_count"],
            "fact_count": metadata["fact_count"],
            "persisted": True,
            "path": str(directory),
            "artifacts": metadata["artifacts"],
            "database": manifest.get("database"),
        }

    def _stage(self, droid, snapshot: Dict) -> Dict[str, Any]:
        """Load and validate every artifact. Raises before touching live state."""
        staged: Dict[str, Any] = {
            "plastic_state": clone_state(snapshot.get("plastic_state", {})),
            "policy_state": clone_state(snapshot.get("policy_state")),
            "config": clone_state(snapshot.get("config")),
            "metadata": clone_state(snapshot.get("metadata", {})),
            "database": None,
        }
        artifacts = snapshot.get("artifacts") or {}
        directory = snapshot.get("path")
        if snapshot.get("persisted"):
            if not directory:
                raise SnapshotValidationError("persisted snapshot has no directory")
            base = Path(directory)
            for name in _SNAPSHOT_REQUIRED:
                if not (base / name).exists():
                    raise SnapshotValidationError(f"missing snapshot artifact {name!r}")
            for name, info in artifacts.items():
                path = base / name
                expected = info.get("sha256") if isinstance(info, dict) else None
                if expected and _sha256(path) != expected:
                    raise SnapshotValidationError(
                        f"snapshot artifact {name!r} failed its checksum"
                    )
            tensors = self._read_tensors(base / SNAPSHOT_TENSORS)
            with open(base / SNAPSHOT_META, "r") as handle:
                meta = json.load(handle)
            staged["plastic_state"] = {**meta, **tensors}
            with open(base / SNAPSHOT_POLICY, "r") as handle:
                staged["policy_state"] = json.load(handle)
            with open(base / SNAPSHOT_CONFIG, "r") as handle:
                staged["config"] = json.load(handle)
            staged["database"] = self._validate_database(base / SNAPSHOT_DB,
                                                         snapshot.get("metadata", {}))
        self._validate_state_against_adapter(droid, staged["plastic_state"])
        self._validate_policy(staged["policy_state"])
        return staged

    @staticmethod
    def _validate_database(path: Path, metadata: Dict[str, Any]) -> Path:
        if not path.exists():
            raise SnapshotValidationError("snapshot database artifact is missing")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise SnapshotValidationError(f"snapshot database is corrupt: {integrity}")
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = [name for name in _REQUIRED_TABLES if name not in tables]
            if missing:
                raise SnapshotValidationError(
                    f"snapshot database lacks tables: {', '.join(missing)}"
                )
            rows = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            expected = (metadata.get("database") or {}).get("fact_rows")
            if expected is None:
                expected = metadata.get("fact_count")
            if expected is not None and int(expected) != int(rows):
                raise SnapshotValidationError(
                    f"snapshot database has {rows} facts but the manifest declares {expected}"
                )
        finally:
            conn.close()
        return path

    @staticmethod
    def _validate_state_against_adapter(droid, plastic_state: Dict[str, Any]) -> None:
        memory = getattr(droid, "memory", None)
        if memory is None or not hasattr(memory, "state_dict"):
            return
        try:
            current = memory.state_dict()
        except Exception:
            return
        for key, value in plastic_state.items():
            if key not in current:
                continue
            live = current[key]
            if isinstance(live, torch.Tensor) and isinstance(value, torch.Tensor):
                snap_shape, live_shape = tuple(value.shape), tuple(live.shape)
                if len(snap_shape) != len(live_shape):
                    raise SnapshotValidationError(
                        f"tensor {key!r} has rank {len(snap_shape)} in the snapshot "
                        f"but {len(live_shape)} live"
                    )
                # Capacity buffers (key/ring registries) legitimately grow and
                # shrink with use, so leading axes may differ. Rank and every
                # trailing dimension must still agree; the adapter's own
                # load_state_dict is the final authority on the rest, and a
                # failure there rolls the live state back (see _activate).
                if len(snap_shape) >= 2 and snap_shape[1:] != live_shape[1:]:
                    raise SnapshotValidationError(
                        f"tensor {key!r} has shape {snap_shape} in the snapshot "
                        f"but {live_shape} live"
                    )
            elif type(live) is not type(value):
                raise SnapshotValidationError(
                    f"snapshot field {key!r} has type {type(value).__name__} "
                    f"but {type(live).__name__} live"
                )

    @staticmethod
    def _validate_policy(policy_state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(policy_state, dict) or not policy_state:
            return
        probs = policy_state.get("probs")
        if probs is not None and not isinstance(probs, dict):
            raise SnapshotValidationError("retrieval policy probs must be an object")
        epsilon = policy_state.get("epsilon")
        if epsilon is not None:
            try:
                epsilon = float(epsilon)
            except (TypeError, ValueError) as exc:
                raise SnapshotValidationError("retrieval policy epsilon is not numeric") from exc
            if not 0.0 <= epsilon <= 1.0:
                raise SnapshotValidationError("retrieval policy epsilon is out of range")

    def restore_snapshot(self, droid, version_id: str) -> bool:
        """Stage, validate, then activate. Live state survives any failure."""
        self.last_error = None
        snapshot = self._resolve_snapshot(droid, version_id)
        if snapshot is None:
            self.last_error = f"snapshot {version_id!r} not found"
            return False
        try:
            staged = self._stage(droid, snapshot)
        except Exception as exc:
            self.last_error = str(exc)
            return False
        return self._activate(droid, staged)

    def _activate(self, droid, staged: Dict[str, Any]) -> bool:
        memory = getattr(droid, "memory", None)
        live_plastic = {}
        if memory is not None and hasattr(memory, "state_dict"):
            try:
                live_plastic = clone_state(memory.state_dict())
            except Exception:
                live_plastic = {}
        live_policy = None
        policy = getattr(droid, "retrieval_policy", None)
        if policy is not None and hasattr(policy, "state_dict"):
            try:
                live_policy = clone_state(policy.state_dict())
            except Exception:
                live_policy = None
        store = getattr(droid, "knowledge", None)
        safety_db = None
        if staged.get("database") is not None and isinstance(
            getattr(store, "conn", None), sqlite3.Connection
        ):
            try:
                handle, safety_db = tempfile.mkstemp(prefix="pre-restore-", suffix=".db")
                os.close(handle)
                self._write_database(Path(safety_db), store)
            except Exception as exc:
                self.last_error = f"could not take a safety backup: {exc}"
                return False
        try:
            if memory is not None and hasattr(memory, "load_state_dict"):
                if staged["plastic_state"]:
                    memory.load_state_dict(clone_state(staged["plastic_state"]))
            if staged.get("policy_state") and policy is not None:
                if hasattr(policy, "load_state_dict"):
                    policy.load_state_dict(staged["policy_state"])
            if staged.get("database") is not None:
                source = sqlite3.connect(f"file:{staged['database']}?mode=ro", uri=True)
                try:
                    source.backup(store.conn)
                    store.conn.commit()
                finally:
                    source.close()
                self._invalidate_store(store)
            metadata = staged.get("metadata", {})
            if "step_count" in metadata:
                droid.step_count = int(metadata["step_count"])
            droid.current_version = metadata.get("version_id", getattr(droid, "current_version", None))
            return True
        except Exception as exc:
            self.last_error = str(exc)
            try:
                if live_plastic and hasattr(memory, "load_state_dict"):
                    memory.load_state_dict(clone_state(live_plastic))
                if live_policy and policy is not None and hasattr(policy, "load_state_dict"):
                    policy.load_state_dict(live_policy)
                if safety_db and isinstance(getattr(store, "conn", None), sqlite3.Connection):
                    source = sqlite3.connect(f"file:{safety_db}?mode=ro", uri=True)
                    try:
                        source.backup(store.conn)
                        store.conn.commit()
                    finally:
                        source.close()
                    self._invalidate_store(store)
            except Exception:
                pass
            return False
        finally:
            if safety_db:
                try:
                    os.remove(safety_db)
                except OSError:
                    pass

    @staticmethod
    def _invalidate_store(store) -> None:
        store._vec_cache = None
        store._id_cache = None
        store._index_signature = None
        store.last_retrieval_trace = {}
        try:
            store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Rollback / branch / listing
    # ------------------------------------------------------------------ #
    def rollback(self, droid, version_id: str, pre_rollback_version: Optional[str] = None) -> Dict[str, Any]:
        """Reversible rollback: snapshot the live state, then restore."""
        if self._resolve_snapshot(droid, version_id) is None:
            self.last_error = f"snapshot {version_id!r} not found"
            return {"restored": False, "version_id": version_id,
                    "pre_rollback_version": None, "error": self.last_error}
        pre_id = pre_rollback_version or f"{version_id}__pre-rollback-{int(time.time() * 1000)}"
        pre = self.create_snapshot(droid, pre_id, parent_version=getattr(droid, "current_version", None))
        restored = self.restore_snapshot(droid, version_id)
        return {
            "restored": bool(restored),
            "version_id": version_id,
            "pre_rollback_version": pre["metadata"]["version_id"],
            "error": self.last_error,
        }

    def list_snapshots(self, droid=None) -> List[Dict]:
        """Metadata for in-memory and on-disk snapshots, oldest first."""
        bases = self._candidate_bases(droid) if droid is not None else (
            [self.base_dir] if self.base_dir is not None else []
        )
        if bases:
            for base in bases:
                if base is None or not base.exists():
                    continue
                for directory in base.iterdir():
                    manifest = directory / SNAPSHOT_MANIFEST
                    if not manifest.is_file():
                        continue
                    version_id = directory.name
                    if version_id in self.snapshots:
                        continue
                    try:
                        snapshot = self._snapshot_from_disk(directory)
                    except Exception:
                        continue
                    self.snapshots[version_id] = snapshot
        result = [clone_state(snap.get("metadata", {})) for snap in self.snapshots.values()]
        result.sort(key=lambda meta: meta.get("timestamp", 0))
        return result

    def branch_from(self, version_id: str, new_version_id: str,
                    droid=None) -> Optional[Dict]:
        """Create an independent copy: cloned tensors and copied artifacts."""
        base = self.snapshots.get(version_id)
        if base is None and droid is not None:
            base = self._resolve_snapshot(droid, version_id)
        if base is None:
            return None
        new_version_id = str(new_version_id)
        source_dir = base.get("path")
        artifacts: Dict[str, Any] = {}
        target_dir = None
        if base.get("persisted") and source_dir and droid is not None:
            target_dir = self._base_for(droid) / _safe_component(new_version_id)
            source = Path(source_dir)
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.copytree(source, target_dir)
            # No shared mutable artifact: regenerate the manifest with the new id.
            with open(target_dir / SNAPSHOT_MANIFEST, "r") as handle:
                manifest = json.load(handle)
            manifest["version_id"] = new_version_id
            manifest["parent_version"] = version_id
            manifest["timestamp"] = time.time()
            artifacts = manifest.get("artifacts", {})
            for name, info in artifacts.items():
                if isinstance(info, dict):
                    info["path"] = str(target_dir / name)
            _atomic_write_json(target_dir / SNAPSHOT_MANIFEST, manifest)
        metadata = {
            "version_id": new_version_id,
            "timestamp": time.time(),
            "step_count": base.get("metadata", {}).get("step_count", 0),
            "fact_count": base.get("metadata", {}).get("fact_count", 0),
            "parent_version": version_id,
            "branch_from": version_id,
            "persisted": bool(target_dir),
            "path": str(target_dir) if target_dir else None,
            "artifacts": artifacts,
        }
        new_snap = {
            "plastic_state": clone_state(base.get("plastic_state", {})),
            "policy_state": clone_state(base.get("policy_state")),
            "config": clone_state(base.get("config")),
            "metadata": metadata,
            "version": new_version_id,
            "version_id": new_version_id,
            "timestamp": metadata["timestamp"],
            "step_count": metadata["step_count"],
            "fact_count": metadata["fact_count"],
            "persisted": metadata["persisted"],
            "path": metadata["path"],
            "artifacts": artifacts,
        }
        self._remember(new_snap)
        return new_snap


class MergeEngine:
    """Multi-profile memory fusion with conflict detection and resolution."""

    POLICIES = ("keep_newest", "keep_highest_novelty", "keep_both_as_alternatives", "require_manual")

    @staticmethod
    def validate_policy(policy: str) -> str:
        """Return ``policy`` unchanged, or raise before any write for an unknown value."""
        if policy not in MergeEngine.POLICIES:
            raise ValueError(
                f"unknown merge policy {policy!r}; expected one of {', '.join(MergeEngine.POLICIES)}"
            )
        return policy
    
    @staticmethod
    def detect_conflicts(store_a, store_b) -> List[Dict]:
        """Detect slot clashes between two stores using SVO logic."""
        # Simplified: use existing _is_contradiction logic from droid.py if available
        # For now, detect by checking similar facts with different objects
        conflicts = []
        try:
            # Get all facts from both stores
            facts_a = store_a.get_audit_trail()
            facts_b = store_b.get_audit_trail()
            # For each fact in A, find similar in B
            for fa in facts_a:
                if fa.get("superseded"):
                    continue
                # Find similar in B via embedding? For simplicity, use text overlap
                for fb in facts_b:
                    if fb.get("superseded"):
                        continue
                    # Check if same subject-predicate but different object
                    # Use simple heuristic: high overlap but not identical
                    # Reuse _svo_slots if available
                    try:
                        from src.model.droid import _svo_slots, _is_contradiction
                        if _is_contradiction(fa["text"], fb["text"]):
                            conflicts.append({
                                "type": "SlotClash" if "capital" in fa["text"].lower() or "is" in fa["text"].lower() else "Polarity",
                                "fact_a": fa,
                                "fact_b": fb,
                                "subject": _svo_slots(fa["text"])[0],
                            })
                    except:
                        # Fallback: check if texts share subject but differ
                        if fa["text"][:20].lower() == fb["text"][:20].lower() and fa["text"] != fb["text"]:
                            conflicts.append({
                                "type": "SlotClash",
                                "fact_a": fa,
                                "fact_b": fb,
                            })
        except Exception as e:
            print(f"Conflict detection error: {e}")
        return conflicts

    @staticmethod
    def merge_memories(store_a, store_b, policy: str = "require_manual") -> Dict:
        """Merge two Droid memories by unioning knowledge stores and fusing plastic states."""
        # Reject an unknown policy before any conflict is resolved or written.
        MergeEngine.validate_policy(policy)
        conflicts = MergeEngine.detect_conflicts(store_a, store_b)
        
        # Resolve per policy
        resolutions = []
        for conflict in conflicts:
            fa = conflict["fact_a"]
            fb = conflict["fact_b"]
            if policy == "keep_newest":
                # Keep later timestamp
                keep = fa if fa.get("timestamp",0) > fb.get("timestamp",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"keep": keep["id"], "discard": discard["id"], "policy": policy})
            elif policy == "keep_highest_novelty":
                keep = fa if fa.get("novelty",0) > fb.get("novelty",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"keep": keep["id"], "discard": discard["id"], "policy": policy})
            elif policy == "keep_both_as_alternatives":
                resolutions.append({"keep": [fa["id"], fb["id"]], "policy": policy, "alternative_of": True})
            else:  # require_manual
                resolutions.append({"conflict": conflict, "policy": "manual_review_needed"})
        
        return {
            "conflicts": conflicts,
            "resolutions": resolutions,
            "policy": policy,
            "conflict_count": len(conflicts),
        }

    @staticmethod
    def fuse_plastic(rtu_a, rtu_b, weight_a: float = 0.5, weight_b: float = 0.5) -> Dict:
        """Fuse plastic associative states by weighted averaging of S matrices and trace vectors."""
        try:
            # Weighted average of S
            S_a = rtu_a.S if hasattr(rtu_a, 'S') else torch.zeros(4,96,96)
            S_b = rtu_b.S if hasattr(rtu_b, 'S') else torch.zeros(4,96,96)
            # Handle shape mismatches
            if S_a.shape != S_b.shape:
                # Use smaller shape or pad
                min_shape = (min(S_a.shape[0], S_b.shape[0]), min(S_a.shape[1], S_b.shape[1]), min(S_a.shape[2], S_b.shape[2]))
                S_a = S_a[:min_shape[0], :min_shape[1], :min_shape[2]]
                S_b = S_b[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            fused_S = S_a * weight_a + S_b * weight_b
            
            # Fuse h_trace
            h_a = rtu_a.h_trace if hasattr(rtu_a, 'h_trace') else rtu_a.states if hasattr(rtu_a, 'states') else torch.zeros(384)
            h_b = rtu_b.h_trace if hasattr(rtu_b, 'h_trace') else rtu_b.states if hasattr(rtu_b, 'states') else torch.zeros(384)
            fused_h = h_a * weight_a + h_b * weight_b
            
            # Fuse momentum
            mom_a = rtu_a.momentum if hasattr(rtu_a, 'momentum') else torch.zeros_like(fused_S)
            mom_b = rtu_b.momentum if hasattr(rtu_b, 'momentum') else torch.zeros_like(fused_S)
            fused_mom = mom_a * weight_a + mom_b * weight_b
            
            return {
                "S": fused_S,
                "h_trace": fused_h,
                "momentum": fused_mom,
                "weight_a": weight_a,
                "weight_b": weight_b,
            }
        except Exception as e:
            return {"error": str(e), "S": None}


class ContradictionResolver:
    """Detection and resolution of conflicting facts."""
    
    def detect_contradictions(self, facts: List[Dict]) -> List[Dict]:
        """Detect contradictions in list of facts."""
        contradictions = []
        for i in range(len(facts)):
            for j in range(i+1, len(facts)):
                try:
                    from src.model.droid import _is_contradiction, _svo_slots
                    if _is_contradiction(facts[i]["text"], facts[j]["text"]):
                        s_new, p_new = _svo_slots(facts[i]["text"])
                        s_old, p_old = _svo_slots(facts[j]["text"])
                        contradictions.append({
                            "fact_a": facts[i],
                            "fact_b": facts[j],
                            "subject": s_new,
                            "predicate": p_new,
                            "type": "SlotClash" if p_new and "of" in p_new else "PolarityInversion"
                        })
                except:
                    continue
        return contradictions

    def resolve(self, contradictions: List[Dict], policy: str = "supersede_newest") -> List[Dict]:
        """Resolve contradictions per policy."""
        resolutions = []
        for c in contradictions:
            fa = c["fact_a"]
            fb = c["fact_b"]
            if policy == "supersede_newest":
                keep = fa if fa.get("timestamp",0) > fb.get("timestamp",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"action": "supersede", "keep": keep["id"], "discard": discard["id"]})
            elif policy == "supersede_highest_confidence":
                keep = fa if fa.get("novelty",0) > fb.get("novelty",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"action": "supersede", "keep": keep["id"], "discard": discard["id"]})
            elif policy == "mark_both_review":
                resolutions.append({"action": "mark_review", "ids": [fa["id"], fb["id"]]})
            elif policy == "keep_as_alternatives":
                resolutions.append({"action": "keep_both", "ids": [fa["id"], fb["id"]], "alternative": True})
        return resolutions


class SemanticStore:
    """Distilled proposition store separate from episodic - KnowledgeStore extension."""
    
    def __init__(self, knowledge_store):
        self.ks = knowledge_store

    def distill(self, similarity_threshold: float = 0.85) -> List[Dict]:
        """Distill episodic facts into semantic propositions by clustering similar facts."""
        # Get all active facts
        facts = [f for f in self.ks.get_audit_trail() if not f.get("superseded")]
        if not facts:
            return []
        
        # Simple clustering by embedding similarity
        # For now, cluster by text similarity heuristic if no embeddings
        try:
            # Get embeddings
            vecs, ids = self.ks._load_dense()
            if len(ids) == 0:
                return []
            
            # Build id -> fact mapping
            id_to_fact = {f["id"]: f for f in facts}
            
            # Cluster: group facts with cosine >= threshold
            clusters = []
            visited = set()
            for i, fid in enumerate(ids):
                if fid in visited or fid not in id_to_fact:
                    continue
                cluster = [fid]
                visited.add(fid)
                # Find similar
                sims = torch.mv(vecs, vecs[i])
                for j, fid2 in enumerate(ids):
                    if fid2 in visited or fid2 not in id_to_fact:
                        continue
                    if sims[j].item() >= similarity_threshold:
                        cluster.append(fid2)
                        visited.add(fid2)
                clusters.append(cluster)
            
            # For each cluster, produce one semantic proposition (consensus)
            semantic_facts = []
            for cluster in clusters:
                if not cluster:
                    continue
                # Consensus: pick most central fact or longest?
                # For simplicity, pick first fact's text as consensus
                # In real implementation, would resolve contradictions
                cluster_facts = [id_to_fact[fid] for fid in cluster if fid in id_to_fact]
                if not cluster_facts:
                    continue
                # Pick fact with highest novelty or access_count as representative
                rep = max(cluster_facts, key=lambda f: f.get("novelty",0) + f.get("access_count",1)*0.1)
                # Create semantic fact
                # Need embedding for semantic fact - use average of cluster embeddings
                cluster_indices = [ids.index(fid) for fid in cluster if fid in ids]
                if cluster_indices:
                    avg_vec = vecs[cluster_indices].mean(dim=0)
                else:
                    avg_vec = torch.randn(self.ks.dim)
                
                # Check for contradictions within cluster and resolve
                resolver = ContradictionResolver()
                contras = resolver.detect_contradictions(cluster_facts)
                # If contradictions, pick highest confidence (novelty)
                if contras:
                    # Resolve by keeping highest novelty
                    rep = max(cluster_facts, key=lambda f: f.get("novelty",0))
                
                sid = self.ks.insert_semantic_fact(
                    text=rep["text"],
                    source_cluster_ids=cluster,
                    confidence=0.8,  # heuristic
                    vec=avg_vec
                )
                semantic_facts.append({
                    "id": sid,
                    "text": rep["text"],
                    "source_cluster_ids": cluster,
                    "confidence": 0.8,
                })
            
            return semantic_facts
        except Exception as e:
            print(f"Distillation error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_semantic(self, query_vec: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """Retrieve from semantic store."""
        return self.ks.recall_semantic(query_vec, top_k=top_k)

    def incremental_distill(self, new_fact_ids: List[int]) -> List[Dict]:
        """Only re-distill clusters overlapping with new facts."""
        # For simplicity, just call full distill
        # In optimized version, would only re-cluster affected areas
        return self.distill()


class RetrievalPolicy:
    """Epsilon-greedy bandit per query type for adaptive retrieval strategy.

    Classification is deterministic and documented (highest precedence first):

    1. ``ambiguous`` — empty/whitespace queries and underspecified ones such as
       ``"what?"``. Punctuation never promotes a query out of this class.
    2. ``temporal`` — explicit time words (``when``, ``before``, ``latest``...).
    3. ``relational`` — comparison/why/between questions.
    4. ``definitional`` — ``what is``/``define``/``meaning`` questions about a
       topic.
    5. ``specific_fact`` — everything else with enough content words.

    Exploration uses a *local* ``random.Random`` seeded per policy, so repeated
    runs with the same seed produce the same exploration sequence and the
    classifier itself never depends on randomness.
    """

    CLASSIFIER_VERSION = 2
    STRATEGIES = (
        "episodic_only", "semantic_primary", "hybrid_rrf",
        "episodic_recency_weighted", "contextual_boosted",
    )
    QUERY_TYPES = ("definitional", "specific_fact", "temporal", "relational", "ambiguous")

    _TEMPORAL = ("when", "before", "after", "recent", "latest", "history",
                 "yesterday", "today", "last week", "last month", "earlier")
    _RELATIONAL = ("how does", "how do", "why", "relationship", "between", "compare",
                   "versus", " vs ", "difference")
    _DEFINITIONAL = ("what is", "what are", "what's", "define", "definition", "meaning")
    _UNDERSPECIFIED = {"what", "what?", "who", "who?", "which", "which?", "why?",
                       "how", "how?", "huh", "?", "hmm"}

    def __init__(self, epsilon: float = 0.1, seed: Optional[int] = None,
                 exploration: str = "epsilon_greedy"):
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if exploration not in ("epsilon_greedy", "off"):
            raise ValueError(f"unknown exploration mode {exploration!r}")
        self.epsilon = float(epsilon)
        self.exploration = exploration
        self.seed = seed
        self._rng = random.Random(seed)
        self.strategies = list(self.STRATEGIES)
        self.query_types = list(self.QUERY_TYPES)
        self.probs: Dict[str, Dict[str, float]] = {
            qt: {s: 1.0 / len(self.strategies) for s in self.strategies}
            for qt in self.query_types
        }
        self.defaults = {
            "definitional": "semantic_primary",
            "specific_fact": "hybrid_rrf",
            "temporal": "episodic_recency_weighted",
            "relational": "hybrid_rrf",
            "ambiguous": "hybrid_rrf",
        }
        self.rewards: Dict[str, Dict[str, List[float]]] = {
            qt: {s: [] for s in self.strategies} for qt in self.query_types
        }
        self.exploration_count = 0
        self.selection_count = 0

    # -- classification -------------------------------------------------- #
    def classify_query(self, query: str) -> str:
        """Deterministic classification with documented precedence."""
        raw = (query or "").strip()
        normalized = raw.lower().strip(" .!,")
        words = normalized.split()
        core = normalized.rstrip("?").strip()
        # 1. Ambiguous: empty, whitespace, or an underspecified question.
        if not normalized or not core:
            return "ambiguous"
        if normalized in self._UNDERSPECIFIED or core in self._UNDERSPECIFIED:
            return "ambiguous"
        if len(words) < 3 and len(core.split()) <= 1:
            return "ambiguous"
        # Punctuation alone never makes a question specific.
        if core in {"what", "who", "which", "how", "why", "huh", "hmm"}:
            return "ambiguous"
        # 2. Temporal.
        if any(token in f" {normalized} " for token in self._TEMPORAL):
            return "temporal"
        # 3. Relational.
        if any(token in normalized for token in self._RELATIONAL):
            return "relational"
        # 4. Definitional.
        if any(normalized.startswith(token) for token in self._DEFINITIONAL):
            return "definitional"
        # 5. Very short questions with no recognised pattern are underspecified.
        if len(words) <= 2:
            return "ambiguous"
        # 6. Specific fact.
        return "specific_fact"

    # -- selection ------------------------------------------------------- #
    def select_strategy(self, query_type: str) -> str:
        """Select a strategy per query type with seeded epsilon-greedy exploration."""
        if query_type not in self.probs:
            query_type = "specific_fact"
        self.selection_count += 1
        if self.exploration == "off":
            return self.defaults[query_type]
        if self._rng.random() < self.epsilon:
            self.exploration_count += 1
            return self._rng.choice(self.strategies)
        probs = self.probs[query_type]
        best = max(probs, key=lambda s: probs[s])
        default = self.defaults[query_type]
        if abs(probs.get(default, 0) - probs[best]) < 0.05:
            return default
        return best

    def update(self, query_type: str, strategy: str, reward: float):
        """Nudge the strategy distribution with a bounded reward in [-1, 1]."""
        if query_type not in self.probs:
            query_type = "specific_fact"
        if strategy not in self.probs[query_type]:
            return
        reward = max(-1.0, min(1.0, float(reward)))
        self.rewards[query_type][strategy].append(reward)
        if len(self.rewards[query_type][strategy]) > 100:
            self.rewards[query_type][strategy] = self.rewards[query_type][strategy][-100:]
        current = self.probs[query_type][strategy]
        new_prob = max(0.05, min(0.95, current + 0.1 * reward))
        self.probs[query_type][strategy] = new_prob
        total = sum(self.probs[query_type].values())
        for name in self.probs[query_type]:
            self.probs[query_type][name] /= total

    # -- persistence ----------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "classifier_version": self.CLASSIFIER_VERSION,
            "probs": self.probs,
            "rewards": self.rewards,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "exploration": self.exploration,
            "exploration_count": self.exploration_count,
            "selection_count": self.selection_count,
        }

    def load_state_dict(self, data: Dict[str, Any]):
        """Restore a state_dict captured by ``state_dict()`` (snapshot restore)."""
        if not isinstance(data, dict):
            return self
        probs = data.get("probs")
        if isinstance(probs, dict):
            for query_type, values in probs.items():
                if query_type in self.probs and isinstance(values, dict):
                    for strategy, value in values.items():
                        if strategy in self.probs[query_type]:
                            try:
                                self.probs[query_type][strategy] = float(value)
                            except (TypeError, ValueError):
                                pass
        rewards = data.get("rewards")
        if isinstance(rewards, dict):
            for query_type, values in rewards.items():
                if query_type in self.rewards and isinstance(values, dict):
                    for strategy, entries in values.items():
                        if strategy in self.rewards[query_type] and isinstance(entries, list):
                            self.rewards[query_type][strategy] = [
                                float(v) for v in entries if isinstance(v, (int, float))
                            ]
        if isinstance(data.get("epsilon"), (int, float)):
            self.epsilon = float(data["epsilon"])
        if data.get("seed") is not None:
            self.seed = data["seed"]
        if data.get("exploration") in ("epsilon_greedy", "off"):
            self.exploration = data["exploration"]
        self.exploration_count = int(data.get("exploration_count", self.exploration_count) or 0)
        self.selection_count = int(data.get("selection_count", self.selection_count) or 0)
        self._rng = random.Random(self.seed)
        for _ in range(self.selection_count):
            self._rng.random()
        return self

    def persist(self, path: str):
        """Persist the learned policy and its exploration seed per profile."""
        data = self.state_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Restore a persisted policy; unknown files are ignored, not invented."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return self
        if not isinstance(data, dict):
            return self
        probs = data.get("probs")
        if isinstance(probs, dict):
            for query_type, values in probs.items():
                if query_type in self.probs and isinstance(values, dict):
                    for strategy, value in values.items():
                        if strategy in self.probs[query_type]:
                            try:
                                self.probs[query_type][strategy] = float(value)
                            except (TypeError, ValueError):
                                pass
        rewards = data.get("rewards")
        if isinstance(rewards, dict):
            for query_type, values in rewards.items():
                if query_type in self.rewards and isinstance(values, dict):
                    for strategy, entries in values.items():
                        if strategy in self.rewards[query_type] and isinstance(entries, list):
                            self.rewards[query_type][strategy] = [
                                float(v) for v in entries if isinstance(v, (int, float))
                            ]
        if isinstance(data.get("epsilon"), (int, float)):
            self.epsilon = float(data["epsilon"])
        if data.get("seed") is not None:
            self.seed = data["seed"]
        if data.get("exploration") in ("epsilon_greedy", "off"):
            self.exploration = data["exploration"]
        self.exploration_count = int(data.get("exploration_count", self.exploration_count) or 0)
        self.selection_count = int(data.get("selection_count", self.selection_count) or 0)
        # Restore the exploration stream so a restarted profile continues the
        # same sequence instead of re-rolling from the start.
        self._rng = random.Random(self.seed)
        for _ in range(self.selection_count):
            self._rng.random()
        return self


class ContinuousStream:
    """Background learning from interaction stream with rate limiting and novelty filtering."""
    
    def __init__(self, droid, max_facts_per_minute: int = 10, surprise_threshold: float = 0.6):
        self.droid = droid
        self.max_fpm = max_facts_per_minute
        self.surprise_threshold = surprise_threshold
        self.queue = queue.Queue()
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.stats = {
            "facts_processed": 0,
            "facts_absorbed": 0,
            "dropped": 0,
            "queue_depth": 0,
        }
        self.lock = threading.Lock()

    def start(self, log_source: Optional[str] = None):
        """Start background thread."""
        if self.running:
            return
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # If log_source provided, ingest it
        if log_source:
            self.ingest_file(log_source)

    def _run_loop(self):
        """Background loop processing queue with rate limiting."""
        last_minute = time.time()
        facts_this_minute = 0
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Rate limiting: check if we exceeded max_fpm
                now = time.time()
                if now - last_minute >= 60:
                    last_minute = now
                    facts_this_minute = 0
                
                if facts_this_minute >= self.max_fpm:
                    time.sleep(1.0)
                    continue
                
                try:
                    text = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                with self.lock:
                    self.stats["queue_depth"] = self.queue.qsize()
                
                # Process text: extract propositions and teach with higher surprise threshold
                # Simplified: treat each sentence as proposition
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 10:
                        continue
                    with self.lock:
                        self.stats["facts_processed"] += 1
                    
                    # Check surprise threshold - need embedding
                    try:
                        if hasattr(self.droid, 'anchor') and hasattr(self.droid, 'memory'):
                            emb = self.droid.anchor.embed(sent)
                            if emb.dim() > 1:
                                emb = emb.squeeze(0)
                            _, surprise = self.droid.memory.compute_surprise(emb)
                            if surprise < self.surprise_threshold:
                                continue
                    except:
                        pass
                    
                    # Teach
                    try:
                        result = self.droid.teach(sent, source="continuous_stream")
                        with self.lock:
                            self.stats["facts_absorbed"] += result.get("propositions", 1)
                        facts_this_minute += 1
                    except Exception as e:
                        print(f"Continuous stream teach error: {e}")
                
                self.queue.task_done()
                
            except Exception as e:
                print(f"Continuous stream loop error: {e}")
                time.sleep(0.5)

    def ingest(self, text: str):
        """Ingest text into stream queue."""
        if not self.running or self.paused:
            # If not running, queue anyway for later
            pass
        
        # Rate limiting: if queue too large, drop
        if self.queue.qsize() > 100:
            with self.lock:
                self.stats["dropped"] += 1
            return False
        
        self.queue.put(text)
        with self.lock:
            self.stats["queue_depth"] = self.queue.qsize()
        return True

    def ingest_file(self, path: str):
        """Ingest chat log file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.ingest(line)
        except Exception as e:
            print(f"Failed to ingest file {path}: {e}")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "running": self.running and not self.paused,
                "paused": self.paused,
                "facts_processed": self.stats["facts_processed"],
                "facts_absorbed": self.stats["facts_absorbed"],
                "queue_depth": self.stats["queue_depth"],
                "dropped": self.stats["dropped"],
            }

    def get_stats(self) -> Dict:
        return self.get_status()
