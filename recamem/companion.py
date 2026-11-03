"""Companion surface: measured state, real actions and provenance-backed checkpoints.

This module is a presentation/adapter layer only. It never invents a metric:
every field of ``measured_state`` comes from an engine call, and an unavailable
signal is reported as ``None`` with a reason. Pet attributes are a pure function
of those signals. Training jobs launch a real subprocess and report only metrics
parsed from its output.
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ACTIONS = ("pet", "treat", "feed", "scold", "correct", "sleep", "play")
REQUIRED_PROVENANCE = ("config", "seed", "dataset", "commit", "command", "gates")
TRAIN_ARGS = {
    "lm_tokens": "--lm-tokens", "data_dir": "--data-dir", "out_dir": "--out-dir",
    "d_model": "--d-model", "n_layers": "--n-layers", "n_heads": "--n-heads",
    "d_ff": "--d-ff", "vocab_size": "--vocab-size", "seq_len": "--seq-len",
    "lr": "--lr", "batch_size": "--batch-size", "grad_accum": "--grad-accum",
    "checkpoint_every": "--checkpoint-every", "seed": "--seed", "core": "--core",
    "resume": "--resume",
}


class CompanionActionError(ValueError):
    """An action that cannot be mapped to a real engine call."""


class CheckpointError(ValueError):
    """A checkpoint that must not be presented as an eligible trained model."""


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a missing signal is reported, not invented
        return default


def _last_sleep_at(engine) -> Optional[float]:
    logs = _safe(lambda: list(getattr(engine, "logs", [])), []) or []
    for entry in reversed(logs):
        message = str(entry.get("message", "")).lower()
        if "sleep" in message or "consolidat" in message:
            value = entry.get("timestamp")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def measured_state(engine, *, anchor: str, backend: str, profile: str,
                   selected_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Read only measured engine signals; never substitute a local counter."""
    knowledge = getattr(engine, "knowledge", None)
    total = _safe(lambda: knowledge.total_count())
    active = _safe(lambda: knowledge.active_count())
    health = _safe(lambda: engine.get_memory_health(), {}) or {}
    ttt = _safe(lambda: engine.ttt_status(), {}) or {}
    retrieval = _safe(lambda: knowledge.retrieval_health(), {}) or {}
    feedback = _safe(lambda: engine.get_feedback_log(), []) or []
    approvals = sum(1 for row in feedback if str(row.get("action", row.get("event", ""))).lower() in {"approve", "accepted"})
    rejections = sum(1 for row in feedback if str(row.get("action", row.get("event", ""))).lower() in {"reject", "rejected"})
    return {
        "anchor": anchor,
        "backend": backend,
        "profile": profile,
        "facts_total": total,
        "facts_active": active,
        "facts_retired": (total - active) if isinstance(total, int) and isinstance(active, int) else None,
        "feedback_counts": {"approve": approvals, "reject": rejections, "total": len(feedback)},
        "memory_health": health,
        "ttt_status": ttt,
        "retrieval_health": retrieval,
        "retrieval_hit_rate": retrieval.get("precision_at_k") if retrieval.get("available") else None,
        "last_sleep_at": _last_sleep_at(engine),
        "selected_checkpoint": selected_checkpoint,
        "measured_at": time.time(),
        "signals": ["knowledge.total_count", "knowledge.active_count", "get_memory_health",
                    "ttt_status", "get_feedback_log", "knowledge.retrieval_health", "engine.logs"],
    }


def derive_pet(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pet attributes are computed from measured signals, or reported unavailable."""
    active = state.get("facts_active")
    feedback = state.get("feedback_counts") or {}
    hit_rate = state.get("retrieval_hit_rate")
    last_sleep = state.get("last_sleep_at")

    knowledge = (min(1.0, float(active) / 20.0) if isinstance(active, (int, float)) else None)
    approvals = feedback.get("approve", 0)
    bond = min(1.0, float(approvals) / 10.0) if isinstance(approvals, (int, float)) else None
    energy = (1.0 / (1.0 + max(0.0, time.time() - last_sleep) / 86400.0)
              if isinstance(last_sleep, (int, float)) else None)
    if isinstance(hit_rate, (int, float)) and isinstance(bond, (int, float)):
        mood = round(min(1.0, 0.5 * float(hit_rate) + 0.5 * float(bond)), 4)
    elif isinstance(bond, (int, float)):
        mood = round(float(bond), 4)
    else:
        mood = None
    return {
        "mood": {"value": mood, "sources": ["retrieval_health.precision_at_k", "feedback_counts.approve"]},
        "energy": {"value": energy, "sources": ["last_sleep_at"]},
        "bond": {"value": bond, "sources": ["feedback_counts.approve"]},
        "knowledge": {"value": knowledge, "sources": ["facts_active"]},
        "note": "A null value means its source signal was unavailable; it is never a decorative counter.",
    }


def dispatch_action(engine, action: str, payload: Dict[str, Any], ctx: Dict[str, Any],
                    state_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Map a game action to a real engine call and return measured before/after."""
    if action not in ACTIONS:
        raise CompanionActionError(f"unknown action {action!r}; expected one of {ACTIONS}")
    text = str(payload.get("text") or "").strip()
    fact_id = payload.get("fact_id", ctx.get("last_fact_id"))
    before = state_fn()
    engine_call = ""
    result: Any

    if action in {"pet", "treat"}:
        if fact_id is None:
            raise CompanionActionError(f"{action} needs a recalled fact_id; nothing to reinforce")
        engine_call = f"approve({fact_id})"
        result = engine.approve(int(fact_id))
    elif action == "feed":
        if not text:
            raise CompanionActionError("feed needs non-empty text")
        engine_call = "teach(text, source='companion')"
        result = engine.teach(text, source="companion")
    elif action == "scold":
        if fact_id is None:
            raise CompanionActionError("scold needs a recalled fact_id")
        engine_call = f"reject({fact_id})"
        result = engine.reject(int(fact_id))
    elif action == "correct":
        if fact_id is None or not text:
            raise CompanionActionError("correct needs a recalled fact_id and replacement text")
        engine_call = f"correct({fact_id}, text)"
        result = engine.correct(int(fact_id), text)
    elif action == "sleep":
        engine_call = "sleep()"
        result = engine.sleep()
    else:  # play
        if not text:
            raise CompanionActionError("play needs a prompt")
        engine_call = "chat(text)"
        result = {"response": engine.chat(text)}

    after = state_fn()
    event_id = uuid.uuid4().hex[:12]
    event = {
        "id": event_id, "action": action, "engine_call": engine_call,
        "target": fact_id if action in {"pet", "treat", "scold", "correct"} else None,
        "timestamp": time.time(),
        "before": {"facts_active": before.get("facts_active"), "feedback_counts": before.get("feedback_counts")},
        "after": {"facts_active": after.get("facts_active"), "feedback_counts": after.get("feedback_counts")},
    }
    events = ctx.setdefault("events", [])
    events.append(event)
    if len(events) > 200:
        del events[:-200]
    return {"event": event, "result": result, "state": after}


# --------------------------------------------------------------------------- #
# Checkpoint provenance
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(directory: Path) -> Dict[str, Any]:
    """A checkpoint is eligible only with a manifest, provenance and measured gates."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    row: Dict[str, Any] = {
        "path": directory.as_posix(), "name": directory.name,
        "eligible": False, "reason": "", "held_out_bpb": None, "gates": {},
    }
    if not manifest_path.is_file():
        row["reason"] = "manifest.json missing"
        return row
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        row["reason"] = f"unreadable manifest: {exc}"
        return row
    files = manifest.get("files") or []
    for entry in files:
        rel = entry.get("path")
        target = directory / str(rel)
        if not target.is_file() or _sha256(target) != entry.get("sha256"):
            row["reason"] = f"checksum mismatch or missing file: {rel}"
            return row
    missing = [key for key in REQUIRED_PROVENANCE if key not in manifest]
    if missing:
        row["reason"] = "missing provenance: " + ", ".join(missing)
        return row
    bpb = manifest.get("held_out_bpb")
    gates = manifest.get("gates") or {}
    g2 = gates.get("G2")
    if not isinstance(bpb, (int, float)) or g2 not in (True, "passed"):
        row["reason"] = "no measured held-out BPB or G2 did not pass"
        return row
    row.update({
        "eligible": True, "reason": "provenance and measured gates present",
        "held_out_bpb": float(bpb), "gates": gates,
        "config": manifest.get("config"), "seed": manifest.get("seed"),
        "dataset": manifest.get("dataset"), "commit": manifest.get("commit"),
        "command": manifest.get("command"),
    })
    return row


def list_checkpoints(root: Path) -> List[Dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        return []
    rows = [inspect_checkpoint(path) for path in sorted(root.iterdir()) if path.is_dir()]
    return rows


def best_checkpoint(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible")]
    if not eligible:
        return {"status": "none", "reason": "no eligible checkpoint with provenance and measured gates",
                "checkpoints": len(rows)}
    winner = min(eligible, key=lambda row: (row["held_out_bpb"], row["name"]))
    return {"status": "ok", "checkpoint": winner, "eligible_count": len(eligible)}


# --------------------------------------------------------------------------- #
# Deterministic fixture corpus and checkpoint provenance
# --------------------------------------------------------------------------- #
def ensure_fixture_corpus(home: Path) -> Path:
    """A stdlib-only, deterministic corpus so a fresh install can run a smoke job.

    It is labelled ``dataset_source=fixture`` and must never be described as a
    language-learning result. Real training points ``data_dir`` at a real corpus.
    """
    directory = Path(home) / "companion-fixture"
    train = directory / "train.bin"
    if not train.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        rng = random.Random(1337)
        array.array("H", (rng.randrange(1024) for _ in range(8192))).tofile(train.open("wb"))
        array.array("H", (rng.randrange(1024) for _ in range(2048))).tofile(
            (directory / "val.bin").open("wb"))
        (directory / "meta.json").write_text(json.dumps({
            "vocab_size": 1024, "train_tokens": 8192, "val_tokens": 2048,
            "dataset_source": "fixture",
            "note": "random uint16 tokens; a pipeline smoke corpus, not language data",
        }, indent=2))
    return directory


def write_training_manifest(out_dir: Path, argv: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Attach provenance to a finished run; never fabricate an unmeasured BPB."""
    from recamem import hub

    out_dir = Path(out_dir)
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        return {"written": False, "reason": "report.json missing"}
    report = json.loads(report_path.read_text())
    bpb = report.get("val_bpb")
    gates_raw = report.get("gates") or {}
    gates = {key: ("passed" if value is True else "failed" if value is False else "not_evaluated")
             for key, value in gates_raw.items()}
    manifest_path = hub.write_manifest(out_dir)
    manifest = json.loads(manifest_path.read_text())
    data_dir = Path(str(params.get("data_dir") or ""))
    dataset = "fixture"
    if (data_dir / "meta.json").is_file():
        try:
            dataset = str(json.loads((data_dir / "meta.json").read_text()).get("dataset_source", "unknown"))
        except (OSError, ValueError):
            dataset = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(out_dir.parent),
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unavailable"
    manifest.update({
        "config": {key: report.get(key) for key in
                   ("d_model", "n_layers", "vocab_size", "seq_len", "params", "lm_tokens_target")},
        "seed": report.get("seed"),
        "dataset": dataset,
        "commit": commit,
        "command": " ".join(argv),
        "gates": gates,
        "held_out_bpb": float(bpb) if isinstance(bpb, (int, float)) else None,
        "source_report": "report.json",
        "note": "Generated from a real train_lm.py report. Gates not evaluated are not claimed.",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"written": True, "held_out_bpb": manifest["held_out_bpb"], "gates": gates}


# --------------------------------------------------------------------------- #
# Real training jobs
# --------------------------------------------------------------------------- #
class TrainingJobRunner:
    """Run ``scripts/train_lm.py`` as a subprocess and report only real output."""

    def __init__(self, repo_root: Path, updater: Callable[[str, Any], None],
                 home: Optional[Path] = None):
        self.repo_root = Path(repo_root)
        self.updater = updater
        self._lock = threading.Lock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._params: Dict[str, Dict[str, Any]] = {}
        self.home = Path(home) if home else Path(
            os.environ.get("RECAMEM_HOME") or (Path.home() / ".recamem"))

    def _emit(self, job_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            events = self._events.setdefault(job_id, [])
            event.setdefault("seq", len(events))
            event.setdefault("timestamp", time.time())
            events.append(event)
            if len(events) > 2000:
                del events[:-2000]

    def events(self, job_id: str, since: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events.get(job_id, []) if event["seq"] >= since]

    def start(self, job_id: str, params: Dict[str, Any]) -> None:
        script = self.repo_root / "scripts" / "train_lm.py"
        if not script.is_file():
            raise CompanionActionError(f"training script missing: {script}")
        unknown = sorted(set(params) - set(TRAIN_ARGS) - {"kind"})
        if unknown:
            raise CompanionActionError(f"unsupported training parameters: {unknown}")
        params = dict(params)
        data_dir = Path(str(params.get("data_dir") or ""))
        if not (data_dir / "train.bin").is_file():
            fixture = ensure_fixture_corpus(self.home)
            params["data_dir"] = fixture.as_posix()
            params.setdefault("vocab_size", 1024)
            self._emit(job_id, {
                "type": "notice",
                "line": "no usable corpus; using the deterministic fixture corpus "
                        "(dataset_source=fixture). This is a pipeline smoke run, "
                        "not a language-learning result.",
            })
        params.setdefault("out_dir", (self.home / "checkpoints" / job_id).as_posix())
        Path(str(params["out_dir"])).mkdir(parents=True, exist_ok=True)
        self._params[job_id] = params
        argv = [sys.executable, "-u", str(script)]
        for key, flag in TRAIN_ARGS.items():
            if key in params and params[key] is not None:
                argv += [flag, str(params[key])]
        target = int(params.get("lm_tokens") or 0)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        for secret in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(secret, None)
        self._emit(job_id, {"type": "started", "command": " ".join(argv)})
        try:
            proc = subprocess.Popen(argv, cwd=str(self.repo_root), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except OSError as exc:
            self._emit(job_id, {"type": "failed", "reason": str(exc)})
            self.updater(job_id, status="failed", error=str(exc))
            return
        with self._lock:
            self._processes[job_id] = proc
        self.updater(job_id, status="running")
        thread = threading.Thread(target=self._pump, args=(job_id, proc, target, argv), daemon=True)
        thread.start()

    def _pump(self, job_id: str, proc: subprocess.Popen, target: int,
              argv: List[str]) -> None:
        metrics: Dict[str, Any] = {}
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            self._emit(job_id, {"type": "log", "line": line})
            tokens = re.search(r"tokens=([\d,]+)", line)
            loss = re.search(r"loss=([\d.]+)", line)
            bpb = re.search(r"bpb=([\d.]+)", line)
            if tokens:
                seen = int(tokens.group(1).replace(",", ""))
                metrics["tokens_seen"] = seen
                progress = min(1.0, seen / target) if target > 0 else 0.0
                self.updater(job_id, progress=progress, tokens_seen=seen)
            if loss:
                metrics["loss"] = float(loss.group(1))
                self._emit(job_id, {"type": "metrics", **metrics})
            if bpb:
                metrics["held_out_bpb"] = float(bpb.group(1))
                self._emit(job_id, {"type": "metrics", **metrics})
        returncode = proc.wait()
        finished_at = time.time()
        with self._lock:
            self._processes.pop(job_id, None)
        status = "done" if returncode == 0 else "failed"
        manifest: Dict[str, Any] = {}
        checkpoint_path = None
        report: Dict[str, Any] = {}
        samples: List[str] = []
        if status == "done":
            params = self._params.get(job_id, {})
            out_dir_path = Path(str(params.get("out_dir") or ""))
            if out_dir_path.is_dir():
                checkpoint_path = str(out_dir_path)
            report_file = out_dir_path / "report.json"
            if report_file.is_file():
                try:
                    report = json.loads(report_file.read_text())
                except Exception:
                    pass
            samples_file = out_dir_path / "samples.txt"
            if samples_file.is_file():
                try:
                    samples = [line.strip() for line in samples_file.read_text().splitlines() if line.strip()][-10:]
                except Exception:
                    pass
            try:
                manifest = write_training_manifest(out_dir_path, argv, params)
            except Exception as exc:  # noqa: BLE001 - report it, never fake a pass
                manifest = {"written": False, "reason": f"{type(exc).__name__}: {exc}"}
            self._emit(job_id, {"type": "checkpoint", "manifest": manifest, "checkpoint_path": checkpoint_path})
        self._emit(job_id, {"type": "finished", "returncode": returncode, "metrics": metrics,
                            "report": report, "samples": samples})
        self.updater(job_id, status=status, finished_at=finished_at,
                     checkpoint_path=checkpoint_path,
                     result={**metrics, "manifest": manifest, "report": report, "samples": samples,
                             "out_dir": self._params.get(job_id, {}).get("out_dir"),
                             "checkpoint_path": checkpoint_path},
                     returncode=returncode)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._emit(job_id, {"type": "cancelled"})
        self.updater(job_id, status="cancelled", finished_at=time.time())
        return True
