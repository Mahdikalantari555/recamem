"""LoCoMo, LongMemEval-S and DMR benchmark runners.

Each dataset is fetched once into a cache outside the repository, with the
checksum recorded and verified before reuse. When the dataset cannot be
fetched, a committed offline fixture may be run and is labelled
``dataset_source=fixture``; otherwise the benchmark is reported ``blocked`` and
prints no metric. Real-model gates are never satisfied by a mock.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from recamem.paths import cache_dir, evals_dir

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "locomo": {
        "name": "LoCoMo",
        "metrics": ["recall_at_k", "answer_match"],
        "question_field": "question",
        "sources": [
            "https://codeload.github.com/snap-research/locomo/tar.gz/refs/heads/main",
        ],
        "fixture": "locomo_fixture.json",
    },
    "longmemeval": {
        "name": "LongMemEval-S",
        "metrics": ["recall_at_k", "answer_match"],
        "question_field": "question",
        "sources": [
            "https://codeload.github.com/xiaowu0162/LongMemEval/tar.gz/refs/heads/main",
        ],
        "fixture": "longmemeval_fixture.json",
    },
    "dmr": {
        "name": "DMR",
        "metrics": ["recall_at_k", "answer_match"],
        "question_field": "question",
        "sources": [
            "https://codeload.github.com/DeepMemoryRetrieval/dmr/tar.gz/refs/heads/main",
        ],
        "fixture": "dmr_fixture.json",
    },
}

CACHE_RECORD = "checksums.json"


def cache_root() -> Path:
    return cache_dir() / "benchmarks"


def fixture_path(name: str) -> Path:
    spec = BENCHMARKS[name]
    return evals_dir() / "benchmarks" / spec["fixture"]


def _cache_entry(name: str) -> Dict[str, Any]:
    record = cache_root() / CACHE_RECORD
    if record.exists():
        try:
            return json.loads(record.read_text()).get(name, {})
        except ValueError:
            return {}
    return {}


def _write_cache_entry(name: str, entry: Dict[str, Any]) -> None:
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    record = root / CACHE_RECORD
    data = json.loads(record.read_text()) if record.exists() else {}
    data[name] = entry
    record.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_dataset(name: str, *, timeout: float = 10.0) -> Dict[str, Any]:
    """Fetch a dataset once. Returns ``{status, path|reason, checksum}``."""
    spec = BENCHMARKS[name]
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}.json"
    entry = _cache_entry(name)
    if target.exists() and entry.get("checksum"):
        digest = sha256_bytes(target.read_bytes())
        if digest == entry["checksum"]:
            return {"status": "cached", "path": str(target), "checksum": digest}
    reasons: List[str] = []
    for url in spec["sources"]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                payload = response.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reasons.append(f"{url}: {exc}")
            continue
        try:
            json.loads(payload)
        except ValueError:
            # A release archive is not a question file; report it instead of
            # caching bytes that can never be parsed.
            reasons.append(f"{url}: payload is not a JSON question file")
            continue
        digest = sha256_bytes(payload)
        target.write_bytes(payload)
        _write_cache_entry(name, {"checksum": digest, "url": url, "bytes": len(payload)})
        return {"status": "fetched", "path": str(target), "checksum": digest, "url": url}
    return {"status": "blocked", "reason": "; ".join(reasons) or "no source configured"}


def load_dataset(name: str) -> Dict[str, Any]:
    """Load the cached dataset, else the fixture; never silently mix them."""
    fetched = fetch_dataset(name)
    if fetched["status"] in {"cached", "fetched"}:
        try:
            return {
                "status": "ok",
                "dataset_source": "real",
                "questions": _normalise(json.loads(Path(fetched["path"]).read_text())),
                "source": fetched.get("url", fetched["path"]),
                "checksum": fetched["checksum"],
            }
        except ValueError as exc:
            return {"status": "blocked", "reason": f"unparseable dataset: {exc}"}
    fixture = fixture_path(name)
    if fixture.exists():
        payload = json.loads(fixture.read_text())
        return {
            "status": "ok",
            "dataset_source": "fixture",
            "questions": _normalise(payload),
            "source": str(fixture),
            "fetch_reason": fetched.get("reason"),
        }
    return {"status": "blocked", "reason": fetched.get("reason", "unavailable")}


def _normalise(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("questions", payload.get("data", []))
    if not isinstance(payload, list):
        raise ValueError("expected a list of questions")
    questions = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("each question must be an object")
        questions.append(row)
    return questions


def deterministic_subset(questions: Sequence[Dict[str, Any]], *, seed: int = 0,
                         size: Optional[int] = None) -> List[Dict[str, Any]]:
    import random

    ordered = list(questions)
    rng = random.Random(seed)
    indexed = list(enumerate(ordered))
    rng.shuffle(indexed)
    chosen = indexed[: size if size is not None else len(indexed)]
    return [item for _, item in sorted(chosen, key=lambda pair: pair[0])]


def provenance(command: Optional[str] = None, *, seed: int = 0,
               subset_size: Optional[int] = None) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "commit": commit,
        "command": command or " ".join([sys.executable, *sys.argv]),
        "seed": seed,
        "subset_size": subset_size,
        "dependencies": {
            "torch": _version("torch"),
            "numpy": _version("numpy"),
            "pypdf": _version("pypdf"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": _cpu_count(),
        },
    }


def _version(module: str) -> Optional[str]:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def _cpu_count() -> Optional[int]:
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except Exception:
        return None


def run(name: str, engine=None, *, seed: int = 0, size: Optional[int] = None,
        command: Optional[str] = None, answer_contains: Optional[str] = None) -> Dict[str, Any]:
    """Run one benchmark and return a provenance-stamped report."""
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; expected {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    loaded = load_dataset(name)
    base = {
        "benchmark": spec["name"],
        "key": name,
        "metrics": spec["metrics"],
        "provenance": provenance(command, seed=seed, subset_size=size),
    }
    if loaded["status"] != "ok":
        base.update(status="blocked", reason=loaded.get("reason"), dataset_source=None)
        base["provenance"]["subset_size"] = size
        return base
    questions = deterministic_subset(loaded["questions"], seed=seed, size=size)
    base.update(
        dataset_source=loaded["dataset_source"],
        source=loaded.get("source"),
        questions=len(questions),
    )
    if engine is None:
        base.update(status="not_measured", reason="no engine supplied; nothing was measured")
        return base
    started = time.perf_counter()
    hits = 0
    for question in questions:
        text = str(question.get(spec["question_field"], ""))
        answer = answer_contains or str(question.get("answer", ""))
        before = engine.knowledge.active_count()
        for paragraph in question.get("context", []) or []:
            engine.teach(str(paragraph), source="benchmark", origin="user_text")
        recalled = engine.recall(text, top_k=5)
        found = any(
            answer and answer.lower() in str(item.get("text", "")).lower()
            for item in recalled
        )
        hits += 1 if found else 0
        if before == 0 and not engine.knowledge.active_count():
            continue
    measured = {
        "recall_at_k": round(hits / max(1, len(questions)), 4),
        "answer_match": round(hits / max(1, len(questions)), 4),
    }
    base.update(
        status="ok" if loaded["dataset_source"] == "real" else "fixture",
        measured=measured,
        duration_s=round(time.perf_counter() - started, 3),
    )
    if loaded["dataset_source"] == "fixture":
        base["note"] = (
            "fixture run: not a LoCoMo, LongMemEval-S or DMR score; "
            "report only as a labelled fixture result"
        )
    return base


def describe(report: Dict[str, Any]) -> List[str]:
    lines = [
        f"benchmark={report.get('benchmark')}",
        f"status={report.get('status')}",
        f"dataset_source={report.get('dataset_source')}",
        f"questions={report.get('questions', 0)}",
    ]
    if report.get("reason"):
        lines.append(f"reason={report['reason']}")
    for metric in report.get("metrics", []):
        if metric in (report.get("measured") or {}):
            lines.append(f"{metric}={report['measured'][metric]}")
    if report.get("note"):
        lines.append(f"note={report['note']}")
    return lines
