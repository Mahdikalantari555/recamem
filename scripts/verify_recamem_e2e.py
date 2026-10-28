#!/usr/bin/env python3
"""End-to-end demo for the recamem identity/ingest/recall surface.

Writes `reports/evidence/recamem-e2e.json` with the measured outputs. Nothing
here needs network or model weights: the anchor is explicitly selected as mock
and the report says so.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run() -> dict:
    os.environ.setdefault("RECAMEM_ANCHOR", "mock")
    from recamem.engine import DroidEngine
    from recamem.ingest import ingest_directory
    from recamem.retrieval import TwoChannelRecall

    evidence: dict = {
        "commit": _commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor": {"mode": "mock", "reason": "no ONNX weights in this sandbox"},
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "python": sys.version.split()[0],
    }

    docs = Path(tempfile.mkdtemp()) / "demo"
    docs.mkdir()
    (docs / "project.md").write_text(
        "# Project\n\nThe project lead is Dr. Mina Rostami. The budget is 5 million.\n"
    )
    (docs / "lab.md").write_text(
        "# Lab\n\nThe calibration constant is 0.98 for the soil moisture sensor.\n"
    )
    (docs / "sensors.md").write_text(
        "# Sensors\n\n```bash\nrm -rf /\n```\n\nThe gateway forwards readings every minute.\n"
    )

    engine = DroidEngine(name="e2e")
    started = time.perf_counter()
    report = ingest_directory(engine, docs)
    ingest_seconds = time.perf_counter() - started
    evidence["ingest"] = {**report.to_dict(), "seconds": round(ingest_seconds, 4)}

    recall = TwoChannelRecall(engine)
    lead = recall.recall("Who is the project lead?")
    unrelated = recall.recall("What is the boiling point of mercury?")
    constant = recall.recall("What is the calibration constant?")
    evidence["recall"] = {
        "project_lead": {
            "texts": [hit["text"] for hit in lead["results"]],
            "sources": [hit["source"] for hit in lead["results"]],
        },
        "unrelated_is_empty": unrelated["results"] == [],
        "calibration": {
            "texts": [hit["text"] for hit in constant["results"]],
            "sources": [hit["source"] for hit in constant["results"]],
        },
    }
    evidence["checks"] = {
        "full_sentence": "Dr. Mina Rostami" in " ".join(evidence["recall"]["project_lead"]["texts"]),
        "unrelated_empty": unrelated["results"] == [],
        "cites_lab_md": any("lab.md" in s for s in evidence["recall"]["calibration"]["sources"]),
        "code_fence_not_stored": not any(
            "rm -rf" in text for (text,) in engine.knowledge.conn.execute("SELECT text FROM facts")
        ),
    }

    # 100-file ingest workload (task 7.7): wall time and fact count.
    bulk = Path(tempfile.mkdtemp()) / "bulk"
    bulk.mkdir()
    for index in range(100):
        (bulk / f"note-{index:02d}.md").write_text(
            f"# Note {index}\n\nStation {index} recorded a moisture reading of {index / 10:.1f}.\n"
        )
    bulk_engine = DroidEngine(name="bulk")
    started = time.perf_counter()
    bulk_report = ingest_directory(bulk_engine, bulk)
    bulk_seconds = time.perf_counter() - started
    evidence["ingest_100_files"] = {
        "files": 100,
        "facts_stored": bulk_report.facts_stored,
        "store_count": bulk_engine.knowledge.total_count(),
        "seconds": round(bulk_seconds, 3),
        "facts_per_second": round(bulk_report.facts_stored / max(bulk_seconds, 1e-9), 2),
    }
    return evidence


if __name__ == "__main__":
    payload = run()
    out = ROOT / "reports" / "evidence" / "recamem-e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("checks", "ingest", "ingest_100_files")}, indent=2))
    print(f"evidence={out}")
