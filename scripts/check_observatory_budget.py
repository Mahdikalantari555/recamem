#!/usr/bin/env python3
"""Measure the observatory's shipped bundle against a declared budget.

This is an offline, deterministic check of build output only: no browser, no
GPU and no weights. It reports what the *committed* Svelte build costs and how
much of that is the Three.js chunk, then compares against the budget recorded
here. The budget was set from the measurements in
`docs/three-dimensional-studio.md`, not from vendor demos.

Exit codes: 0 when every budget holds, 1 when one does not or the build is
missing. The JSON printed on stdout is a release-evidence record (`tier:
benchmark`, `outcome: passed|failed`).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "companion" / "dist"

# Budgets in bytes / counts. Three.js is lazy-loaded, so it is measured apart
# from the entry chunk that a first visit must download.
BUDGETS = {
    "entry_js_gzip_bytes": 64 * 1024,
    "three_chunk_gzip_bytes": 220 * 1024,
    "total_css_gzip_bytes": 12 * 1024,
    "total_assets_bytes": 1_100 * 1024,
    "webgl_contexts": 1,
}


def gzip_size(path: Path) -> int:
    """Compressed size of the file as a static server would send it."""
    import gzip

    return len(gzip.compress(path.read_bytes(), compresslevel=9))


def measure() -> dict:
    index = DIST / "index.html"
    if not index.is_file():
        raise SystemExit("companion build missing; run npm --prefix companion run build")
    assets = sorted((DIST / "assets").glob("*"))
    js = [p for p in assets if p.suffix == ".js"]
    css = [p for p in assets if p.suffix == ".css"]
    three = [p for p in js if p.name.startswith("three.module")]
    entry = [p for p in js if p not in three]
    if not entry or not three:
        raise SystemExit("expected an entry chunk and a three.module chunk in the build")

    entry_gz = sum(gzip_size(p) for p in entry)
    three_gz = sum(gzip_size(p) for p in three)
    css_gz = sum(gzip_size(p) for p in css)
    total = sum(p.stat().st_size for p in assets if p.is_file())

    measured = {
        "entry_js_gzip_bytes": entry_gz,
        "three_chunk_gzip_bytes": three_gz,
        "total_css_gzip_bytes": css_gz,
        "total_assets_bytes": total,
    }
    results = []
    for name, value in measured.items():
        limit = BUDGETS[name]
        results.append({
            "metric": name,
            "value": value,
            "budget": limit,
            "met": value <= limit,
        })
    # The scene is created once per mount; more than one live context would be
    # a lifecycle bug. Asserted from source because this check has no browser.
    component = (ROOT / "companion" / "src" / "lib" / "observatoryScene.ts").read_text()
    scene_source = (ROOT / "companion" / "src" / "components" / "Observatory.svelte").read_text()
    contexts = component.count("new THREE.WebGLRenderer(")
    creates = scene_source.count("createObservatoryScene(")
    results.append({
        "metric": "webgl_contexts", "value": contexts,
        "budget": BUDGETS["webgl_contexts"], "met": contexts == 1,
    })
    results.append({
        "metric": "scene_creation_calls", "value": creates,
        "budget": 1, "met": creates == 1,
    })
    return {
        "measurements": results,
        "chunks": [
            {"file": p.name, "bytes": p.stat().st_size, "gzip_bytes": gzip_size(p),
             "lazy": p.name.startswith("three.module")}
            for p in js + css
        ],
    }


def main() -> int:
    build = measure()
    met = all(item["met"] for item in build["measurements"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "0" * 40
    record = {
        "tier": "benchmark",
        "outcome": "passed" if met else "failed",
        "commit": commit,
        "command": "python scripts/check_observatory_budget.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip() or "unknown",
            "browser": "none (build-output measurement; no WebGL executed)",
        },
        "configuration": {
            "budgets": BUDGETS,
            "note": "Budgets are byte ceilings for the committed dist. They bound download cost, "
                    "not frame rate: frame timing needs a real browser and is listed as unmeasured.",
        },
        "measurements": build,
        "unmeasured": [
            "frame timing and interaction latency (no browser/GPU in this environment)",
            "16 GB target-device rendering and CPU-training contention",
            "real-browser context-loss recovery",
        ],
    }
    print(json.dumps(record, indent=2))
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
