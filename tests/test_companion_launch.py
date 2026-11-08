"""One-command launcher contract (no Node needed at runtime)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, extra_env=None):
    env = dict(os.environ)
    env.pop("RECAMEM_API_ALLOW_REMOTE", None)
    env["RECAMEM_ANCHOR"] = "mock"
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-m", "recamem.cli", *args],
                          cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)


def test_dry_run_reports_dist_host_and_anchor():
    result = _run(["companion", "--dry-run"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "url=http://127.0.0.1:8000/" in result.stdout
    assert "host_policy=loopback-only" in result.stdout
    assert "exists=yes" in result.stdout
    assert "anchor=mock" in result.stdout


def test_remote_bind_requires_explicit_opt_in():
    result = _run(["companion", "--host", "0.0.0.0", "--dry-run"])
    assert result.returncode == 1
    assert "RECAMEM_API_ALLOW_REMOTE=1" in result.stderr


def test_committed_dist_is_servable_without_node():
    dist = ROOT / "companion" / "dist"
    assert (dist / "index.html").is_file()
    html = (dist / "index.html").read_text()
    assert "RecaMem Companion" in html
    assert "http://127.0.0.1" not in html and "localhost" not in html.lower()


def test_committed_dist_ships_the_observatory():
    """The 3D studio must be in the build the Python server actually serves."""
    dist = ROOT / "companion" / "dist"
    html = (dist / "index.html").read_text()
    assert "RecaMem Companion" in html
    bundles = list((dist / "assets").glob("*.js"))
    assert bundles, "no JS bundle in the committed dist"
    text = "\n".join(path.read_text(errors="ignore") for path in bundles)
    # The scene is created lazily from the observatory component; the label and
    # the no-WebGL copy must ship, and three.js must come from the same origin.
    assert "learning observatory" in text.lower()
    assert "Accessible 2D" in text
    assert "cdn.jsdelivr" not in text and "unpkg.com" not in text
    assert any(path.name.startswith("three.module") for path in (dist / "assets").glob("*.js"))
