"""Download a real GGUF model from Hugging Face and record the outcome as evidence.

Used for the `real-model` verification tier (docs/testing.md). Every attempt writes
a JSON record: `status` is `downloaded` (with byte count and SHA-256 of the file) or
`blocked` (with the exception class and message). A failed download is NEVER
recorded as a pass, and the process exits 1 when blocked.

Weights are written under `models/` (git-ignored, see .gitignore) and are never
committed. Set `HF_TOKEN` in the environment for gated repositories; the token is
read by huggingface_hub and is not written to the evidence file.

Usage (default: MiniCPM5-1B Q4_K_M, the smallest official MiniCPM GGUF):
    .venv/bin/python scripts/fetch_real_model.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO = "openbmb/MiniCPM5-1B-GGUF"
DEFAULT_FILENAME = "MiniCPM5-1B-Q4_K_M.gguf"
DEFAULT_LOCAL_DIR = "models/minicpm5-1b"
DEFAULT_EVIDENCE = "reports/evidence/2026-10-22-minicpm5-1b-download.json"

_HASH_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hub_version() -> Optional[str]:
    try:
        import huggingface_hub
    except ImportError:
        return None
    return getattr(huggingface_hub, "__version__", None)


def _default_downloader(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def fetch(
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    local_dir: str = DEFAULT_LOCAL_DIR,
    evidence_path: Path = ROOT / DEFAULT_EVIDENCE,
    downloader: Optional[Callable[[str, str, str], str]] = None,
    root: Path = ROOT,
) -> dict:
    """Attempt one download and write the evidence record. Returns the record."""
    if downloader is None:
        downloader = _default_downloader
    record: dict = {
        "tier": "real-model",
        "attempted_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": local_dir,
        "huggingface_hub_version": _hub_version(),
        "python": sys.version.split()[0],
    }
    try:
        path = Path(downloader(repo_id, filename, str(root / local_dir)))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"downloader returned no usable file: {path}")
        record.update(
            status="downloaded",
            path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - every failure is evidence, not a crash
        message = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ""
        record.update(
            status="blocked",
            path=None,
            bytes=None,
            sha256=None,
            error={"type": type(exc).__name__, "message": message},
        )

    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--evidence", type=Path, default=ROOT / DEFAULT_EVIDENCE)
    args = parser.parse_args(argv)
    record = fetch(args.repo_id, args.filename, args.local_dir, args.evidence)
    if record["status"] == "downloaded":
        print(f"downloaded {record['path']} ({record['bytes']} bytes, sha256 {record['sha256']})")
        return 0
    err = record["error"]
    print(f"blocked: {err['type']}: {err['message']}", file=sys.stderr)
    print(f"evidence written to {args.evidence}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
