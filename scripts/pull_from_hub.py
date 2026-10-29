#!/usr/bin/env python3
"""Pull a checkpoint from HuggingFace Hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recamem import hub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    hub.pull_checkpoint(args.repo, Path(args.dir), revision=args.revision, token=args.token)
    print(f"Pulled {args.repo} -> {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
