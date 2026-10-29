#!/usr/bin/env python3
"""Push a checkpoint directory to HuggingFace Hub."""
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
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    hub.push_checkpoint(args.repo, Path(args.dir), token=args.token, private=args.private)
    print(f"Pushed {args.dir} -> {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
