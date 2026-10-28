#!/usr/bin/env python3
"""Thin wrapper around the headless runner (``recamem run``).

Kept for scripts and CI that do not want to depend on the console entry point::

    python scripts/run_recamem.py --prompt "status?" --name demo
    printf 'a\\nb\\n' | python scripts/run_recamem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recamem.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
