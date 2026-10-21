#!/usr/bin/env python3
"""Live smoke check for web-grounded learning.

Performs exactly one real search and one fetch, then prints the sources it
consulted. Guarded by an environment variable so a test run never goes online:

    TMT_ALLOW_NETWORK=1 PYTHONPATH=. python scripts/smoke_web_research.py "lidar bathymetry"

With the variable unset the script names it and exits 2 without performing any
request.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("TMT_ALLOW_NETWORK") != "1":
        print(
            "refusing to touch the network: set TMT_ALLOW_NETWORK=1 to run this smoke check",
            file=sys.stderr,
        )
        return 2
    topic = " ".join(argv) or "lidar bathymetry"

    from src.model.web_teacher import DuckDuckGoHTMLProvider, fetch_page

    provider = DuckDuckGoHTMLProvider(fetcher=lambda url, **kwargs: fetch_page(url, **kwargs))
    results = provider.search(topic, limit=3)
    print(f"topic={topic}")
    print(f"results={len(results)}")
    consulted = []
    for result in results:
        fetched = fetch_page(result.url, timeout=8.0, max_bytes=200_000, max_chars=4000)
        status = "ok" if fetched.ok else f"failed: {fetched.reason}"
        print(f"- {result.url} ({status}, {len(fetched.text)} chars)")
        if fetched.ok:
            consulted.append(result.url)
    print(f"sources={','.join(consulted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
