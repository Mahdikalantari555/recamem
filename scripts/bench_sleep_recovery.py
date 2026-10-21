#!/usr/bin/env python
"""Measure autonomous sleep recovery on 100 correlated patterns.

The evaluator stores the patterns and only *afterwards* scores the IDs the
sleep cycle retrieved; the expected list is never passed into replay.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.model.droid import DroidEngine
from src.model.sleep_consolidation import SleepConsolidation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", type=int, default=100)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/evidence/sleep-recovery-100.json")
    args = parser.parse_args()

    droid = DroidEngine(name="sleep-benchmark", db_path=":memory:", dim=args.dim)
    generator = torch.Generator().manual_seed(args.seed)
    base = torch.randn(args.dim, generator=generator)
    base = base / base.norm()
    expected = []
    for index in range(args.patterns):
        noise = torch.randn(args.dim, generator=generator)
        vector = base + 0.5 * noise / noise.norm()
        vector = vector / vector.norm()
        expected.append(droid.knowledge.insert_fact(
            f"correlated pattern {index}", "benchmark", time.time(), index, 0.5, vector
        ))
    droid.knowledge._vec_cache = None

    started = time.perf_counter()
    report = SleepConsolidation(
        droid, max_replay_steps=args.steps, max_seconds=args.seconds
    ).autonomous_retrieval()
    elapsed = time.perf_counter() - started

    recovered = set(report["retrieved_ids"])
    expected_set = set(expected)
    results = {
        "protocol": {
            "patterns": args.patterns, "dim": args.dim, "seed": args.seed,
            "max_replay_steps": args.steps, "max_seconds": args.seconds,
            "expected_patterns_passed_to_retrieval": False,
        },
        "expected_count": len(expected_set),
        "recovered_distinct_expected": len(recovered & expected_set),
        "recovered_count": report["unique_recovered"],
        "coverage": report["coverage"],
        "duplicates": report["duplicates"],
        "steps": report["steps"],
        "stop_reason": report["stop_reason"],
        "replay_writes": report["replay_writes"],
        "duration_sleep_s": report["duration_ms"] / 1000.0,
        "duration_total_s": elapsed,
        "target": 0.95,
        "target_met": (len(recovered & expected_set) / max(1, len(expected_set))) >= 0.95,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
