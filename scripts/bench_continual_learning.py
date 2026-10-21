#!/usr/bin/env python
"""Run the versioned 20-task continual-learning benchmark.

Usage:
    PYTHONPATH=. python scripts/bench_continual_learning.py \
        --out reports/evidence/continual-learning-20task.json
"""

import argparse
import json
from pathlib import Path

from src.model.continual_bench import BenchmarkConfig, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--patterns", type=int, default=40)
    parser.add_argument("--eval-patterns", type=int, default=10)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="reports/evidence/continual-learning-20task.json")
    args = parser.parse_args()

    config = BenchmarkConfig(
        tasks=args.tasks, patterns_per_task=args.patterns,
        eval_per_task=args.eval_patterns, dim=args.dim, seed=args.seed,
    )
    results = run_benchmark(config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    summary = results["summary"]
    print(f"Delta memory loss : {summary['delta_memory_loss']:.4f} "
          f"(target < {summary['delta_target']}, met={summary['delta_target_met']})")
    print(f"Sparse memory loss: {summary['sparse_memory_loss']:.4f}")
    print(f"EWC memory loss   : {summary['ewc_memory_loss']:.6f} "
          f"(reference {summary['ewc_reference_target']}, "
          f"matched={summary['ewc_reference_matched']})")
    print(f"Interference      : {summary['interference']}")
    print(f"Written           : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
