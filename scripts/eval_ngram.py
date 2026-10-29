#!/usr/bin/env python3
"""Compute n-gram (unigram..4-gram) byte-level BPB on a val corpus.

Usage:
    PYTHONPATH=. python scripts/eval_ngram.py [--splits-dir .cache/tinystories/splits] [--max-n 4]
    PYTHONPATH=. python scripts/eval_ngram.py --data-dir data/tinystories  (token-level fallback)

Computes smoothed byte-level n-gram BPB on val stories using train stories for
counts. Writes <splits_dir>/ngram_baselines.json (or <data_dir>/ngram_baselines.json).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.model.tournament.large_workloads import ensure_tinystories_real_data_dir


def load_stories(path: Path) -> list[str]:
    stories: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                stories.append(text)
    return stories


def compute_ngram_bpb_bytes(train_bytes: bytes, val_bytes: bytes, n: int) -> float:
    V = 256
    # counts of n-grams in train
    counts: Counter = Counter()
    ctx_counts: Counter = Counter()
    for i in range(len(train_bytes) - n + 1):
        ctx = train_bytes[i:i + n - 1] if n > 1 else b""
        tgt = train_bytes[i + n - 1]
        counts[(ctx, tgt)] += 1
        ctx_counts[ctx] += 1

    total_bits = 0.0
    n_val = 0
    for i in range(n - 1, len(val_bytes)):
        ctx = val_bytes[i - n + 1:i] if n > 1 else b""
        tgt = val_bytes[i]
        c = counts.get((ctx, tgt), 0)
        ctx_c = ctx_counts.get(ctx, 0)
        p = (c + 1.0) / (ctx_c + V)
        total_bits -= math.log2(max(1e-12, p))
        n_val += 1

    if n_val == 0:
        return float("inf")
    return float(total_bits / max(1, len(val_bytes)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute n-gram baselines")
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-n", type=int, default=4)
    args = parser.parse_args()

    if args.splits_dir is None and args.data_dir is None:
        splits_dir = ensure_tinystories_real_data_dir()
        if splits_dir is None:
            print("ERROR: no data")
            return 1
        args.splits_dir = str(splits_dir)

    if args.splits_dir is not None:
        sd = Path(args.splits_dir)
        train_stories = load_stories(sd / "train.jsonl")
        val_stories = load_stories(sd / "val_cal.jsonl")
        train_bytes = "\n\n".join(train_stories).encode("utf-8")
        val_bytes = "\n\n".join(val_stories).encode("utf-8")
        out_path = sd / "ngram_baselines.json"
    else:
        # token-level fallback
        import json as _json
        dd = Path(args.data_dir)
        meta = _json.loads((dd / "meta.json").read_text())
        train = np.fromfile(dd / "train.bin", dtype=np.uint16)
        val = np.fromfile(dd / "val.bin", dtype=np.uint16)
        V = meta["vocab_size"]
        results = {}
        for n in range(1, args.max_n + 1):
            # reuse token-level
            counts = Counter()
            ctx_counts = Counter()
            for i in range(len(train) - n + 1):
                ctx = tuple(int(x) for x in train[i:i + n - 1]) if n > 1 else ()
                tgt = int(train[i + n - 1])
                counts[(ctx, tgt)] += 1
                ctx_counts[ctx] += 1
            total_bits = 0.0
            n_val = 0
            for i in range(n - 1, len(val)):
                ctx = tuple(int(x) for x in val[i - n + 1:i]) if n > 1 else ()
                tgt = int(val[i])
                c = counts.get((ctx, tgt), 0)
                ctx_c = ctx_counts.get(ctx, 0)
                p = (c + 1.0) / (ctx_c + V)
                total_bits -= math.log2(max(1e-12, p))
                n_val += 1
            results[f"{n}-gram"] = float(total_bits / max(1, n_val))
        out_path = dd / "ngram_baselines.json"
        uni = results["1-gram"]
        out = {"vocab_size": V, "mode": "token", "results": results,
               "thresholds": {"G1": uni * 0.75, "G2": results["3-gram"], "G3": results["4-gram"]}}
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 0

    results = {}
    for n in range(1, args.max_n + 1):
        bpb = compute_ngram_bpb_bytes(train_bytes, val_bytes, n)
        results[f"{n}-gram"] = bpb
        print(f"{n}-gram byte BPB: {bpb:.4f}")

    uni = results["1-gram"]
    print(f"\nG1 threshold (25% below unigram): {uni * 0.75:.4f}")
    print(f"G2 threshold (below trigram):     {results['3-gram']:.4f}")
    print(f"G3 threshold (below 4-gram):      {results['4-gram']:.4f}")

    out = {"mode": "byte", "results": results, "thresholds": {
        "G1": uni * 0.75, "G2": results["3-gram"], "G3": results["4-gram"],
    }}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
