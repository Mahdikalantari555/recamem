#!/usr/bin/env python3
"""Prepare TinyStories corpus as tokenized uint16 memmap files for LM training.

Usage:
    PYTHONPATH=. python scripts/prepare_tinystories.py [--vocab-size 1024] [--out-dir data/tinystories]

Outputs:
    <out_dir>/train.bin   (uint16 memmap)
    <out_dir>/val.bin     (uint16 memmap)
    <out_dir>/meta.json   (vocab size, counts, sha256)
    configs/recamem_selected_bpe.json  (tokenizer artifact, V=1024)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model.cores.tokenizer import CompactBPETokenizer
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TinyStories memmap corpus")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out-dir", type=str, default="data/tinystories")
    parser.add_argument("--max-train-bytes", type=int, default=250_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = ensure_tinystories_real_data_dir()
    if splits_dir is None:
        print("ERROR: could not provision TinyStories data")
        return 1

    train_stories = load_stories(splits_dir / "train.jsonl")
    val_stories = load_stories(splits_dir / "val_cal.jsonl")
    print(f"Loaded {len(train_stories)} train docs, {len(val_stories)} val docs")

    tok = CompactBPETokenizer.train_from_texts(
        train_stories, vocab_size=args.vocab_size, max_training_bytes=args.max_train_bytes
    )
    print(f"Tokenizer V={tok.vocab_size}, merges={len(tok.merges)}")

    bpe_path = out_dir / "tokenizer.json"
    tok.save(bpe_path)
    print(f"Saved tokenizer to {bpe_path}")

    variant_path = REPO_ROOT / "configs" / "recamem_selected_bpe_1024.json"
    tok.save(variant_path)
    print(f"Saved V=1024 variant to {variant_path}")

    sample = train_stories[0]
    ids = tok.encode(sample)
    dec = tok.decode(ids)
    assert dec == sample, f"Round-trip failed: {dec[:80]!r} != {sample[:80]!r}"
    print("Round-trip lossless: OK")

    train_tokens: list[int] = []
    for s in train_stories:
        train_tokens.extend(tok.encode(s, add_eos=True))
    val_tokens: list[int] = []
    for s in val_stories:
        val_tokens.extend(tok.encode(s, add_eos=True))

    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "vocab_size": tok.vocab_size,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_docs": len(train_stories),
        "val_docs": len(val_stories),
        "sha256_train": hashlib.sha256(train_arr.tobytes()).hexdigest(),
        "sha256_val": hashlib.sha256(val_arr.tobytes()).hexdigest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {train_path} ({len(train_tokens)} tokens)")
    print(f"Wrote {val_path} ({len(val_tokens)} tokens)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
