#!/usr/bin/env python3
"""Generate text samples from a trained LM checkpoint.

Usage:
    PYTHONPATH=. python scripts/generate_samples.py --checkpoint runs/lm/checkpoint.pt \\
        --data-dir data/tinystories --n 20 --temperature 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model.cores.registry import build_candidate_lm


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--core", type=str, default="C4")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    V = meta["vocab_size"]

    cfg = {"core_id": args.core, "d_model": args.d_model, "n_layers": args.n_layers,
           "n_heads": args.n_heads, "d_ff": args.d_ff, "vocab_size": V,
           "window_size": args.seq_len, "use_conv": True, "conv_kernel": 4, "state_dim": 16}
    model = build_candidate_lm(cfg, tied_embeddings=True, chunk_size=64)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    prompt = train[:16].tolist()

    for i in range(args.n):
        ids = list(prompt)
        with torch.no_grad():
            for _ in range(args.max_new):
                x = torch.tensor([ids[-args.seq_len:]], dtype=torch.long)
                logits = model(x)[:, -1, :]
                probs = torch.softmax(logits / args.temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
                ids.append(nxt)
                if nxt == 0:  # eos-ish
                    break
        print(f"--- sample {i+1} ---")
        print(" ".join(str(t) for t in ids[len(prompt):]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
