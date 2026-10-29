#!/usr/bin/env python3
"""Colab-friendly training entry point.

Wraps scripts/train_lm.py with env-var config, optional resume from a
HuggingFace repo, and auto-push on completion.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recamem import hub


def main() -> int:
    ap = argparse.ArgumentParser(description="Cloud training entry point")
    ap.add_argument("--lm-tokens", type=int, default=int(os.environ.get("LM_TOKENS", "1000000")))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/tinystories"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "runs/cloud"))
    ap.add_argument("--core", default=os.environ.get("CORE", "C4"))
    ap.add_argument("--d-model", type=int, default=int(os.environ.get("D_MODEL", "192")))
    ap.add_argument("--n-layers", type=int, default=int(os.environ.get("N_LAYERS", "6")))
    ap.add_argument("--n-heads", type=int, default=int(os.environ.get("N_HEADS", "4")))
    ap.add_argument("--d-ff", type=int, default=int(os.environ.get("D_FF", "768")))
    ap.add_argument("--vocab-size", type=int, default=int(os.environ.get("VOCAB_SIZE", "1024")))
    ap.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "256")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "0.003")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "4")))
    ap.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("CKPT_EVERY", "1000")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1337")))
    ap.add_argument("--resume-from", default=os.environ.get("RESUME_FROM"))
    ap.add_argument("--push-to-hub", default=os.environ.get("PUSH_TO_HUB"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from HF repo if needed
    resume_path = None
    if args.resume_from:
        p = Path(args.resume_from)
        if p.is_file():
            resume_path = str(p)
        elif "/" in args.resume_from:
            # treat as repo id
            target = out_dir / "resumed_ckpt"
            hub.pull_checkpoint(args.resume_from, target, token=args.hf_token)
            # find checkpoint.pt
            cands = list(target.rglob("checkpoint.pt"))
            if cands:
                resume_path = str(cands[0])
        else:
            resume_path = args.resume_from

    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "train_lm.py"),
        "--core", args.core,
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads),
        "--d-ff", str(args.d_ff),
        "--vocab-size", str(args.vocab_size),
        "--seq-len", str(args.seq_len),
        "--lm-tokens", str(args.lm_tokens),
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--checkpoint-every", str(args.checkpoint_every),
        "--seed", str(args.seed),
    ]
    if resume_path:
        cmd += ["--resume", resume_path]
    # push handled after training completes below

    env = dict(os.environ)
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    env["PYTHONUNBUFFERED"] = "1"

    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        return rc

    # Push to hub on completion
    if args.push_to_hub:
        try:
            hub.write_manifest(out_dir)
            hub.push_checkpoint(args.push_to_hub, out_dir, token=args.hf_token)
            print(f"Pushed {out_dir} -> {args.push_to_hub}")
        except Exception as e:
            print(f"Final push failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
