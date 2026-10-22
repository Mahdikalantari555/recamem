#!/usr/bin/env python3
"""Train a byte-level RTU checkpoint for a Droid profile.

This is a build step, not a shipped weight: the checkpoint is written into
``droids/<name>/rtu.safetensors`` and the manifest is updated afterwards.
``droids/`` is a generated artifact directory and is git-ignored.

    PYTHONPATH=. python scripts/train_byte_rtu.py \
        --profile droids/droid-rs --corpus plans docs README.md \
        --steps 2000 --seq-len 512 --dim 384 --layers 8 --spread 32 --seed 0

Training reuses the losses in ``src/model/rtu.py``. ``--seq-len`` is the
truncated-BPTT window: the model state is reset at every window boundary.

``--chunked`` switches to ``Model.train_chunk``: one vectorized forward/backward
per ``--seq-len`` window with the associative scan carrying the state, one
optimizer step per window. It trains orders of magnitude more bytes per second
than the per-token loop, at the cost of a different optimizer trajectory (the
chunked pass is documented in ``src/model/rtu.py`` as loss- and
gradient-equivalent but not step-equivalent). In this mode ``--steps`` counts
windows, not byte transitions, and the log reports mean cross-entropy per byte.

``--losses full`` (default) trains the variance, prediction, cross-entropy and
stop heads. Stop targets are annotated from the corpus itself: a blank-line
paragraph boundary and the last byte of a window end the sequence, so the stop
head learns where prose ends instead of terminating after one byte (line ends
are *not* sequence ends: an answer follows the query line). ``--losses ce`` keeps cross-entropy only
(``ce_only=True``): it is faster but leaves the stop head untrained, so the
resulting checkpoint stops almost immediately when sampled.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.rtu import Model  # noqa: E402

DEFAULT_CORPUS = ["README.md", "VISION.md", "AGENTS.md", "plans", "docs", "openspec/specs"]


def collect_files(paths):
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.rglob("*.md")))
        elif p.is_file():
            files.append(str(p))
        else:
            raise FileNotFoundError(f"Corpus path not found: {raw}")
    if not files:
        raise FileNotFoundError("No corpus files found")
    return files


def read_corpus(files):
    chunks = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())
    data = "\n\n".join(chunks).encode("utf-8")
    if len(data) < 2:
        raise ValueError("Corpus is empty")
    return data


def write_train_log(path, entries):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2)
    os.replace(str(tmp), str(path))


def write_manifest(profile: Path, updates):
    """Update config.json atomically, last (change 1.4 save order)."""
    config_path = profile / "config.json"
    config = {}
    if config_path.is_file():
        with open(config_path, "r") as f:
            config = json.load(f)
    config.update(updates)
    tmp = profile / "config.json.tmp"
    with open(tmp, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(str(tmp), str(config_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Target profile directory, e.g. droids/droid-rs")
    parser.add_argument("--corpus", nargs="+", default=DEFAULT_CORPUS,
                        help="Files and/or directories of UTF-8 text (default: project corpus)")
    parser.add_argument("--steps", type=int, default=2000, help="Byte transitions to train (default 2000)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Truncated-BPTT window; model state resets here (default 512)")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--spread", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--losses", choices=["full", "ce"], default="full",
                        help="full = variance+pred+ce+stop (default); ce = cross-entropy only")
    parser.add_argument("--chunked", action="store_true",
                        help="One vectorized optimizer step per window (much faster)")
    parser.add_argument("--chunk", type=int, default=128,
                        help="Associative-scan chunk size (chunked mode only, default 128)")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seq_len < 2:
        parser.error("--seq-len must be >= 2")

    profile = Path(args.profile)
    profile.mkdir(parents=True, exist_ok=True)
    checkpoint = profile / "rtu.safetensors"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    files = collect_files(args.corpus)
    data = read_corpus(files)
    print(f"Corpus: {len(files)} files, {len(data)} bytes")
    print(f"Model: dim={args.dim} layers={args.layers} spread={args.spread}")
    if args.chunked:
        print(f"Training: {args.steps} windows of {args.seq_len} bytes (chunked pass)")
    else:
        print(f"Training: {args.steps} byte transitions, window {args.seq_len}")

    model = Model(dim=args.dim, layers=args.layers, spread=args.spread)
    log_path = profile / "train_log.json"
    entries = []
    started = time.time()

    if args.chunked:
        if len(data) < args.seq_len + 2:
            parser.error("--seq-len is longer than the corpus")
        position = random.randrange(0, len(data) - args.seq_len - 1)
        for step in range(args.steps):
            if position + args.seq_len + 1 > len(data):
                position = 0
                model.reset()
            window = data[position:position + args.seq_len]
            following = data[position + 1:position + args.seq_len + 1]
            # Stop targets: a blank-line boundary inside the window, and the
            # window end itself (the same annotations the per-token path uses).
            ends = [window[i] == 0x0A and following[i] == 0x0A
                    for i in range(len(window) - 1)] + [True]
            info = model.train_chunk(
                torch.tensor(list(window), dtype=torch.long),
                next_tokens=torch.tensor(list(following), dtype=torch.long),
                ends=torch.tensor(ends, dtype=torch.bool),
                chunk=args.chunk,
                ce_only=(args.losses == "ce"),
            )
            tokens = max(1, int(info["tokens"]))
            model.last_loss = float(info["loss_terms"]["total"]) / tokens
            position += args.seq_len

            if (step + 1) % args.log_every == 0 or step == args.steps - 1:
                entry = {
                    "step": step + 1,
                    "loss": round(float(info["loss_terms"]["total"]) / tokens, 6),
                    "ce_per_byte": round(float(info["loss_terms"]["ce"]) / tokens, 6),
                    "bytes_trained": int(info["step_count"]),
                    "elapsed_s": round(time.time() - started, 2),
                }
                entries.append(entry)
                write_train_log(log_path, entries)
                print(f"window {entry['step']:>6}  loss {entry['loss']:.4f}  "
                      f"ce/byte {entry['ce_per_byte']:.4f}  {entry['elapsed_s']}s")

        model.save(str(checkpoint))
        write_manifest(profile, {
            "engine": "byte_rtu",
            "fusion_mode": "prompt_stuffing",
            "byte_rtu_checkpoint": checkpoint.name,
            "rtu_checkpoint": checkpoint.name,
            "rtu_dim": args.dim,
            "rtu_layers": args.layers,
            "rtu_spread": args.spread,
            "injection_strategy": "prompt",
            "training": {"mode": "chunked", "windows": args.steps,
                         "window_bytes": args.seq_len, "bytes_trained": int(model.step_count),
                         "losses": args.losses},
            "updated_at": time.time(),
        })
        final = entries[-1]["loss"] if entries else float("nan")
        print(f"Wrote {checkpoint} ({checkpoint.stat().st_size / 1e6:.1f} MB)")
        print(f"Updated {profile / 'config.json'} (final train loss {final:.4f})")
        print("Export RTU_CHECKPOINT=<profile>/rtu.safetensors to run the grounding gate.")
        return 0

    position = random.randrange(0, max(1, len(data) - 2))
    in_window = 0
    model.reset()
    for step in range(args.steps):
        if in_window >= args.seq_len:
            model.reset()
            in_window = 0
        if position >= len(data) - 1:
            position = 0
            model.reset()
            in_window = 0
        c, n = data[position], data[position + 1]
        # Boundary annotations: blank-line paragraph boundary and window end.
        paragraph_end = (
            c == 0x0A and position + 1 < len(data) and data[position + 1] == 0x0A
        )
        end = paragraph_end or in_window + 1 >= args.seq_len
        model(c, nextb=n, end=end, frozen=False, ce_only=(args.losses == "ce"))
        position += 1
        in_window += 1

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            entry = {
                "step": step + 1,
                "loss": round(float(model.last_loss), 6),
                "elapsed_s": round(time.time() - started, 2),
            }
            entries.append(entry)
            write_train_log(log_path, entries)
            print(f"step {entry['step']:>6}  loss {entry['loss']:.4f}  {entry['elapsed_s']}s")

    # 1. Artifact first.
    model.save(str(checkpoint))
    # 2. Manifest last.
    write_manifest(profile, {
        "engine": "byte_rtu",
        "fusion_mode": "prompt_stuffing",
        "byte_rtu_checkpoint": checkpoint.name,
        # Legacy alias, read-only on load, kept for one release.
        "rtu_checkpoint": checkpoint.name,
        "rtu_dim": args.dim,
        "rtu_layers": args.layers,
        "rtu_spread": args.spread,
        "injection_strategy": "prompt",
        "updated_at": time.time(),
    })

    final = entries[-1]["loss"] if entries else float("nan")
    print(f"Wrote {checkpoint} ({checkpoint.stat().st_size / 1e6:.1f} MB)")
    print(f"Updated {profile / 'config.json'} (final train loss {final:.4f})")
    print("Export RTU_CHECKPOINT=<profile>/rtu.safetensors to run the grounding gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
