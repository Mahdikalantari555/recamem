#!/usr/bin/env python3
"""Benchmark the chunked RTU forward against the sequential per-token loop.

Both paths run the same model (``dim=384``, eight layers, ``spread=32``) over a
deterministic byte slice of the project corpus and produce the same hidden
states and logits. The sequential path advances the recurrence one token at a
time; the chunked path uses ``associative_scan`` and applies the per-token
output transform as one vectorized operation.

Usage::

    PYTHONPATH=. python scripts/bench_parallel_scan.py
    PYTHONPATH=. python scripts/bench_parallel_scan.py --lengths 512 --runs 5

The output is CSV with a ``path,L,seconds,factor,cv`` header followed by the
max absolute logit difference between the two paths at ``L=512``.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/bench_parallel_scan.py`
    sys.path.insert(0, str(ROOT))

from src.model.rtu import Model  # noqa: E402

DIM = 384
LAYERS = 8
VOCAB = 256
SPREAD = 32
CORPUS_FILES = ("README.md", "plans/10_droid_rtu_fusion.md")
DEFAULT_LENGTHS = (128, 512, 1024)


def build_corpus(length: int, files: tuple[str, ...] = CORPUS_FILES) -> torch.Tensor:
    """A deterministic ``(1, length)`` byte slice of the project corpus."""
    data = b"\n".join((ROOT / name).read_bytes() for name in files)
    if not data:
        raise ValueError("corpus files are empty")
    repeats = length // len(data) + 1
    return torch.tensor(list((data * repeats)[:length]), dtype=torch.long).unsqueeze(0)


def build_model(seed: int = 0) -> Model:
    torch.manual_seed(seed)
    model = Model(dim=DIM, layers=LAYERS, spread=SPREAD, vocab_size=VOCAB)
    model.eval()
    return model


def sequential_forward(model: Model, tokens: torch.Tensor):
    """Token-by-token forward: the reference path."""
    batch, length = tokens.shape
    enc = model.encoder(tokens)
    states = [layer.states.expand(batch, model.dim).clone() for layer in model.blocks]
    outputs = []
    for t in range(length):
        x = enc[:, t]
        for i, layer in enumerate(model.blocks):
            decay = torch.sigmoid(layer.decay)
            states[i] = decay * states[i] + enc[:, t]
            x = x + layer.silu(layer.weights(layer.norm(states[i])))
        outputs.append(x)
    x = torch.stack(outputs, dim=1)
    logits, _ = model.decoder(x)
    return x, logits, states


def chunked_forward(model: Model, tokens: torch.Tensor, chunk: int = 128):
    """Chunked forward: one scan per layer, vectorized outputs."""
    x, states, (logits, _) = model.forward_chunk(tokens, chunk=chunk)
    return x, logits, states


def timed(fn, model: Model, tokens: torch.Tensor, runs: int = 3, warmup: int = 1):
    """Median wall clock over ``runs`` measurements plus the CV."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(model, tokens)
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(model, tokens)
            samples.append(time.perf_counter() - start)
    mean = statistics.fmean(samples)
    cv = statistics.stdev(samples) / mean if len(samples) > 1 and mean > 0 else 0.0
    return statistics.median(samples), cv


def hardware() -> str:
    return (
        f"{platform.machine()} | {platform.processor() or 'unknown cpu'} | "
        f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
        f"python {platform.python_version()}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads; defaults to torch's own default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    model = build_model(args.seed)
    print(f"# hardware: {hardware()}")
    print(f"# model: dim={DIM} layers={LAYERS} spread={SPREAD} chunk={args.chunk} runs={args.runs}")
    print("path,L,seconds,factor,cv")

    for length in args.lengths:
        tokens = build_corpus(length)
        seq_seconds, seq_cv = timed(sequential_forward, model, tokens, runs=args.runs)
        scan_seconds, scan_cv = timed(
            lambda m, t: chunked_forward(m, t, chunk=args.chunk), model, tokens, runs=args.runs
        )
        print(f"sequential,{length},{seq_seconds:.6f},1.00,{seq_cv:.3f}")
        print(f"parallel-scan,{length},{scan_seconds:.6f},{seq_seconds / scan_seconds:.2f},{scan_cv:.3f}")

    tokens = build_corpus(512)
    with torch.no_grad():
        _, seq_logits, _ = sequential_forward(model, tokens)
        _, scan_logits, _ = chunked_forward(model, tokens, chunk=args.chunk)
    max_diff = (seq_logits - scan_logits).abs().max().item()
    print(f"# equivalence: max |logit diff| at L=512 = {max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
