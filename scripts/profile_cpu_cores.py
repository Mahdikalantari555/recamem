#!/usr/bin/env python3
"""CPU-only synthetic C4/Muir profiling, not a language quality benchmark."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.cores.registry import build_candidate_lm


def _elapsed(fn, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return samples


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    # Nearest-rank p95, not an interpolation or a claim of statistical confidence.
    rank = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {"samples": len(samples), "median_ms": statistics.median(samples) * 1000,
            "p95_ms": ordered[rank] * 1000}


def _revision() -> dict:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def profile_core(core: str, *, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, seq_len: int, iterations: int, warmup: int, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = {"core_id": core, "d_model": d_model, "n_layers": n_layers,
           "n_heads": n_heads, "d_ff": d_ff, "vocab_size": 256,
           "use_conv": True, "conv_kernel": 4}
    proc = psutil.Process()
    before = proc.memory_info().rss
    model = build_candidate_lm(cfg).cpu()
    ids = torch.randint(0, 256, (1, seq_len), dtype=torch.long)
    model.eval()

    def forward():
        with torch.no_grad():
            return model(ids, return_details=True)

    def decode():
        with torch.no_grad():
            state = None
            output = []
            for i in range(seq_len):
                step = model.step(ids[:, i], state=state)
                state = step['state']
                output.append(step['logits'].unsqueeze(1))
            return torch.cat(output, dim=1), state

    def train_step():
        model.zero_grad(set_to_none=True)
        logits = model(ids)
        torch.nn.functional.cross_entropy(logits.reshape(-1, 256), ids.reshape(-1)).backward()

    for _ in range(warmup):
        forward(); decode(); train_step()
    result = forward()
    stepped, state = decode()
    errors = {}
    for key in result['state']:
        errors[key] = float((result['state'][key] - state[key]).abs().max().item()) if state[key].numel() else 0.0
    parity = float((result['logits'] - stepped).abs().max().item())
    forward_times = _elapsed(forward, iterations)
    step_times = _elapsed(decode, iterations)
    train_times = _elapsed(train_step, iterations)
    # Separate profiling pass; operator totals include time inside the PyTorch
    # profiler and exclude Python dispatch, so they are not latency samples.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        train_step()
    ops = sorted(prof.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)[:10]
    return {
        "config": cfg, "parameters": sum(p.numel() for p in model.parameters()),
        "state_bytes_fp32_per_sequence": result['state_spec'].bytes_per_sequence,
        "logit_max_abs_error": parity, "state_max_abs_error": errors,
        "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
        "forward": _summary(forward_times), "incremental_sequence": _summary(step_times),
        "forward_backward": _summary(train_times),
        "rss_before_model_bytes": before, "rss_after_profile_bytes": proc.memory_info().rss,
        "profiler_top_ops": [{"op": op.key, "calls": op.count,
                              "self_cpu_time_us": op.self_cpu_time_total} for op in ops],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cores', nargs='+', choices=['C4', 'C7'], default=['C4', 'C7'])
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if (args.d_model < 1 or args.n_heads < 1 or args.d_model % args.n_heads or
            args.n_layers < 1 or args.d_ff < 1 or args.seq_len < 1 or args.iterations < 1 or
            args.warmup < 0 or args.threads < 1):
        p.error('dimensions, iteration counts and threads must be positive; d-model must divide n-heads')
    torch.set_num_threads(args.threads)
    results = {core: profile_core(core, d_model=args.d_model, n_layers=args.n_layers,
               n_heads=args.n_heads, d_ff=args.d_ff, seq_len=args.seq_len,
               iterations=args.iterations, warmup=args.warmup, seed=args.seed) for core in args.cores}
    report = {
        'dataset': 'synthetic seeded random token IDs; not a language benchmark',
        'comparison': 'same configured dimensions, not parameter matched',
        'memory_note': 'process RSS snapshots are not peak memory and include earlier runs',
        'profiler_note': 'separate torch operator profile; excludes Python dispatch',
        'seed': args.seed, 'threads': args.threads, 'seq_len': args.seq_len,
        'warmup': args.warmup, 'platform': platform.platform(), 'cpu': platform.processor() or None,
        'physical_cpus': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
        'system_memory_bytes': psutil.virtual_memory().total, 'torch_version': torch.__version__,
        'python_version': platform.python_version(), 'revision': _revision(), 'cores': results,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
