#!/usr/bin/env python3
"""Deterministic CPU held-out byte LM and smoothed n-gram comparison.

Only load trusted local checkpoints. This is not a promotion gate: it cannot
verify training-data history, benchmark independence, retention, or capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from recamem.native_lm import NativeLMBackend

MAX_BYTES = 64 * 1024 * 1024


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _read_split(data_dir: Path, name: str, meta: dict) -> np.ndarray:
    path = data_dir / f'{name}.bin'
    if not path.is_file() or path.stat().st_size % 2 or path.stat().st_size > MAX_BYTES * 2:
        raise ValueError(f'{name} split is missing, malformed or exceeds {MAX_BYTES} bytes')
    data = np.fromfile(path, dtype=np.uint16)
    if not len(data) or np.any(data > 255):
        raise ValueError(f'{name} split must contain byte tokens')
    raw = data.astype(np.uint8).tobytes()
    if meta.get(f'{name}_tokens') != len(data) or meta.get(f'{name}_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError(f'{name} split length or SHA-256 does not match metadata')
    return data


def evaluate(checkpoint: Path, data_dir: Path, *, context: int = 64,
             max_predictions: int = 4096, threads: int = 1) -> dict:
    if not 1 <= context <= 1024 or not 1 <= max_predictions <= 1000000 or not 1 <= threads <= 32:
        raise ValueError('context must be 1–1024, max predictions 1–1000000, and threads 1–32')
    checkpoint, data_dir = Path(checkpoint), Path(data_dir)
    meta = json.loads((data_dir / 'meta.json').read_text(encoding='utf-8'))
    if meta.get('tokenizer') != 'utf8_bytes_256' or meta.get('vocab_size') != 256:
        raise ValueError('evaluation requires UTF-8 byte tokenizer metadata')
    train, val = _read_split(data_dir, 'train', meta), _read_split(data_dir, 'val', meta)
    if len(val) < 2:
        raise ValueError('validation must contain at least two bytes')
    # torch's thread count is process-global; restore it for library callers.
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        backend = NativeLMBackend(checkpoint)
        model = backend.model
        n = min(max_predictions, len(val) - 1)
        unigram_counts = np.bincount(train, minlength=256).astype(np.float64) + 1.0
        bigram_counts = np.ones((256, 256), dtype=np.float64)
        if len(train) > 1:
            np.add.at(bigram_counts, (train[:-1], train[1:]), 1.0)
        unigram_prob = unigram_counts / unigram_counts.sum()
        bigram_prob = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)
        targets = val[1:n + 1]
        preceding = val[:n]
        unigram_nll = float(-np.log(unigram_prob[targets]).sum())
        bigram_nll = float(-np.log(bigram_prob[preceding, targets]).sum())
        model_nll = 0.0
        with torch.inference_mode():
            for start in range(0, n, context):
                size = min(context, n - start)
                x = torch.from_numpy(val[start:start + size].astype(np.int64)).unsqueeze(0)
                y = torch.from_numpy(val[start + 1:start + size + 1].astype(np.int64))
                logits = model(x).reshape(-1, 256)
                model_nll += F.cross_entropy(logits, y, reduction='sum').item()
        if not math.isfinite(model_nll):
            raise ValueError('model produced non-finite held-out loss')
    finally:
        torch.set_num_threads(old_threads)
    def scores(nll):
        return {'nll_nats_per_byte': nll / n, 'bits_per_byte': nll / n / math.log(2)}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        'evaluation': 'deterministic contiguous validation prefix; disjoint context windows; no generation or retention test',
        'warning': 'Dataset rights and checkpoint training history are not independently verified. Not a promotion gate or general language benchmark.',
        'checkpoint_sha256': _digest(checkpoint), 'checkpoint_config': backend.config,
        'checkpoint_training_progress': backend.training_progress, 'dataset_meta': meta,
        'context': context, 'predicted_bytes': n, 'threads': threads, 'device': 'cpu',
        'baselines': {'unigram_add_one': scores(unigram_nll), 'bigram_add_one': scores(bigram_nll)},
        'model': scores(model_nll), 'torch_version': torch.__version__,
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'physical_cpus': psutil.cpu_count(logical=False), 'system_memory_bytes': psutil.virtual_memory().total,
        'revision': {'commit': revision, 'dirty': dirty},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='trusted local CPU checkpoint')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--max-predictions', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='optional new JSON file; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.checkpoint, args.data_dir, context=args.context,
                          max_predictions=args.max_predictions, threads=args.threads)
        content = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as out:
                out.write(content)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(content, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
