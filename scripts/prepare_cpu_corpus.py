#!/usr/bin/env python3
"""Prepare separately supplied UTF-8 train/validation documents for CPU LM tests.

One document per non-empty line. Source and license are user assertions, not
legal verification. Exact normalized-line overlap is rejected; this is not a
near-duplicate or semantic leakage detector. Never downloads data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

MAX_SPLIT_BYTES = 64 * 1024 * 1024


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f'missing input file: {path}')
    if path.stat().st_size > MAX_SPLIT_BYTES:
        raise ValueError(f'input exceeds {MAX_SPLIT_BYTES} bytes: {path}')
    data = path.read_bytes()
    if len(data) > MAX_SPLIT_BYTES:
        raise ValueError('input grew beyond size limit')
    try:
        data.decode('utf-8', errors='strict')
    except UnicodeDecodeError as exc:
        raise ValueError(f'input is not valid UTF-8: {path}') from exc
    if not data.strip():
        raise ValueError(f'input is empty: {path}')
    return data


def _documents(raw: bytes) -> set[str]:
    return {' '.join(line.split()).casefold() for line in raw.decode('utf-8').splitlines() if line.strip()}


def prepare(train_path: Path, val_path: Path, output: Path, *, source: str, license_name: str) -> dict:
    if not source.strip() or not license_name.strip():
        raise ValueError('source and license must be specified')
    train_path, val_path, output = Path(train_path), Path(val_path), Path(output)
    if output.exists():
        raise ValueError('output directory already exists; choose a new directory')
    train, val = _read(train_path), _read(val_path)
    overlap = _documents(train) & _documents(val)
    if overlap:
        raise ValueError(f'train/validation contain {len(overlap)} exact normalized document overlaps')
    metadata = {'vocab_size': 256, 'tokenizer': 'utf8_bytes_256', 'bytes_per_token': 1.0,
                'kind': 'user_supplied', 'dataset_source': 'user_supplied',
                'source': source.strip(), 'license': license_name.strip(),
                'rights_verification': 'user_asserted_not_independently_verified',
                'leakage_check': 'exact normalized non-empty lines only; near duplicates not checked'}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cpu-corpus-', dir=output.parent))
    try:
        for split, raw in [('train', train), ('val', val)]:
            np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).tofile(staging / f'{split}.bin')
            metadata[f'{split}_tokens'] = len(raw)
            metadata[f'{split}_sha256'] = hashlib.sha256(raw).hexdigest()
            metadata[f'{split}_documents'] = len(_documents(raw))
        (staging / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        # mkdir is an atomic no-overwrite claim; readers should require meta.json.
        output.mkdir()
        try:
            for name in ('train.bin', 'val.bin', 'meta.json'):
                os.replace(staging / name, output / name)
        except BaseException:
            # Remove only files created in this new output directory.
            for name in ('train.bin', 'val.bin', 'meta.json'):
                (output / name).unlink(missing_ok=True)
            output.rmdir()
            raise
    finally:
        shutil.rmtree(staging)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--source', required=True, help='provenance description, not a download')
    parser.add_argument('--license', required=True, help='license or permission you have verified')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.train, args.val, args.out_dir, source=args.source, license_name=args.license)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
