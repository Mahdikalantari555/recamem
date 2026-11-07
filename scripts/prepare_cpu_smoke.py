#!/usr/bin/env python3
"""Pack the self-authored byte corpus for a CPU pipeline smoke test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def prepare(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {'vocab_size': 256, 'tokenizer': 'utf8_bytes_256', 'kind': 'smoke_fixture',
            'license': 'MIT', 'dataset_source': 'fixture', 'source': 'data/cpu-smoke', 'bytes_per_token': 1.0}
    for split in ('train', 'val'):
        raw = (ROOT / 'data' / 'cpu-smoke' / (split + '.txt')).read_bytes()
        np.asarray(list(raw), dtype=np.uint16).tofile(out / (split + '.bin'))
        meta[split + '_tokens'] = len(raw)
        meta[split + '_sha256'] = hashlib.sha256(raw).hexdigest()
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='runs/cpu-smoke-data')
    args = p.parse_args()
    print(json.dumps(prepare(args.out_dir), indent=2))

if __name__ == '__main__':
    main()
