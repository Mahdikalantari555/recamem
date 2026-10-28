import argparse
import glob
import itertools
import os
import random
import sys
import time
from datetime import datetime
import torch

from src.model.rtu import Model, Layer, Encoder, Decoder

# RecaMem: PyTorch runtime for the RecaMem memory engine (optional embedded micro-model).

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.parallel_scan = bool(kwargs.pop('parallel_scan', False))
        self.scan_chunk = int(kwargs.pop('scan_chunk', 128))
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold
        self.step = 0

    def save(self):
        self.step += 1
        if self.step % 500 == 0:
            self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, save: bool, frozen: bool, ce_only: bool,
             planned_states: list[torch.Tensor] | None = None):
        outputs = self.model.forward_planned(c, n, end, frozen, ce_only, planned_states=planned_states)
        if save:
            self.save()
        return outputs

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool, ce_only: bool):
        timestamp = None

        while True:
            text = input(f'\n[{self.now()} | {0 if timestamp is None else time.time() - timestamp:.4f}s]\nUser >> ')
            timestamp = time.time()

            data = (text + '\n').encode('utf-8')

            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, save, frozen, ce_only)

            print(f'\n[{self.now()}]\nModel >> ', end='', flush=True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen, ce_only)
                self.write(b)

                if stop > self.threshold:
                    print()
                    break

    def train_sequential(self, data: bytes, save: bool, frozen: bool, ce_only: bool):
        for i, (c, n) in enumerate(itertools.pairwise(data)):
            b, _ = self.call(c, n, i == len(data) - 2, save, frozen, ce_only)
            self.write(b)

    def train_parallel(self, data: bytes, save: bool, frozen: bool, ce_only: bool):
        """Training path where each chunk's state trajectory comes from the scan.

        The pairwise token loop, losses, trace-buffer commits and optimizer
        steps are the same calls the sequential path makes; only the
        state-trajectory computation is replaced by
        ``Model.scan_states`` (``associative_scan``) once per chunk.
        """
        last = len(data) - 2
        for start in range(0, len(data) - 1, self.scan_chunk):
            stop = min(start + self.scan_chunk, len(data) - 1)
            chunk = torch.tensor(list(data[start:stop]), dtype=torch.long).unsqueeze(0)
            trajectories = self.model.scan_states(chunk, chunk=self.scan_chunk)

            for offset, i in enumerate(range(start, stop)):
                # The chunk is (1, C); the per-token path carries (D,) state.
                planned = [trajectory[0, offset] for trajectory in trajectories]
                b, _ = self.call(data[i], data[i + 1], i == last, save, frozen, ce_only,
                                 planned_states=planned)
                self.write(b)

    def train(self, save: bool, frozen: bool, dataset: str, ce_only: bool):
        files = glob.glob(dataset, recursive=True)

        if not files:
            raise FileNotFoundError(
                f'Could not find training files with the following glob: {dataset!r}. Try downloading a dataset first.'
            )

        random.shuffle(files)

        while True:
            for file in files:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        if len(data) < 2:
                            continue

                        if self.parallel_scan:
                            self.train_parallel(data, save, frozen, ce_only)
                        else:
                            self.train_sequential(data, save, frozen, ce_only)

    def now(self):
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool, ce_only: bool):
        # Model.load() is strict: a missing checkpoint raises. The CLI treats a
        # missing path as an explicit request to train from random init.
        if os.path.exists(self.path):
            self.model.load(self.path)
        else:
            print(f'No checkpoint at {self.path!r}; starting from random initialisation.')
        print(f'parameters: {self.model.count():,}\n')

        try:
            match mode:
                case 'train': self.train(save, frozen, dataset, ce_only)
                case 'chat': self.chat(save, frozen, ce_only)

        finally:
            if save:
                self.model.save(self.path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='recamem')
    parser.add_argument('path')
    parser.add_argument('mode', choices=['train', 'chat'])

    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--ce-only', action='store_true')
    parser.add_argument('--no-save', action='store_false')
    parser.add_argument('--dataset', default='wikipedia_clean/**/wiki_*')
    parser.add_argument('--parallel-scan', action='store_true',
                        help='Compute each training chunk state trajectory with the associative scan (default: off).')
    parser.add_argument('--scan-chunk', type=int, default=128,
                        help='Tokens per chunk for --parallel-scan (default: 128).')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    Runtime(
        path=args.path, threshold=0.35,
        dim=512, layers=16, spread=32, temp=0.75,
        rate=5e-4, bound=(40000, 120000),
        parallel_scan=args.parallel_scan, scan_chunk=args.scan_chunk,
    )(args.mode, args.dataset, args.no_save, args.frozen, args.ce_only)
