#!/usr/bin/env python3
"""Edge-scale pretraining run: prepare, train, checkpoint, report.

    PYTHONPATH=. python scripts/train_edge_scale.py \
        --corpus docs plans README.md \
        --dim 384 --layers 8 --chunk-size 128 --chunked \
        --seed 0 --checkpoint-interval 1000000 \
        --report reports/edge-pretraining.json \
        --byte-target 2000000

The corpus is prepared deterministically (a document-level split derived from
the seed and the document identifiers), the held-out split, the grammar probe
file and any ``--reserved`` corpus are excluded from the training bytes, and the
run is resumable from ``--resume-from``. The report uses the benchmark report
format and states the chunked path's equivalence boundary; when the equivalence
check was not run on this environment, the report says so instead of claiming it.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.edge_pretraining import (  # noqa: E402
    DEFAULT_BYTE_TARGET,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HELD_OUT_FRACTION,
    EDGE_SCALE_CONFIG,
    HARDWARE_CLASS,
    run_scale,
)

DEFAULT_PROBE = "evals/grammar_probes.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", nargs="+", required=True,
                        help="files or directories of the training corpus")
    parser.add_argument("--dim", type=int, default=EDGE_SCALE_CONFIG["dim"])
    parser.add_argument("--layers", type=int, default=EDGE_SCALE_CONFIG["layers"])
    parser.add_argument("--spread", type=int, default=EDGE_SCALE_CONFIG["spread"])
    parser.add_argument("--vocab-size", type=int, default=EDGE_SCALE_CONFIG["vocab_size"])
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="bytes per chunk (one optimizer step per chunk)")
    parser.add_argument("--chunked", dest="chunked", action="store_true", default=True,
                        help="train with the chunked pass (default)")
    parser.add_argument("--per-token", dest="chunked", action="store_false",
                        help="train with the per-token reference path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-fraction", type=float, default=DEFAULT_HELD_OUT_FRACTION)
    parser.add_argument("--probe", nargs="*", default=[DEFAULT_PROBE],
                        help="grammar probe files, excluded from the training bytes")
    parser.add_argument("--reserved", nargs="*", default=[],
                        help="corpora reserved for benchmarking, excluded from training")
    parser.add_argument("--byte-target", type=int, default=DEFAULT_BYTE_TARGET,
                        help="bytes consumed before the run is complete")
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="bytes between checkpoints (0 disables)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path; also the resume source with --resume")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the checkpoint's recorded corpus position")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="stop after this many chunks (a short run)")
    parser.add_argument("--verify-equivalence", action="store_true",
                        help="run the per-token equivalence check on this environment")
    parser.add_argument("--report", default="reports/edge-pretraining.json",
                        help="report path (written atomically)")
    parser.add_argument("--hardware-class", default=HARDWARE_CLASS,
                        help="the documented hardware class this run completed on")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = " ".join([sys.executable, *sys.argv])
    checkpoint = args.checkpoint
    if args.resume and not checkpoint:
        raise SystemExit("--resume requires --checkpoint")
    if checkpoint and not args.resume and os.path.exists(checkpoint):
        # A fresh run must not silently continue on top of an old checkpoint.
        raise SystemExit(
            f"{checkpoint} exists; pass --resume to continue it or remove it"
        )
    config = {
        "dim": args.dim,
        "layers": args.layers,
        "spread": args.spread,
        "vocab_size": args.vocab_size,
    }
    report = run_scale(
        args.corpus,
        command=command,
        config=config,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        probe_files=args.probe,
        reserved_sources=args.reserved,
        chunk_size=args.chunk_size,
        chunked=args.chunked,
        byte_target=args.byte_target,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=checkpoint,
        report_path=args.report,
        max_chunks=args.max_chunks,
        verify_equivalence=args.verify_equivalence,
        hardware_class=args.hardware_class,
    )
    measurements = report.get("measurements", {})
    print(f"outcome={report['outcome']}")
    print(f"report={args.report}")
    for name in ("bytes_consumed", "wall_clock_seconds", "final_training_loss",
                 "held_out_perplexity", "parameter_count", "hardware_class"):
        record = measurements.get(name, {})
        print(f"{name}={record.get('value', record.get('status'))}")
    if report.get("unmet_criteria"):
        print("unmet=" + ",".join(report["unmet_criteria"]))
    return 0 if report["outcome"] in ("passed", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
