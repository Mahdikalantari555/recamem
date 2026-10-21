#!/usr/bin/env python3
"""Runnable edge benchmark harness.

Measures, in one run and one report, the numbers the project argues about:

* generation throughput on the sequential per-token path;
* training throughput on the per-token path;
* the chunked forward against the sequential forward, with the ratio;
* the parameter count of the configuration;
* the resident memory footprint with the artifacts that were loaded;
* byte-level perplexity on a hash-selected held-out document split;
* the contrastive grammar probe score, per phenomenon.

Every measurement carries its conditions; a measurement that cannot be taken is
reported as skipped with a reason and no value, and the run still writes its
report. Nothing here touches the network.

Usage::

    PYTHONPATH=. python scripts/bench_edge.py --help
    PYTHONPATH=. python scripts/bench_edge.py --checkpoint droids/droid-rs/rtu.safetensors \\
        --corpus README.md --tokens 2000 --probes evals/grammar_probes.jsonl \\
        --report reports/bench-edge.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/bench_edge.py`
    sys.path.insert(0, str(ROOT))

from src.model import benchmarks  # noqa: E402
from src.model.probes import load_probes, score_probes  # noqa: E402
from src.model.rtu import Model  # noqa: E402

DEFAULT_CORPUS_FILES = ("README.md", "VISION.md")
PROBE_FILE = ROOT / "evals" / "grammar_probes.jsonl"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_edge.py",
        description="Measure throughput, size, held-out perplexity and grammar probes.",
    )
    parser.add_argument("--profile", default=None,
                        help="Droid profile directory; its manifest names the checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Byte-RTU checkpoint to measure (safetensors)")
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Corpus files for throughput and the perplexity split")
    parser.add_argument("--tokens", type=int, default=2000,
                        help="Tokens per timed throughput run (default 2000)")
    parser.add_argument("--probes", default=str(PROBE_FILE),
                        help="Grammar probe JSONL file")
    parser.add_argument("--report", default="reports/bench-edge.json",
                        help="Where to write the JSON report")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per measurement")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs")
    parser.add_argument("--chunk", type=int, default=128, help="Chunk size for the scan")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension when constructing")
    parser.add_argument("--layers", type=int, default=2, help="Model layer count")
    parser.add_argument("--spread", type=int, default=32, help="Decay half-life spread")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--seed", type=int, default=0, help="Seed for construction and split")
    parser.add_argument("--held-out-fraction", type=float, default=0.25,
                        help="Document-level held-out fraction")
    parser.add_argument("--allow-busy-machine", action="store_true",
                        help="Record the load guard result instead of skipping on load")
    return parser


def resolve_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    """The checkpoint to measure, or ``None`` with the reason left to the caller."""
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.profile:
        manifest = Path(args.profile) / "config.json"
        if manifest.is_file():
            try:
                config = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                return None
            declared = config.get("byte_rtu_checkpoint") or config.get("rtu_checkpoint")
            if declared:
                return Path(args.profile) / str(declared)
    return None


def load_corpus(files: Optional[List[str]]) -> Dict[str, bytes]:
    documents: Dict[str, bytes] = {}
    for name in files or list(DEFAULT_CORPUS_FILES):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            documents[path.name] = path.read_bytes()
    if not documents:
        # A deterministic synthetic corpus keeps the harness runnable anywhere.
        documents["synthetic.txt"] = (
            b"The quick brown fox jumps over the lazy dog. " * 64
        )
    return documents


def corpus_bytes(documents: Dict[str, bytes]) -> bytes:
    return b"\n".join(documents[key] for key in sorted(documents))


def build_model(args: argparse.Namespace) -> Model:
    torch.manual_seed(args.seed)
    return Model(dim=args.dim, layers=args.layers, spread=args.spread, vocab_size=args.vocab)


def checkpointed_model(args: argparse.Namespace, checkpoint: Path) -> Model:
    model = build_model(args)
    model.load(str(checkpoint))
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def frozen_pass(model: Model, tokens: torch.Tensor) -> None:
    """One sequential frozen generation pass over ``tokens``."""
    with torch.no_grad():
        model.reset()
        for byte in tokens.tolist():
            model.step(torch.tensor(byte, dtype=torch.long), frozen=True)


def chunked_pass(model: Model, tokens: torch.Tensor, chunk: int) -> None:
    """One chunked forward pass over ``tokens`` (one batch element)."""
    with torch.no_grad():
        model.forward_chunk(tokens.view(1, -1), chunk=chunk)


def training_pass(model: Model, tokens: torch.Tensor) -> None:
    """One per-token training pass (one optimizer step per byte)."""
    model.reset()
    data = tokens.tolist()
    for current, following in zip(data, data[1:]):
        model.forward(current, nextb=following)


def perplexity_score(model: Model, documents: Dict[str, bytes]) -> Tuple[float, int]:
    """Total negative log-likelihood and scored byte count on held-out docs."""
    total = 0.0
    counted = 0
    with torch.no_grad():
        for document_id in sorted(documents):
            data = documents[document_id]
            if len(data) < 2:
                continue
            model.reset()
            logits = None
            for index, byte in enumerate(data):
                if logits is not None:
                    log_probs = torch.log_softmax(logits.reshape(-1).float(), dim=-1)
                    total += float(-log_probs[byte].item())
                    counted += 1
                if index + 1 < len(data):
                    token = torch.tensor(byte, dtype=torch.long)
                    (_, _, _), (logits, _) = model.step(token, frozen=True)
    return total, counted


def throughput_tokens(corpus: bytes, count: int) -> torch.Tensor:
    if not corpus:
        corpus = b"bench"
    repeats = count // len(corpus) + 1
    return torch.tensor(list((corpus * repeats)[:count]), dtype=torch.long)


def collect_measurements(args: argparse.Namespace, documents: Dict[str, bytes]
                         ) -> Tuple[Dict[str, Any], List[str], bool]:
    measurements: Dict[str, Any] = {}
    notes: List[str] = []
    failed = False
    data = corpus_bytes(documents)
    tokens = throughput_tokens(data, args.tokens)

    checkpoint = resolve_checkpoint(args)
    if checkpoint is None:
        reason = "no checkpoint supplied (use --checkpoint or a profile with a manifest)"
        measurements["throughput"] = benchmarks.skipped(reason)
        measurements["training_throughput"] = benchmarks.skipped(reason)
        measurements["perplexity"] = benchmarks.skipped(reason)
        measurements["probe_score"] = benchmarks.skipped(reason)
    elif not checkpoint.is_file():
        reason = f"checkpoint not found: {checkpoint}"
        measurements["throughput"] = benchmarks.skipped(reason)
        measurements["training_throughput"] = benchmarks.skipped(reason)
        measurements["perplexity"] = benchmarks.skipped(reason)
        measurements["probe_score"] = benchmarks.skipped(reason)
    else:
        model = checkpointed_model(args, checkpoint)
        checkpoint_hash = benchmarks.sha256_file(checkpoint)
        measurements["params"] = benchmarks.measurement(
            benchmarks.count_trainable_parameters(model),
            metric="params",
            config={"dim": args.dim, "layers": args.layers, "vocab": args.vocab},
        )
        measurements["throughput"] = benchmarks.measure_throughput(
            lambda: frozen_pass(model, tokens),
            tokens.numel(),
            runs=args.runs,
            warmup=args.warmup,
            path="sequential",
            sequence_length=tokens.numel(),
            threads=torch.get_num_threads(),
        )
        train_tokens = tokens[: max(2, min(args.tokens, 64))]
        measurements["training_throughput"] = benchmarks.measure_throughput(
            lambda: training_pass(model, train_tokens),
            max(1, train_tokens.numel() - 1),
            runs=max(1, args.runs - 1),
            warmup=0,
            path="per_token",
            sequence_length=train_tokens.numel(),
            threads=torch.get_num_threads(),
        )
        try:
            measurements["perplexity"] = benchmarks.measure_held_out_perplexity(
                lambda held: perplexity_score(model, held),
                documents,
                held_out_fraction=args.held_out_fraction,
                seed=args.seed,
            )
        except Exception as exc:  # a failed measurement is recorded, not hidden
            failed = True
            measurements["perplexity"] = benchmarks.skipped(
                f"perplexity measurement raised {type(exc).__name__}: {exc}"
            )

        probe_path = Path(args.probes)
        if not probe_path.is_absolute():
            probe_path = ROOT / probe_path
        if probe_path.is_file():
            probe_set = load_probes(probe_path)
            notes.extend(
                f"probe line {error['line']} rejected: {error['reason']}"
                for error in probe_set.errors
            )
            measurements["probe_score"] = score_probes(
                model, probe_set, checkpoint_path=checkpoint, probe_path=probe_path
            )
        else:
            measurements["probe_score"] = benchmarks.skipped(
                f"probe file not found: {probe_path}"
            )

        measurements["rss"] = benchmarks.measure_resident_memory([str(checkpoint)])
        measurements["throughput"]["checkpoint_hash"] = checkpoint_hash

    if "params" not in measurements:
        measurements["params"] = benchmarks.measurement(
            _configuration_params(args),
            metric="params",
            config={"dim": args.dim, "layers": args.layers, "vocab": args.vocab},
        )
    if "rss" not in measurements:
        measurements["rss"] = benchmarks.measure_resident_memory([])

    # The reference comparison is meaningful without a checkpoint: it times the
    # two forward implementations on a freshly constructed model.
    model = build_model(args)
    measurements["reference_comparison"] = benchmarks.compare_reference_paths(
        lambda: frozen_pass(model, tokens),
        lambda: chunked_pass(model, tokens, args.chunk),
        tokens.numel(),
        runs=args.runs,
        warmup=args.warmup,
        sequence_length=tokens.numel(),
        threads=torch.get_num_threads(),
        load={} if args.allow_busy_machine else None,
    )
    return measurements, notes, failed


def _configuration_params(args: argparse.Namespace) -> int:
    """Parameter count of the configuration, without loading a checkpoint."""
    torch.manual_seed(args.seed)
    model = Model(dim=args.dim, layers=args.layers, spread=args.spread,
                  vocab_size=args.vocab)
    return benchmarks.count_trainable_parameters(model)


def summary_lines(measurements: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    throughput = measurements.get("throughput", {})
    if benchmarks.is_measured(throughput):
        lines.append(f"tokens_per_second={throughput['value']:.2f}")
    else:
        lines.append(f"tokens_per_second_skipped={throughput.get('reason', 'not measured')}")
    train = measurements.get("training_throughput", {})
    if benchmarks.is_measured(train):
        lines.append(f"training_tokens_per_second={train['value']:.2f}")
    else:
        lines.append(
            f"training_tokens_per_second_skipped={train.get('reason', 'not measured')}"
        )
    params = measurements.get("params", {})
    if benchmarks.is_measured(params):
        lines.append(f"params={int(params['value'])}")
    rss = measurements.get("rss", {})
    if benchmarks.is_measured(rss):
        lines.append(f"rss_mb={rss['value']:.1f}")
    perplexity = measurements.get("perplexity", {})
    if benchmarks.is_measured(perplexity):
        lines.append(f"held_out_perplexity={perplexity['value']:.3f}")
    else:
        lines.append(
            f"held_out_perplexity_skipped={perplexity.get('reason', 'not measured')}"
        )
    comparison = measurements.get("reference_comparison", {})
    if benchmarks.is_measured(comparison):
        lines.append(f"chunked_speed_factor={comparison['value']:.2f}")
    else:
        lines.append(
            f"chunked_speed_factor_skipped={comparison.get('reason', 'not measured')}"
        )
    probes = measurements.get("probe_score", {})
    if benchmarks.is_measured(probes):
        for name, stats in probes.get("per_phenomenon", {}).items():
            lines.append(f"grammar_accuracy.{name}={stats['accuracy']:.3f}")
        lines.append(f"grammar_accuracy={probes['accuracy']:.3f}")
    else:
        lines.append(f"grammar_accuracy_skipped={probes.get('reason', 'not measured')}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    documents = load_corpus(args.corpus)
    measurements, notes, failed = collect_measurements(args, documents)
    config = {
        "seed": args.seed,
        "tokens": args.tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "chunk": args.chunk,
        "model": {"dim": args.dim, "layers": args.layers,
                  "spread": args.spread, "vocab": args.vocab},
        "corpus": sorted(documents),
        "held_out_fraction": args.held_out_fraction,
        "probes": str(args.probes),
        "checkpoint": str(resolve_checkpoint(args) or ""),
    }
    blocked = not documents
    report = benchmarks.build_report(
        command=" ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        config=config,
        measurements=measurements,
        commit=benchmarks.git_commit(ROOT),
        hardware=benchmarks.hardware_description(),
        dependencies=benchmarks.dependency_versions(),
        failed=failed,
        blocked=blocked,
        skip_reason="no corpus documents were available" if blocked else None,
    )
    path = benchmarks.write_report(report, args.report)

    for line in summary_lines(measurements):
        print(line)
    for note in notes:
        print(f"# {note}", file=sys.stderr)
    print(f"outcome={report['outcome']}")
    print(f"report={path}")
    print(f"# hardware: {report['hardware']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
