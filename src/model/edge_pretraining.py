"""Edge-scale pretraining protocol: corpus, resumable run, report, acceptance.

This module is the harness behind ``scripts/train_edge_scale.py`` and the
``training/edge-pretraining`` capability. It provides:

* deterministic document preparation and a document-level split, both derived
  from a hash of the document identifier and the seed (never from filesystem
  order);
* explicit exclusion of the held-out split, the grammar probe file and any
  reserved benchmark corpus from the training bytes, with a contamination check
  that names an overlapping document;
* resumable run state and atomic checkpoints carrying the weights, the
  optimizer state, the byte step count and the corpus position;
* a scale run that trains with the chunked path and writes a report in the
  benchmark report format, including the equivalence boundary;
* acceptance criteria that decide ``passed`` / ``failed`` / ``blocked`` and
  name every criterion that was not met.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from src.model import benchmarks
from src.model.rtu import CHUNK_EQUIVALENT_QUANTITIES, CHUNK_NON_EQUIVALENT_QUANTITIES, Model

DEFAULT_HELD_OUT_FRACTION = 0.2
DEFAULT_CHUNK_SIZE = 128
DEFAULT_EVAL_CHUNK = 256
DEFAULT_BYTE_TARGET = 25_000_000
HARDWARE_CLASS = "cpu-only"
REPORT_OUTCOMES = ("passed", "failed", "blocked", "skipped")
EQUIVALENCE_REFERENCE = "Model.per_token_gradient_oracle"

#: The configuration the project's headline claim is about.
EDGE_SCALE_CONFIG: Dict[str, Any] = {
    "dim": 1024,
    "layers": 24,
    "spread": 32,
    "vocab_size": 256,
}


def parameter_count(config: Mapping[str, Any]) -> int:
    """The parameter count ``Model.count()`` reports for a configuration."""
    dim = int(config["dim"])
    layers = int(config["layers"])
    vocab = int(config.get("vocab_size", 256))
    return vocab * dim + layers * (dim * dim + 3 * dim) + vocab * dim + vocab + dim + 1


# --------------------------------------------------------------------------- #
# Corpus preparation
# --------------------------------------------------------------------------- #
class CorpusContaminationError(RuntimeError):
    """Raised when an excluded document appears in the training bytes."""

    def __init__(self, offenders: Sequence[Mapping[str, Any]]):
        self.offenders = [dict(item) for item in offenders]
        names = ", ".join(sorted({str(item["document_id"]) for item in self.offenders}))
        super().__init__(f"excluded documents appear in the training set: {names}")


def _document_id(path: Path, root: Optional[Path]) -> str:
    try:
        if root is not None:
            return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return path.name


def collect_document_paths(sources: Iterable[str | os.PathLike],
                           root: Optional[str | os.PathLike] = None) -> Dict[str, Path]:
    """Map every document identifier to its path, independent of disk order."""
    root_path = Path(root) if root is not None else None
    found: Dict[str, Path] = {}
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if not candidate.is_file() or "__pycache__" in candidate.parts:
                    continue
                found[_document_id(candidate, root_path)] = candidate
        elif path.is_file():
            found[_document_id(path, root_path)] = path
        else:
            raise FileNotFoundError(f"Corpus path not found: {raw}")
    if not found:
        raise FileNotFoundError("No corpus documents found")
    return found


def read_documents(paths: Mapping[str, Path]) -> Dict[str, bytes]:
    return {doc_id: Path(path).read_bytes() for doc_id, path in paths.items()}


def order_documents(documents: Mapping[str, Any], seed: int) -> List[str]:
    """Document order from ``sha256(seed:document_id)``, never from disk."""
    return sorted(
        documents,
        key=lambda doc_id: hashlib.sha256(
            f"{int(seed)}:{doc_id}".encode("utf-8")
        ).hexdigest(),
    )


def split_documents(document_ids: Sequence[str], *, seed: int,
                    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION) -> Dict[str, Any]:
    """Document-level split, reusing the benchmark harness's split rule."""
    return benchmarks.document_split(
        {doc_id: b"" for doc_id in document_ids},
        held_out_fraction=held_out_fraction,
        seed=int(seed),
    )


def corpus_identity(documents: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for doc_id in sorted(documents):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(bytes(documents[doc_id])).digest())
    return digest.hexdigest()


def check_contamination(train_ids: Iterable[str], train_bytes: bytes,
                        excluded_documents: Mapping[str, bytes]) -> List[Dict[str, Any]]:
    """Excluded documents whose identifier or bytes reappear in the training set."""
    train_id_set = set(train_ids)
    offenders: List[Dict[str, Any]] = []
    for doc_id, data in excluded_documents.items():
        data = bytes(data)
        if doc_id in train_id_set:
            offenders.append({"document_id": doc_id, "reason": "document id is in the training set"})
        elif data and data in train_bytes:
            offenders.append({"document_id": doc_id, "reason": "document bytes are in the training set"})
    return offenders


def _read_optional(sources: Optional[Iterable[str | os.PathLike]],
                   root: Optional[Path]) -> Tuple[Dict[str, Path], Dict[str, bytes]]:
    if not sources:
        return {}, {}
    paths = collect_document_paths(sources, root=root)
    return paths, read_documents(paths)


def prepare_corpus(sources: Iterable[str | os.PathLike], *,
                   seed: int = 0,
                   held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
                   probe_files: Optional[Iterable[str | os.PathLike]] = None,
                   reserved_sources: Optional[Iterable[str | os.PathLike]] = None,
                   root: Optional[str | os.PathLike] = None,
                   strict: bool = True) -> Dict[str, Any]:
    """Prepare a deterministic training corpus and its excluded sets.

    The returned mapping carries the training bytes, the held-out documents,
    the excluded sets, the split rule and the corpus identity. ``strict``
    raises :class:`CorpusContaminationError` when the contamination check fails;
    a non-strict preparation records the offenders instead, so a run can still
    write a failing report that names them.
    """
    root_path = Path(root) if root is not None else None
    paths = collect_document_paths(sources, root=root_path)
    documents = read_documents(paths)

    probe_paths, probe_documents = _read_optional(probe_files, root_path)
    reserved_paths, reserved_documents = _read_optional(reserved_sources, root_path)

    identifiers = order_documents(documents, seed)
    split = split_documents(identifiers, seed=seed, held_out_fraction=held_out_fraction)

    held_out_documents = {doc_id: documents[doc_id] for doc_id in split["held_out_ids"]}
    train_ids = [doc_id for doc_id in split["train_ids"]]

    excluded_kinds: Dict[str, str] = {}
    for doc_id in split["held_out_ids"]:
        excluded_kinds[doc_id] = "held_out_split"
    for doc_id in probe_documents:
        excluded_kinds.setdefault(doc_id, "grammar_probe_file")
    for doc_id in reserved_documents:
        excluded_kinds.setdefault(doc_id, "reserved_benchmark_corpus")

    excluded_documents: Dict[str, bytes] = {}
    for doc_id, kind in excluded_kinds.items():
        if doc_id in documents:
            excluded_documents[doc_id] = documents[doc_id]
        elif kind == "grammar_probe_file":
            excluded_documents[doc_id] = probe_documents[doc_id]
        elif kind == "reserved_benchmark_corpus":
            excluded_documents[doc_id] = reserved_documents[doc_id]

    # The training bytes are the non-excluded documents, in hash order, joined
    # with a newline so the result depends only on the identifiers and contents.
    train_ids = [doc_id for doc_id in train_ids if doc_id not in excluded_kinds]
    train_bytes = b"\n".join(documents[doc_id] for doc_id in train_ids)

    offenders = check_contamination(train_ids, train_bytes, excluded_documents)
    if offenders and strict:
        raise CorpusContaminationError(offenders)

    excluded_sets = {
        "held_out_split": sorted(split["held_out_ids"]),
        "grammar_probe_file": sorted(probe_documents),
        "reserved_benchmark_corpus": sorted(reserved_documents),
    }
    if not excluded_sets["grammar_probe_file"]:
        excluded_sets["grammar_probe_file"] = []

    return {
        "seed": int(seed),
        "held_out_fraction": float(held_out_fraction),
        "split_rule": split["split_rule"],
        "order": identifiers,
        "train_ids": train_ids,
        "held_out_ids": list(split["held_out_ids"]),
        "documents": documents,
        "held_out_documents": held_out_documents,
        "probe_documents": probe_documents,
        "reserved_documents": reserved_documents,
        "excluded_documents": excluded_documents,
        "excluded_kinds": excluded_kinds,
        "excluded_sets": excluded_sets,
        "excluded_from_training": True,
        "train_bytes": train_bytes,
        "train_bytes_count": len(train_bytes),
        "held_out_bytes_count": sum(len(data) for data in held_out_documents.values()),
        "corpus_id": corpus_identity(documents),
        "train_sha256": benchmarks.sha256_bytes(train_bytes),
        "contamination": offenders,
        "contaminated": bool(offenders),
    }


def excluded_report(corpus: Mapping[str, Any]) -> Dict[str, Any]:
    """The exclusion ledger a report carries."""
    return {
        "excluded_from_training": True,
        "held_out_split": list(corpus["excluded_sets"]["held_out_split"]),
        "grammar_probe_file": list(corpus["excluded_sets"]["grammar_probe_file"]),
        "reserved_benchmark_corpus": list(corpus["excluded_sets"]["reserved_benchmark_corpus"]),
        "contamination_check": "failed" if corpus["contaminated"] else "passed",
        "contaminated_documents": [item["document_id"] for item in corpus["contamination"]],
    }


# --------------------------------------------------------------------------- #
# Run state and checkpoints
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return str(path)


@dataclass
class RunState:
    """Corpus position, byte step count, seeds and configuration."""

    corpus_position: int = 0
    step_count: int = 0
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_position": int(self.corpus_position),
            "step_count": int(self.step_count),
            "seed": int(self.seed),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunState":
        return cls(
            corpus_position=int(payload.get("corpus_position", 0)),
            step_count=int(payload.get("step_count", 0)),
            seed=int(payload.get("seed", 0)),
            config=dict(payload.get("config", {})),
        )

    def save(self, path: str | os.PathLike) -> str:
        return _write_json_atomic(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RunState":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def state_path_for(checkpoint: str | os.PathLike) -> Path:
    return Path(str(checkpoint) + ".state.json")


def write_checkpoint(checkpoint: str | os.PathLike, model: Model,
                     state: RunState) -> Dict[str, Any]:
    """Write the state, then the weights, both atomically.

    The state file goes first so a crash cannot leave weights that claim a
    corpus position they do not have. ``Model.save`` writes a temporary file and
    renames it, so an interrupted weights write leaves the previous checkpoint
    intact and readable.
    """
    target = Path(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    state.step_count = int(model.step_count)
    state_path = state_path_for(target)
    # The state is staged next to its destination and only moved into place
    # after the weights land, so a failed weights write cannot leave a corpus
    # position that claims bytes the checkpoint never saw.
    tmp_state = state_path.with_name(state_path.name + ".pending")
    state.save(tmp_state)
    try:
        model.save(str(target))
    except BaseException:
        tmp_state.unlink(missing_ok=True)
        raise
    os.replace(tmp_state, state_path)
    return {
        "checkpoint": str(target),
        "state": str(state_path),
        "corpus_position": int(state.corpus_position),
        "step_count": int(state.step_count),
    }


def read_checkpoint(checkpoint: str | os.PathLike,
                    model: Optional[Model] = None) -> RunState:
    target = Path(checkpoint)
    state_path = state_path_for(target)
    if not state_path.is_file():
        raise FileNotFoundError(f"Run state not found: {state_path}")
    state = RunState.load(state_path)
    if model is not None:
        model.load(str(target))
        model.step_count = state.step_count
    return state


# --------------------------------------------------------------------------- #
# Held-out perplexity
# --------------------------------------------------------------------------- #
def byte_nll(model: Model, data: bytes, *, chunk: int = DEFAULT_EVAL_CHUNK) -> Tuple[float, int]:
    """Cross-entropy of ``data`` under the model, in nats, and the byte count."""
    data = bytes(data)
    if len(data) < 2:
        return 0.0, 0
    total = 0.0
    scored = 0
    with torch.no_grad():
        for start in range(0, len(data) - 1, chunk):
            window = data[start:start + chunk + 1]
            if len(window) < 2:
                break
            tokens = torch.tensor([list(window[:-1])], dtype=torch.long)
            targets = torch.tensor([list(window[1:])], dtype=torch.long)
            _, _, (logits, _stop) = model.forward_chunk(tokens, chunk=chunk)
            mask = torch.ones_like(targets, dtype=torch.bool)
            total += float(model._batched_ce(logits, targets, mask).sum().item())
            scored += int(targets.numel())
    return total, scored


def held_out_perplexity(model: Model, corpus: Mapping[str, Any], *,
                        chunk: int = DEFAULT_EVAL_CHUNK) -> Dict[str, Any]:
    """Held-out perplexity on the split this corpus prepared, measured only."""
    documents = {
        doc_id: corpus["documents"][doc_id]
        for doc_id in corpus["held_out_ids"] + corpus["train_ids"]
    }

    def score(held_out: Mapping[str, bytes]) -> Tuple[float, int]:
        total = 0.0
        scored = 0
        for data in held_out.values():
            nll, count = byte_nll(model, data, chunk=chunk)
            total += nll
            scored += count
        return total, scored

    record = benchmarks.measure_held_out_perplexity(
        score,
        documents,
        held_out_fraction=float(corpus["held_out_fraction"]),
        seed=int(corpus["seed"]),
        corpus_id=str(corpus["corpus_id"]),
    )
    record["excluded_from_training"] = True
    record["held_out_documents"] = list(corpus["held_out_ids"])
    return record


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_model(config: Mapping[str, Any]) -> Model:
    return Model(
        dim=int(config["dim"]),
        layers=int(config["layers"]),
        spread=int(config.get("spread", 32)),
        vocab_size=int(config.get("vocab_size", 256)),
    )


def _chunk_losses(model: Model, tokens: torch.Tensor, next_tokens: torch.Tensor,
                  *, chunk: int, chunked: bool) -> Tuple[float, Optional[Dict[str, Any]]]:
    if chunked:
        info = model.train_chunk(tokens, next_tokens, chunk=chunk)
        return float(info["loss"]), info

    total = 0.0
    for index in range(tokens.shape[1]):
        _, _ = model.forward(
            tokens[0, index], nextb=next_tokens[0, index],
            end=index == tokens.shape[1] - 1,
        )
        total += float(model.last_loss)
    return total, None


def train_stream(model: Model, corpus: Mapping[str, Any], state: RunState, *,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunked: bool = True,
                 byte_target: int = DEFAULT_BYTE_TARGET,
                 checkpoint_interval: int = 0,
                 checkpoint_path: Optional[str] = None,
                 max_chunks: Optional[int] = None,
                 interrupt_after_bytes: Optional[int] = None,
                 ) -> Dict[str, Any]:
    """Train over the prepared bytes from ``state.corpus_position``.

    Bytes before the recorded position are never read again: a resumed run
    starts exactly at the checkpoint's corpus position.
    """
    data = bytes(corpus["train_bytes"])
    if not data:
        raise ValueError("the training corpus is empty")
    start_position = int(state.corpus_position)
    if start_position >= len(data):
        start_position = 0
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    positions: List[int] = []
    losses: List[float] = []
    consumed_this_run = 0
    checkpoints: List[Dict[str, Any]] = []
    interrupted = False
    position = start_position
    chunks = 0
    started_at = time.perf_counter()

    while position + 1 < len(data):
        if state.step_count >= int(byte_target):
            break
        if max_chunks is not None and chunks >= int(max_chunks):
            break
        if interrupt_after_bytes is not None and state.step_count >= int(interrupt_after_bytes):
            interrupted = True
            break

        window = data[position:position + chunk_size + 1]
        if len(window) < 2:
            break
        tokens = torch.tensor([list(window[:-1])], dtype=torch.long)
        next_tokens = torch.tensor([list(window[1:])], dtype=torch.long)
        positions.append(position)
        loss, _info = _chunk_losses(
            model, tokens, next_tokens, chunk=chunk_size, chunked=bool(chunked)
        )
        losses.append(loss)
        position += tokens.shape[1]
        consumed_this_run += tokens.shape[1]
        chunks += 1
        state.corpus_position = position
        state.step_count = int(model.step_count)

        if checkpoint_path and checkpoint_interval > 0 and \
                state.step_count >= (len(checkpoints) + 1) * checkpoint_interval:
            checkpoints.append(write_checkpoint(checkpoint_path, model, state))

    elapsed = time.perf_counter() - started_at
    return {
        "start_position": start_position,
        "end_position": position,
        "positions": positions,
        "chunks": chunks,
        "bytes_consumed": int(state.step_count),
        "bytes_consumed_this_run": int(consumed_this_run),
        "wall_clock_seconds": elapsed,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "checkpoints": checkpoints,
        "interrupted": interrupted,
        "state": state,
    }


# --------------------------------------------------------------------------- #
# Acceptance and report
# --------------------------------------------------------------------------- #
def evaluate_acceptance(*, bytes_consumed: int, byte_target: int,
                        parameter_count_actual: int, parameter_count_expected: int,
                        initial_loss: Optional[float], final_loss: Optional[float],
                        perplexity: Mapping[str, Any], hardware_class: str,
                        documented_hardware_class: str = HARDWARE_CLASS,
                        completed: bool = True,
                        contaminated: bool = False) -> List[Dict[str, Any]]:
    """One record per criterion, each with the reason it did not hold."""
    criteria: List[Dict[str, Any]] = []

    def record(name: str, met: bool, **details: Any) -> None:
        entry = {"criterion": name, "met": bool(met)}
        entry.update(details)
        criteria.append(entry)

    record("byte_target", bytes_consumed >= byte_target,
           bytes_consumed=int(bytes_consumed), target=int(byte_target))
    record("parameter_count", parameter_count_actual == parameter_count_expected,
           actual=int(parameter_count_actual), expected=int(parameter_count_expected))
    decreased = (
        initial_loss is not None and final_loss is not None
        and math.isfinite(float(final_loss)) and float(final_loss) < float(initial_loss)
    )
    record("loss_decreased", decreased, initial_loss=initial_loss, final_loss=final_loss)
    measured_perplexity = benchmarks.is_measured(perplexity)
    finite = measured_perplexity and math.isfinite(float(perplexity.get("value", float("nan"))))
    record(
        "held_out_perplexity",
        bool(finite and perplexity.get("excluded_from_training")),
        status=perplexity.get("status"),
        value=perplexity.get("value"),
        excluded_from_training=bool(perplexity.get("excluded_from_training")),
    )
    record("hardware_class", hardware_class == documented_hardware_class,
           hardware_class=hardware_class, documented=documented_hardware_class)
    record("completed", bool(completed), interrupted=not completed)
    record("contamination", not contaminated, contaminated=bool(contaminated))
    return criteria


def scale_report(*, command: str, config: Mapping[str, Any], corpus: Mapping[str, Any],
                 run: Mapping[str, Any], model: Model, perplexity: Mapping[str, Any],
                 hardware_class: str = HARDWARE_CLASS,
                 equivalence: Optional[Mapping[str, Any]] = None,
                 checkpoint: Optional[str] = None,
                 outcome_override: Optional[str] = None,
                 acceptance: Optional[Sequence[Mapping[str, Any]]] = None,
                 notes: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the report document for a scale run."""
    parameter_count_actual = int(model.count())
    parameter_count_expected = int(parameter_count(config))
    criteria = list(acceptance) if acceptance is not None else evaluate_acceptance(
        bytes_consumed=int(run["bytes_consumed"]),
        byte_target=int(config["byte_target"]),
        parameter_count_actual=parameter_count_actual,
        parameter_count_expected=parameter_count_expected,
        initial_loss=run.get("initial_loss"),
        final_loss=run.get("final_loss"),
        perplexity=perplexity,
        hardware_class=hardware_class,
        completed=not bool(run.get("interrupted")),
        contaminated=bool(corpus.get("contaminated")),
    )
    unmet = [item["criterion"] for item in criteria if not item["met"]]
    if outcome_override:
        outcome = outcome_override
    elif run.get("interrupted"):
        outcome = "blocked"
    elif unmet:
        outcome = "failed"
    else:
        outcome = "passed"

    measurements: Dict[str, Dict[str, Any]] = {
        "bytes_consumed": benchmarks.measurement(
            int(run["bytes_consumed"]), target=int(config["byte_target"]),
        ),
        "wall_clock_seconds": benchmarks.measurement(
            float(run["wall_clock_seconds"]), path="chunked" if config.get("chunked") else "per_token",
        ),
        "final_training_loss": benchmarks.measurement(
            run.get("final_loss") if run.get("final_loss") is not None else float("nan")
        ),
        "initial_training_loss": benchmarks.measurement(
            run.get("initial_loss") if run.get("initial_loss") is not None else float("nan")
        ),
        "held_out_perplexity": dict(perplexity),
        "parameter_count": benchmarks.measurement(
            parameter_count_actual, expected=parameter_count_expected,
        ),
        "hardware_class": benchmarks.measurement(
            hardware_class, documented=HARDWARE_CLASS,
        ),
    }
    for measurement_name, run_key in (
        ("final_training_loss", "final_loss"),
        ("initial_training_loss", "initial_loss"),
    ):
        if run.get(run_key) is None:
            measurements[measurement_name] = benchmarks.skipped(
                "the run processed no chunk", path="none"
            )

    boundary = dict(equivalence or {})
    if boundary:
        measurements["equivalence_boundary"] = dict(boundary)

    report = benchmarks.build_report(
        command=command,
        config={
            **{k: v for k, v in config.items() if k != "byte_target"},
            "seed": int(corpus["seed"]),
            "byte_target": int(config["byte_target"]),
        },
        measurements=measurements,
        outcome=outcome,
        failed=outcome == "failed",
        blocked=outcome == "blocked",
        artifacts={
            "checkpoint": checkpoint,
            "code": "scripts/train_edge_scale.py",
        },
        skip_reason="; ".join(unmet) if unmet and outcome != "passed" else None,
    )
    report["corpus"] = {
        "identity": corpus["corpus_id"],
        "train_sha256": corpus["train_sha256"],
        "train_bytes_available": corpus["train_bytes_count"],
        "held_out_bytes_available": corpus["held_out_bytes_count"],
        "split_rule": corpus["split_rule"],
        "seed": corpus["seed"],
        **excluded_report(corpus),
    }
    report["acceptance"] = [dict(item) for item in criteria]
    report["acceptance_met"] = not unmet
    report["unmet_criteria"] = unmet
    report["equivalence"] = boundary
    report["run"] = {
        "start_position": run.get("start_position"),
        "end_position": run.get("end_position"),
        "chunks": run.get("chunks"),
        "bytes_consumed": run.get("bytes_consumed"),
        "interrupted": bool(run.get("interrupted")),
        "wall_clock_seconds": run.get("wall_clock_seconds"),
    }
    if notes:
        report["notes"] = dict(notes)
    return report


def equivalence_report(chunked: bool, verification: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The equivalence boundary, with the missing verification named."""
    if not chunked:
        return {
            "path": "per_token",
            "equivalent": [],
            "not_equivalent": [],
            "claim": False,
            "note": "the run used the per-token reference path; no chunked claim is made",
        }
    if verification and verification.get("verified"):
        return {
            "path": "chunked",
            "equivalent": list(CHUNK_EQUIVALENT_QUANTITIES),
            "not_equivalent": list(CHUNK_NON_EQUIVALENT_QUANTITIES),
            "claim": True,
            "reference": EQUIVALENCE_REFERENCE,
            "tolerance": verification.get("tolerance"),
            "note": "losses, accumulated gradients and trace buffers equal the per-token reference",
        }
    return {
        "path": "chunked",
        "equivalent": [],
        "not_equivalent": list(CHUNK_NON_EQUIVALENT_QUANTITIES),
        "claim": False,
        "missing_verification": "per-token equivalence check was not run on this environment",
        "note": "the chunked path is not claimed equivalent to the per-token path",
    }


def run_scale(sources: Iterable[str | os.PathLike], *, command: str = "",
              config: Optional[Mapping[str, Any]] = None,
              seed: int = 0,
              held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
              probe_files: Optional[Iterable[str | os.PathLike]] = None,
              reserved_sources: Optional[Iterable[str | os.PathLike]] = None,
              chunk_size: int = DEFAULT_CHUNK_SIZE,
              chunked: bool = True,
              byte_target: int = DEFAULT_BYTE_TARGET,
              checkpoint_interval: int = 0,
              checkpoint_path: Optional[str] = None,
              report_path: Optional[str] = None,
              max_chunks: Optional[int] = None,
              interrupt_after_bytes: Optional[int] = None,
              verify_equivalence: bool = False,
              hardware_class: str = HARDWARE_CLASS,
              model: Optional[Model] = None,
              root: Optional[str | os.PathLike] = None) -> Dict[str, Any]:
    """Run the documented protocol and return the report.

    The report is written to ``report_path`` when one is given, also for a
    blocked or failed outcome.
    """
    model_config = dict(config or EDGE_SCALE_CONFIG)
    model_config.setdefault("vocab_size", 256)
    model_config["byte_target"] = int(byte_target)
    model_config["chunk_size"] = int(chunk_size)
    model_config["chunked"] = bool(chunked)

    corpus = prepare_corpus(
        sources, seed=seed, held_out_fraction=held_out_fraction,
        probe_files=probe_files, reserved_sources=reserved_sources,
        root=root, strict=False,
    )
    state = RunState(
        corpus_position=0,
        step_count=0,
        seed=int(seed),
        config=dict(model_config),
    )
    if checkpoint_path and Path(checkpoint_path).is_file():
        state = read_checkpoint(checkpoint_path)

    trained = model or build_model(model_config)
    run = {
        "start_position": state.corpus_position,
        "end_position": state.corpus_position,
        "chunks": 0,
        "bytes_consumed": int(state.step_count),
        "wall_clock_seconds": 0.0,
        "initial_loss": None,
        "final_loss": None,
        "positions": [],
        "interrupted": False,
    }
    verification: Optional[Dict[str, Any]] = None
    verification_chunk: Optional[Dict[str, Any]] = None

    if corpus["contaminated"]:
        outcome_override = "failed"
        perplexity = benchmarks.skipped(
            "the contamination check failed, so no held-out score is reported",
            contaminated_documents=[item["document_id"] for item in corpus["contamination"]],
        )
        notes = {"contamination": corpus["contamination"]}
    else:
        if verify_equivalence and chunked and corpus["train_bytes"]:
            data = corpus["train_bytes"]
            window = data[: min(len(data), chunk_size + 1)]
            if len(window) >= 2:
                # A dedicated probe model: the equivalence check must not take
                # an optimizer step on the model the run is about to train.
                probe_model = build_model(model_config)
                verification_chunk = probe_model.train_chunk(
                    torch.tensor([list(window[:-1])], dtype=torch.long),
                    torch.tensor([list(window[1:])], dtype=torch.long),
                    chunk=chunk_size,
                    verify_equivalence=True,
                )
                verification = verification_chunk["verification"]
        run = train_stream(
            trained, corpus, state,
            chunk_size=chunk_size, chunked=chunked,
            byte_target=byte_target, checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path, max_chunks=max_chunks,
            interrupt_after_bytes=interrupt_after_bytes,
        )
        perplexity = held_out_perplexity(trained, corpus)
        outcome_override = None
        notes = {}

    report = scale_report(
        command=command or "scripts/train_edge_scale.py",
        config=model_config,
        corpus=corpus,
        run=run,
        model=trained,
        perplexity=perplexity,
        hardware_class=hardware_class,
        equivalence=equivalence_report(chunked, verification),
        checkpoint=checkpoint_path,
        outcome_override=outcome_override,
        notes=notes,
    )
    if verification_chunk is not None:
        report["notes"] = {**report.get("notes", {}), "equivalence_verification": {
            "verified": bool(verification_chunk["verification"].get("verified")),
            "tolerance": verification_chunk["verification"].get("tolerance"),
            "problems": verification_chunk["verification"].get("problems", []),
        }}
    if report_path:
        benchmarks.write_report(report, report_path)
    report["_corpus"] = corpus
    report["_run"] = run
    return report
