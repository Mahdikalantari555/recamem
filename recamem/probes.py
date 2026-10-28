"""Contrastive cloze grammar probes: format, loader, and scoring.

A probe is a context, exactly one correct continuation, and at least one
incorrect continuation. The model scores each continuation by its mean
log-probability per byte, and the probe counts as correct when the correct
continuation's mean log-probability exceeds the best incorrect one. Reporting a
margin against a distractor rather than raw likelihood means a model that likes
everything equally cannot look competent, and reporting the result per grammar
phenomenon means an aggregate cannot hide a failed construction.

The probe data lives in the repository (``evals/grammar_probes.jsonl``), one
JSON object per line. Scoring is frozen: no optimizer step is taken, no
parameter changes, and the recurrent buffers are restored afterwards. A score
is tied to the checkpoint and probe file it was produced from by hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "PHENOMENA",
    "Probe",
    "ProbeSet",
    "ProbeFormatError",
    "load_probes",
    "probe_file_hash",
    "score_continuation",
    "score_probe",
    "score_probes",
    "parameter_hash",
    "PROBE_FILE",
]

# The documented phenomenon set. A probe naming anything else is rejected with
# the offending value named, so a typo cannot silently create a new bucket.
PHENOMENA: Tuple[str, ...] = (
    "subject_verb_agreement",
    "active_voice",
    "passive_voice",
    "question_formation",
)

from recamem.paths import evals_dir

PROBE_FILE = evals_dir() / "grammar_probes.jsonl"

# A primer used only when a probe has an empty context, so the model has a
# state to condition on. It is never scored and never appears in the data file.
_EMPTY_CONTEXT_PRIMER = b"\n"


class ProbeFormatError(ValueError):
    """A probe line does not satisfy the documented shape."""


@dataclass(frozen=True)
class Probe:
    id: str
    phenomenon: str
    context: str
    correct: str
    incorrect: Tuple[str, ...]

    def continuations(self) -> Tuple[str, ...]:
        return (self.correct,) + self.incorrect


@dataclass
class ProbeSet:
    """A loaded probe file: the valid probes plus one error per rejected line."""

    probes: List[Probe] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.probes)

    def __iter__(self) -> Iterable[Probe]:
        return iter(self.probes)

    def phenomena(self) -> List[str]:
        present = {probe.phenomenon for probe in self.probes}
        return [name for name in PHENOMENA if name in present]

    def by_phenomenon(self) -> Dict[str, List[Probe]]:
        grouped: Dict[str, List[Probe]] = {name: [] for name in PHENOMENA}
        for probe in self.probes:
            grouped[probe.phenomenon].append(probe)
        return grouped


def _validate_record(entry: Any, lineno: int) -> Probe:
    where = f"line {lineno}"
    if not isinstance(entry, dict):
        raise ProbeFormatError(f"{where}: expected a JSON object")
    probe_id = entry.get("id")
    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ProbeFormatError(f"{where}: 'id' must be a non-empty string")
    phenomenon = entry.get("phenomenon")
    if not isinstance(phenomenon, str) or not phenomenon.strip():
        raise ProbeFormatError(f"{where}: 'phenomenon' must be a non-empty string")
    if phenomenon not in PHENOMENA:
        raise ProbeFormatError(
            f"{where}: unknown phenomenon {phenomenon!r}; expected one of {list(PHENOMENA)}"
        )
    context = entry.get("context")
    if not isinstance(context, str) or not context.strip():
        raise ProbeFormatError(f"{where}: 'context' must be a non-empty string")
    correct = entry.get("correct")
    if not isinstance(correct, str) or not correct.strip():
        raise ProbeFormatError(f"{where}: 'correct' must be a non-empty string")
    incorrect = entry.get("incorrect")
    if isinstance(incorrect, str):
        raise ProbeFormatError(f"{where}: 'incorrect' must be a list of strings, not a string")
    if not isinstance(incorrect, list) or not incorrect:
        raise ProbeFormatError(f"{where}: 'incorrect' must be a non-empty list of strings")
    for item in incorrect:
        if not isinstance(item, str) or not item.strip():
            raise ProbeFormatError(f"{where}: every 'incorrect' entry must be a non-empty string")
    if correct in incorrect:
        raise ProbeFormatError(f"{where}: the correct continuation also appears in 'incorrect'")
    return Probe(
        id=probe_id,
        phenomenon=phenomenon,
        context=context,
        correct=correct,
        incorrect=tuple(incorrect),
    )


def load_probes(path: str | Path) -> ProbeSet:
    """Load a probe file, rejecting malformed lines one at a time.

    A broken line contributes an error with its line number and the reason and
    does not stop the rest of the file from loading. The caller decides whether
    a partial load is acceptable; :func:`score_probes` refuses to score when no
    probe survived.
    """
    source = Path(path)
    result = ProbeSet(path=str(source))
    seen = set()
    with open(source, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                result.errors.append(
                    {"line": lineno, "reason": f"invalid JSON ({exc.msg})"}
                )
                continue
            try:
                probe = _validate_record(entry, lineno)
            except ProbeFormatError as exc:
                result.errors.append({"line": lineno, "reason": str(exc)})
                continue
            if probe.id in seen:
                result.errors.append(
                    {"line": lineno, "reason": f"duplicate probe id {probe.id!r}"}
                )
                continue
            seen.add(probe.id)
            result.probes.append(probe)
    return result


def probe_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(model: Any) -> str:
    """Stable hash over the model's parameters and buffers."""
    digest = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _buffer_snapshot(model: Any) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.named_buffers()}


def _restore_buffers(model: Any, snapshot: Mapping[str, torch.Tensor]) -> None:
    current = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in snapshot.items():
            if name in current and current[name].shape == value.shape:
                current[name].copy_(value)


def _step_logits(model: Any, byte: int) -> torch.Tensor:
    """Consume one byte and return the logits that predict the next byte."""
    device = _device(model)
    token = torch.tensor(byte, dtype=torch.long, device=device)
    output = model.step(token, frozen=True)
    # Model.step returns ((x, states, decays), (logits, stop)); a fake model in
    # the tests may return the logits directly.
    if isinstance(output, tuple) and len(output) == 2:
        second = output[1]
        if isinstance(second, tuple):
            return second[0]
        return second
    return output


def score_continuation(model: Any, context: str, continuation: str) -> float:
    """Mean log-probability per byte of ``continuation`` given ``context``.

    The model is primed with the context, then each target byte is scored from
    the state that precedes it. Scoring runs under ``torch.no_grad`` and uses
    the frozen step, so no parameter is updated.
    """
    data = continuation.encode("utf-8")
    if not data:
        return float("-inf")
    context_bytes = context.encode("utf-8") or _EMPTY_CONTEXT_PRIMER
    with torch.no_grad():
        for byte in context_bytes[:-1]:
            _step_logits(model, byte)
        # The last context byte produces the logits that score the first target.
        logits = _step_logits(model, context_bytes[-1])
        total = 0.0
        for index, byte in enumerate(data):
            log_probs = torch.log_softmax(logits.reshape(-1).float(), dim=-1)
            total += float(log_probs[byte].item())
            if index + 1 < len(data):
                logits = _step_logits(model, byte)
    return total / len(data)


def score_probe(model: Any, probe: Probe) -> Dict[str, Any]:
    """Score one probe and report the margin against the best distractor."""
    if hasattr(model, "reset"):
        model.reset()
    correct = score_continuation(model, probe.context, probe.correct)
    incorrect_scores = [
        score_continuation(model, probe.context, continuation)
        for continuation in probe.incorrect
    ]
    best_incorrect = max(incorrect_scores) if incorrect_scores else float("-inf")
    margin = correct - best_incorrect
    return {
        "id": probe.id,
        "phenomenon": probe.phenomenon,
        "correct_score": correct,
        "incorrect_scores": incorrect_scores,
        "best_incorrect_score": best_incorrect,
        "margin": margin,
        "correct": bool(margin > 0.0),
    }


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def score_probes(model: Any, probe_set: ProbeSet | Sequence[Probe], *,
                 checkpoint_path: Optional[str | Path] = None,
                 probe_path: Optional[str | Path] = None,
                 restore_state: bool = True) -> Dict[str, Any]:
    """Score a probe set and return the report fragment.

    The result names the probe count, the correct count and the accuracy, the
    margins, and the per-phenomenon breakdown. The aggregate is never the only
    accuracy in the result. When nothing could be scored the result is a skip
    with a reason and no accuracy.
    """
    probes = list(probe_set.probes if isinstance(probe_set, ProbeSet) else probe_set)
    if not probes:
        return {
            "status": "skipped",
            "reason": "the probe set contained no scoreable probes",
            "probes": 0,
        }

    snapshot = _buffer_snapshot(model) if restore_state else {}
    before = parameter_hash(model)
    results: List[Dict[str, Any]] = []
    try:
        for probe in probes:
            results.append(score_probe(model, probe))
    finally:
        if restore_state:
            _restore_buffers(model, snapshot)
    after = parameter_hash(model)

    correct = sum(1 for result in results if result["correct"])
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in PHENOMENA}
    for result in results:
        grouped.setdefault(result["phenomenon"], []).append(result)
    per_phenomenon = {
        name: {
            "probes": len(items),
            "correct": sum(1 for item in items if item["correct"]),
            "accuracy": _accuracy(sum(1 for item in items if item["correct"]), len(items)),
        }
        for name, items in grouped.items()
        if items
    }

    accuracy = _accuracy(correct, len(results))
    report: Dict[str, Any] = {
        "status": "passed",
        "value": accuracy,
        "probes": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "margins": results,
        "per_phenomenon": per_phenomenon,
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "frozen": before == after,
    }
    if checkpoint_path is not None:
        from recamem.bench.metrics import sha256_file

        report["checkpoint_hash"] = sha256_file(checkpoint_path)
    if probe_path is not None:
        report["probe_file_hash"] = probe_file_hash(probe_path)
    return report
