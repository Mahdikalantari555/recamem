"""Bounded verifier-guided diverse search over test-time candidates.

Implements ``model/test-time-compute``.  Every cycle is bounded by candidate,
depth, token and wall-clock budgets.  Candidates are deduplicated before the
verifier scores them, and nothing is ever written to persistent knowledge
without an explicit approval call.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

REJECTED_THRESHOLD_DEFAULT = 0.0


class Verifier:
    """Verifier interface: higher scores are better, in any finite range."""

    name = "abstract"

    def score(self, prompt: str, candidate: str) -> float:
        raise NotImplementedError

    def evidence(self, prompt: str, candidate: str) -> Dict[str, Any]:
        return {"verifier": self.name, "score": self.score(prompt, candidate)}


class CallableVerifier(Verifier):
    def __init__(self, fn: Callable[[str, str], float], name: str = "callable"):
        self._fn = fn
        self.name = name

    def score(self, prompt: str, candidate: str) -> float:
        return float(self._fn(prompt, candidate))


class KeywordVerifier(Verifier):
    """Deterministic reference verifier: token overlap plus length sanity."""

    name = "keyword-overlap"

    def __init__(self, expected: Sequence[str] = (), penalty_per_token: float = 0.01):
        self.expected = [str(item).lower() for item in expected]
        self.penalty_per_token = float(penalty_per_token)

    def score(self, prompt: str, candidate: str) -> float:
        lowered = candidate.lower()
        overlap = sum(1 for token in self.expected if token in lowered)
        coverage = overlap / max(1, len(self.expected))
        length_penalty = self.penalty_per_token * len(candidate.split())
        return coverage - length_penalty


@dataclass
class TTCBudget:
    max_candidates: int = 8
    max_depth: int = 2
    max_tokens: int = 512
    max_seconds: float = 5.0
    min_verifier_score: float = REJECTED_THRESHOLD_DEFAULT

    def __post_init__(self):
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_depth": self.max_depth,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
            "min_verifier_score": self.min_verifier_score,
        }


def _default_token_counter(text: str) -> int:
    # Conservative whitespace/4-char hybrid; callers can inject a tokenizer.
    return max(1, len(text.split())) if text else 0


def _hash_embedding(text: str, dim: int) -> torch.Tensor:
    """Deterministic stand-alone embedding used when no anchor is supplied."""
    vector = torch.zeros(dim)
    for token in text.lower().split():
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % dim
        vector[index] += 1.0
    norm = vector.norm()
    return vector / norm if float(norm) > 0 else vector


class TestTimeCompute:
    """Diverse branching tree search with an explicit verifier and strict budgets."""

    __test__ = False  # not a pytest test class despite the name

    def __init__(self, generate: Callable[..., str], verifier: Verifier,
                 budget: Optional[TTCBudget] = None,
                 diversity: str = "dedupe", token_counter=None,
                 temperature: float = 0.8, seed: int = 0):
        if generate is None or not callable(generate):
            raise ValueError("TestTimeCompute requires a generate callable")
        if verifier is None:
            raise ValueError("TestTimeCompute requires a verifier")
        if diversity not in ("dedupe", "none"):
            raise ValueError("diversity must be 'dedupe' or 'none'")
        self.generate = generate
        self.verifier = verifier
        self.budget = budget or TTCBudget()
        self.diversity = diversity
        self.token_counter = token_counter or _default_token_counter
        self.temperature = float(temperature)
        self.seed = int(seed)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def _call_generate(self, prompt: str, tokens: int, sample_index: int) -> str:
        try:
            return str(self.generate(
                prompt, max_tokens=tokens, temperature=self.temperature,
                seed=self.seed + sample_index,
            ))
        except TypeError:
            # Minimal generate(prompt) callables remain usable.
            return str(self.generate(prompt))

    # -- search ---------------------------------------------------------
    def search(self, prompt: str, budget: Optional[TTCBudget] = None,
               cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
        budget = budget or self.budget
        started = time.perf_counter()
        tokens_used = 0
        candidates: List[Dict[str, Any]] = []
        seen_norm: set = set()
        stop_reason = "budget_exhausted"
        cancelled = False
        budget_exhausted = False
        next_id = 0

        def elapsed_or_cancelled() -> bool:
            nonlocal cancelled
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                return True
            return (time.perf_counter() - started) >= budget.max_seconds

        for depth in range(1, budget.max_depth + 1):
            if cancelled or budget_exhausted:
                break
            if elapsed_or_cancelled():
                break
            # Expand the best partial candidate at deeper levels, otherwise the
            # original prompt: a bounded, deterministic tree.
            if depth == 1 or not candidates:
                base_prompt = prompt
                parent = None
            else:
                best = sorted(
                    candidates,
                    key=lambda item: (-item["score"], item["id"]),
                )[0]
                base_prompt = f"{prompt}\n{best['text']}"
                parent = best["id"]
            per_depth = max(1, budget.max_candidates // budget.max_depth)
            for _ in range(per_depth):
                if elapsed_or_cancelled():
                    break
                if len(candidates) >= budget.max_candidates:
                    budget_exhausted = True
                    break
                remaining_tokens = budget.max_tokens - tokens_used
                if remaining_tokens <= 0:
                    budget_exhausted = True
                    break
                request_tokens = min(
                    max(1, budget.max_tokens // (budget.max_candidates * budget.max_depth)),
                    remaining_tokens,
                )
                text = self._call_generate(base_prompt, request_tokens, len(candidates))
                spent = self.token_counter(text)
                tokens_used += max(0, spent)
                normalized = self._normalize(text)
                if not normalized:
                    continue
                if self.diversity == "dedupe" and normalized in seen_norm:
                    # Duplicates never consume verifier work.
                    continue
                seen_norm.add(normalized)
                evidence = self.verifier.evidence(prompt, text)
                score = float(evidence.get("score", 0.0))
                candidates.append({
                    "id": next_id,
                    "text": text,
                    "depth": depth,
                    "parent": parent,
                    "tokens": spent,
                    "score": score,
                    "verifier": evidence,
                    "rejected": score < budget.min_verifier_score,
                    "selected": False,
                })
                next_id += 1

        consumed = {
            "candidates": len(candidates),
            "depth": max((c["depth"] for c in candidates), default=0),
            "tokens": tokens_used,
            "seconds": time.perf_counter() - started,
        }
        if cancelled:
            status = "cancelled"
            stop_reason = "cancelled"
        elif not candidates:
            status = "no_candidate"
            stop_reason = "no_candidate"
        else:
            status = "selected"
            stop_reason = "budget_exhausted" if budget_exhausted else "search_complete"

        selected = None
        if candidates:
            ordered = sorted(
                (c for c in candidates if not c["rejected"]),
                key=lambda item: (-item["score"], item["id"]),
            )
            if ordered:
                ordered[0]["selected"] = True
                selected = ordered[0]
            else:
                status = "no_candidate"
                stop_reason = "all_candidates_rejected"

        return {
            "status": status,
            "stop_reason": stop_reason,
            "prompt": prompt,
            "selected": selected,
            "candidates": candidates,
            "budget": budget.to_dict(),
            "consumed": consumed,
            "verifier": getattr(self.verifier, "name", "unknown"),
            "trace": {
                "selected_id": selected["id"] if selected else None,
                "selected_score": selected["score"] if selected else None,
                "verifier": getattr(self.verifier, "name", "unknown"),
                "candidate_ids": [c["id"] for c in candidates],
                "budget_use": consumed,
            },
        }

    # -- persistence policy --------------------------------------------
    @staticmethod
    def persist_selected(store, result: Dict[str, Any], approved: bool = False,
                         source: str = "ttc", embed=None) -> Dict[str, Any]:
        """Insert the selected answer only after an explicit approval."""
        if not approved:
            raise PermissionError(
                "Test-time candidates are not persisted without explicit approval"
            )
        selected = result.get("selected")
        if not selected or selected.get("rejected"):
            return {"status": "skipped", "reason": "no approved selected candidate"}
        if embed is None:
            anchor = getattr(store, "anchor", None)
            if anchor is not None:
                embed = anchor.embed
        if embed is not None:
            embedding = torch.as_tensor(embed(selected["text"])).detach().float().reshape(-1)
        else:
            embedding = _hash_embedding(selected["text"], int(getattr(store, "dim", 384)))
        fact_id = store.insert_fact(
            selected["text"], source, time.time(), 0,
            float(selected.get("score", 0.0)), embedding,
        )
        return {"status": "persisted", "fact_id": fact_id, "text": selected["text"]}


__all__ = [
    "CallableVerifier",
    "KeywordVerifier",
    "TTCBudget",
    "TestTimeCompute",
    "Verifier",
]
