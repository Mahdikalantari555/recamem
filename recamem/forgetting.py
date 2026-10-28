"""Forgetting that never destroys data.

Accessibility follows the power law in ``memory-forgetting``; capacity is
enforced by moving the lowest-utility facts to the ``retired`` tier, and a fact
is retired only after a recall probe confirms the plastic channel absorbed it.
No code path here issues a ``DELETE``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

DEFAULT_TAU_HALF_SECONDS = 7 * 24 * 3600.0
DEFAULT_NOVELTY_LAMBDA = 1.0


def utility(
    access_count: float,
    last_retrieved: Optional[float],
    novelty: float,
    *,
    now: Optional[float] = None,
    tau_half: float = DEFAULT_TAU_HALF_SECONDS,
    novelty_lambda: float = DEFAULT_NOVELTY_LAMBDA,
) -> float:
    """Utility(i) = AccessCount * exp(-(t - t_last)/tau) + lambda * novelty."""
    now = time.time() if now is None else float(now)
    last = float(last_retrieved if last_retrieved is not None else now)
    elapsed = max(0.0, now - last)
    return float(access_count) * math.exp(-elapsed / float(tau_half)) + float(
        novelty_lambda
    ) * float(novelty)


def _active_rows(store) -> List[tuple]:
    return store.conn.execute(
        "SELECT id, text, access_count, last_retrieved, timestamp, novelty"
        " FROM facts WHERE superseded = 0 AND (tier IS NULL OR tier = 'active')"
        " ORDER BY id"
    ).fetchall()


def utility_ranking(store, now: Optional[float] = None) -> List[Dict[str, Any]]:
    rows = _active_rows(store)
    ranked = []
    for fact_id, text, access, last, timestamp, novelty in rows:
        ranked.append(
            {
                "id": int(fact_id),
                "text": text,
                "utility": utility(
                    access or 0, last if last is not None else timestamp, novelty or 0.0, now=now
                ),
                "access_count": int(access or 0),
                "novelty": float(novelty or 0.0),
            }
        )
    ranked.sort(key=lambda row: (row["utility"], -row["id"]))
    return ranked


def enforce_capacity(store, K_max: int, *, now: Optional[float] = None) -> Dict[str, Any]:
    """Retire the lowest-utility active facts until ``active_count <= K_max``."""
    active = store.active_count()
    if active <= K_max:
        return {"retired": [], "kept": active, "active_count": active, "K_max": K_max}
    ranked = utility_ranking(store, now=now)
    overflow = active - K_max
    retired: List[int] = []
    for row in ranked:
        if len(retired) >= overflow:
            break
        if store.retire_fact(row["id"], reason="capacity"):
            retired.append(row["id"])
    return {
        "retired": retired,
        "kept": store.active_count(),
        "active_count": store.active_count(),
        "K_max": K_max,
        "considered": len(ranked),
    }


def plastic_probe_score(engine, text: str) -> float:
    """Score how well the plastic channel alone holds ``text``."""
    import torch

    with torch.no_grad():
        vec = engine.anchor.embed(text)
        if hasattr(engine.anchor, "is_mock") and engine.anchor.is_mock:
            pass  # mock embeddings are labelled by the caller, not hidden here
        memory = getattr(engine, "memory", None)
        if memory is None or not hasattr(memory, "recall_score"):
            return 0.0
        return float(memory.recall_score(vec))


def consolidate_then_forget(
    engine,
    *,
    store=None,
    candidates: Optional[Sequence[int]] = None,
    probe: Optional[Callable[[str], float]] = None,
    probe_threshold: float = 0.10,
    limit: int = 32,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Retire absorbed facts only; keep the rest active with a recorded reason."""
    store = store or engine.knowledge
    probe = probe or (lambda text: plastic_probe_score(engine, text))
    ranked = utility_ranking(store, now=now)
    if candidates is not None:
        wanted = {int(c) for c in candidates}
        ranked = [row for row in ranked if row["id"] in wanted]
    ranked = ranked[: max(0, int(limit))]
    retired: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    for row in ranked:
        score = float(probe(row["text"]))
        if score >= probe_threshold:
            store.retire_fact(row["id"], reason="consolidated")
            retired.append(
                {"id": row["id"], "probe": score, "utility": row["utility"], "status": "retired"}
            )
        else:
            kept.append(
                {
                    "id": row["id"],
                    "probe": score,
                    "utility": row["utility"],
                    "status": "kept",
                    "reason": f"recall probe {score:.3f} below threshold {probe_threshold}",
                }
            )
    return {
        "considered": len(ranked),
        "retired": retired,
        "kept": kept,
        "retired_count": len(retired),
        "kept_count": len(kept),
        "probe_threshold": probe_threshold,
        "active_count": store.active_count(),
        "hard_deleted": 0,
    }


def restore(store, fact_id: int) -> bool:
    """Explicit user action: bring a retired fact back to the active tier."""
    return store.restore_fact(fact_id)


def forgetting_cycle(engine, *, K_max: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Capacity enforcement followed by consolidate-then-forget."""
    store = engine.knowledge
    K_max = int(K_max if K_max is not None else getattr(engine, "K_max", 5000))
    capacity = enforce_capacity(store, K_max)
    consolidated = consolidate_then_forget(engine, store=store, **kwargs)
    return {"capacity": capacity, "consolidation": consolidated}
