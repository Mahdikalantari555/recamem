"""Two-channel recall for recamem.

The episodic channel returns verbatim facts with their source; the plastic
channel contributes a prior score from the associative memory. The two are
fused by a small logistic scorer trained on labelled queries (and recorded
feedback) with a held-out split. Below the minimum held-out size the scorer is
disabled, the plastic prior is used alone, and the reason is reported.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

FEATURES = (
    "episodic_fused",
    "dense_similarity",
    "plastic_prior",
    "lexical_rank",
    "recency",
    "access_count",
)

MIN_TRAIN_ROWS = 6
MIN_HELDOUT_ROWS = 3


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LogisticScorer:
    """Deterministic logistic scorer over a fixed feature vector."""

    def __init__(self, weights: Optional[Sequence[float]] = None,
                 bias: float = -1.0, feature_names: Sequence[str] = FEATURES):
        self.feature_names = tuple(feature_names)
        self.weights = np.array(
            weights if weights is not None else np.zeros(len(self.feature_names)),
            dtype=np.float64,
        )
        self.bias = float(bias)
        self.trained_rows = 0

    def vectorize(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([float(features.get(name, 0.0)) for name in self.feature_names])

    def predict(self, features: Dict[str, float]) -> float:
        return float(_sigmoid(self.vectorize(features) @ self.weights + self.bias))

    def fit(self, rows: List[Dict[str, float]], labels: Sequence[int],
            *, epochs: int = 400, lr: float = 0.5, l2: float = 1e-3) -> Dict[str, Any]:
        X = np.stack([self.vectorize(r) for r in rows]) if rows else np.zeros((0, len(self.feature_names)))
        y = np.asarray(labels, dtype=np.float64)
        w = np.zeros(len(self.feature_names))
        b = 0.0
        n = max(1, len(y))
        for _ in range(epochs):
            p = _sigmoid(X @ w + b)
            grad_w = X.T @ (p - y) / n + l2 * w
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b
        self.weights = w
        self.bias = b
        self.trained_rows = len(y)
        return {"weights": w.tolist(), "bias": b, "rows": len(y)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "trained_rows": self.trained_rows,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticScorer":
        scorer = cls(
            weights=payload.get("weights"),
            bias=payload.get("bias", -1.0),
            feature_names=payload.get("feature_names", FEATURES),
        )
        scorer.trained_rows = int(payload.get("trained_rows", 0))
        return scorer


@dataclass
class RecallResult:
    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    plastic_only: bool = False
    scorer: str = "plastic_prior"
    reason: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not self.results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": self.results,
            "plastic_only": self.plastic_only,
            "scorer": self.scorer,
            "reason": self.reason,
            "empty": self.empty,
        }


class TwoChannelRecall:
    """Fuse the plastic prior and episodic verbatim channels."""

    def __init__(self, engine, *, min_score: float = 0.30,
                 scorer: Optional[LogisticScorer] = None,
                 scorer_path: Optional[str] = None):
        self.engine = engine
        self.store = engine.knowledge
        self.min_score = float(min_score)
        self.scorer = scorer
        self.scorer_path = Path(scorer_path) if scorer_path else None
        self.training_report: Optional[Dict[str, Any]] = None
        if self.scorer is None and self.scorer_path and self.scorer_path.exists():
            self.scorer = LogisticScorer.from_dict(json.loads(self.scorer_path.read_text()))
            self.training_report = {"loaded_from": str(self.scorer_path)}

    # ---------------------------------------------------------------- #
    # Channels
    # ---------------------------------------------------------------- #
    def _plastic_read(self, query_vec: torch.Tensor) -> Optional[torch.Tensor]:
        memory = getattr(self.engine, "memory", None)
        if memory is None or not hasattr(memory, "read"):
            return None
        with torch.no_grad():
            try:
                return memory.read(query_vec).detach().float().reshape(-1)
            except Exception:
                return None

    def _plastic_prior(self, query_vec: torch.Tensor, fact_vec: torch.Tensor) -> float:
        read = self._plastic_read(query_vec)
        if read is None:
            return 0.0
        denom = (torch.linalg.vector_norm(read) * torch.linalg.vector_norm(fact_vec)).clamp(min=1e-8)
        return float(torch.dot(read, fact_vec) / denom)

    def _recency(self, timestamp: Optional[float]) -> float:
        if not timestamp:
            return 0.0
        age = max(0.0, time.time() - float(timestamp))
        return math.exp(-age / (7 * 24 * 3600.0))

    def _candidates(self, query: str, query_vec: torch.Tensor, pool: int) -> List[Dict[str, Any]]:
        hits = self.store.recall(
            query_vec, top_k=pool, threshold=0.0, query_text=query,
            strategy=getattr(self.engine, "retrieval_strategy", "hybrid_rrf"),
        )
        out = []
        for rank, hit in enumerate(hits):
            features = {
                "episodic_fused": float(hit.get("fused_score", hit.get("similarity", 0.0)) or 0.0),
                "dense_similarity": float(hit.get("dense_similarity", hit.get("similarity", hit.get("dscore", 0.0))) or 0.0),
                "lexical_rank": 1.0 / (1.0 + rank),
                "recency": self._recency(hit.get("timestamp")),
                "access_count": math.log1p(float(hit.get("access_count", 1) or 1)),
            }
            out.append({"hit": hit, "features": features, "rank": rank})
        return out

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #
    def recall(self, query: str, *, top_k: int = 3,
               min_score: Optional[float] = None) -> Dict[str, Any]:
        floor = self.min_score if min_score is None else float(min_score)
        with torch.no_grad():
            query_vec = self.engine.anchor.embed(query)
        candidates = self._candidates(query, query_vec, max(top_k * 8, 24))
        for cand in candidates:
            fact_vec = self._fact_vector(cand["hit"]["id"])
            prior = self._plastic_prior(query_vec, fact_vec) if fact_vec is not None else 0.0
            cand["features"]["plastic_prior"] = prior
            if self.scorer is not None:
                cand["fused"] = self.scorer.predict(cand["features"])
            else:
                # Scorer disabled: the plastic prior is used alone as a bonus
                # on top of the episodic dense agreement. Raw RRF scores are
                # rank-based (~1/61), so they are not comparable to the floor.
                cand["fused"] = (
                    0.7 * max(0.0, cand["features"]["dense_similarity"])
                    + 0.3 * max(0.0, prior)
                )
        kept = [c for c in candidates if c["fused"] >= floor]
        kept.sort(key=lambda c: c["fused"], reverse=True)

        results = []
        for cand in kept[: max(top_k, 1)]:
            hit = cand["hit"]
            results.append(
                {
                    "id": int(hit["id"]),
                    "text": hit["text"],
                    "source": hit.get("source"),
                    "origin": hit.get("origin"),
                    "scores": {
                        "fused": float(cand["fused"]),
                        "episodic": cand["features"]["episodic_fused"],
                        "plastic": cand["features"]["plastic_prior"],
                        "dense": cand["features"]["dense_similarity"],
                    },
                    "plastic_only": False,
                    "channels": ["episodic", "plastic"] if cand["features"]["plastic_prior"] > 0 else ["episodic"],
                }
            )
        plastic_only = False
        reason = None
        if not results:
            prior_max = max((c["features"]["plastic_prior"] for c in candidates), default=0.0)
            if prior_max >= floor and candidates:
                plastic_only = True
                reason = (
                    "only the plastic channel scored above the minimum; "
                    "no stored fact is presented"
                )
        if self.scorer is None:
            scorer_name = "plastic_prior"
            if self.training_report and not self.training_report.get("enabled", True):
                reason = reason or self.training_report.get("reason")
        else:
            scorer_name = "logistic"
        # Successful recall reinforces the fact and is audited.
        for item in results:
            self._record_access(item["id"], query)
        return RecallResult(
            query=query,
            results=results,
            plastic_only=plastic_only,
            scorer=scorer_name,
            reason=reason,
        ).to_dict()

    def _fact_vector(self, fact_id: int) -> Optional[torch.Tensor]:
        try:
            vecs, ids = self.store.get_all_embeddings()
        except Exception:
            return None
        for idx, fid in enumerate(ids):
            if int(fid) == int(fact_id):
                return vecs[idx]
        return None

    def _record_access(self, fact_id: int, query: str) -> None:
        try:
            self.store.reinforce(int(fact_id))
        except Exception:
            pass
        try:
            self.store.conn.execute(
                "INSERT INTO feedback_log(fact_id, action, correction_text, timestamp, strategy_used)"
                " VALUES (?,?,?,?,?)",
                (int(fact_id), "recall_access", query, time.time(), "two_channel"),
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- #
    # Scorer training
    # ---------------------------------------------------------------- #
    def train_scorer(self, labelled: Sequence[Dict[str, Any]],
                     *, heldout_fraction: float = 0.34,
                     seed: int = 0) -> Dict[str, Any]:
        """Fit the logistic scorer on a split and report held-out metrics."""
        rows: List[Dict[str, float]] = []
        labels: List[int] = []
        for item in labelled:
            query = item["query"]
            relevant = set(int(x) for x in item.get("relevant_ids", []))
            with torch.no_grad():
                qvec = self.engine.anchor.embed(query)
            for cand in self._candidates(query, qvec, max(8, len(relevant) * 4)):
                fact_vec = self._fact_vector(cand["hit"]["id"])
                cand["features"]["plastic_prior"] = (
                    self._plastic_prior(qvec, fact_vec) if fact_vec is not None else 0.0
                )
                rows.append(cand["features"])
                labels.append(1 if int(cand["hit"]["id"]) in relevant else 0)
        total = len(rows)
        min_needed = MIN_TRAIN_ROWS + MIN_HELDOUT_ROWS
        if total < min_needed or len(set(labels)) < 2:
            self.scorer = None
            self.training_report = {
                "enabled": False,
                "reason": (
                    f"labelled rows {total} below minimum {min_needed} "
                    "or only one class present; plastic prior used alone"
                ),
                "rows": total,
            }
            return dict(self.training_report)

        rng = np.random.default_rng(seed)
        order = rng.permutation(total)
        heldout_size = max(MIN_HELDOUT_ROWS, int(round(total * heldout_fraction)))
        heldout_idx = order[:heldout_size]
        train_idx = order[heldout_size:]
        scorer = LogisticScorer()
        fit = scorer.fit([rows[i] for i in train_idx], [labels[i] for i in train_idx])
        metrics = self._ranking_metrics(
            scorer, [rows[i] for i in heldout_idx], [labels[i] for i in heldout_idx]
        )
        self.scorer = scorer
        self.training_report = {
            "enabled": True,
            "rows": total,
            "train_rows": len(train_idx),
            "heldout_rows": len(heldout_idx),
            "metrics": metrics,
            "fit": fit,
        }
        if self.scorer_path:
            self.scorer_path.parent.mkdir(parents=True, exist_ok=True)
            self.scorer_path.write_text(json.dumps(scorer.to_dict(), indent=2) + "\n")
        return dict(self.training_report)

    @staticmethod
    def _ranking_metrics(scorer: LogisticScorer, rows, labels) -> Dict[str, float]:
        scores = [scorer.predict(r) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
        top = order[: max(1, len(order) // 2)]
        tp = sum(1 for i in top if labels[i] == 1)
        precision = tp / max(1, len(top))
        first_rel = next((pos + 1 for pos, i in enumerate(order) if labels[i] == 1), None)
        mrr = 1.0 / first_rel if first_rel else 0.0
        relevant_total = sum(labels)
        recall = tp / relevant_total if relevant_total else 0.0
        return {
            "precision_at_half": round(precision, 4),
            "recall_at_half": round(recall, 4),
            "mrr": round(mrr, 4),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "scorer": "logistic" if self.scorer is not None else "plastic_prior",
            "training": self.training_report,
            "min_score": self.min_score,
        }


def measure_index_recall(store, queries: Iterable[torch.Tensor], k: int = 5) -> Dict[str, Any]:
    """Recall@k of the configured ANN index against exact dense search."""
    index = getattr(store, "dense_index", None)
    if index is None:
        return {"indexed": False, "reason": "no ANN index configured", "k": k}
    overlaps: List[float] = []
    for query in queries:
        exact = store._dense_candidates(query, k)[0]  # exact MVM baseline
        approx = None
        if hasattr(index, "search"):
            approx = [int(i) for i in index.search(query.detach().cpu().numpy(), k)[0]]
        if not approx:
            continue
        overlaps.append(len(set(exact) & set(approx)) / max(1, min(len(exact), len(approx))))
    if not overlaps:
        return {"indexed": True, "recall_at_k": None, "reason": "index returned no candidates", "k": k}
    return {
        "indexed": True,
        "k": k,
        "queries": len(overlaps),
        "recall_at_k": round(sum(overlaps) / len(overlaps), 4),
    }
