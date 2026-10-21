"""Scalable dense candidate retrieval, reranking and Matryoshka fast paths.

Implements ``memory/hybrid-retrieval``:

* :class:`ExactDenseIndex` is the correctness baseline (one matrix-vector
  product, identical scores to the pre-change ``KnowledgeStore`` path).
* :class:`HNSWIndex` is an *optional* approximate index. When ``hnswlib`` is
  unavailable, invalid, or disabled it reports that explicitly and serves
  queries from the exact index, so recall never silently degrades into an
  error.
* :class:`CrossEncoderReranker` reranks only a bounded candidate set and
  labels whether the score came from a real cross-encoder or the documented
  lexical fallback.
* :class:`MatryoshkaAnchor` produces a *validated projection* 64-dimensional
  fast representation alongside the 384-dimensional one.  An arbitrary stored
  384-D vector is never truncated to 64 dimensions.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

INDEX_FORMAT_VERSION = 1
MATRYOSHKA_VERSION = "minilm-l6+projection-v1"


class EmbeddingVersionError(ValueError):
    """Stored embeddings are incompatible with the configured representation."""


class IndexUnavailableError(RuntimeError):
    """The requested dense index backend is not installed or not usable."""


def _normalize(matrix: torch.Tensor) -> torch.Tensor:
    return F.normalize(matrix.float(), dim=-1, eps=1e-8)


def _hnswlib():
    try:
        import hnswlib  # type: ignore
        return hnswlib
    except Exception:
        return None


class DenseIndex:
    """Interface shared by the exact and approximate dense candidate paths."""

    kind = "abstract"

    def __init__(self, dim: int = 384):
        self.dim = int(dim)
        self.ids: List[int] = []
        self.vectors = torch.zeros((0, self.dim), dtype=torch.float32)
        self._positions: Dict[int, int] = {}

    @property
    def available(self) -> bool:
        return True

    @property
    def mode(self) -> str:
        return self.kind

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        vectors = _normalize(torch.as_tensor(vectors).reshape(len(ids), self.dim))
        for fact_id, vector in zip(ids, vectors):
            fact_id = int(fact_id)
            if fact_id in self._positions:
                # Upsert semantics: never create two rows for the same fact.
                self.vectors[self._positions[fact_id]] = vector
                continue
            self._positions[fact_id] = len(self.ids)
            self.ids.append(fact_id)
            self.vectors = torch.cat([self.vectors, vector.unsqueeze(0)], dim=0)

    def update(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        vectors = _normalize(torch.as_tensor(vectors).reshape(len(ids), self.dim))
        for fact_id, vector in zip(ids, vectors):
            fact_id = int(fact_id)
            position = self._positions.get(fact_id)
            if position is None:
                self.add([fact_id], vector.unsqueeze(0))
            else:
                self.vectors[position] = vector

    def delete(self, ids: Iterable[int]) -> int:
        removed = set(int(i) for i in ids)
        keep = [i for i, fact_id in enumerate(self.ids) if fact_id not in removed]
        self.ids = [self.ids[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else torch.zeros((0, self.dim))
        self._positions = {fact_id: pos for pos, fact_id in enumerate(self.ids)}
        return len(removed)

    def rebuild(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        """Atomic rebuild: build the new state fully, then swap it in."""
        ids = [int(i) for i in ids]
        matrix = _normalize(torch.as_tensor(vectors).reshape(len(ids), self.dim))
        self.ids = ids
        self.vectors = matrix
        self._positions = {fact_id: pos for pos, fact_id in enumerate(ids)}

    def search(self, query: torch.Tensor, k: int = 200) -> Tuple[List[int], List[float]]:
        if not self.ids:
            return [], []
        q = F.normalize(torch.as_tensor(query).float().reshape(-1), dim=0, eps=1e-8)
        sims = self.vectors @ q
        k = min(k, len(self.ids))
        top = torch.topk(sims, k)
        return [self.ids[i] for i in top.indices.tolist()], top.values.tolist()

    def persist(self, path: str) -> Dict[str, Any]:
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"vectors": self.vectors.contiguous()}, str(path) + ".safetensors")
        meta = {
            "version": INDEX_FORMAT_VERSION, "kind": self.kind, "dim": self.dim,
            "count": len(self.ids), "ids": self.ids,
        }
        Path(str(path) + ".json").write_text(json.dumps(meta))
        return {"path": str(path), "count": len(self.ids), "kind": self.kind}

    def load(self, path: str) -> Dict[str, Any]:
        from safetensors.torch import load_file
        path = Path(path)
        meta_file = Path(str(path) + ".json")
        vector_file = Path(str(path) + ".safetensors")
        meta = json.loads(meta_file.read_text())
        if int(meta.get("version", 0)) > INDEX_FORMAT_VERSION:
            raise ValueError(f"Index version {meta.get('version')} is not supported")
        if int(meta.get("dim", 0)) != self.dim:
            raise ValueError(
                f"Index dim {meta.get('dim')} does not match configured dim {self.dim}"
            )
        tensors = load_file(str(vector_file))
        vectors = tensors["vectors"]
        if tuple(vectors.shape) != (len(meta["ids"]), self.dim):
            raise ValueError("Index artifact is corrupt: vector count does not match ids")
        self.rebuild(meta["ids"], vectors)
        return {"count": len(self.ids), "kind": self.kind}


class ExactDenseIndex(DenseIndex):
    kind = "exact"


class HNSWIndex(DenseIndex):
    """Optional HNSW index with an explicit exact fallback.

    ``available`` is False when ``hnswlib`` cannot be imported; ``mode`` then
    reads ``exact-fallback`` and ``fallback_reason`` explains why.  Profiles can
    still query without the native dependency.
    """

    kind = "hnsw"

    def __init__(self, dim: int = 384, M: int = 16, ef_construction: int = 200,
                 ef_search: int = 64, allow_fallback: bool = True):
        super().__init__(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.allow_fallback = bool(allow_fallback)
        self._lib = _hnswlib()
        self._index = None
        self._fallback = ExactDenseIndex(dim)
        self._deleted: set = set()
        self.fallback_reason: Optional[str] = None
        if self._lib is None:
            self.fallback_reason = "hnswlib is not installed"

    @property
    def available(self) -> bool:
        return self._lib is not None

    @property
    def mode(self) -> str:
        return "hnsw" if self.available else "exact-fallback"

    def rebuild(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        self._fallback.rebuild(ids, vectors)
        self.ids = list(self._fallback.ids)
        self.vectors = self._fallback.vectors
        self._positions = dict(self._fallback._positions)
        self._deleted = set()
        if not self.available:
            return
        if not self.ids:
            self._index = None
            return
        index = self._lib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=max(1, len(self.ids)), ef_construction=self.ef_construction,
                         M=self.M, random_seed=0)
        index.add_items(self.vectors.numpy(), self.ids)
        index.set_ef(self.ef_search)
        self._index = index

    def add(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        self._fallback.add(ids, vectors)
        self.rebuild(self._fallback.ids, self._fallback.vectors)

    def update(self, ids: Sequence[int], vectors: torch.Tensor) -> None:
        self._fallback.update(ids, vectors)
        self.rebuild(self._fallback.ids, self._fallback.vectors)

    def delete(self, ids: Iterable[int]) -> int:
        removed = self._fallback.delete(ids)
        self.rebuild(self._fallback.ids, self._fallback.vectors)
        return removed

    def search(self, query: torch.Tensor, k: int = 200) -> Tuple[List[int], List[float]]:
        if not self.available or self._index is None:
            return self._fallback.search(query, k)
        k = min(k, len(self.ids))
        if k <= 0:
            return [], []
        q = F.normalize(torch.as_tensor(query).float().reshape(-1), dim=0, eps=1e-8)
        self._index.set_ef(max(self.ef_search, k))
        labels, distances = self._index.knn_query(q.numpy(), k=k)
        ids = [int(label) for label in labels[0]]
        scores = [1.0 - float(distance) for distance in distances[0]]
        return ids, scores

    def load(self, path: str) -> Dict[str, Any]:
        # Persistence is defined by the exact vectors; the graph is rebuilt on
        # load so a corrupt graph can never take the store down.
        try:
            return super().load(path)
        except Exception as exc:
            self.fallback_reason = f"corrupt index: {exc}"
            if not self.allow_fallback:
                raise
            return {"count": len(self.ids), "kind": self.kind, "recovered": True}

    def persist(self, path: str) -> Dict[str, Any]:
        info = super().persist(path)
        info["mode"] = self.mode
        return info

    def stats(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "mode": self.mode, "available": self.available,
            "count": len(self.ids), "M": self.M, "ef_search": self.ef_search,
            "fallback_reason": self.fallback_reason,
        }


def build_dense_index(kind: str = "auto", dim: int = 384, **kwargs) -> DenseIndex:
    """Construct a dense index; ``auto`` prefers HNSW when importable."""
    kind = (kind or "auto").lower()
    if kind == "exact":
        return ExactDenseIndex(dim)
    if kind in ("auto", "hnsw"):
        index = HNSWIndex(dim, **kwargs)
        if isinstance(index, HNSWIndex) and index.available:
            return index
        if kind == "hnsw" and not index.allow_fallback:
            raise IndexUnavailableError(index.fallback_reason or "hnswlib unavailable")
        return index
    raise ValueError(f"Unknown dense index kind {kind!r}")


class CrossEncoderReranker:
    """Bounded optional reranker with explicit scorer provenance."""

    def __init__(self, model_name: Optional[str] = None, limit: int = 50,
                 allow_fallback: bool = True, fusion: str = "replace"):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.model_name = model_name
        self.limit = int(limit)
        self.allow_fallback = bool(allow_fallback)
        if fusion not in ("replace", "blend"):
            raise ValueError("fusion must be 'replace' or 'blend'")
        self.fusion = fusion
        self._model = None
        self._load_error: Optional[str] = None
        if model_name:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, local_files_only=True
                )
                self._model.eval()
            except Exception as exc:
                self._load_error = str(exc)
                if not allow_fallback:
                    raise IndexUnavailableError(
                        f"Cross-encoder {model_name!r} is unavailable: {exc}"
                    ) from exc

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def scorer(self) -> str:
        return "cross-encoder" if self.available else "fallback-lexical"

    def score(self, query: str, candidates: Sequence[str]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if self.available:
            with torch.no_grad():
                encoded = self._tokenizer(
                    [query] * len(candidates), list(candidates),
                    padding=True, truncation=True, return_tensors="pt",
                )
                logits = self._model(**encoded).logits.reshape(-1)
            return [{"score": float(value), "scorer": "cross-encoder"}
                    for value in logits]
        if not self.allow_fallback:
            raise IndexUnavailableError(self._load_error or "cross-encoder unavailable")
        tokens = set(_lexical_tokens(query))
        results = []
        for candidate in candidates:
            overlap = len(tokens & set(_lexical_tokens(candidate)))
            results.append({
                "score": overlap / max(1, len(tokens)),
                "scorer": "fallback-lexical",
            })
        return results

    def rerank(self, query: str, results: List[Dict[str, Any]],
               text_key: str = "text") -> Dict[str, Any]:
        """Rerank at most ``limit`` candidates and return an auditable trace."""
        if not results:
            return {"results": [], "scorer": self.scorer, "scored": 0,
                    "timing_ms": 0.0, "available": self.available}
        started = time.perf_counter()
        subset = results[: self.limit]
        scored = self.score(query, [str(item.get(text_key, "")) for item in subset])
        ranked = []
        for index, (item, score) in enumerate(zip(subset, scored)):
            enriched = dict(item)
            enriched["pre_rerank_rank"] = index + 1
            enriched["rerank_score"] = score["score"]
            enriched["rerank_scorer"] = score["scorer"]
            if self.fusion == "blend":
                base = float(item.get("fused_score", 0.0) or 0.0)
                enriched["rerank_score"] = 0.5 * base + 0.5 * score["score"]
            ranked.append(enriched)
        ranked.sort(key=lambda item: (-item["rerank_score"], item["pre_rerank_rank"]))
        for rank, item in enumerate(ranked, start=1):
            item["post_rerank_rank"] = rank
        tail = [dict(item, pre_rerank_rank=None, post_rerank_rank=None)
                for item in results[self.limit:]]
        return {
            "results": ranked + tail,
            "scorer": self.scorer,
            "scored": len(ranked),
            "available": self.available,
            "timing_ms": (time.perf_counter() - started) * 1000,
        }


def _lexical_tokens(text: str) -> List[str]:
    return [token for token in "".join(
        character.lower() if character.isalnum() else " " for character in text
    ).split() if token]


class MatryoshkaAnchor:
    """Validated 64-D fast representation plus the 384-D slow representation.

    The 64-D vector is a fixed random projection with orthonormal rows (never a
    truncation), versioned together with the base encoder identity.  A stored
    embedding produced by a different representation version is rejected.
    """

    def __init__(self, base_anchor=None, fast_dim: int = 64, full_dim: int = 384,
                 seed: int = 1234, version: str = MATRYOSHKA_VERSION):
        if not 0 < fast_dim <= full_dim:
            raise ValueError("fast_dim must be in (0, full_dim]")
        self.fast_dim = int(fast_dim)
        self.full_dim = int(full_dim)
        self.version = str(version)
        self._base = base_anchor
        generator = torch.Generator().manual_seed(int(seed))
        projection = torch.randn(self.fast_dim, self.full_dim, generator=generator)
        q, _ = torch.linalg.qr(projection.T)
        self.projection = q.T.contiguous()  # fast_dim x full_dim, orthonormal rows
        self.seed = int(seed)

    @property
    def base(self):
        if self._base is None:
            from src.model.onnx_anchor import OnnxMiniLM
            self._base = OnnxMiniLM()
        return self._base

    @property
    def projection_checksum(self) -> str:
        digest = hashlib.sha256(self.projection.numpy().tobytes()).hexdigest()
        return digest[:16]

    def metadata(self) -> Dict[str, Any]:
        return {
            "version": self.version, "fast_dim": self.fast_dim,
            "full_dim": self.full_dim, "seed": self.seed,
            "projection_checksum": self.projection_checksum,
        }

    def embed_full(self, texts) -> torch.Tensor:
        vector = self.base.embed(texts)
        vector = torch.as_tensor(vector).float()
        return F.normalize(vector.reshape(-1, self.full_dim), dim=-1, eps=1e-8)

    def project(self, full: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(full).float()
        flat = tensor.reshape(-1, self.full_dim)
        projected = F.normalize(flat @ self.projection.T, dim=-1, eps=1e-8)
        return projected.squeeze(0) if tensor.dim() == 1 else projected

    def embed(self, texts, dim: Optional[int] = None) -> torch.Tensor:
        full = self.embed_full(texts)
        dim = self.full_dim if dim is None else int(dim)
        if dim == self.full_dim:
            return full.squeeze(0) if len(full) == 1 else full
        if dim == self.fast_dim:
            fast = self.project(full)
            return fast.squeeze(0) if len(fast) == 1 else fast
        raise EmbeddingVersionError(
            f"Unsupported Matryoshka dimension {dim}; use {self.fast_dim} or {self.full_dim}"
        )

    def embed_pair(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        full = self.embed_full([text])[0]
        return self.project(full)[0], full

    def validate(self, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise EmbeddingVersionError("Embedding metadata must be an object")
        stored_version = metadata.get("version")
        if stored_version and stored_version != self.version:
            raise EmbeddingVersionError(
                f"Stored embedding version {stored_version!r} does not match "
                f"{self.version!r}; re-embed explicitly"
            )
        if int(metadata.get("full_dim", self.full_dim)) != self.full_dim:
            raise EmbeddingVersionError("Stored embedding dimension is incompatible")
        checksum = metadata.get("projection_checksum")
        if checksum and checksum != self.projection_checksum:
            raise EmbeddingVersionError(
                "Stored 64-D projection checksum differs; refusing to reinterpret it"
            )

    def save(self, path: str) -> None:
        from safetensors.torch import save_file
        save_file({"projection": self.projection.contiguous()}, str(path))
        Path(str(path) + ".json").write_text(json.dumps(self.metadata()))

    @classmethod
    def load(cls, path: str, base_anchor=None) -> "MatryoshkaAnchor":
        from safetensors.torch import load_file
        meta = json.loads(Path(str(path) + ".json").read_text())
        anchor = cls(base_anchor, fast_dim=int(meta["fast_dim"]),
                     full_dim=int(meta["full_dim"]), seed=int(meta.get("seed", 0)),
                     version=meta["version"])
        tensors = load_file(str(path))
        if tuple(tensors["projection"].shape) != (anchor.fast_dim, anchor.full_dim):
            raise EmbeddingVersionError("Stored projection shape is incompatible")
        anchor.projection = tensors["projection"].float()
        return anchor


class DualIndex:
    """Versioned fast (64-D) + full (384-D) index pair with no silent truncation."""

    FORMAT_VERSION = 1

    def __init__(self, anchor: MatryoshkaAnchor, kind: str = "exact"):
        self.anchor = anchor
        self.kind = kind
        self.fast = build_dense_index(kind, dim=anchor.fast_dim)
        self.full = build_dense_index(kind, dim=anchor.full_dim)
        self.version = anchor.version
        self.ids: List[int] = []

    def rebuild(self, ids: Sequence[int], full_vectors: torch.Tensor) -> None:
        ids = [int(i) for i in ids]
        full = _normalize(torch.as_tensor(full_vectors).reshape(len(ids), self.anchor.full_dim))
        fast = self.anchor.project(full)
        self.ids = ids
        self.fast.rebuild(ids, fast)
        self.full.rebuild(ids, full)

    def add(self, ids: Sequence[int], full_vectors: torch.Tensor) -> None:
        ids = [int(i) for i in ids]
        full = _normalize(torch.as_tensor(full_vectors).reshape(len(ids), self.anchor.full_dim))
        self.ids.extend(ids)
        self.fast.add(ids, self.anchor.project(full))
        self.full.add(ids, full)

    def delete(self, ids: Iterable[int]) -> int:
        removed = set(int(i) for i in ids)
        self.ids = [i for i in self.ids if i not in removed]
        self.fast.delete(removed)
        self.full.delete(removed)
        return len(removed)

    def search(self, query: torch.Tensor, k: int = 10, stage: str = "fast",
               confirm: int = 50) -> Dict[str, Any]:
        if stage == "fast":
            ids, scores = self.fast.search(self.anchor.project(query), k)
            return {"ids": ids, "scores": scores, "stage": "fast"}
        if stage == "full":
            ids, scores = self.full.search(query, k)
            return {"ids": ids, "scores": scores, "stage": "full"}
        if stage == "fast_then_full":
            first_ids, _ = self.fast.search(self.anchor.project(query), max(confirm, k))
            candidates = [fact_id for fact_id in first_ids
                          if fact_id in self.full._positions]
            if not candidates:
                return {"ids": [], "scores": [], "stage": "fast_then_full",
                        "confirmed": 0}
            positions = [self.full._positions[fact_id] for fact_id in candidates]
            q = F.normalize(torch.as_tensor(query).float().reshape(-1), dim=0)
            scores = self.full.vectors[positions] @ q
            order = torch.argsort(scores, descending=True)[:k].tolist()
            return {
                "ids": [candidates[i] for i in order],
                "scores": [float(scores[i]) for i in order],
                "stage": "fast_then_full",
                "confirmed": len(candidates),
            }
        raise ValueError(f"Unknown stage {stage!r}")

    def save(self, path: str) -> Dict[str, Any]:
        base = Path(path)
        self.fast.persist(str(base) + ".fast")
        self.full.persist(str(base) + ".full")
        meta = {
            "format_version": self.FORMAT_VERSION,
            "matryoshka": self.anchor.metadata(),
            "count": len(self.ids),
            "ids": self.ids,
        }
        base.write_text(json.dumps(meta))
        return {"path": str(base), "count": len(self.ids), "version": self.version}

    @classmethod
    def load(cls, path: str, anchor: MatryoshkaAnchor, kind: str = "exact") -> "DualIndex":
        base = Path(path)
        meta = json.loads(base.read_text())
        if int(meta.get("format_version", 0)) > cls.FORMAT_VERSION:
            raise EmbeddingVersionError("Dual index format is newer than supported")
        anchor.validate(meta.get("matryoshka", {}))
        dual = cls(anchor, kind=kind)
        dual.fast.load(str(base) + ".fast")
        dual.full.load(str(base) + ".full")
        if dual.fast.ids != dual.full.ids:
            raise EmbeddingVersionError("Fast and full indexes disagree on fact ids")
        dual.ids = list(dual.fast.ids)
        return dual

    def migrate(self, ids: Sequence[int], full_vectors: torch.Tensor) -> Dict[str, Any]:
        """Re-embed into the configured representation version."""
        self.rebuild(ids, full_vectors)
        return {"status": "migrated", "version": self.version, "count": len(self.ids)}


__all__ = [
    "CrossEncoderReranker",
    "DenseIndex",
    "DualIndex",
    "EmbeddingVersionError",
    "ExactDenseIndex",
    "HNSWIndex",
    "IndexUnavailableError",
    "MatryoshkaAnchor",
    "build_dense_index",
]
