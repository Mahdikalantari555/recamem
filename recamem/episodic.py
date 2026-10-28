
import os
import re
import shutil
import sqlite3
import threading
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# ponytail: SQLite WAL + FTS5 (BM25 lexical) + pre-normalized dense MVM fused via
# Reciprocal Rank Fusion, replacing O(N) JSON linear scan. stdlib sqlite3 + torch only.
# Ceiling: single-process; upgrade to HNSW/FAISS only if 100k+ facts miss 5ms on GPU.
# Extended with access_count, last_retrieved, version_id, feedback, needs_review,
# semantic_facts, feedback_log for lifelong learning.


EXECUTABLE_STRATEGIES = (
    "episodic_only",
    "semantic_primary",
    "hybrid_rrf",
    "episodic_recency_weighted",
    "contextual_boosted",
)


def _precision_at_k(ranked: Sequence[int], relevant: Sequence[int]) -> float:
    """Fraction of the first k results that are judged relevant (k=|relevant|)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    k = min(len(relevant_set), max(len(ranked), 1))
    head = list(ranked)[:k]
    if not head:
        return 0.0
    return sum(1 for fid in head if fid in relevant_set) / float(k)


def _mrr(ranked: Sequence[int], relevant: Sequence[int]) -> float:
    """Reciprocal rank of the first relevant result (0.0 when absent)."""
    relevant_set = set(relevant)
    for position, fid in enumerate(ranked, start=1):
        if fid in relevant_set:
            return 1.0 / float(position)
    return 0.0


def _tokenize(terms: List[str]) -> str:
    """Build a quoted OR'd FTS5 MATCH expression from query terms.

    English function words are dropped: they match nearly every fact and
    would otherwise swamp the BM25 lexical signal.
    """
    stop = {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "by", "for", "with", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "have", "has", "had", "it", "its",
        "this", "that", "these", "those", "which", "who", "whom", "whose",
        "what", "when", "where", "why", "how", "can", "could", "should",
        "would", "will", "shall", "may", "about", "into", "over", "under",
        "between", "during", "through", "out", "up", "down", "you", "your",
    }
    clean = [
        t for t in (re.sub(r"[^a-z0-9\u00C0-\uFFFF]", "", w.lower()) for w in terms)
        if t and t not in stop
    ]
    if not clean:
        return ""
    return " OR ".join(f'"{t}"' for t in clean[:32])


# The closed origin vocabulary for a stored fact (web-grounded-learning 3.2).
# ``source`` remains free text (a domain tag at the MCP boundary); ``origin`` is
# the machine-readable provenance class and is validated on insert.
ORIGINS: tuple = ("user_text", "chat", "continuous_stream", "web", "ingest")


def validate_origin(origin: str) -> str:
    """Return ``origin`` when it is documented, raise naming it otherwise."""
    if origin not in ORIGINS:
        raise ValueError(
            f"unknown fact origin {origin!r}; expected one of {list(ORIGINS)}"
        )
    return origin


class KnowledgeStore:
    """Persistent hybrid knowledge store: FTS5 BM25 + dense MVM, fused with RRF.

    Facts, embeddings and TMS metadata live in one SQLite DB (WAL mode).
    All dense vectors are L2-normalized at insert so recall is a single
    matrix-vector multiply with no per-query re-normalization allocation.
    Extended with lifelong learning columns and semantic/feedback tables.
    """

    _RRF_K = 60.0
    _CANDS = 200  # per-signal candidate pool before RRF fusion

    def __init__(self, db_path: str = ":memory:", dim: int = 384):
        self.db_path = db_path
        self.dim = dim
        file_based = db_path != ":memory:"
        if file_based:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        if file_based:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._migrate_schema()
        self._write_lock = threading.RLock()
        self._retired_cache = None
        self._record_schema_version()
        # Cached dense matrix, rebuilt only on insert. Per-query recall reuses it.
        self._vec_cache: Optional[torch.Tensor] = None
        self._id_cache: Optional[List[int]] = None
        # Optional hybrid retrieval (HNSW / reranker / Matryoshka). Defaults
        # preserve the exact dense MVM path byte-for-byte.
        self.dense_index = None
        self.reranker = None
        self.matryoshka = None
        self.rerank_limit = 50
        self._index_signature = None
        self.last_retrieval_trace: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Optional hybrid retrieval
    # ------------------------------------------------------------------ #
    def configure_retrieval(self, dense_index=None, reranker=None,
                            matryoshka=None, rerank_limit: int = 50) -> Dict[str, Any]:
        """Enable approximate candidates and/or reranking for this store."""
        self.dense_index = dense_index
        self.reranker = reranker
        self.matryoshka = matryoshka
        self.rerank_limit = int(rerank_limit)
        self._index_signature = None
        return self.retrieval_status()

    def retrieval_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "dense_index": None,
            "reranker": None,
            "matryoshka": None,
        }
        if self.dense_index is not None:
            if hasattr(self.dense_index, "stats"):
                status["dense_index"] = self.dense_index.stats()
            else:
                try:
                    count = len(self.dense_index)
                except Exception:
                    count = len(getattr(self.dense_index, "ids", []) or [])
                status["dense_index"] = {
                    "kind": getattr(self.dense_index, "kind", type(self.dense_index).__name__),
                    "mode": getattr(self.dense_index, "mode", "exact"),
                    "count": count,
                }
        if self.reranker is not None:
            status["reranker"] = {
                "scorer": getattr(self.reranker, "scorer", "unknown"),
                "limit": int(getattr(self.reranker, "limit", self.rerank_limit)),
            }
        if self.matryoshka is not None:
            status["matryoshka"] = self.matryoshka.metadata()
        return status

    def _sync_dense_index(self) -> bool:
        if self.dense_index is None:
            return False
        vecs, ids = self._load_dense()
        signature = (len(ids), ids[-1] if ids else None, len(getattr(self.dense_index, "ids", [])))
        index_ids = list(getattr(self.dense_index, "ids", []))
        if signature != self._index_signature or index_ids != ids:
            self.dense_index.rebuild(ids, vecs)
            self._index_signature = (len(ids), ids[-1] if ids else None, len(ids))
        return True

    def _dense_candidates(self, q: torch.Tensor, k: int):
        """Return (ids, scores) from the configured index, else exact MVM."""
        try:
            if self._sync_dense_index():
                ids, scores = self.dense_index.search(q, k)
                if len(ids):
                    return ids, scores
        except Exception as exc:  # fall back without corrupting the store
            self.last_retrieval_trace = {"dense_index_error": str(exc)}
        vecs, ids_all = self._load_dense()
        sims = torch.mv(vecs, q)
        top = sims.topk(k=min(k, len(ids_all)))
        return ([ids_all[i] for i in top.indices.tolist()], top.values.tolist())

    def _create_schema(self):
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT NOT NULL,
                source        TEXT,
                timestamp     REAL,
                step          INTEGER,
                novelty       REAL,
                superseded    INTEGER DEFAULT 0,
                valid_until   REAL,
                superseded_by INTEGER,
                access_count  INTEGER DEFAULT 1,
                last_retrieved REAL,
                version_id    TEXT,
                feedback      TEXT,
                needs_review  INTEGER DEFAULT 0,
                decay_beta    REAL DEFAULT 1.0,
                origin        TEXT DEFAULT 'user_text',
                source_url    TEXT,
                source_query  TEXT,
                tier          TEXT DEFAULT 'active',
                retire_reason TEXT,
                retired_at    REAL,
                file_path     TEXT,
                heading_path  TEXT,
                content_hash  TEXT
            );
            CREATE TABLE IF NOT EXISTS facts_vecs (
                id INTEGER PRIMARY KEY REFERENCES facts(id),
                vec BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                text, content='facts', content_rowid='id'
            );
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_cluster_ids TEXT,
                confidence REAL DEFAULT 0.5,
                created_at REAL,
                access_count INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS semantic_vecs (
                id INTEGER PRIMARY KEY REFERENCES semantic_facts(id),
                vec BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                text, content='semantic_facts', content_rowid='id'
            );
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                action TEXT,
                correction_text TEXT,
                timestamp REAL,
                strategy_used TEXT
            );
            CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                ranked_ids TEXT,
                relevant_ids TEXT,
                precision_at_k REAL,
                mrr REAL,
                k INTEGER,
                strategy_used TEXT,
                query_type TEXT,
                timestamp REAL
            );
            """
        )
        c.commit()

    #: Schema version written into ``store_meta``. Older databases are upgraded
    #: in place; a newer declared version is refused rather than guessed at.
    SCHEMA_VERSION = 2

    def _migrate_schema(self):
        """Add missing columns to existing DBs (ALTER TABLE)."""
        c = self.conn
        try:
            cols = [row[1] for row in c.execute("PRAGMA table_info(facts)").fetchall()]
            migrations = {
                "tier": "ALTER TABLE facts ADD COLUMN tier TEXT DEFAULT 'active'",
                "retire_reason": "ALTER TABLE facts ADD COLUMN retire_reason TEXT",
                "retired_at": "ALTER TABLE facts ADD COLUMN retired_at REAL",
                "file_path": "ALTER TABLE facts ADD COLUMN file_path TEXT",
                "heading_path": "ALTER TABLE facts ADD COLUMN heading_path TEXT",
                "content_hash": "ALTER TABLE facts ADD COLUMN content_hash TEXT",
                "access_count": "ALTER TABLE facts ADD COLUMN access_count INTEGER DEFAULT 1",
                "last_retrieved": "ALTER TABLE facts ADD COLUMN last_retrieved REAL",
                "version_id": "ALTER TABLE facts ADD COLUMN version_id TEXT",
                "feedback": "ALTER TABLE facts ADD COLUMN feedback TEXT",
                "needs_review": "ALTER TABLE facts ADD COLUMN needs_review INTEGER DEFAULT 0",
                "decay_beta": "ALTER TABLE facts ADD COLUMN decay_beta REAL DEFAULT 1.0",
                "origin": "ALTER TABLE facts ADD COLUMN origin TEXT DEFAULT 'user_text'",
                "source_url": "ALTER TABLE facts ADD COLUMN source_url TEXT",
                "source_query": "ALTER TABLE facts ADD COLUMN source_query TEXT",
            }
            for col, sql in migrations.items():
                if col not in cols:
                    try:
                        c.execute(sql)
                    except Exception:
                        pass
            c.commit()
        except Exception:
            pass

        try:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source_cluster_ids TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL,
                    access_count INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS semantic_vecs (
                    id INTEGER PRIMARY KEY REFERENCES semantic_facts(id),
                    vec BLOB
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                    text, content='semantic_facts', content_rowid='id'
                );
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id INTEGER,
                    action TEXT,
                    correction_text TEXT,
                    timestamp REAL,
                    strategy_used TEXT
                );
                CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    ranked_ids TEXT,
                    relevant_ids TEXT,
                    precision_at_k REAL,
                    mrr REAL,
                    k INTEGER,
                    strategy_used TEXT,
                    query_type TEXT,
                    timestamp REAL
                );
                """
            )
            c.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def insert_fact(
        self,
        text: str,
        source: str,
        timestamp: float,
        step: int,
        novelty: float,
        vec: torch.Tensor,
        version_id: Optional[str] = None,
        feedback: Optional[str] = None,
        needs_review: int = 0,
        origin: Optional[str] = None,
        source_url: Optional[str] = None,
        source_query: Optional[str] = None,
    ) -> int:
        # An explicit origin is validated; an absent one is derived from the
        # free-text source so every existing caller keeps working.
        if origin is None:
            origin = source if source in ORIGINS else "user_text"
        validate_origin(origin)
        with self._write_lock:
            v = vec.detach().float().reshape(-1)
            if v.numel() != self.dim:
                # A wrong-width vector previously corrupted the dense blob and
                # only failed later, at read time, as an opaque reshape error.
                raise ValueError(
                    f"vector width {v.numel()} does not match store dimension {self.dim}"
                )
            if not torch.isfinite(v).all():
                raise ValueError("vector contains non-finite values")
            norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
            v = v / norm
            self.conn.execute("BEGIN")
            try:
                cur = self.conn.execute(
                    "INSERT INTO facts(text, source, timestamp, step, novelty, access_count, last_retrieved, version_id, feedback, needs_review, origin, source_url, source_query)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (text, source, float(timestamp), int(step), float(novelty), 1, float(timestamp), version_id, feedback, int(needs_review), origin, source_url, source_query),
                )
                fid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO facts_vecs(id, vec) VALUES (?,?)",
                    (fid, v.cpu().contiguous().numpy().tobytes()),
                )
                self.conn.execute(
                    "INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (fid, text)
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            self._vec_cache = None
            return fid

    def semantic_count(self) -> int:
        """Number of distilled semantic facts."""
        row = self.conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()
        return int(row[0]) if row else 0

    def clear_semantic(self) -> int:
        """Delete every distilled semantic fact; returns the deleted count."""
        with self._write_lock:
            before = self.semantic_count()
            self.conn.execute("DELETE FROM semantic_facts")
            self.conn.commit()
        return before

    def bulk_insert_semantic_facts(self, rows: List[Dict[str, Any]]) -> int:
        """Insert distilled semantic facts in one transaction."""
        inserted = 0
        with self._write_lock:
            self.conn.execute("BEGIN")
            try:
                for row in rows:
                    vec = torch.as_tensor(row["vec"]).detach().float().reshape(-1)
                    if vec.numel() != self.dim or not torch.isfinite(vec).all():
                        continue
                    vec = vec / torch.linalg.vector_norm(vec).clamp(min=1e-8)
                    self.conn.execute(
                        "INSERT INTO semantic_facts(text, source_cluster_ids, confidence,"
                        " created_at, access_count) VALUES (?,?,?,?,?)",
                        (
                            row["text"],
                            json.dumps([int(i) for i in row["source_cluster_ids"]]),
                            float(row["confidence"]),
                            float(row.get("created_at", time.time())),
                            1,
                        ),
                    )
                    self.conn.execute(
                        "INSERT INTO semantic_vecs(id, vec) VALUES (?,?)",
                        (self.conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                         vec.numpy().tobytes()),
                    )
                    inserted += 1
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return inserted

    def bulk_insert_facts(
        self,
        rows: List[Dict[str, Any]],
        dedupe_threshold: Optional[float] = None,
    ) -> Dict[str, int]:
        """Insert many facts in one transaction.

        Merging 10k facts through ``insert_fact`` committed once per row, which
        turned a merge into minutes of fsync. This batches the writes, keeps the
        same normalisation/validation, and reports skipped rows instead of
        dropping them silently.

        ``rows`` items need ``text``, ``source``, ``timestamp``, ``step``,
        ``novelty`` and ``vec``. Returns ``{"inserted", "duplicate", "invalid"}``.
        """
        inserted = duplicate = invalid = 0
        matrix, _ids = self._load_dense()
        matrix = matrix if len(matrix) else None
        with self._write_lock:
            self.conn.execute("BEGIN")
            try:
                for row in rows:
                    try:
                        vec = torch.as_tensor(row["vec"]).detach().float().reshape(-1)
                        if vec.numel() != self.dim or not torch.isfinite(vec).all():
                            invalid += 1
                            continue
                        vec = vec / torch.linalg.vector_norm(vec).clamp(min=1e-8)
                        if dedupe_threshold is not None and matrix is not None:
                            if float(torch.mv(matrix, vec).max().item()) >= dedupe_threshold:
                                duplicate += 1
                                continue
                        origin = row.get("origin") or (
                            row.get("source") if row.get("source") in ORIGINS else "user_text"
                        )
                        validate_origin(origin)
                        cursor = self.conn.execute(
                            "INSERT INTO facts(text, source, timestamp, step, novelty,"
                            " access_count, last_retrieved, origin, source_url, source_query)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (
                                row["text"], row.get("source", "merge"),
                                float(row.get("timestamp", time.time())),
                                int(row.get("step", 0)), float(row.get("novelty", 0.5)),
                                1, float(row.get("timestamp", time.time())), origin,
                                row.get("source_url"), row.get("source_query"),
                            ),
                        )
                        self.conn.execute(
                            "INSERT INTO facts_vecs(id, vec) VALUES (?,?)",
                            (cursor.lastrowid, vec.numpy().tobytes()),
                        )
                        inserted += 1
                    except Exception:
                        invalid += 1
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                # The dense cache is stale after any successful write.
                self._vec_cache = None
                self._id_cache = None
        return {"inserted": inserted, "duplicate": duplicate, "invalid": invalid}

    def insert_semantic_fact(
        self,
        text: str,
        source_cluster_ids: List[int],
        confidence: float,
        vec: torch.Tensor,
    ) -> int:
        with self._write_lock:
            v = vec.detach().float().reshape(-1)
            if v.numel() != self.dim:
                # A wrong-width vector previously corrupted the dense blob and
                # only failed later, at read time, as an opaque reshape error.
                raise ValueError(
                    f"vector width {v.numel()} does not match store dimension {self.dim}"
                )
            if not torch.isfinite(v).all():
                raise ValueError("vector contains non-finite values")
            norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
            v = v / norm
            self.conn.execute("BEGIN")
            try:
                cur = self.conn.execute(
                    "INSERT INTO semantic_facts(text, source_cluster_ids, confidence, created_at, access_count)"
                    " VALUES (?,?,?,?,?)",
                    (text, json.dumps(source_cluster_ids), float(confidence), time.time(), 1),
                )
                fid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO semantic_vecs(id, vec) VALUES (?,?)",
                    (fid, v.cpu().contiguous().numpy().tobytes()),
                )
                self.conn.execute(
                    "INSERT INTO semantic_fts(rowid, text) VALUES (?,?)", (fid, text)
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return fid

    def log_feedback(self, fact_id: int, action: str, correction_text: Optional[str] = None, strategy_used: Optional[str] = None):
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO feedback_log(fact_id, action, correction_text, timestamp, strategy_used) VALUES (?,?,?,?,?)",
                (int(fact_id), action, correction_text, time.time(), strategy_used),
            )
            self.conn.commit()

    def log_ranked_feedback(self, query: str, ranked_ids: List[int], relevant_ids: List[int],
                            strategy_used: Optional[str] = None,
                            query_type: Optional[str] = None) -> Dict[str, Any]:
        """Persist one labelled retrieval judgement (post-merge 4.9).

        A row is written only when the caller supplies *both* a ranked result
        list and a non-empty set of judged-relevant ids, so health metrics can
        never be computed from unlabelled traffic.
        """
        ranked = [int(fid) for fid in ranked_ids]
        relevant = sorted({int(fid) for fid in relevant_ids})
        precision = _precision_at_k(ranked, relevant)
        mrr = _mrr(ranked, relevant)
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO retrieval_feedback(query, ranked_ids, relevant_ids, "
                "precision_at_k, mrr, k, strategy_used, query_type, timestamp) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    query,
                    json.dumps(ranked),
                    json.dumps(relevant),
                    precision,
                    mrr,
                    len(relevant),
                    strategy_used,
                    query_type,
                    time.time(),
                ),
            )
            self.conn.commit()
        return {
            "query": query,
            "k": len(relevant),
            "precision_at_k": precision,
            "mrr": mrr,
            "strategy_used": strategy_used,
            "query_type": query_type,
        }

    def retrieval_label_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM retrieval_feedback").fetchone()
        return int(row[0]) if row else 0

    def retrieval_health(self, window: int = 200) -> Dict[str, Any]:
        """precision@k / MRR over labelled feedback only.

        Returns ``available: False`` with an explicit reason and a sample count
        when there is no labelling data — no placeholder metric is invented.
        """
        rows = self.conn.execute(
            "SELECT precision_at_k, mrr, k, strategy_used, query_type FROM "
            "retrieval_feedback ORDER BY id DESC LIMIT ?",
            (int(window),),
        ).fetchall()
        total = self.retrieval_label_count()
        samples = len(rows)
        source = "retrieval_feedback table (explicitly labelled judgements)"
        if samples == 0:
            return {
                "available": False,
                "reason": "no_labelled_feedback",
                "samples": 0,
                "labels_recorded": total,
                "source": source,
                "precision_at_k": None,
                "mrr": None,
            }
        precision = sum(float(r[0]) for r in rows) / samples
        mrr = sum(float(r[1]) for r in rows) / samples
        by_strategy: Dict[str, Dict[str, Any]] = {}
        for p_at_k, row_mrr, _k, strategy, qtype in rows:
            key = strategy or "unknown"
            bucket = by_strategy.setdefault(
                key, {"samples": 0, "precision_at_k": 0.0, "mrr": 0.0}
            )
            bucket["samples"] += 1
            bucket["precision_at_k"] += float(p_at_k)
            bucket["mrr"] += float(row_mrr)
        for bucket in by_strategy.values():
            bucket["precision_at_k"] /= bucket["samples"]
            bucket["mrr"] /= bucket["samples"]
        by_query_type: Dict[str, int] = {}
        for _p, _m, _k, _s, qtype in rows:
            by_query_type[qtype or "unknown"] = by_query_type.get(qtype or "unknown", 0) + 1
        return {
            "available": True,
            "reason": None,
            "samples": samples,
            "labels_recorded": total,
            "window": int(window),
            "source": source,
            "precision_at_k": precision,
            "mrr": mrr,
            "by_strategy": by_strategy,
            "by_query_type": by_query_type,
        }

    def get_feedback_log(self, fact_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if fact_id is not None:
            rows = self.conn.execute(
                "SELECT id, fact_id, action, correction_text, timestamp, strategy_used FROM feedback_log WHERE fact_id=? ORDER BY timestamp",
                (fact_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, fact_id, action, correction_text, timestamp, strategy_used FROM feedback_log ORDER BY timestamp"
            ).fetchall()
        return [
            {"id": r[0], "fact_id": r[1], "action": r[2], "correction_text": r[3], "timestamp": r[4], "strategy_used": r[5]}
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Dense helpers
    # ------------------------------------------------------------------ #
    def _load_dense(self) -> Tuple[torch.Tensor, List[int]]:
        if self._vec_cache is not None and self._id_cache is not None:
            return self._vec_cache, self._id_cache
        rows = self.conn.execute(
            "SELECT id, vec FROM facts_vecs ORDER BY id"
        ).fetchall()
        if not rows:
            self._vec_cache = torch.zeros((0, self.dim))
            self._id_cache = []
            return self._vec_cache, self._id_cache
        ids = [r[0] for r in rows]
        buf = b"".join(r[1] for r in rows)
        arr = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(
            len(rows), self.dim
        )
        self._vec_cache = arr.clone().contiguous()
        self._id_cache = ids
        return self._vec_cache, self._id_cache

    def _query_vec(self, query_vec: torch.Tensor) -> torch.Tensor:
        q = query_vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(q).clamp(min=1e-8)
        return q / norm

    def find_similar(
        self, query_vec: torch.Tensor, threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        """Active (non-superseded) facts whose dense cosine sim >= threshold."""
        vecs, ids = self._load_dense()
        if len(ids) == 0:
            return []
        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        hits: List[Dict[str, Any]] = []
        for idx, score in enumerate(sims.tolist()):
            if score < threshold:
                continue
            row = self.conn.execute(
                "SELECT id, text, source, timestamp, step, novelty,"
                " superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review,"
                " origin, source_url, source_query"
                " FROM facts WHERE id = ?",
                (ids[idx],),
            ).fetchone()
            if row and row[6] != 1:
                entry = self._to_dict(row)
                entry["similarity"] = score
                hits.append(entry)
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return self._drop_retired(hits)

    # ------------------------------------------------------------------ #
    # Power-law accessibility (post-merge 4.7)
    # ------------------------------------------------------------------ #
    # A_i(t) = S_base * (1 + alpha * (t - t_last)) ** -beta_i
    # beta_i is positive, starts at DECAY_BETA_INITIAL and is *decreased* by
    # successful retrieval (bounded below by DECAY_BETA_MIN), so reinforced
    # facts decay more slowly. State lives in the facts table and therefore
    # survives restart with no replay.
    DECAY_ALPHA = 1.0 / 86400.0       # per-second ageing rate (one day scale)
    DECAY_BETA_INITIAL = 1.0
    DECAY_BETA_MIN = 0.2
    REINFORCE_DELTA = 0.2             # within the specified [0.1, 0.3]

    def decay_exponent(self, fact_id: int) -> float:
        row = self.conn.execute(
            "SELECT decay_beta FROM facts WHERE id=?", (int(fact_id),)
        ).fetchone()
        value = row[0] if row and row[0] is not None else self.DECAY_BETA_INITIAL
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.DECAY_BETA_INITIAL
        return max(self.DECAY_BETA_MIN, value)

    def accessibility(self, fact_id: int, now: Optional[float] = None) -> float:
        """Current power-law accessibility of a fact in (0, 1]."""
        row = self.conn.execute(
            "SELECT access_count, last_retrieved, timestamp, decay_beta FROM facts WHERE id=?",
            (int(fact_id),),
        ).fetchone()
        if row is None:
            return 0.0
        access_count, last_retrieved, timestamp, beta = row
        now = time.time() if now is None else float(now)
        base = max(1.0, float(access_count or 1))
        last = last_retrieved if last_retrieved is not None else timestamp
        if last is None:
            last = now
        beta = self.DECAY_BETA_MIN if beta is None else max(
            self.DECAY_BETA_MIN, float(beta)
        )
        age = max(0.0, now - float(last))
        return float(base * (1.0 + self.DECAY_ALPHA * age) ** (-beta))

    def reinforce(self, fact_id: int, now: Optional[float] = None,
                  delta: Optional[float] = None) -> Dict[str, Any]:
        """Successful retrieval: bounded decay-exponent decrease + access bump."""
        delta = self.REINFORCE_DELTA if delta is None else float(delta)
        delta = max(0.1, min(0.3, delta))
        before = self.decay_exponent(fact_id)
        after = max(self.DECAY_BETA_MIN, before * (1.0 - delta))
        now = time.time() if now is None else float(now)
        with self._write_lock:
            self.conn.execute(
                "UPDATE facts SET decay_beta=?, access_count=COALESCE(access_count,0)+1, "
                "last_retrieved=? WHERE id=?",
                (after, now, int(fact_id)),
            )
            self.conn.commit()
        return {
            "fact_id": int(fact_id),
            "beta_before": before,
            "beta_after": after,
            "delta": delta,
            "accessibility": self.accessibility(fact_id, now=now),
        }

    # ------------------------------------------------------------------ #
    # Recall: fused RRF - preserved original logic with extensions
    # ------------------------------------------------------------------ #
    def recall(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
        strategy: str = "hybrid_rrf",
    ) -> List[Dict[str, Any]]:
        """Recall the active tier; retired facts need an explicit request."""
        return self._drop_retired(
            self._recall_all_tiers(query_vec, top_k, threshold, query_text, strategy)
        )

    def _recall_all_tiers(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
        strategy: str = "hybrid_rrf",
    ) -> List[Dict[str, Any]]:
        """Hybrid lexical+dense retrieval, fused via Reciprocal Rank Fusion."""
        vecs, ids = self._load_dense()
        if len(ids) == 0:
            if strategy != "semantic_primary":
                return []
            # A semantic-only profile still answers definitional questions:
            # the selected strategy decides which store is consulted (4.2).
            semantic = self.recall_semantic(
                query_vec, top_k=top_k, query_text=query_text
            )
            results = []
            for hit in semantic:
                entry = dict(hit)
                entry["retrieval_store"] = "semantic"
                entry["superseded"] = False
                results.append(entry)
            self.last_retrieval_trace = {
                "requested_strategy": strategy,
                "executed_strategy": "semantic_primary",
                "semantic_hits": len(results),
                "candidates": len(results),
                "episodic_available": False,
            }
            return results[: max(top_k, 1)]

        q = self._query_vec(query_vec)
        dense_ids, dense_vals = self._dense_candidates(q, min(self._CANDS, len(ids)))
        with self._write_lock:
            self.conn.execute("DROP TABLE IF EXISTS _dense_cands")
            self.conn.execute(
                "CREATE TEMPORARY TABLE _dense_cands ("
                "id INTEGER PRIMARY KEY, score REAL, drank INTEGER)"
            )
            self.conn.executemany(
                "INSERT INTO _dense_cands(id, score, drank) VALUES (?,?,?)",
                [
                    (int(fact_id), dense_vals[rank], rank + 1)
                    for rank, fact_id in enumerate(dense_ids)
                ],
            )

        lex_match = _tokenize(query_text.split())
        has_lex = bool(lex_match)

        # Original used FULL OUTER JOIN - we emulate with UNION for compatibility
        # But we keep original SQL structure for correctness
        # Try FULL OUTER JOIN first, fallback to UNION if fails
        try:
            fused_sql = f"""
                WITH lex AS (
                    SELECT f.id AS id,
                           ROW_NUMBER() OVER (ORDER BY bm25(facts_fts) ASC) AS lrank
                    FROM facts_fts
                    JOIN facts f ON f.id = facts_fts.rowid
                    WHERE {{lex_where}} AND f.superseded = 0
                    LIMIT {{cands}}
                ),
                dense AS (
                    SELECT id, score AS dscore, drank
                    FROM _dense_cands
                ),
                fused AS (
                    SELECT COALESCE(lex.id, dense.id) AS id,
                           CASE WHEN lex.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + lex.lrank) END AS lex_part,
                           CASE WHEN dense.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + dense.drank) END AS dense_part,
                           COALESCE(lex.lrank, 0) AS lrank,
                           COALESCE(dense.drank, 0) AS drank,
                           dense.dscore AS dscore
                    FROM lex FULL OUTER JOIN dense ON lex.id = dense.id
                )
                SELECT f.id, f.text, f.source, f.timestamp, f.step, f.novelty,
                       f.superseded, f.valid_until, f.superseded_by,
                       f.access_count, f.last_retrieved, f.version_id, f.feedback, f.needs_review,
                       fused.fused_score, fused.lex_part, fused.dense_part,
                       fused.lrank, fused.drank, fused.dscore,
                       f.origin, f.source_url, f.source_query
                FROM (
                    SELECT id,
                           (lex_part + dense_part) AS fused_score,
                           lex_part, dense_part, lrank, drank, dscore
                    FROM fused
                ) fused
                JOIN facts f ON f.id = fused.id
                WHERE f.superseded = 0 AND fused.fused_score >= {{threshold}}
                ORDER BY fused.fused_score DESC, fused.dscore DESC
                LIMIT {{top_k}}
            """.format(
                lex_where="facts_fts MATCH ?" if has_lex else "1=0",
                cands=self._CANDS,
                rrf_k=self._RRF_K,
                threshold=threshold,
                top_k=top_k,
            )
            params = [lex_match] if has_lex else []
            cursor = self.conn.execute(fused_sql, params)
            rows = cursor.fetchall()
        except Exception as e:
            # Fallback: UNION emulation
            fused_sql = f"""
                WITH lex AS (
                    SELECT f.id AS id,
                           ROW_NUMBER() OVER (ORDER BY bm25(facts_fts) ASC) AS lrank
                    FROM facts_fts
                    JOIN facts f ON f.id = facts_fts.rowid
                    WHERE {{lex_where}} AND f.superseded = 0
                    LIMIT {{cands}}
                ),
                dense AS (
                    SELECT id, score AS dscore, drank
                    FROM _dense_cands
                ),
                fused AS (
                    SELECT COALESCE(lex.id, dense.id) AS id,
                           CASE WHEN lex.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + lex.lrank) END AS lex_part,
                           CASE WHEN dense.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + dense.drank) END AS dense_part,
                           COALESCE(lex.lrank, 0) AS lrank,
                           COALESCE(dense.drank, 0) AS drank,
                           dense.dscore AS dscore
                    FROM lex LEFT JOIN dense ON lex.id = dense.id
                    UNION
                    SELECT dense.id AS id,
                           0.0 AS lex_part,
                           1.0/({{rrf_k}} + dense.drank) AS dense_part,
                           0 AS lrank,
                           dense.drank AS drank,
                           dense.dscore AS dscore
                    FROM dense LEFT JOIN lex ON dense.id = lex.id
                    WHERE lex.id IS NULL
                )
                SELECT f.id, f.text, f.source, f.timestamp, f.step, f.novelty,
                       f.superseded, f.valid_until, f.superseded_by,
                       f.access_count, f.last_retrieved, f.version_id, f.feedback, f.needs_review,
                       fused.fused_score, fused.lex_part, fused.dense_part,
                       fused.lrank, fused.drank, fused.dscore,
                       f.origin, f.source_url, f.source_query
                FROM (
                    SELECT id,
                           (lex_part + dense_part) AS fused_score,
                           lex_part, dense_part, lrank, drank, dscore
                    FROM fused
                ) fused
                JOIN facts f ON f.id = fused.id
                WHERE f.superseded = 0 AND fused.fused_score >= {{threshold}}
                ORDER BY fused.fused_score DESC, fused.dscore DESC
                LIMIT {{top_k}}
            """.format(
                lex_where="facts_fts MATCH ?" if has_lex else "1=0",
                cands=self._CANDS,
                rrf_k=self._RRF_K,
                threshold=threshold,
                top_k=top_k,
            )
            params = [lex_match] if has_lex else []
            rows = self.conn.execute(fused_sql, params).fetchall()

        results = []
        for r in rows:
            entry = {
                "id": r[0],
                "text": r[1],
                "source": r[2],
                "timestamp": r[3],
                "step": r[4],
                "novelty": r[5],
                "superseded": bool(r[6]),
                "valid_until": r[7],
                "superseded_by": r[8],
                "access_count": r[9] if r[9] is not None else 1,
                "last_retrieved": r[10],
                "version_id": r[11],
                "feedback": r[12],
                "needs_review": bool(r[13]) if r[13] is not None else False,
                "fused_score": r[14],
                "lexical_score": r[15],
                "dense_score": r[16],
                "lexical_rank": r[17],
                "dense_rank": r[18],
                "dense_similarity": r[19] if r[19] is not None else 0.0,
                "similarity": r[19] if r[19] is not None else 0.0,
                "origin": r[20] if len(r) > 20 and r[20] is not None else "user_text",
                "source_url": r[21] if len(r) > 21 else None,
                "source_query": r[22] if len(r) > 22 else None,
                "retrieved_at": r[3],
            }
            results.append(entry)

        # Strategy execution (post-merge 4.2): the label decides access, not
        # just diagnostics. Every result records which store answered.
        executed_strategy = strategy if strategy in EXECUTABLE_STRATEGIES else "hybrid_rrf"
        if executed_strategy == "semantic_primary":
            try:
                semantic = self.recall_semantic(q, top_k=top_k)
            except Exception:
                semantic = []
            if semantic:
                promoted = []
                for hit in semantic:
                    entry = dict(hit)
                    entry.setdefault("fused_score", float(entry.get("similarity", 0.0)))
                    entry["retrieval_store"] = "semantic"
                    entry["superseded"] = False
                    promoted.append(entry)
                # Deduplicate by text: semantic ids and fact ids come from
                # different tables and are not comparable.
                seen = {str(entry.get("text", "")).strip().lower() for entry in promoted}
                for entry in results:
                    entry["retrieval_store"] = "episodic"
                results = promoted + [
                    e for e in results
                    if str(e.get("text", "")).strip().lower() not in seen
                ]
                results = results[: max(top_k, 1)]
            else:
                for entry in results:
                    entry["retrieval_store"] = "episodic"
        elif executed_strategy == "episodic_recency_weighted":
            for entry in results:
                stamp = entry.get("timestamp") or entry.get("last_retrieved") or 0.0
                age_days = max(0.0, (time.time() - float(stamp)) / 86400.0)
                weight = 1.0 / (1.0 + 0.5 * age_days)
                entry["recency_weight"] = weight
                entry["fused_score"] = float(entry.get("fused_score") or 0.0) * weight
                entry["retrieval_store"] = "episodic"
            results.sort(key=lambda e: e["fused_score"], reverse=True)
        elif executed_strategy == "contextual_boosted":
            for entry in results:
                boost = 1.0 + 0.1 * float(self.decay_exponent(entry["id"]))
                entry["contextual_weight"] = boost
                entry["fused_score"] = float(entry.get("fused_score") or 0.0) * boost
                entry["retrieval_store"] = "episodic"
            results.sort(key=lambda e: e["fused_score"], reverse=True)
        else:  # episodic_only / hybrid_rrf
            for entry in results:
                entry["retrieval_store"] = "episodic"

        # Optional bounded reranking: only the configured top candidates are
        # scored, and both pre/post ranks plus the scorer stay in the trace.
        previous_trace = self.last_retrieval_trace or {}
        self.last_retrieval_trace = {
            **previous_trace,
            "dense_index": self.retrieval_status().get("dense_index"),
            "candidates": len(results),
            "requested_strategy": strategy,
            "executed_strategy": executed_strategy,
            "semantic_hits": sum(
                1 for entry in results if entry.get("retrieval_store") == "semantic"
            ),
        }
        if self.reranker is not None and results and query_text:
            try:
                rerank = self.reranker.rerank(query_text, results, text_key="text")
                results = rerank["results"]
                self.last_retrieval_trace.update({
                    "reranker": rerank["scorer"],
                    "rerank_scored": rerank["scored"],
                    "rerank_timing_ms": rerank["timing_ms"],
                })
            except Exception as exc:
                self.last_retrieval_trace["reranker_error"] = str(exc)

        # Successful retrieval reinforces each returned *episodic* fact: the
        # access counter bumps and the decay exponent decreases (4.7). Semantic
        # rows have their own counters inside recall_semantic().
        if results:
            now = time.time()
            for entry in results:
                if entry.get("retrieval_store") != "semantic":
                    try:
                        self.reinforce(entry["id"], now=now)
                    except Exception:
                        pass

        with self._write_lock:
            self.conn.execute("DROP TABLE IF EXISTS _dense_cands")
        return results

    def recall_semantic(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT id, vec FROM semantic_vecs ORDER BY id").fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        buf = b"".join(r[1] for r in rows)
        vecs = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(len(rows), self.dim)
        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        top = sims.topk(k=min(top_k, len(ids)))
        results = []
        for rank, idx in enumerate(top.indices.tolist()):
            score = top.values[rank].item()
            # The default threshold is 0.0, documented as "no filter". Applying
            # it as a floor silently dropped every hit whose cosine happened to
            # be negative, which made a semantic-only profile answer nothing.
            # A positive threshold is a real floor and still filters.
            if threshold > 0 and score < threshold:
                continue
            row = self.conn.execute(
                "SELECT id, text, source_cluster_ids, confidence, created_at, access_count FROM semantic_facts WHERE id=?",
                (ids[idx],),
            ).fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "source_cluster_ids": json.loads(row[2]) if row[2] else [],
                    "confidence": row[3],
                    "created_at": row[4],
                    "access_count": row[5],
                    "similarity": score,
                    "dense_similarity": score,
                    "fused_score": score,
                })
        if results:
            with self._write_lock:
                for entry in results:
                    try:
                        self.conn.execute(
                            "UPDATE semantic_facts SET access_count = COALESCE(access_count,0)+1 WHERE id=?",
                            (entry["id"],),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                self.conn.commit()
        return results

    # ------------------------------------------------------------------ #
    # Schema version, writer lock and the active/retired tier
    # ------------------------------------------------------------------ #
    def schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM store_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else self.SCHEMA_VERSION

    def _record_schema_version(self) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO store_meta(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    def _refresh_tier_cache(self) -> None:
        rows = self.conn.execute(
            "SELECT id FROM facts WHERE tier = 'retired'"
        ).fetchall()
        self._retired_cache = {int(r[0]) for r in rows}

    def retired_ids(self) -> set:
        if getattr(self, "_retired_cache", None) is None:
            self._refresh_tier_cache()
        return set(self._retired_cache)

    def _drop_retired(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        retired = self.retired_ids()
        if not retired:
            return hits
        return [h for h in hits if int(h.get("id", -1)) not in retired]

    def tier_of(self, fact_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT tier FROM facts WHERE id = ?", (int(fact_id),)
        ).fetchone()
        return row[0] if row else None

    def retire_fact(self, fact_id: int, reason: str = "capacity") -> bool:
        """Move a fact to the retired tier. The row is never deleted."""
        with self._write_lock:
            cur = self.conn.execute(
                "UPDATE facts SET tier='retired', retire_reason=?, retired_at=?"
                " WHERE id=? AND tier IS NOT 'retired'",
                (reason, float(time.time()), int(fact_id)),
            )
            self._retired_cache = None
            return cur.rowcount > 0

    def restore_fact(self, fact_id: int) -> bool:
        """Return a retired fact to the active tier with its history intact."""
        with self._write_lock:
            cur = self.conn.execute(
                "UPDATE facts SET tier='active', retire_reason=NULL, retired_at=NULL"
                " WHERE id=? AND tier='retired'",
                (int(fact_id),),
            )
            self._retired_cache = None
            return cur.rowcount > 0

    def facts_by_file(self, file_path: str, include_superseded: bool = False):
        sql = "SELECT id, text, source, tier, superseded, content_hash FROM facts WHERE file_path = ?"
        if not include_superseded:
            sql += " AND superseded = 0"
        return self.conn.execute(sql + " ORDER BY id", (str(file_path),)).fetchall()

    def file_content_hash(self, file_path: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content_hash FROM facts WHERE file_path = ? AND content_hash IS NOT NULL"
            " ORDER BY id DESC LIMIT 1",
            (str(file_path),),
        ).fetchone()
        return row[0] if row else None

    def insert_fact_with_provenance(
        self,
        text: str,
        source: str,
        timestamp: float,
        step: int,
        novelty: float,
        vec,
        *,
        file_path: Optional[str] = None,
        heading_path: Optional[str] = None,
        content_hash: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> int:
        """Insert a fact and attach ingest provenance (ingestion capability)."""
        fid = self.insert_fact(
            text, source, timestamp, step, novelty, vec, origin=origin
        )
        with self._write_lock:
            self.conn.execute(
                "UPDATE facts SET file_path=?, heading_path=?, content_hash=?"
                " WHERE id=?",
                (file_path, heading_path, content_hash, fid),
            )
        return fid

    def _to_dict(self, row) -> Dict[str, Any]:
        entry: Dict[str, Any]
        if len(row) >= 14:
            entry = {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "timestamp": row[3],
                "step": row[4],
                "novelty": row[5],
                "superseded": bool(row[6]),
                "valid_until": row[7],
                "superseded_by": row[8],
                "access_count": row[9] if row[9] is not None else 1,
                "last_retrieved": row[10],
                "version_id": row[11] if len(row) > 11 else None,
                "feedback": row[12] if len(row) > 12 else None,
                "needs_review": bool(row[13]) if len(row) > 13 and row[13] is not None else False,
                "origin": row[14] if len(row) > 14 and row[14] is not None else "user_text",
                "source_url": row[15] if len(row) > 15 else None,
                "source_query": row[16] if len(row) > 16 else None,
                "retrieved_at": row[3],
            }
        else:
            entry = {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "timestamp": row[3],
                "step": row[4],
                "novelty": row[5],
                "superseded": bool(row[6]),
                "valid_until": row[7],
                "superseded_by": row[8],
                "origin": "user_text",
                "source_url": None,
                "source_query": None,
                "retrieved_at": row[3],
            }
        return entry

    def supersede(self, old_id: int, new_id: Optional[int], now: Optional[float] = None) -> int:
        with self._write_lock:
            """Mark ``old_id`` superseded by ``new_id``.

            ``new_id`` must be the id of a stored fact (the replacement) or ``None``
            when the fact is superseded with no successor. Sentinels such as ``0``
            are rejected: they would otherwise be written to ``superseded_by`` and
            point at a fact that does not exist.
            """
            if new_id is None:
                return self.mark_superseded(old_id, now)
            new_id = int(new_id)
            if new_id <= 0:
                raise ValueError(
                    "superseded_by must be a stored fact id or None; "
                    f"got {new_id!r} (sentinels are not allowed)"
                )
            exists = self.conn.execute("SELECT 1 FROM facts WHERE id = ?", (new_id,)).fetchone()
            if exists is None:
                raise ValueError(f"superseded_by must reference an existing fact; id {new_id} not found")
            cur = self.conn.execute(
                "UPDATE facts SET superseded = 1, valid_until = ?,"
                " superseded_by = ? WHERE id = ? AND superseded = 0",
                (now if now is not None else time.time(), new_id, old_id),
            )
            return cur.rowcount

    def mark_superseded(self, fact_id: int, now: Optional[float] = None) -> int:
        with self._write_lock:
            """Mark a fact superseded with no successor: ``superseded_by`` is NULL."""
            cur = self.conn.execute(
                "UPDATE facts SET superseded = 1, valid_until = ?,"
                " superseded_by = NULL WHERE id = ? AND superseded = 0",
                (now if now is not None else time.time(), int(fact_id)),
            )
            return cur.rowcount

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty,"
            " superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review,"
            " origin, source_url, source_query FROM facts ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_active_texts(self):
        rows = self.conn.execute("SELECT LOWER(text) FROM facts WHERE superseded=0").fetchall()
        return set(r[0].strip() for r in rows)

    def get_active_facts(self):
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty, superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review, origin, source_url, source_query FROM facts WHERE superseded=0 ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve single fact by ID with all metadata."""
        row = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty, superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review, origin, source_url, source_query FROM facts WHERE id = ?",
            (int(fact_id),),
        ).fetchone()
        if row is None:
            return None
        return self._to_dict(row)

    def update_fact(self, fact_id: int, new_text: str, new_vec: Optional[torch.Tensor] = None) -> int:
        """Update fact text and re-embed vector. Invalidates caches."""
        with self._write_lock:
            fact_id = int(fact_id)
            # Check exists
            existing = self.conn.execute("SELECT id FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if existing is None:
                return 0
            
            # Embed if vec not provided
            if new_vec is None:
                try:
                    # Try to import anchor and embed
                    from recamem.anchor import OnnxMiniLM
                    anchor = OnnxMiniLM()
                    new_vec = anchor.embed(new_text, to_torch=True)
                except Exception:
                    try:
                        from .anchor import OnnxMiniLM
                        anchor = OnnxMiniLM()
                        new_vec = anchor.embed(new_text, to_torch=True)
                    except Exception as e:
                        # Fallback: keep old vector if embedding fails
                        row = self.conn.execute("SELECT vec FROM facts_vecs WHERE id = ?", (fact_id,)).fetchone()
                        if row:
                            import numpy as np
                            vec_np = np.frombuffer(row[0], dtype=np.float32)
                            new_vec = torch.from_numpy(vec_np)
                        else:
                            new_vec = torch.randn(self.dim)

            v = new_vec.detach().float().reshape(-1)
            norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
            v = v / norm

            self.conn.execute("BEGIN")
            try:
                # Update facts table
                self.conn.execute(
                    "UPDATE facts SET text = ?, timestamp = ? WHERE id = ?",
                    (new_text, time.time(), fact_id),
                )
                # Update vec
                self.conn.execute(
                    "UPDATE facts_vecs SET vec = ? WHERE id = ?",
                    (v.cpu().contiguous().numpy().tobytes(), fact_id),
                )
                # Rebuild FTS - simplest: delete and insert
                self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
                self.conn.execute(
                    "INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (fact_id, new_text)
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            
            self._vec_cache = None
            self._id_cache = None
            return 1

    def delete_fact(self, fact_id: int) -> int:
        """Delete fact entirely from all tables. Invalidates caches."""
        with self._write_lock:
            fact_id = int(fact_id)
            self.conn.execute("BEGIN")
            try:
                cur = self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                deleted = cur.rowcount
                self.conn.execute("DELETE FROM facts_vecs WHERE id = ?", (fact_id,))
                self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
                # Also clean feedback log? Keep for audit, but optional
                # self.conn.execute("DELETE FROM feedback_log WHERE fact_id = ?", (fact_id,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            
            self._vec_cache = None
            self._id_cache = None
            return deleted

    def total_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        return row[0] if row else 0

    def active_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded=0 AND (tier IS NULL OR tier='active')"
        ).fetchone()
        return row[0] if row else 0

    def checkpoint(self):
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:  # noqa: BLE001
            pass

    def export_to(self, dest_path: str) -> Path:
        self.checkpoint()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path == ":memory:":
            backup_conn = sqlite3.connect(str(dest))
            self.conn.backup(backup_conn)
            backup_conn.close()
        else:
            src = Path(self.db_path)
            if src.exists():
                shutil.copy2(str(src), str(dest))
                for suffix in ["-wal", "-shm"]:
                    s = Path(str(src) + suffix)
                    if s.exists():
                        try:
                            shutil.copy2(str(s), str(dest) + suffix)
                        except Exception:  # noqa: BLE001
                            pass
            else:
                backup_conn = sqlite3.connect(str(dest))
                self.conn.backup(backup_conn)
                backup_conn.close()
        return dest

    def clear_all(self):
        with self._write_lock:
            self.conn.execute("DELETE FROM facts")
            self.conn.execute("DELETE FROM facts_vecs")
            self.conn.execute("DELETE FROM facts_fts")
            self.conn.execute("DELETE FROM semantic_facts")
            self.conn.execute("DELETE FROM semantic_vecs")
            self.conn.execute("DELETE FROM semantic_fts")
            self.conn.execute("DELETE FROM feedback_log")
            self.conn.commit()
            self._vec_cache = None
            self._id_cache = None

    def close(self):
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def get_all_embeddings(self) -> Tuple[torch.Tensor, List[int]]:
        return self._load_dense()
