import os
import tempfile

import pytest
import torch

from src.model.knowledge_store import KnowledgeStore


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    ks = KnowledgeStore(os.path.join(d, "k.db"), dim=8)
    yield ks
    ks.close()


def _vec(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(8, generator=g)


def test_insert_and_recall(store):
    v = _vec(1)
    store.insert_fact("Remote sensing uses satellites", "chat", 1.0, 0, 0.9, v)
    store.insert_fact("Lidar measures distance with laser pulses", "chat", 2.0, 1, 0.8, _vec(2))

    hits = store.recall(v, top_k=2, query_text="remote sensing satellites")
    assert len(hits) == 2
    assert hits[0]["text"].startswith("Remote sensing")
    assert hits[0]["dense_similarity"] > 0.9
    # spec: response includes both lexical rank and dense similarity scores
    assert "lexical_rank" in hits[0] and "dense_similarity" in hits[0]


def test_recall_empty_store(store):
    assert store.recall(_vec(1), top_k=3, query_text="anything") == []


def test_supersede_excludes_from_recall_but_keeps_audit(store):
    v1 = _vec(1)
    id1 = store.insert_fact("The capital of West Germany is Bonn", "chat", 1.0, 0, 0.5, v1)
    id2 = store.insert_fact("The capital of West Germany is Berlin", "chat", 2.0, 1, 0.5, _vec(2))

    assert store.supersede(id1, id2) == 1
    # double-supersede is a no-op
    assert store.supersede(id1, id2) == 0
    # unknown id is a no-op
    assert store.supersede(9999, id2) == 0

    hits = store.recall(_vec(3), top_k=5, query_text="west germany capital")
    assert all(h["id"] != id1 for h in hits)

    trail = store.get_audit_trail()
    assert len(trail) == 2
    old = [t for t in trail if t["id"] == id1][0]
    assert old["superseded"] is True
    assert old["superseded_by"] == id2
    assert old["valid_until"] is not None
    # original text/timestamp/step unchanged (audit trail completeness)
    assert old["text"] == "The capital of West Germany is Bonn"
    assert old["timestamp"] == 1.0
    assert old["step"] == 0


def test_rrf_fusion_ranks_lexical_and_dense(store):
    # fact A: dense match to query vec, no lexical overlap
    # fact B: lexical match to query text, weak dense
    v = _vec(42)
    store.insert_fact("Quantum entanglement enables secure communication", "chat", 1.0, 0, 0.5, v)
    store.insert_fact("Remote sensing uses satellites for Earth observation", "chat", 2.0, 1, 0.5, _vec(7))

    hits = store.recall(v, top_k=2, query_text="remote sensing satellites")
    texts = [h["text"] for h in hits]
    assert len(hits) == 2
    # both signals present in results; fused score combines both parts
    for h in hits:
        assert h["fused_score"] > 0
        assert h["lexical_score"] >= 0 and h["dense_score"] >= 0


def test_pre_normalized_vectors(store):
    v = _vec(5) * 100.0  # deliberately unnormalized
    store.insert_fact("Normalized fact", "chat", 1.0, 0, 0.5, v)
    vecs, ids = store._load_dense()
    norm = torch.linalg.vector_norm(vecs[0])
    assert abs(norm.item() - 1.0) < 1e-4


def test_export_roundtrip(store):
    store.insert_fact("Persistent fact", "chat", 1.0, 0, 0.5, _vec(9))
    store.insert_fact("Another fact", "chat", 2.0, 1, 0.5, _vec(10))
    store.supersede(1, 2)

    d = tempfile.mkdtemp()
    dest = store.export_to(os.path.join(d, "exported.db"))
    ks2 = KnowledgeStore(str(dest), dim=8)
    try:
        assert ks2.total_count() == 2
        assert ks2.active_count() == 1
        trail = ks2.get_audit_trail()
        assert trail[0]["superseded"] is True
        assert trail[0]["superseded_by"] == 2
        # recall works on restored DB
        hits = ks2.recall(_vec(9), top_k=1, query_text="persistent fact")
        assert len(hits) == 1
    finally:
        ks2.close()


def test_clear_all(store):
    store.insert_fact("Fact one", "chat", 1.0, 0, 0.5, _vec(1))
    store.insert_fact("Fact two", "chat", 2.0, 1, 0.5, _vec(2))
    store.clear_all()
    assert store.total_count() == 0
    assert store.active_count() == 0
    assert store.recall(_vec(1), top_k=3, query_text="fact") == []
    assert store.get_audit_trail() == []


# --------------------------------------------------------------------------- #
# Provenance (adaptation/web-grounded-learning)
# --------------------------------------------------------------------------- #
import sqlite3

import pytest
import torch

from src.model.knowledge_store import ORIGINS, KnowledgeStore, validate_origin


def _vec(seed: int = 0, dim: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=generator)


def test_an_existing_database_gains_the_provenance_columns_and_keeps_its_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT,
            timestamp REAL,
            step INTEGER,
            novelty REAL,
            superseded INTEGER DEFAULT 0,
            valid_until REAL,
            superseded_by INTEGER
        );
        INSERT INTO facts(text, source, timestamp, step, novelty)
        VALUES ('Legacy row about radar.', 'user_text', 1.0, 0, 0.5);
        """
    )
    conn.commit()
    conn.close()

    store = KnowledgeStore(str(db_path), dim=8)
    columns = {row[1] for row in store.conn.execute("PRAGMA table_info(facts)")}
    assert {"origin", "source_url", "source_query"} <= columns
    fact = store.get_fact(1)
    assert fact["text"] == "Legacy row about radar."
    assert fact["origin"] == "user_text"
    assert fact["source_url"] is None
    store.close()


def test_origin_vocabulary_is_closed_and_an_unknown_origin_is_rejected():
    assert ORIGINS == ("user_text", "chat", "continuous_stream", "web")
    with pytest.raises(ValueError) as excinfo:
        validate_origin("scraped")
    assert "scraped" in str(excinfo.value)

    store = KnowledgeStore(":memory:", dim=8)
    with pytest.raises(ValueError) as excinfo:
        store.insert_fact("x", "user_text", 1.0, 0, 0.5, _vec(), origin="scraped")
    assert "scraped" in str(excinfo.value)
    store.close()


def test_a_web_fact_reports_url_query_and_retrieval_timestamp():
    store = KnowledgeStore(":memory:", dim=8)
    fact_id = store.insert_fact(
        "Radar uses radio waves.", "web", 1234.5, 3, 0.8, _vec(1),
        origin="web", source_url="https://example.test/radar",
        source_query="what is radar",
    )
    fact = store.get_fact(fact_id)
    assert fact["origin"] == "web"
    assert fact["source_url"] == "https://example.test/radar"
    assert fact["source_query"] == "what is radar"
    assert fact["retrieved_at"] == 1234.5

    taught_id = store.insert_fact("Lidar uses lasers.", "user_text", 2.0, 1, 0.5, _vec(2))
    taught = store.get_fact(taught_id)
    assert taught["origin"] == "user_text"
    assert taught["source_url"] is None
    store.close()


def test_recall_returns_the_origin_and_url_of_each_hit():
    store = KnowledgeStore(":memory:", dim=8)
    store.insert_fact(
        "Radar uses radio waves.", "web", 10.0, 0, 0.9, _vec(3),
        origin="web", source_url="https://example.test/radar", source_query="radar",
    )
    store.insert_fact("Lidar uses lasers.", "user_text", 11.0, 1, 0.5, _vec(4))
    hits = store.recall(_vec(3), top_k=5, query_text="radar radio waves")
    by_text = {hit["text"]: hit for hit in hits}
    assert by_text["Radar uses radio waves."]["origin"] == "web"
    assert by_text["Radar uses radio waves."]["source_url"] == "https://example.test/radar"
    assert by_text["Lidar uses lasers."]["origin"] == "user_text"
    assert by_text["Lidar uses lasers."]["source_url"] is None
    store.close()
