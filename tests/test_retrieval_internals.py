"""Unit tests for the retrieval internals added by the 5.x performance work.

These cover the pieces that used to be implicit inside ``recall``:

- ``_fts_tokens`` / ``_lexical_candidates``: FTS-first candidate generation
  from the most selective query term, with a hard candidate budget;
- ``_dense_scores`` / ``_batch_max_similarity``: the BLAS helpers that replaced
  ``torch.mv`` in the hot path, pinned against the exact torch products;
- the duplicate-aware surprise gate in ``teach``;
- ``bulk_insert_facts`` batched dedupe, including the FTS rebuild contract.
"""

import math
import time

import pytest
import torch
import torch.nn.functional as F

from recamem.engine import DroidEngine
from recamem.episodic import KnowledgeStore, _fts_tokens


def _matrix(n=40, dim=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=generator), dim=-1)


def _unit(vec):
    return vec / vec.norm().clamp(min=1e-8)


@pytest.fixture()
def store(tmp_path):
    kb = KnowledgeStore(str(tmp_path / "kb.db"), dim=24)
    vectors = _matrix(30)
    for index, vector in enumerate(vectors):
        text = (
            f"NDVI-{1000 + index} reading at station {index} is nominal"
            if index < 5 else f"fact number {index} about sensors"
        )
        kb.insert_fact(text, "test", time.time(), index, 0.5, vector)
    yield kb
    kb.close()


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_fts_tokens_split_code_like_words_and_drop_stopwords():
    assert _fts_tokens(["NDVI-1234"]) == ["ndvi", "1234"]
    assert _fts_tokens(["the", "sensor", "of"]) == ["sensor"]
    # Duplicates collapse and order is preserved.
    assert _fts_tokens(["sensor", "SENSOR", "pump"]) == ["sensor", "pump"]


# --------------------------------------------------------------------------- #
# Dense helpers
# --------------------------------------------------------------------------- #
def test_dense_scores_matches_torch_mv(store):
    matrix = _matrix(50, 24, seed=3)
    query = _unit(torch.randn(24, generator=torch.Generator().manual_seed(4)))
    expected = torch.mv(matrix, query)
    got = store._dense_scores(matrix, query)
    assert torch.allclose(got, expected, atol=1e-6)


def test_batch_max_similarity_matches_per_row_products(store):
    matrix = _matrix(57, 24, seed=5)
    batch = _matrix(37, 24, seed=6)
    got = store._batch_max_similarity(matrix, batch)
    expected = torch.stack([torch.mv(matrix, row).max() for row in batch])
    assert torch.allclose(got, expected, atol=1e-6)


def test_batch_max_similarity_is_chunk_invariant(store):
    matrix = _matrix(31, 24, seed=7)
    batch = _matrix(20, 24, seed=8)
    assert torch.allclose(
        store._batch_max_similarity(matrix, batch, chunk=512),
        store._batch_max_similarity(matrix, batch, chunk=3),
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# FTS-first candidates
# --------------------------------------------------------------------------- #
def test_lexical_candidates_are_opt_in(store):
    assert store.fts_first is False
    assert store._lexical_candidates(["ndvi", "1234"]) == (None, None)


def test_lexical_candidates_pick_the_selective_term(store):
    store.configure_retrieval(fts_first=True)
    # "ndvi" matches every code fact; "1002" matches exactly one of them.
    ids, term = store._lexical_candidates(["ndvi", "1002"])
    assert ids, "the code-like term must produce candidates"
    assert term == '"1002"', f"expected the rare token, got {term}"
    assert len(ids) == 1


def test_lexical_candidates_do_not_narrow_on_a_corpus_wide_term(store):
    store.configure_retrieval(fts_first=True)
    # "ndvi" matches all five code facts and "fact"/"sensors" the rest, but the
    # candidate budget is 512, so this stays selective; a term matching more
    # rows than the budget must instead decline to narrow.
    store._FTS_FIRST_LIMIT = 2
    ids, term = store._lexical_candidates(["ndvi"])
    assert (ids, term) == (None, None)


def test_retrieval_status_reports_the_fts_policy(store):
    status = store.retrieval_status()
    assert status["fts_first"] is False
    store.configure_retrieval(fts_first=True)
    assert store.retrieval_status()["fts_first"] is True


def test_retrieval_status_tolerates_a_dense_index_without_metadata(store):
    from recamem.hybrid_retrieval import ExactDenseIndex

    index = ExactDenseIndex(dim=24)
    index.rebuild(
        [row[0] for row in store.conn.execute("SELECT id FROM facts").fetchall()],
        _matrix(len(store.conn.execute("SELECT id FROM facts").fetchall())),
    )
    store.configure_retrieval(dense_index=index)
    status = store.retrieval_status()
    assert status["dense_index"]["count"] == len(index)


# --------------------------------------------------------------------------- #
# Batched dedupe in bulk_insert_facts
# --------------------------------------------------------------------------- #
def _bulk_row(text, vector, step):
    return {
        "text": text, "source": "test", "timestamp": time.time(), "step": step,
        "novelty": 0.5, "vec": vector,
    }


def test_bulk_insert_facts_skips_rows_already_in_the_store(store):
    existing = store.conn.execute("SELECT text FROM facts").fetchone()[0]
    vectors, ids = store._load_dense()
    duplicate_of = vectors[0]
    result = store.bulk_insert_facts([
        _bulk_row(existing, duplicate_of, 100),
        _bulk_row("a genuinely new fact about pumps", _matrix(1, 24, seed=99)[0], 101),
    ], dedupe_threshold=0.99)
    assert result["inserted"] == 1
    assert result["duplicate"] == 1
    assert result["invalid"] == 0


def test_bulk_insert_facts_rejects_wrong_dimension_vectors(store):
    result = store.bulk_insert_facts([
        _bulk_row("bad row", torch.randn(7), 200),
    ])
    assert result == {"inserted": 0, "duplicate": 0, "invalid": 1}


def test_bulk_insert_facts_rebuilds_the_fts_index(store):
    store.bulk_insert_facts([
        _bulk_row("fresh geothermal telemetry", _matrix(1, 24, seed=11)[0], 300),
    ])
    hits = store.conn.execute(
        "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH '\"geothermal\"'"
    ).fetchone()[0]
    assert hits == 1


# --------------------------------------------------------------------------- #
# Duplicate-aware surprise gate
# --------------------------------------------------------------------------- #
def test_teach_skips_duplicate_propositions_but_keeps_new_ones(tmp_path):
    engine = DroidEngine(db_path=str(tmp_path / "gate.db"))
    paragraph = (
        "The outlet pressure is 42 bar. The outlet pressure is 42 bar. "
        "The inlet temperature is 11 celsius."
    )
    first = engine.teach(paragraph)
    assert first["duplicate_threshold"] == pytest.approx(0.95)
    # Two distinct facts survive, the restatement does not.
    assert first["episodic_absorbed"] == 2
    assert first["episodic_skipped_duplicate"] >= 1
    assert engine.knowledge.active_count() == 2

    again = engine.teach(paragraph)
    assert again["episodic_absorbed"] == 0
    assert again["episodic_skipped_duplicate"] == again["propositions"]
    assert engine.knowledge.active_count() == 2
    engine.knowledge.close()


def test_teach_keeps_facts_that_differ_only_in_a_quantity(tmp_path):
    engine = DroidEngine(db_path=str(tmp_path / "qty.db"))
    engine.teach("Pump 0 runs at site 0 on line 0.")
    result = engine.teach("Pump 0 runs at site 0 on line 10.")
    assert result["episodic_absorbed"] == 1
    assert result["episodic_skipped_duplicate"] == 0
    engine.knowledge.close()
