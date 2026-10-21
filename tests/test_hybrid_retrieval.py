"""Hybrid retrieval: exact/HNSW candidates, reranking and Matryoshka paths."""
import json
import time

import pytest
import torch

from src.model.hybrid_retrieval import (
    CrossEncoderReranker,
    DualIndex,
    EmbeddingVersionError,
    ExactDenseIndex,
    HNSWIndex,
    IndexUnavailableError,
    MatryoshkaAnchor,
    build_dense_index,
)
from src.model.knowledge_store import KnowledgeStore


def _matrix(n=50, dim=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1)


# ---------------------------------------------------------------------------
# Dense indexes
# ---------------------------------------------------------------------------
def test_exact_index_matches_matrix_vector_search():
    vectors = _matrix(40, 32)
    index = ExactDenseIndex(dim=32)
    index.rebuild(list(range(40)), vectors)
    query = vectors[7]
    ids, scores = index.search(query, k=5)
    exact = torch.topk(vectors @ query, 5)
    assert ids == [int(i) for i in exact.indices.tolist()]
    assert scores[0] == pytest.approx(float(exact.values[0]), abs=1e-6)


def test_exact_index_add_update_delete_and_rebuild():
    index = ExactDenseIndex(dim=8)
    base = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    index.add([1, 2, 3], base)
    assert len(index) == 3
    replacement = torch.nn.functional.normalize(torch.randn(8), dim=0)
    index.update([2], replacement.unsqueeze(0))
    ids, _ = index.search(replacement, k=1)
    assert ids == [2]
    assert index.delete([1]) == 1
    assert len(index) == 2
    assert 1 not in index.ids
    # Re-adding an existing id updates instead of duplicating.
    index.add([2], replacement.unsqueeze(0))
    assert index.ids.count(2) == 1
    index.rebuild([9, 10], torch.nn.functional.normalize(torch.randn(2, 8), dim=-1))
    assert index.ids == [9, 10]


def test_hnsw_reports_exact_fallback_when_dependency_missing():
    index = HNSWIndex(dim=16, allow_fallback=True)
    vectors = _matrix(10, 16)
    index.rebuild(list(range(10)), vectors)
    if index.available:
        pytest.skip("hnswlib is installed; fallback path not exercised")
    assert index.mode == "exact-fallback"
    assert "hnswlib" in (index.fallback_reason or "")
    ids, scores = index.search(vectors[3], k=3)
    baseline = ExactDenseIndex(16)
    baseline.rebuild(list(range(10)), vectors)
    exact_ids, exact_scores = baseline.search(vectors[3], k=3)
    assert ids == exact_ids
    assert scores == pytest.approx(exact_scores, abs=1e-6)
    assert index.stats()["available"] is False


def test_hnsw_explicit_requirement_fails_without_backend():
    index = HNSWIndex(dim=16, allow_fallback=False)
    if index.available:
        pytest.skip("hnswlib is installed")
    with pytest.raises(IndexUnavailableError):
        build_dense_index("hnsw", dim=16, allow_fallback=False)


def test_hnsw_persistence_and_corruption_fallback(tmp_path):
    index = HNSWIndex(dim=16, allow_fallback=True)
    vectors = _matrix(12, 16)
    index.rebuild(list(range(12)), vectors)
    path = str(tmp_path / "index")
    info = index.persist(path)
    assert info["count"] == 12

    fresh = HNSWIndex(dim=16, allow_fallback=True)
    loaded = fresh.load(path)
    assert loaded["count"] == 12
    if not fresh.available:
        assert fresh.mode == "exact-fallback"
    assert fresh.search(vectors[2], k=2)[0][0] == 2

    # Corrupt the vector payload: the fallback must report it instead of
    # returning wrong candidates.
    (tmp_path / "index.safetensors").write_bytes(b"not a safetensors file")
    recovered = HNSWIndex(dim=16, allow_fallback=True)
    result = recovered.load(path)
    if not recovered.available:
        assert result.get("recovered") is True
        assert "corrupt index" in (recovered.fallback_reason or "")
    else:
        assert "corrupt index" in (recovered.fallback_reason or result.get("kind", ""))


def test_build_dense_index_kinds():
    assert isinstance(build_dense_index("exact", dim=8), ExactDenseIndex)
    auto = build_dense_index("auto", dim=8)
    assert auto.dim == 8
    with pytest.raises(ValueError):
        build_dense_index("bogus", dim=8)


# ---------------------------------------------------------------------------
# KnowledgeStore integration
# ---------------------------------------------------------------------------
def _populated_store(tmp_path, count=30, dim=32):
    store = KnowledgeStore(str(tmp_path / "kb.db"), dim=dim)
    generator = torch.Generator().manual_seed(3)
    vectors = torch.nn.functional.normalize(torch.randn(count, dim, generator=generator), dim=-1)
    for i in range(count):
        store.insert_fact(f"fact number {i} about sensors", "test", time.time(), i, 0.5, vectors[i])
    return store, vectors


@pytest.mark.parametrize("kind", ["exact", "hnsw"])
def test_index_integration_preserves_recall_contract(tmp_path, kind):
    baseline, vectors = _populated_store(tmp_path, count=30)
    query = vectors[5]
    reference = baseline.recall(query, top_k=5, query_text="sensors", strategy="hybrid_rrf")

    store = KnowledgeStore(str(tmp_path / f"{kind}.db"), dim=32)
    g = torch.Generator().manual_seed(3)
    vecs = torch.nn.functional.normalize(torch.randn(30, 32, generator=g), dim=-1)
    for i in range(30):
        store.insert_fact(f"fact number {i} about sensors", "test", time.time(), i, 0.5, vecs[i])
    index = build_dense_index(kind, dim=32)
    store.configure_retrieval(dense_index=index)
    results = store.recall(vecs[5], top_k=5, query_text="sensors", strategy="hybrid_rrf")
    reference_ids = [r["id"] for r in reference]
    result_ids = [r["id"] for r in results]
    if kind == "exact":
        # An exact index is the reference: identical ids and order.
        assert result_ids == reference_ids
    else:
        # An approximate index may reorder near-ties, so the contract is the
        # published recall@5 target (>= 0.95) rather than identity with the
        # exact ranking. Recall is measured against the exact baseline.
        overlap = len(set(result_ids) & set(reference_ids))
        recall = overlap / max(1, len(reference_ids))
        assert recall >= 0.95, (
            f"{kind} recall@5={recall:.2f} against the exact baseline "
            f"(got {result_ids}, expected {reference_ids})"
        )
    assert set(results[0]) >= {
        "id", "text", "superseded", "fused_score", "lexical_score",
        "dense_score", "lexical_rank", "dense_rank", "dense_similarity", "similarity",
    }
    assert store.retrieval_status()["dense_index"]["kind"] == kind


def test_index_excludes_superseded_facts(tmp_path):
    store, vectors = _populated_store(tmp_path, count=10)
    store.configure_retrieval(dense_index=ExactDenseIndex(32))
    store.supersede(3, 4)
    results = store.recall(vectors[3], top_k=10, query_text="", strategy="hybrid_rrf")
    assert all(r["id"] != 3 for r in results)
    assert all(r["superseded"] is False for r in results)


def test_index_error_falls_back_without_losing_results(tmp_path):
    store, vectors = _populated_store(tmp_path, count=8)

    class Broken:
        kind = "broken"
        mode = "broken"
        ids = []

        def rebuild(self, ids, vecs):
            raise RuntimeError("boom")

        def search(self, q, k):
            raise RuntimeError("boom")

    store.configure_retrieval(dense_index=Broken())
    results = store.recall(vectors[1], top_k=3, query_text="sensors")
    assert results
    assert "dense_index_error" in (store.last_retrieval_trace or {})


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
def _result(fact_id, text, score):
    return {"id": fact_id, "text": text, "fused_score": score}


def test_reranker_scores_only_bounded_candidates_and_reports_provenance():
    reranker = CrossEncoderReranker(model_name=None, limit=2)  # lexical fallback
    results = [_result(1, "unrelated text", 0.9), _result(2, "berlin capital", 0.5),
               _result(3, "berlin city", 0.4), _result(4, "other", 0.3)]
    trace = reranker.rerank("berlin", results)
    assert trace["scored"] == 2
    assert trace["scorer"] == "fallback-lexical"
    assert trace["results"][0]["id"] in (2, 3)  # lexical overlap wins
    assert trace["results"][0]["pre_rerank_rank"] is not None
    assert trace["results"][0]["post_rerank_rank"] == 1
    assert trace["results"][-1]["pre_rerank_rank"] is None


def test_reranker_without_fallback_rejects_missing_model():
    with pytest.raises(IndexUnavailableError):
        CrossEncoderReranker(model_name="definitely/not-a-local-model",
                             allow_fallback=False)


def test_reranker_integration_adds_trace_fields(tmp_path):
    store, vectors = _populated_store(tmp_path, count=10)
    store.configure_retrieval(reranker=CrossEncoderReranker(model_name=None, limit=5))
    results = store.recall(vectors[2], top_k=5, query_text="sensor fact")
    assert results
    assert all("rerank_score" in r for r in results)
    assert store.last_retrieval_trace["reranker"] == "fallback-lexical"


# ---------------------------------------------------------------------------
# Matryoshka
# ---------------------------------------------------------------------------
def test_matryoshka_projects_instead_of_truncating():
    anchor = MatryoshkaAnchor(fast_dim=64, full_dim=384)
    full = torch.nn.functional.normalize(torch.randn(384), dim=0)
    fast = anchor.project(full)
    assert fast.shape == (64,)
    assert not torch.allclose(fast, full[:64], atol=1e-3)
    # Orthonormal rows preserve norms of the projected signal.
    gram = anchor.projection @ anchor.projection.T
    assert torch.allclose(gram, torch.eye(64), atol=1e-4)
    assert anchor.metadata()["fast_dim"] == 64


def test_matryoshka_rejects_incompatible_versions():
    anchor = MatryoshkaAnchor(fast_dim=64, full_dim=384, version="v1")
    with pytest.raises(EmbeddingVersionError):
        anchor.validate({"version": "v2", "full_dim": 384,
                         "projection_checksum": anchor.projection_checksum})
    with pytest.raises(EmbeddingVersionError):
        anchor.validate({"version": "v1", "full_dim": 768})
    with pytest.raises(EmbeddingVersionError):
        anchor.validate({"version": "v1", "full_dim": 384,
                         "projection_checksum": "deadbeef"})
    anchor.validate(anchor.metadata())


def test_matryoshka_save_load_round_trip(tmp_path):
    anchor = MatryoshkaAnchor(fast_dim=32, full_dim=128)
    path = tmp_path / "matryoshka.safetensors"
    anchor.save(str(path))
    loaded = MatryoshkaAnchor.load(str(path))
    assert loaded.version == anchor.version
    assert torch.allclose(loaded.projection, anchor.projection)
    assert json.loads((tmp_path / "matryoshka.safetensors.json").read_text())["fast_dim"] == 32


def test_dual_index_fast_full_and_confirmation():
    anchor = MatryoshkaAnchor(fast_dim=16, full_dim=64)
    generator = torch.Generator().manual_seed(5)
    vectors = torch.nn.functional.normalize(torch.randn(40, 64, generator=generator), dim=-1)
    dual = DualIndex(anchor)
    dual.rebuild(list(range(40)), vectors)
    fast = dual.search(vectors[3], k=3, stage="fast")
    full = dual.search(vectors[3], k=3, stage="full")
    confirmed = dual.search(vectors[3], k=3, stage="fast_then_full", confirm=20)
    assert fast["ids"][0] == 3
    assert full["ids"][0] == 3
    assert confirmed["ids"][0] == 3
    assert confirmed["stage"] == "fast_then_full"
    assert dual.delete([3]) == 1
    assert 3 not in dual.ids


def test_dual_index_migration_and_version_guard(tmp_path):
    anchor = MatryoshkaAnchor(fast_dim=16, full_dim=64, version="v1")
    generator = torch.Generator().manual_seed(6)
    vectors = torch.nn.functional.normalize(torch.randn(10, 64, generator=generator), dim=-1)
    dual = DualIndex(anchor)
    dual.rebuild(list(range(10)), vectors)
    path = tmp_path / "dual.idx"
    dual.save(str(path))
    loaded = DualIndex.load(str(path), anchor)
    assert loaded.ids == dual.ids

    other = MatryoshkaAnchor(fast_dim=16, full_dim=64, version="v2")
    with pytest.raises(EmbeddingVersionError):
        DualIndex.load(str(path), other)

    migrated = dual.migrate(list(range(10)), vectors)
    assert migrated == {"status": "migrated", "version": "v1", "count": 10}
