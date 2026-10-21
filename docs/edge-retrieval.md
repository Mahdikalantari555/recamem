# Edge retrieval: optional dependencies, fallbacks and versioning

This page documents the optional pieces added for `edge-slm-optimization`:
the dense index, cross-encoder reranking and the Matryoshka dual index.

## Optional dependencies

| Feature | Package | Behavior without it |
| --- | --- | --- |
| `HNSWIndex` | `hnswlib` | `mode == "exact-fallback"`; the index still answers exactly via matrix-vector search. |
| `CrossEncoderReranker` | `sentence-transformers` | `scorer == "fallback-lexical"`; no silent "cross-encoder" label. |
| Real embeddings | ONNX MiniLM weights | `OnnxMiniLM.is_mock` is `True` and every numeric result is mock-embedding evidence. |

`build_dense_index("auto")` prefers HNSW and records `fallback_reason`;
`build_dense_index("hnsw")` raises `IndexUnavailableError` instead of silently
downgrading when the caller demanded the approximate index.

```python
from src.model.hybrid_retrieval import build_dense_index

index = build_dense_index("auto", dim=384)
index.mode            # "hnsw" or "exact-fallback"
index.fallback_reason # populated when the fallback was used
```

## Wiring into the knowledge store

The store stays authoritative and byte-compatible by default:

```python
store.configure_retrieval(dense_index=index, reranker=reranker, matryoshka=anchor)
store.retrieval_status()   # index kind, mode, reranker scorer, drift
store.last_retrieval_trace # candidates, dedupe, index errors, pre/post ranks
```

`configure_retrieval()` with no arguments restores the exact matrix-vector
path. Recall result keys are unchanged; reranking only adds
`rerank_score`, `pre_rerank_rank` and `post_rerank_rank`. If the index raises,
the store falls back to exact search and records `dense_index_error` in
`last_retrieval_trace` rather than returning fewer results.

Superseded and deleted facts are filtered before scoring, and the index is
rebuilt atomically when the fact set's id/len signature changes.

## Persistence and migration

* `INDEX_FORMAT_VERSION = 1`. `DenseIndex.save()` writes `<path>.safetensors`
  plus `<path>.json` metadata and rejects a newer metadata version.
* A corrupted index file is ignored on load and rebuilt from the store; it does
  not abort recall (`test_hnsw_persistence_and_corruption_fallback`).
* `MatryoshkaAnchor` persists its projection plus a `projection_checksum`.
  A different rotation is rejected with `EmbeddingVersionError` — embeddings are
  projected, never silently truncated.
* `DualIndex.save()/load()` store both stages under one versioned artifact;
  `DualIndex.migrate(ids, full_vectors)` re-embeds a corpus into both stages and
  reports what changed. Rollback is loading the previous artifact.

## Profile flags

`gguf_config` accepts `dense_index`, `reranker` and `matryoshka` settings; a
profile that pins nothing keeps exact search. Recommended models from
`EdgeSLMRegistry.recommend()` map to a `GgufBackend` configuration through
`ModelEntry.gguf_config()` without downloading or loading weights.

## Measured numbers

`reports/evidence/hybrid-retrieval-100k.json` records the 100k-fact, 384-D run
on this machine (`scripts/bench_hybrid_retrieval.py`). Read the `mode` field
first: with `hnswlib` absent the dense stage is the exact fallback, so the
latency figures are brute-force timings, not HNSW timings. Targets that were
missed are recorded as misses in `reports/completion-evidence.json`.
