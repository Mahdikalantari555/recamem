# Release matrix (2026-10-22)

Every row names the requirement or task it answers, the measurement or
command that produced the evidence, and what the result actually is. A miss is
a miss: nothing here is rounded up, and a blocked row is never described as
shipped.

Evidence records live in `reports/evidence/`; the machine-readable ledger is
`reports/completion-evidence.json` and is validated by
`scripts/check_release_evidence.py`.

## Verification tiers

| Tier | Status | Evidence | What it covers |
| --- | --- | --- | --- |
| offline | **passing** | `2026-10-21-offline-suite-reconciled.json` | 917 passed, 1 skipped, 55 s, commit `8c56b0f` lineage |
| benchmark | **passing** | `2026-10-21-workload-benchmarks.json` | every workload below, with hardware and configuration |
| concurrency-migration | **passing** | `2026-10-21-concurrency-migration.json` | profile locks, stream ownership, migrations |
| grounding | **blocked** | — | needs a trained `RTU_CHECKPOINT`; generation quality cannot be asserted from random weights |
| real-model | **blocked** | — | needs GGUF/ONNX weights and a `llama.cpp` build; neither exists in this environment |

`scripts/check_release_evidence.py --require-complete` therefore exits 1 by
design. That is the gate working, not a bug.

## Performance targets

Measured by `PYTHONPATH=. python scripts/bench_workloads.py` on 2 vCPU
(Xeon @ 2.60 GHz, 1 torch thread), mock embeddings, exact-fallback index.

| Target | Required | Measured | Result |
| --- | --- | --- | --- |
| Retrieval p95, 10k facts | < 5 ms | 2.38 ms | met |
| Retrieval p95, 50k facts | < 5 ms | 8.87 ms | **miss** |
| Retrieval p95, 100k facts | < 5 ms | 17.46 ms | **miss** — `hnswlib` is unbuildable here (no `Python.h`) |
| Lexical p95, 100k facts | < 5 ms | 16.89 ms | **miss** — dense scan dominates the fused call |
| recall@5 vs exact MVM, 20k | >= 0.95 | 1.0 | met |
| Matryoshka 64-D speedup, p50 | >= 6x | 5.96x | **miss on this runner** (8.08x on the previous one) |
| Matryoshka relative quality loss | <= 5% | 0.0% | met |
| Reranker Precision@5 gain | +20% | +57.6% | met, lexical fallback scorer (no cross-encoder weights) |
| Delta 20-task memory loss | < 0.10 | 0.1198 | **miss**, published |
| Sparse delta memory loss | < 0.10 | 0.0420 | met |
| Sleep recovery, 100 correlated patterns | >= 95 | 100 / 100 | met |
| Merge 2x10k facts | <= 5 s | 7.4 s | **miss** (286 s / 0 writes before the JOIN rewrite) |
| Distill 10k -> 2k | <= 10 s | 3.93 s | met |
| Stream throughput | >= 100 msg/min | 558,916 msg/min | met |
| Surprise-gate insert reduction | >= 0.4 | 0.0 | **miss** |
| Decision head p50, K=5/10/20 | — | 0.034 ms at every K | published, no target asserted |
| Catalog recommendation p95 | routing overhead only | 0.09 ms | met |
| TTC search p95 | within the declared budget | 0.08 ms, 6/8 candidates, 48/512 tokens | met |

## Functional contract status

| Area | Status | Evidence |
| --- | --- | --- |
| Backend capability and injection contracts | passing | `tests/test_backend_contracts.py`, `test_injection_engine.py`, `test_gguf_capabilities.py` |
| Console, feedback and lifecycle | passing | `tests/test_app_ui.py`, `test_console_lifecycle.py`, `test_memory_console_contracts.py` |
| Snapshots, rollback, merge, import/export | passing | `tests/test_memory_snapshots.py`, `test_manager_lifecycle_ops.py` |
| Adaptive retrieval, health, decay, TTT budgets | passing | `tests/test_adaptive_retrieval.py`, `test_health_metrics.py` |
| Semantic distillation and contradictions | passing | `tests/test_semantic_distillation.py` |
| Cross-feature chain, both adapters | passing | `tests/test_cross_feature_e2e.py` |
| Remote-sensing end-to-end | passing | `tests/test_e2e_remote_sensing.py` (runs the shipped script) |
| Real GGUF generation and KV prefill | **blocked** | no `llama.cpp` binding, no GGUF artifact (tasks 3.4, 3.7, 8.1) |
| Matched-budget 1B+TTC vs an 8B reference | **blocked** | no weights, no inference budget (tasks 7.10, 9.2, edge-slm 4.3/4.4) |

## Migration and rollback

* Adapter and metadata migration: `docs/delta-migration.md`, verified by
  `tests/test_migration_fixtures.py` (legacy, default and custom fixtures with
  recorded checksums in `tests/fixtures/profile_checksums.json`).
* Lifecycle semantics, snapshot layout and rollback ordering:
  `docs/lifecycle.md`, verified by `tests/test_console_lifecycle.py` and
  `tests/test_memory_snapshots.py`.
* Optional dependency matrix and the exact-fallback path:
  `docs/optional-dependencies.md`, verified by
  `tests/test_optional_dependencies.py` and
  `tests/test_hybrid_retrieval.py::test_hnsw_reports_exact_fallback_when_dependency_missing`.

## What this matrix does not claim

`openspec/changes/post-merge-completion` stays active. Its own
`verification/release-evidence` requirement says the change "SHALL remain
incomplete and SHALL NOT be archived as fully implemented" while dependency
closure is unresolved, and the following tasks remain open for artifacts that
do not exist here: 3.4, 3.7, 7.10, 8.1, 9.2, 9.4, 9.5.
