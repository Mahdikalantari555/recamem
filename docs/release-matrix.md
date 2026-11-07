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
| offline | **passing** | `2026-11-07-offline-closure-2.json` | full `pytest tests -q` plus both gates and the remote-sensing E2E; exact counts live in the record, not in prose |
| benchmark | **passing** | `2026-11-07-workload-benchmarks.json`, `2026-11-07-continual-learning.json` | every workload below, re-measured on the declared 2 vCPU runner with mock embeddings |
| concurrency-migration | **passing** | `2026-10-21-concurrency-migration.json` | profile locks, stream ownership, migrations (not re-run on 2026-11-07) |
| grounding | **blocked** | `2026-11-07-grounding-blocked.json` | needs a trained `RTU_CHECKPOINT`; generation quality cannot be asserted from random weights |
| real-model | **blocked** | `2026-11-07-real-model-blocked.json` | needs ONNX MiniLM and a named GGUF plus `llama-cpp-python`; none exists in this environment |

`scripts/check_release_evidence.py --require-complete` therefore exits 1 by
design. That is the gate working, not a bug: two tiers have no passing
evidence, and the five `edge-slm-optimization` rows remain `partial`/`blocked`.

## Performance targets

Re-measured 2026-11-07 by `PYTHONPATH=. RECAMEM_ANCHOR=mock python scripts/bench_workloads.py`
(tier `benchmark`, commit `725f3b8`, 2 vCPU Xeon @ 2.60 GHz, 1 torch thread, mock
embeddings, exact-fallback index, `hnswlib` absent). Evidence:
`reports/evidence/2026-11-07-workload-benchmarks.json`. The continual-learning
row comes from `scripts/bench_continual_learning.py`
(`reports/evidence/2026-11-07-continual-learning.json`).

| Target | Required | Measured 2026-11-07 | Result |
| --- | --- | --- | --- |
| Retrieval p95, 10k facts | < 5 ms | 2.95 ms | met |
| Retrieval p95, 50k facts | < 5 ms | 9.02 ms | **miss** |
| Retrieval p95, 100k facts | < 5 ms | 16.34 ms | **miss** — `hnswlib` cannot be built here: no `Python.h` and no `python3-dev`; the exact fallback is what was measured |
| Lexical p95, 100k facts | < 5 ms | 14.66 ms | **miss** — denser scan dominates the fused call; FTS-first candidate generation is not implemented |
| recall@5 vs exact MVM, 20k | >= 0.95 | 1.0 | met |
| Matryoshka 64-D speedup, p50 | >= 6x | 7.17x | met (was 5.96x on the 2026-10-21 run; same command and hardware class) |
| Matryoshka relative quality loss | <= 5% | 0.0% | met |
| Reranker Precision@5 gain | +20% | +57.6% | met — 2026-10-21 lexical fallback scorer, not re-run on 2026-11-07 |
| Delta 20-task memory loss | < 0.10 | 0.1198 | **miss**, reproduced exactly; the delta adapter is not fixed |
| Sparse delta memory loss | < 0.10 | 0.0420 | met |
| Sleep recovery, 100 correlated patterns | >= 95 | 100 / 100 | met — 2026-10-21 run, not re-run on 2026-11-07 |
| Merge 2x10k facts | <= 5 s | 7.65 s | **miss** |
| Distill 10k -> 2k | <= 10 s | 4.78 s | met |
| Stream throughput | >= 100 msg/min | 482,015 msg/min | met |
| Surprise-gate insert reduction | >= 0.4 | 0.0 | **miss** — no insert is gated at the default threshold |
| Decision head p50, K=5/10/20 | — | 0.045 ms at every K | published, no target asserted |
| Catalog recommendation p95 | routing overhead only | 0.101 ms | met |
| TTC search p95 | within the declared budget | 0.099 ms, 6/8 candidates, 48/512 tokens | met |
| Adapter instantiation p50 | < 10 ms | 6.53 ms | met |

Every **miss** above is a published `partial` outcome: the target is retained,
the measurement is recorded with its provenance, and no threshold was lowered.
Closing them needs either an implementation change with a before/after
measurement (lexical path, merge, surprise gate, delta adapter, ANN index) or a
reviewed requirement revision that keeps the old target in the record.

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

## Open targets (2026-11-07)

Every row marked **miss** above is still an open measured target. They are tracked as rows in `reports/completion-gaps.json` and must be met or re-baselined through an approved requirement revision; no threshold has been lowered and no case removed. `edge-slm-optimization` rows 1.4, 2.5 and 5.2 stay `partial`; rows 4.3 and 4.4 stay `blocked` until a matched-budget comparison exists.
