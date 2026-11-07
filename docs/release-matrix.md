# Release matrix (2026-11-07)

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
| benchmark | **passing** | `2026-11-07-after-optimizations-workload.json`, `2026-11-07-continual-learning.json`, `2026-11-07-delta-erase-scale-sweep.json` | every workload below, re-measured on the declared 2 vCPU runner with mock embeddings; the pre-optimisation run is kept as `2026-11-07-workload-benchmarks.json` |
| concurrency-migration | **passing** | `2026-10-21-concurrency-migration.json` | profile locks, stream ownership, migrations (not re-run on 2026-11-07) |
| grounding | **blocked** | `2026-11-07-grounding-blocked.json` | needs a trained `RTU_CHECKPOINT`; generation quality cannot be asserted from random weights |
| real-model | **blocked** | `2026-11-07-real-model-blocked.json` | needs ONNX MiniLM and a named GGUF plus `llama-cpp-python`; none exists in this environment |

`scripts/check_release_evidence.py --require-complete` therefore exits 1 by
design. That is the gate working, not a bug: two tiers have no passing
evidence, and the five `edge-slm-optimization` rows remain `partial`/`blocked`.

## Performance targets

Two runs on the same runner, same command, mock embeddings: the **before** run
(commit `725f3b8`, evidence `reports/evidence/2026-11-07-workload-benchmarks.json`)
and the **after** run (commit `0ff833f`, evidence
`reports/evidence/2026-11-07-after-optimizations-workload.json`). Both come from
`PYTHONPATH=. RECAMEM_ANCHOR=mock python scripts/bench_workloads.py`
(tier `benchmark`, 2 vCPU Xeon @ 2.60 GHz, 1 torch thread, exact-fallback index,
`hnswlib` absent). The continual-learning rows come from
`scripts/bench_continual_learning.py`
(`reports/evidence/2026-11-07-continual-learning.json` before,
`reports/evidence/2026-11-07-delta-erase-scale-sweep.json` after).

| Target | Required | Before | After | Result |
| --- | --- | --- | --- | --- |
| Retrieval p95, 10k facts | < 5 ms | 2.95 ms | 2.20 ms | met |
| Retrieval p95, 50k facts | < 5 ms | 9.02 ms | 11.31 ms | **miss** — p50 5.26 ms; the five repeats include one 11.31 ms outlier, the array is published |
| Retrieval p95, 100k facts | < 5 ms | 16.34 ms | 11.00 ms | **miss** — 33% faster with the BLAS `sgemv` dense path, still over target; `hnswlib` cannot be built here (no `Python.h`), so the exact fallback is what is measured |
| Lexical p95, 100k facts | < 5 ms | 14.66 ms | 137.12 ms | **miss** — the old number was not a lexical measurement: `NDVI-1234` matched nothing because the FTS5 document tokenizer splits the code (see `lexical_ablation`). With the channel live and FTS-first off, the fused OR-expression costs ~130 ms |
| Lexical p95, 100k facts (FTS-first, opt-in) | < 5 ms | — | 1.37 ms | met on the opt-in path only — `configure_retrieval(fts_first=True)`; exact scan of the same query is 143.59 ms p95 |
| FTS-first top-5 agreement with the exact scan | — | — | 0.2 | published, no target asserted: the candidate set is not equivalent to the full scan, which is why FTS-first stays off by default (gating on the rare term drops semantically relevant facts; four hybrid tests assert that behaviour) |
| Two-stage scan p95 (64-D candidates + full rerank), 50k / 100k | < 5 ms | — | 5.82 / 15.75 ms | not shipped — recall@5 against the exact scan is 0.20, so the pruned stage drops 4 of the 5 exact hits |
| recall@5 vs exact MVM, 20k | >= 0.95 | 1.0 | 1.0 | met |
| ANN index (numpy IVF probe), 50k / 100k | p95 < 5 ms at recall@5 >= 0.95 | — | recall@5 0.445 / 0.450 at nprobe=32 (p95 3.41 / 7.78 ms) | **miss** — not shipped: uniformly random 384-D vectors give a coarse quantizer nothing to cluster, so no probe budget reaches the recall contract; `hnswlib` still cannot be built here (`reports/evidence/2026-11-07-ann-index-probe.json`) |
| Matryoshka 64-D speedup, p50 | >= 6x | 7.17x | 7.39x | met |
| Matryoshka relative quality loss | <= 5% | 0.0% | 0.0% | met |
| Reranker Precision@5 gain | +20% | +57.6% | not re-run | met — 2026-10-21 lexical fallback scorer |
| Delta 20-task memory loss | < 0.10 | 0.1198 | 0.1211 (mean of four seeds, max 0.1249) | **miss** — the shipped default keeps the delta rule's exact-overwrite guarantee; the opt-in rows below reach the target |
| Delta 20-task memory loss (`erase_scale=0.25`) | < 0.10 | — | 0.0920 (mean of four seeds, max 0.0984) | met as an opt-in configuration — see the trade below |
| Delta 20-task memory loss (`erase_scale=0.15`) | < 0.10 | — | 0.0730 (mean of four seeds, max 0.0785) | met as an opt-in configuration |
| Delta association recall, pair top-1 | — | 0.125 | 0.354 at `erase_scale=0.25` | published: the retention gain and the recall gain come from the same change; chance is 0.025 |
| Sparse delta memory loss | < 0.10 | 0.0420 | 0.0318 at `erase_scale=0.25` | met |
| Sleep recovery, 100 correlated patterns | >= 95 | 100 / 100 | not re-run | met — 2026-10-21 run |
| Merge 2x10k facts | <= 5 s | 7.65 s | 1.51 s | met — batched dedupe in `bulk_insert_facts` (one GEMM instead of 10k `torch.mv` calls); merged facts are also written to `facts_fts` now, which they were not before |
| Distill 10k -> 2k | <= 10 s | 4.78 s | 4.80 s | met |
| Stream throughput | >= 100 msg/min | 482,015 msg/min | 235,910 msg/min | met |
| Surprise-gate insert reduction (redundant stream) | >= 0.4 | 0.0 | 0.50 | met on the documented fixture: each fact is stated twice, 200 of 400 propositions are refused |
| Surprise-gate insert reduction (unique stream) | — | 0.0 | 0.0 | published, correct by construction: no proposition is redundant, so nothing is refused |
| Decision head p50, K=5/10/20 | — | 0.045 ms | 0.046 ms | published, no target asserted |
| Catalog recommendation p95 | routing overhead only | 0.101 ms | 0.094 ms | met |
| TTC search p95 | within the declared budget | 0.099 ms | 0.087 ms | met |
| Adapter instantiation p50 | < 10 ms | 6.53 ms | 6.63 ms | met |

### Why the delta row stays partial

The delta rule erases the stored component along the written key
(`S <- S - beta * k (k^T S) + beta * k v^T`). At the default `erase_scale=1.0`
a write replaces its association exactly, which is what the
`adaptation/delta-memory` requirement asks for: a re-written key returns the new
value and no longer returns the old one, and repeated writes are bounded.

`erase_scale` scales only the erase term, and the default leaves every
existing invariant untouched (`tests/test_delta_memory.py` pins the exact
overwrite, the bounded repeated write and the full-gate read-back). The sweep in
`reports/evidence/2026-11-07-delta-erase-scale-sweep.json` shows:

| `erase_scale` | memory loss (4 seeds) | pair recall |
| --- | --- | --- |
| 1.0 (default) | 0.1211 mean, 0.1249 max | 0.125 |
| 0.5 | 0.1193 mean, 0.1263 max | 0.228 |
| 0.25 | 0.0920 mean, 0.0984 max | 0.354 |
| 0.15 | 0.0730 mean, 0.0785 max | 0.413 |

Both retention and recall improve, because less of the older content is
destroyed. The cost is that a re-write keeps `1 - erase_scale` of the old
association and repeated writes converge to `v / erase_scale` instead of `v`,
so the target is met only as an explicit, documented configuration and the
default row is published as a **miss**.

Two other ways to reach the number were measured and rejected (same evidence
record): splitting the memory into more heads (`delta_heads=16`) reaches
0.0377-0.0442 but collapses pair recall to 0.02-0.045, i.e. chance, so it
improves the drift metric while destroying the association; and sweeping the
write gate `beta` in 0.6-1.0 does not move the loss at all (0.120-0.126).

## Functional contract status## Functional contract status

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
