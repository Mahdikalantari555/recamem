# OpenSpec apply run — 2026-10-21

Branch: `arena/d08b6a3d-test-model-thing-torch`
Implementation commit: `f31b00c` (measurements were taken at that commit)
Ledger: `reports/completion-evidence.json` (489 entries, schema/coverage valid)

This run implements the pending tasks of four active changes:
`delta-plastic-memory`, `autonomous-sleep-consolidation`,
`edge-slm-optimization` and the implementation portion of
`post-merge-completion`.

## What landed

| Area | Code | Tests |
| --- | --- | --- |
| Delta adapter, sparse coding, registry | `src/model/delta_memory.py`, `src/model/plastic_adapter.py` | `tests/test_delta_memory.py` (37) |
| 20-task benchmark + EWC reference | `src/model/continual_bench.py`, `scripts/bench_continual_learning.py` | `tests/test_continual_bench.py` (11) |
| Sleep, MAS, decreasing backprop, worker | `src/model/sleep_consolidation.py`, `src/model/droid.py`, `src/model/droid_manager.py`, `src/cli.py` | `tests/test_sleep_consolidation.py` (37) |
| Dense index, reranking, Matryoshka | `src/model/hybrid_retrieval.py`, `src/model/knowledge_store.py` | `tests/test_hybrid_retrieval.py` (18) |
| Edge catalog + recommendation | `src/model/edge_slm.py` | `tests/test_edge_slm.py` (14) |
| Test-time compute | `src/model/test_time_compute.py` | `tests/test_test_time_compute.py` (14) |

Full suite: **622 passed, 1 skipped**.

## Measurements (this sandbox, mock embeddings)

| Metric | Target | Measured | Status |
| --- | --- | --- | --- |
| Delta 20-task memory loss | < 0.10 | 0.1198 | **miss**, published |
| Sparse delta memory loss | < 0.10 | 0.0420 | met |
| EWC reference loss | ~0.24 | 0.1016 | **not reproduced** |
| Interference overlap (dense → sparse) | — | 1.000 → 0.034 (96.6% reduction) | met |
| Sparse code sparsity | 93.75% | 93.75% (128/2048, all 2048 codes used) | met |
| Sleep recovery, 100 correlated patterns | ≥95 | 100 / 100 (coverage 1.0, 160 steps) | met |
| Dense 100k-fact p95 | < 5 ms | 13.3 ms (exact fallback) | **miss** — hnswlib unavailable |
| recall@5 vs exact | ≥ 0.95 | 1.0 | met (exact fallback) |
| Matryoshka 64-D speedup p50 | ≥ 6x | 4.83x | **miss** |
| Matryoshka relative quality loss | ≤ 5% | 0.0% (synthetic cluster labels) | met |
| Reranker P@5 gain | +20% | +57.6% (lexical fallback scorer) | met, caveated |

`OnnxMiniLM.is_mock` is `True` in this environment, so every retrieval number
above is mock-embedding evidence. The ledger records the tier of each claim and
the misses are kept as misses — no target is asserted as met.

## Deliberately not claimed

* HNSW latency (`hnswlib` has no wheel here) — the index reports
  `mode == "exact-fallback"` and the exact timings are labelled as such.
* Cross-encoder quality — `scorer == "fallback-lexical"`; no cross-encoder
  weights were available.
* TTC versus an 8B reference (tasks 4.3/4.4, 7.10) — no local 8B weights, so the
  comparison is `blocked` rather than fabricated.
* Knowledge-store 100k lexical SLA — not re-measured on this hardware.

## Still open (unchecked)

* `post-merge-completion` §1.2/1.3 (archive/task reconciliation), §1.5 (CI job
  exists but has not been observed green on the integration branch), §3.4/3.7
  (KV prefill and local GGUF/hookable artifacts — no binding in this sandbox),
  §7.9/7.10 (benchmarks), §8.x and §9.x (historical contracts and
  whole-product acceptance).
* Sections 2.5–2.9, 3 (except 3.4/3.7) and 4.1–4.12 landed later the same day;
  see the addendum below.
* `delta-plastic-memory` 5.3 (CPU/RSS memory benchmark of the 2048-D codes).
* `edge-slm-optimization` 1.4, 2.5, 5.2, 5.3 (targets above) and 4.3/4.4.
* `post-merge-completion` 7.9 (partial: published misses) and 7.10 (blocked).

## Evidence artifacts

```
reports/evidence/continual-learning-20task.json       raw per-task benchmark
reports/evidence/sleep-recovery-100.json              raw recovery run
reports/evidence/hybrid-retrieval-100k.json           raw 100k-fact run
reports/evidence/edge-slm-catalog.json                catalog dump
reports/evidence/2026-10-21-delta-benchmark.json      tier=benchmark, passed
reports/evidence/2026-10-21-sleep-recovery.json       tier=benchmark, passed
reports/evidence/2026-10-21-retrieval-benchmark.json  tier=benchmark, failed (misses)
reports/evidence/2026-10-21-reranker-benchmark.json   tier=benchmark, passed
reports/evidence/2026-10-21-catalog-audit.json        tier=offline, passed
reports/evidence/2026-10-21-offline-suite.json        tier=offline, passed
reports/evidence/2026-10-21-concurrency-migration.json tier=concurrency-migration, passed
```

Reproduce with:

```bash
PYTHONPATH=. python scripts/bench_continual_learning.py
PYTHONPATH=. python scripts/bench_sleep_recovery.py
PYTHONPATH=. python scripts/bench_hybrid_retrieval.py
PYTHONPATH=. python scripts/check_release_evidence.py
```

## Archive (2026-10-21)

Archived with `git mv` into `openspec/changes/archive/2026-10-21-*` (the
`openspec` CLI is not installed in this environment, so the archive was done by
hand following the CLI's semantics):

* `2026-10-21-delta-plastic-memory` — delta specs synced to
  `openspec/specs/adaptation/delta-memory/spec.md` and
  `openspec/specs/adaptation/sparse-coding/spec.md`.
* `2026-10-21-autonomous-sleep-consolidation` — synced to
  `openspec/specs/memory/{sleep-consolidation,memory-aware-synapses,decreasing-backprop}/spec.md`.
* `2026-10-21-edge-slm-optimization` — synced to
  `openspec/specs/memory/hybrid-retrieval/spec.md`,
  `openspec/specs/model/edge-optimization/spec.md` and
  `openspec/specs/model/test-time-compute/spec.md`.

Each archive directory keeps its proposal, design, delta specs and ticked
tasks.md, plus an archive-status section recording the measured numbers and the
targets that were missed. `inventory.json` and `reports/completion-evidence.json`
were repointed from `openspec/changes/<change>/tasks.md` to the archived paths;
the ledger still validates (489 entries, schema/coverage valid).

`post-merge-completion` is intentionally **not** archived: sections 1.2-1.5,
3.1-3.8 and 4.1-4.12 are unimplemented, TTC-vs-8B is blocked, and its own
`verification/release-evidence` spec forbids archiving while dependency
acceptance evidence is incomplete.

## Merge status

`origin/main` was at `f4389a3`, an ancestor of this branch: the branch had no
open pull request and no conflict, so no automated merge could occur. The fix is
to publish the branch to `main` explicitly (fast-forward merge, no new merge
commit) rather than resolving a nonexistent conflict.

## Addendum (2026-10-21, second run): post-merge §2.7–2.9, §3 and §4

Commits on top of this report's first run:

```
3bbadbc  Injection becomes a backend operation with declared capabilities
c6a682f  Check the effective requirement set across core/optional includes
e1d8523  Retrieval strategies, labelled health, persistent snapshots, power-law decay
0bcc5f2  Transactional profiles, merge refusals, measured consolidation, single-owner workers
```

Full offline suite: **750 passed, 1 skipped**
(`PYTHONPATH=. python -m pytest tests -q`, ~31 s). Evidence:
`reports/evidence/2026-10-21-offline-suite-post-merge.json`.

| Area | Code | Tests |
| --- | --- | --- |
| Backend capability + injection contracts | `src/model/backend_contracts.py`, `memory_injection.py`, `torch_decoder_backend.py` | `tests/test_backend_contracts.py` (34), `test_injection_engine.py` (10), `test_torch_decoder_backend.py` (26), `test_gguf_capabilities.py` (8) |
| Core vs optional install | `requirements-{core,optional}.txt`, `docs/optional-dependencies.md` | `tests/test_optional_dependencies.py` (3) |
| Adaptive retrieval (4.1–4.3) | `src/model/memory_lifecycle.py`, `knowledge_store.py`, `droid.py` | `tests/test_adaptive_retrieval.py` (18) |
| Snapshots, rollback, decay (4.4/4.5/4.7) | `src/model/memory_lifecycle.py`, `knowledge_store.py`, `app.py` | `tests/test_memory_snapshots.py` (20) |
| Health metrics + TTT budgets (4.9/4.10) | `knowledge_store.py`, `droid.py`, `confidence.py` | `tests/test_adaptive_retrieval.py` |
| Profiles, merge, consolidation, workers (2.7–2.9, 4.6/4.8/4.11) | `droid.py`, `droid_manager.py`, `memory_lifecycle.py` | `tests/test_lifecycle_completion.py` (18) |
| Console lifecycle (4.12) | `app.py` | `tests/test_console_lifecycle.py` (6) |
| Migration fixtures (1.4) | `tests/fixtures/profile_checksums.json` | `tests/test_migration_fixtures.py` (6) |
| Offline/CI docs (1.5, partial) | `docs/testing.md`, `docs/lifecycle.md`, `.github/workflows/offline-contracts.yml`, `scripts/validate_openspec.py` | structure validator + ledger guard |

Concrete behaviour changes worth knowing about:

* `KnowledgeStore.recall` now *executes* the selected strategy and records
  `requested_strategy`/`executed_strategy`/`retrieval_store`; `semantic_primary`
  works even when the episodic table is empty.
* `RetrievalPolicy` v2 makes `ambiguous` reachable for empty/underspecified
  queries and never lets `?` promote a query to `specific_fact`.
* Snapshots are complete directories (SQLite backup incl. WAL + safetensors +
  adapter metadata + policy + lifecycle config) with a checksummed manifest
  written last; restore validates before activating and rollback snapshots the
  live state first.
* `get_memory_health()["retrieval_quality"]` is `None` rather than `0.5`/`0.7`
  until labelled retrieval judgements exist; `retrieval_health()` reports the
  sample count and provenance.
* One `_consume_ttt_step()` gate bounds every plastic update path.
* The console delegates stream control to the engine instead of owning a
  second worker.

Deliberately not claimed in this addendum:

* The CI workflow has **not** been observed running (no Actions run in this
  sandbox), so task 1.5 stays unchecked.
* `streamlit` is not installed, so no `AppTest` run; console tests are
  structural plus engine-level.
* KV prefill (3.4) and local GGUF/hookable artifacts (3.7) remain blocked for
  lack of a binding/model file; the capability matrix reports them as
  unsupported rather than faking success.

Ledger: 489 entries, 70 `verified` (12 new rows flipped from `blocked` with
implementation paths, existing test node IDs and the offline evidence record);
`scripts/check_release_evidence.py` still exits 0 in schema mode and
`--require-complete` still exits 1, as intended.

## Addendum (2026-10-22, third run): reconciliation, workloads and closure

Branch: `arena/a4e77c3f-test-model-thing-torch`
Code commit: `2c11003` (suite and benchmarks were run against it)

### Suite and structure

```
PYTHONPATH=. python -m pytest tests -q   ->  917 passed, 1 skipped (55 s)
PYTHONPATH=. python scripts/validate_openspec.py -> structure valid
```

The single skip is the model-gated grounded-answer assertion in
`tests/test_droid_rtu_fusion.py`: `RTU_CHECKPOINT` is unset and no weights exist
in this sandbox. It is reported as a skip, never as a pass.

### Ledger reconciliation

`scripts/reconcile_evidence.py --apply` rewrote all 424 inventory rows from the
current tree instead of from prose similarity:

```
424 rows: superseded=11, verified=413
reports/completion-evidence.json: 489 entries, schema/coverage valid
```

Every `verified` row now points at a test node that `scripts/check_release_evidence.py`
confirms exists (the guard rejects invented ids), at an implementation file on
disk, and at an evidence record with a full commit hash. The 11 superseded rows
each name the replacement requirement that the main specs carry.

### Workloads (`reports/evidence/2026-10-21-workload-benchmarks.json`)

| Metric | Target | Measured | Status |
| --- | --- | --- | --- |
| Retrieval p95, 10k facts | < 5 ms | 2.38 ms | met |
| Retrieval p95, 50k facts | < 5 ms | 8.87 ms | **miss** |
| Retrieval p95, 100k facts | < 5 ms | 17.46 ms | **miss** (exact fallback) |
| Lexical p95, 100k facts | < 5 ms | 16.89 ms | **miss** |
| Dense index recall@5, 20k | >= 0.95 | 1.0 | met |
| Matryoshka 64-D speedup | >= 6x | 5.96x | **miss on this runner** (8.08x previously) |
| Matryoshka quality loss | <= 5% | 0.0% | met |
| Catalog recommendation p95 | routing overhead | 0.09 ms | met |
| TTC search p95 | within budget | 0.08 ms, 6/8 candidates | met |
| Merge 2x10k facts | <= 5 s | 7.4 s | **miss** (was 286 s / 0 writes before the JOIN rewrite) |
| Distill 10k -> 2k | <= 10 s | 3.93 s | met |
| Stream, 100 messages | >= 100 msg/min | 558,916 msg/min | met |
| Decision head p50, K=5/10/20 | — | 0.034 ms at every K | published, no target |
| Surprise-gate insert reduction | >= 0.4 | 0.0 | **miss** |

Misses are recorded as misses with the configuration that produced them. No
threshold was loosened and no measurement was promoted to a claim.

### What still blocks a completion declaration

`scripts/check_release_evidence.py --require-complete` still exits 1:

* `edge-slm-optimization#1.4` and `#5.2` — measured, targets missed (above);
* `edge-slm-optimization#4.3`/`#4.4` — matched-budget 1B+TTC versus an
  identified 8B reference; no weights and no inference budget here;
* `real-model` and `grounding` tiers — need a GGUF checkpoint, real ONNX
  weights, or a trained `RTU_CHECKPOINT`, none of which exist in the repo;
* `post-merge-completion` 3.4, 3.7, 7.9, 7.10, 8.1, 9.2, 9.4, 9.5 stay
  unchecked for the same reasons, and 1.5 waits on this pull request's CI job.

`openspec/changes/post-merge-completion` therefore stays **active**: its own
release-evidence requirement forbids archiving on incomplete dependency
acceptance.
