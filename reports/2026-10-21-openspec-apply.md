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

* `post-merge-completion` §1.2–1.5 (archive reconciliation), §2.5–2.9
  (adapter metadata/migration hardening beyond what delta needed), §3.1–3.8
  (real backend injection contracts; KV prefill needs a GGUF binding),
  §4.1–4.12 (retrieval policy, snapshots, merge lifecycle, health, TTT budgets,
  stream/lock unification).
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
