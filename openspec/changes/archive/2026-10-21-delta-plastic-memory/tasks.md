# Tasks

## 1. Delta adapter

- [x] 1.1 Implement `DeltaPlasticAdapter` against the existing `PlasticAdapter` interface, including projection/state initialization and dimension validation.
- [x] 1.2 Implement the per-head gated Delta Rule using the previous memory prediction `K^T S`, residual value, and sigmoid `W_beta` gate.
- [x] 1.3 Implement `erase(key, strength)` with a Householder-reflected direction, bounded attenuation, and an auditable result report.
- [x] 1.4 Add versioned state save/load and backward-compatibility tests for existing `associative_rtu` profile loading.
- [x] 1.5 Add unit tests for zero/full beta, repeated writes, changed-value overwrite, erase of a selected key, and unrelated-key preservation.

## 2. Sparse coding and registry

- [x] 2.1 Implement `SparseCoder` with a 384-D input, 2048-D latent representation, deterministic TopK 128, reconstruction, and code-usage metrics.
- [x] 2.2 Implement the sparse/delta composition and FlyModel-style configured adapter while retaining the `PlasticAdapter` API.
- [x] 2.3 Register `delta_rtu`, `sparse_delta`, and `fly_model`; test discovery, construction, unknown-name errors, and profile serialization.
- [x] 2.4 Add tests for exact code dimension, exactly 128 active entries, 93.75% expected sparsity, round-trip shape, and stable tie handling.

## 3. Droid/profile integration

- [x] 3.1 Allow new profiles to choose `delta_rtu` without changing the adapter selected by existing profile metadata.
- [x] 3.2 Route `DroidEngine` teaching, recall, and snapshots through the adapter contract and preserve the current default until profile migration is explicitly selected.
- [x] 3.3 Add migration/rollback documentation and tests for incompatible checkpoint dimensions.

## 4. Continual-learning evaluation

- [x] 4.1 Add a versioned 20-task benchmark with fixed seeds, task order, train/evaluation splits, and scoring protocol.
- [x] 4.2 Implement the EWC reference under the same budget and document how memory loss is calculated.
- [x] 4.3 Measure dense delta and sparse/delta memory loss, interference, latency, and code utilization; target memory loss <0.10 and compare with the ~0.24 EWC reference.
- [x] 4.4 Publish full benchmark configuration and per-task results; treat unverified figures as targets, not release claims.

## 5. Validation

- [x] 5.1 Run adapter, DroidEngine, checkpoint, and profile test suites.
- [x] 5.2 Run the full project test suite and verify old `associative_rtu` checkpoints still load.
- [x] 5.3 Benchmark CPU memory, update latency, retrieval quality, and interference for 384-D dense vs 2048-D TopK-128 codes.

## Archive status (2026-10-21)

Archived after implementation. Every task except `5.3` is checked with tests and
evidence recorded in `reports/completion-evidence.json` (see
`reports/2026-10-21-openspec-apply.md`).

Measured on the fixed 20-task protocol (mock embeddings, commit `f31b00c`):

| Metric | Target | Measured |
| --- | --- | --- |
| Delta mean memory loss | < 0.10 | **0.1198 (target missed)** |
| Sparse delta mean memory loss | < 0.10 | 0.0420 |
| EWC reference loss | ~0.24 | **0.1016 (reference not reproduced)** |
| Interference overlap, dense -> sparse | not specified | 1.000 -> 0.034 |
| Sparse code sparsity | 93.75% | 93.75% (128/2048) |

The <0.10 target is retained as an unmet target, not a shipped claim. Task `5.3`
(process CPU/RSS memory of the 2048-D codes) remains open and is tracked by
`openspec/changes/post-merge-completion` ledger row
`openspec/changes/delta-plastic-memory/tasks.md#5.3` (`partial`).
