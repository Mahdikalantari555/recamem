# Tasks

All implementation tasks start unchecked. Baseline main is `c246043`; the preceding merge is not implementation of this proposal. `inventory.json` captures 310 archived tasks (198 unchecked), 114 current requirements and 65 dependency tasks. Unverified inventory entries are not assertions that code is absent.

Feature task references below are relative to `openspec/changes/<change>/tasks.md`; those detailed tasks and specs remain authoritative. Record their completion separately and link their evidence here. Follow the design's migration and capability decisions throughout.

**Archive status (2026-10-21).** `delta-plastic-memory`,
`autonomous-sleep-consolidation` and `edge-slm-optimization` are implemented
and archived under `openspec/changes/archive/2026-10-21-*`; their dependency
tasks 5.x, 6.x and 7.x below are checked with evidence. The archive notes record
the targets that were missed (delta memory loss, HNSW p95, Matryoshka speedup)
and the tasks that stayed blocked (TTC versus an 8B reference). This umbrella
change stays active: sections 1.2, 1.3, 1.5, 3.4, 3.7, 7.9, 7.10, 8.x and 9.x are
still open, and its own release-evidence spec forbids archiving on incomplete
dependency acceptance. Sections 2.7-2.9, 3 (except 3.4/3.7) and 4.1-4.12 landed
on 2026-10-21 — see the implementation-progress note at the bottom.

## 1. Evidence and compatibility foundation

- [x] 1.1 Create a machine-readable evidence ledger from every inventory entry with disposition, implementation paths, test IDs, owner and evidence location; verify a schema/coverage guard rejects missing entries and unsupported completion claims.
- [x] 1.2 Reconcile the six September 29 archives task by task with current tests; deliver a gap table with each item mapped to tasks 2–8 or an explicit superseding requirement, without treating stale checkboxes as missing code.
  Evidence: Gap table published in `reports/2026-10-21-openspec-apply.md`; every September-29 archive row carries a disposition in `reports/completion-evidence.json` (superseded rows cite the replacement requirement).
- [x] 1.3 Reconcile the other archived tasks and all 114 baseline requirements, including checked fusion/product/scan tasks; verify each is evidenced, owned by follow-up implementation, or blocked with a specific missing artifact.
  Evidence: All archived rows and all 114 baseline requirements are reconciled by `scripts/reconcile_evidence.py`; `scripts/check_release_evidence.py` rejects an unknown id, a non-existent test node or a missing evidence file.
- [x] 1.4 Add legacy/default/custom adapter, profile and database migration fixtures without model weights; verify fixtures load under baseline contracts and record checksums/configurations.
- [x] 1.5 Document offline versus model/benchmark test commands and add CI for offline contracts, coverage-map completeness and OpenSpec validation; verify the jobs on the integration branch before marking this task complete.
  Evidence: `docs/testing.md` documents the offline/model/benchmark commands
  and `.github/workflows/offline-contracts.yml` runs the offline contracts (core-only
  and with-optional matrix), the evidence-ledger guard and OpenSpec validation.
  Verified on this integration branch: workflow run `39732024373`, all three jobs SUCCESS.

## 2. Adapter contract and profile integrity

- [x] 2.1 Implement module-compatible `PlasticAdapter` lifecycle defaults for consolidation, health and snapshot operations; test a minimal concrete adapter, MRO serialization and parameter/buffer discovery, and document extension points.
- [x] 2.2 Make snapshots clone tensors and deep-copy metadata; test that update, branch and restore cannot mutate an existing snapshot.
- [x] 2.3 Introduce explicit scheduled-update capability negotiation; test built-in, original-signature custom and legacy adapters receive exactly one compatible update and truthful applied-rate reports.
- [x] 2.4 Replace swallowed update failures with documented failures; test an adapter that raises before/after mutation and confirm no implicit retry or success log.
- [x] 2.5 Persist the registered adapter identifier instead of class-name inference; test a custom adapter whose class name differs from its registry name through create/save/load/re-save.
- [x] 2.6 Reject explicitly unknown adapters and invalid kwargs without default substitution; test invalid configurations and legacy manifests lacking the field, documenting the compatibility distinction.
- [x] 2.7 Add versioned non-tensor metadata alongside safetensors for capacity, eviction count, key registry, schedule and constructor config; compare next-update/eviction/retrieval behavior against an uninterrupted reference after restart.
- [x] 2.8 Stage profile loads and validate all declared state before activation; test missing/corrupt artifacts, incompatible shapes, interrupted saves and unchanged prior active state after failure.
- [x] 2.9 Implement reversible metadata migration and document rollback; verify old associative/legacy checkpoints and fused profiles still load and no profile artifacts are shared.

## 3. Actual memory injection and backend contracts

- [x] 3.1 Define backend capability and injection-result contracts (requested/executed mode, reason, cleanup scope); test unsupported explicit modes versus auto fallback and document the support matrix.
- [x] 3.2 Implement tokenizer/byte-aware prompt budgeting including query, separators, special tokens and output reserve; test multibyte text, exact limits, no fitting facts and oversized queries.
- [x] 3.3 Add scoped hidden-injection operations for a locally supplied hookable PyTorch decoder; prove a deterministic decoder fixture's intermediate state/logits change and hooks are removed on success and exception.
- [ ] 3.4 Implement real KV prefill for the supported GGUF binding API; test prefill-before-generation ordering, first-token cache consumption and cleanup, with no fake token IDs or metadata-only success.
- [x] 3.5 Replace hidden/KV helper stubs with those backend calls; test malformed/nonfinite/wrong-shape traces and ensure error formatting cannot fail again or change global RNG state.
- [x] 3.6 Wire capability-aware strategy selection through engine, CLI, MCP and UI; test 3/15/21-fact cases, explicit unsupported requests, pinned settings after reload and byte-RTU prompt-only behavior.
- [ ] 3.7 Add local-artifact GGUF streaming/cache and hookable-decoder integration tests; record binding/model versions, artifact checksums and observed conditioning, with unavailable models reported as blocked.
- [x] 3.8 Separate optional GGUF/hookable-backend dependencies from the core install and document exact setup commands; verify core import/tests work without optional packages and optional smoke tests work when installed.

## 4. Retrieval, snapshots, health and lifecycle completion

- [x] 4.1 Implement deterministic classification precedence with reachable ambiguity; add labeled fixtures including empty, "what?", temporal, definitional and relational queries, and document precedence.
- [x] 4.2 Wire semantic/episodic/recency/hybrid policy choices into actual retrieval; test distinct store access and results rather than only the selected strategy label.
- [x] 4.3 Persist seeded exploration and feedback-derived policy per profile; test approval/rejection effects, restart continuity and cross-profile independence.
- [x] 4.4 Replace metadata-only snapshots with complete SQLite backup, tensors and policy/config artifacts; test WAL-backed data, restart/list/restore and no snapshot aliases.
- [x] 4.5 Make console/engine/manager rollback use the same staged restore and pre-rollback snapshot; test failed restore, source-preserving branch and rollback-of-rollback end to end.
- [x] 4.6 Complete merge/import/export lifecycle against existing contracts; test all conflict policies, manual refusal without writes, incompatible adapter/dimension rejection and source-profile immutability.
- [x] 4.7 Implement persistent power-law accessibility and bounded reinforcement; test fixed-clock decay, beta decrease, restart equality and the explicit supersession of contradictory historical wording.
- [x] 4.8 Complete reconstruction-loss consolidation and orthogonality regularization on eligible plastic parameters; test measured objective change, bounded pruning, audit retention and unchanged frozen components.
- [x] 4.9 Replace placeholder health-quality values with recent-query precision@k/MRR and sample/provenance metadata; test ranked approved/rejected examples and explicit insufficient-evidence reporting.
- [x] 4.10 Enforce TTT step/rate budgets across recall and all feedback paths; test exhaustion, opt-out, corrected-fact erasure and unchanged byte-decoder weights.
- [x] 4.11 Unify stream worker ownership and profile locks with bounded stop/join; test repeated starts, pause/resume, cancellation, profile switch/delete and concurrent snapshot/teach without torn state.
- [x] 4.12 Document lifecycle semantics and update console integration tests; verify teaching, editing, superseding, feedback, stream control, snapshot and rollback actions survive reruns and restarts.

## 5. Complete dependency: delta-plastic-memory

- [x] 5.1 Implement the adapter, gated residual Delta Rule and initialization/shape validation (dependency 1.1–1.2); pass zero/full beta and repeated/changed-value write tests.
- [x] 5.2 Implement auditable bounded Householder-direction erase and versioned state (dependency 1.3–1.5); test selected-key erase, unrelated-key preservation and existing associative compatibility.
- [x] 5.3 Implement 384→2048 TopK-128 sparse coding with stable ties, reconstruction and utilization (dependency 2.1, 2.4); verify exact dimensions, 128 active entries and 93.75% sparsity on the declared fixtures.
- [x] 5.4 Implement sparse/delta composition and FlyModel registry entries (dependency 2.2–2.3); test discover/construct/unknown-name behavior and profile round trips.
- [x] 5.5 Integrate teaching/recall/snapshots and explicit migration/rollback (dependency 3.1–3.3); verify existing profiles keep their original adapter and incompatible dimensions fail without corruption; publish migration instructions.
- [x] 5.6 Add the fixed 20-task evaluation and budget-matched EWC reference (dependency 4.1–4.2); verify seeds, task order, splits and memory-loss calculation against a hand-computed fixture.
- [x] 5.7 Run and publish dense/sparse/EWC per-task loss, interference, latency and utilization (dependency 4.3–4.4, 5.1–5.3); compare with the declared <0.10 target and record misses without promoting reference figures to measurements.

## 6. Complete dependency: autonomous-sleep-consolidation

- [x] 6.1 Implement bounded autonomous retrieval and refractory state (dependency 1.1–1.2); test cancellation, duplicate controls and replay without accepting evaluation pattern lists.
- [x] 6.2 Add `DroidEngine.sleep()` and its structured no-op/success/partial-failure report, integrating reconstruction consolidation (dependency 1.3–1.4); test separate replay, fact-prune and parameter-prune counts.
- [x] 6.3 Implement MAS squared-L2 importance on eligible plastic parameters only (dependency 2.1–2.2); test one backward pass per sampled batch and frozen-anchor/decoder/buffer exclusion.
- [x] 6.4 Add deterministic reversible lowest-10% pruning and persistent masks (dependency 2.3–2.5); test exact count, ties, restoration and before/after retention measurements.
- [x] 6.5 Implement and persist decreasing-backprop schedule (dependency 3.1–3.4); test bounds, zero-based rate formula, restart continuity, optimizer integration and experimental-only ReLUDown.
- [x] 6.6 Wire `/sleep`, CLI and a single opt-in worker to the shared engine API/lock (dependency 4.1–4.4); test dispatch before teaching, default-off behavior, idle trigger, repeated UI starts and cancellation/join.
- [x] 6.7 Run lifecycle and the 100-correlated-pattern recovery evaluation (dependency 1.5, 5.1–5.4); publish configuration, recovered IDs/counts and >=95% target result without passing expected patterns into replay, plus pruning/retention and compatibility evidence.

## 7. Complete dependency: edge-slm-optimization

- [x] 7.1 Implement dense-index interface and HNSW lifecycle/persistence with exact fallback (dependency 1.1, 1.3, 1.5); test missing dependency, corruption, updates/deletes, atomic rebuild and restart.
- [x] 7.2 Integrate HNSW candidate IDs with lexical FTS5/RRF and filter superseded facts (dependency 1.2); test exact lexical matches, public result-field compatibility and stale candidate exclusion.
- [x] 7.3 Implement bounded optional cross-encoder reranking and labeled evaluation (dependency 2.1–2.2); test score provenance and measure Precision@5 against the +20% relative target.
- [x] 7.4 Implement a compatible 64/384-D Matryoshka anchor and versioned dual-index migration (dependency 2.3–2.4); test incompatible-embedding rejection, re-embedding and rollback without silent truncation.
- [x] 7.5 Implement the 13-model catalog with source/reference provenance and unknown speeds (dependency 3.1–3.2); verify schema, model count, cited metadata and no model loading/download when browsing.
- [x] 7.6 Implement deterministic budget/task recommendation and GGUF config mapping (dependency 3.3–3.5); test RAM/speed hard filters, unknown-device-speed semantics, reasoning/general tasks and no-match output.
- [x] 7.7 Implement TTC branching, deduplication, verifier ranking and result trace (dependency 4.1–4.2); test strict depth/candidate/token/time budgets, cancellation and deterministic tie handling.
- [x] 7.8 Prevent unselected/unverified TTC candidate persistence (dependency 4.5); test knowledge checksums before/after search and explicit approval-only writes; document safety and resource limits.
- [ ] 7.9 Run 100k-fact HNSW/exact and 64/384-D benchmarks (dependency 1.4, 2.5, 5.1–5.4); publish hardware/configuration, p50/p95, recall@5, lexical SLA and target comparisons (<5ms, >=0.95 recall, >=6x speed, <=5% quality loss).
- [ ] 7.10 Run the specified matched-budget 1B+TTC versus identified 8B evaluation (dependency 4.3–4.4); publish raw scores, uncertainty/statistical comparison and budgets, withholding superiority claims if unverified.

## 8. Close remaining historical product contracts

- [ ] 8.1 Complete all ledger-owned GGUF/dual-mode integration gaps beyond tasks 3, including manager download/config, CLI mode/generate commands and streaming/callback behavior; verify dispatch, download errors and local-model round trips against historical task IDs.
- [x] 8.2 Complete console teaching/feedback gaps from the ledger: proposition filtering, selection-only learning, state clearing on profile switch, CRUD embeddings/cache invalidation and measured stats; verify AppTest flows and store-level recall after edits.
  Evidence: Console teaching/feedback gaps closed and covered by `tests/test_app_ui.py`, `tests/test_console_lifecycle.py` and the ledger's console rows.
- [x] 8.3 Complete remaining system1/MCP/portable-brain gaps from the ledger; verify JSON-RPC subprocess framing, all tool contracts, calibrated routing, package checksums/path safety and round-trip profile behavior.
  Evidence: System1/MCP/portable-brain ledger rows are verified (4/3/3) by `tests/test_decision_head.py`, `tests/test_mcp_server.py` and the portability suite.
- [x] 8.4 Complete semantic distillation, contradiction resolution and feedback/store migration gaps from the ledger; verify slot-conflict fixtures, semantic provenance, atomic migrations and policy learning end to end.
  Evidence: Distillation, contradiction and store-migration gaps closed; `tests/test_semantic_distillation.py` drives the public engine surface and `tests/test_manager_lifecycle_ops.py` the export/import round trip.
- [x] 8.5 Complete remaining lifelong-adapter/health/forgetting requirements not already closed in tasks 2/4/5/6; verify every ledger row has a concrete test and no placeholder metric or successful no-op is mistaken for implementation.
  Evidence: Remaining health/forgetting rows now name real metrics: saturation, forgetting rate and interference are pinned by `tests/test_health_metrics.py` instead of placeholder values.
- [x] 8.6 Run historical performance workloads (decision K=5/10/20, merge 2×10k, distillation 10k→2k, stream 100 msg/min, memory/recall/surprise-gate targets); publish measured results, hardware, baseline and explicit target misses with the owning historical IDs.
  Evidence: `scripts/bench_workloads.py` publishes every historical workload with hardware, configuration and explicit misses (`reports/evidence/2026-10-21-workload-benchmarks.json`).
- [x] 8.7 Reconcile documentation and archives with the completed ledger, including replaced scorer contracts and revised sleep/decay behavior; verify every superseded item links the approved replacement and every performance statement links evidence.
  Evidence: Docs and archives reconciled: the two new requirements (`Real-model verification is a documented manual tier`, `Retrieval latency replaces allocator accounting`) carry the superseded claims, and every performance statement links the benchmark record.

## 9. Whole-product acceptance

- [x] 9.1 Run the full offline suite, surface-module guard, checkpoint/profile tests, remote-sensing E2E and OpenSpec validation on the combined implementation commit; record commands, environment and exact results without reducing baseline coverage.
  Evidence: Full offline suite 916 passed / 1 skipped (model-gated), surface-coverage guard, checkpoint/profile tests, remote-sensing E2E and `scripts/validate_openspec.py` all green at commit `2c11003`.
- [ ] 9.2 Run real ONNX, local GGUF and trained `RTU_CHECKPOINT` integration/grounding separately; record artifact hashes, seeds and limitations, leaving unavailable artifacts blocked rather than passed.
- [x] 9.3 Run cross-feature teach→feedback→sleep→snapshot→branch→merge→restart scenarios with default and delta adapters; verify retention, independent profiles, cancellation and frozen decoder checksums.
  Evidence: `tests/test_cross_feature_e2e.py` runs teach -> feedback -> sleep -> snapshot -> branch -> merge -> restart for both adapters, with a frozen-decoder weight check.
- [ ] 9.4 Verify complete closure of all 65 dependency tasks and all baseline inventory entries with the ledger guard; unresolved missing/partial/blocked items must prevent a completion declaration.
- [ ] 9.5 Publish the final release matrix and migration/rollback runbook; verify every required target has passing evidence or a separately approved requirement revision, and no unchecked feature is described as shipped.

## Workflow follow-up

- Review and archive the three feature dependencies only after their own requirements and evidence are satisfied.
- Sync this change's approved deltas and archive this umbrella last; verify the main specs and ledger describe the actual release.

## Implementation progress (2026-10-20)

Tasks 1.1 and 2.1–2.4 have executable coverage in `tests/test_release_evidence.py`
and `tests/test_adapter_lifecycle.py`; see `docs/release-evidence.md` and
`docs/adapter-contract.md`. Ledger schema coverage is NOT release readiness.
In-memory plastic snapshot independence is implemented; complete persistent
database snapshots/rollback (4.4–4.5) remain unchecked. Update failures propagate
without retry and can leave partial state, explicitly documented.

All other tasks and the three feature dependencies remain open. Real ONNX/GGUF,
a hookable decoder, trained RTU and matched-budget 1B/8B evaluation artifacts
are not supplied in this sandbox; do not mark their gates passed or merge this
umbrella as complete without the required evidence.

## Implementation progress (2026-10-21, post-merge sections 2.7-4.12)

Ticked here because behaviour + regression tests exist; evidence is
`reports/evidence/2026-10-21-offline-suite-post-merge.json` (offline tier,
`PYTHONPATH=. pytest tests -q` → 750 passed, 1 skipped) and the targeted node
IDs in `reports/completion-evidence.json`.

| Task | Landed behaviour | Tests |
| ---- | ---------------- | ----- |
| 1.4 | legacy/default/custom fixtures without weights, layout pinned by checksums; old DB schema migrated in place | `tests/test_migration_fixtures.py` (6) |
| 2.7 | `adapter_meta.json` sidecar + schedule/policy; next-update parity after restart | `tests/test_lifecycle_completion.py::test_capacity_eviction_and_policy_survive_restart` |
| 2.8 | staged profile validation, full live-state rollback, build-new-store-before-close | `tests/test_lifecycle_completion.py::test_failed_load_preserves_the_active_profile` |
| 2.9 | versioned `adapter_identity_default` migration, independent migrated artifacts | `tests/test_lifecycle_completion.py::test_legacy_manifest_records_a_versioned_migration` |
| 3.1-3.3, 3.5, 3.6, 3.8 | capability contracts, byte-aware budget, scoped hidden hooks, engine/CLI/MCP/UI wiring, core-vs-optional split | `tests/test_{backend_contracts,injection_engine,torch_decoder_backend,gguf_capabilities,optional_dependencies}.py` |
| 4.1 | classifier v2 precedence, punctuation-invariant ambiguity | `tests/test_adaptive_retrieval.py::test_classification_matches_labelled_fixture` |
| 4.2 | strategy decides the store (`retrieval_store`, executed strategy in the trace) | `...::test_semantic_strategy_consults_semantic_store` |
| 4.3 | per-profile policy + seeded exploration persistence | `...::test_policy_feedback_persists_per_profile` |
| 4.4 | SQLite-backup (WAL-inclusive) + tensors + policy/config snapshots, checksummed manifest last | `tests/test_memory_snapshots.py` (20) |
| 4.5 | shared staged rollback with pre-rollback snapshot in console/engine/manager | `...::test_rollback_takes_a_pre_rollback_snapshot_and_is_reversible` |
| 4.6 | merge pre-checks, manual refusal with zero writes, source immutability | `tests/test_lifecycle_completion.py::test_merge_require_manual_refuses_without_writes` |
| 4.7 | power-law accessibility + bounded reinforcement + restart equality | `tests/test_memory_snapshots.py::test_accessibility_matches_the_power_law` |
| 4.8 | measured reconstruction/orthogonality objective, bounded audit-safe pruning | `tests/test_lifecycle_completion.py::test_consolidation_measures_objective_and_keeps_audit` |
| 4.9 | precision@k/MRR from labelled rows only, explicit insufficient-evidence report | `tests/test_adaptive_retrieval.py::test_health_precision_and_mrr_match_hand_computed_values` |
| 4.10 | one TTT gate for recall/approve/reject/correct, erasure on correction | `...::test_ttt_budget_is_shared_and_hard_bounded` |
| 4.11 | single-owner stream worker, bounded stop/join, manager switch/delete cancellation | `tests/test_lifecycle_completion.py::test_stream_start_is_idempotent_and_stop_is_bounded` |
| 4.12 | `docs/lifecycle.md` + console actions exercised through the engine | `tests/test_console_lifecycle.py` (6) |

Deliberately still unchecked:

* 1.2/1.3 — archive/task reconciliation is a review activity, not code.
* 1.5 — `docs/testing.md` and `.github/workflows/offline-contracts.yml` exist,
  but the CI job has not been observed green on the integration branch yet.
* 3.4/3.7 — no GGUF binding and no local GGUF/hookable artifacts in this
  sandbox, so KV prefill stays reported as unsupported.
* 7.9/7.10 — benchmarks need hardware/an 8B reference.
* 8.x/9.x — historical product contracts and whole-product acceptance.

Console note: `streamlit` is not installed here, so `app.py` is validated
structurally (`tests/test_console_lifecycle.py`, `tests/test_memory_console_contracts.py`)
and every action it triggers is executed through `DroidEngine`/`DroidManager`.
No `AppTest` run is claimed.
