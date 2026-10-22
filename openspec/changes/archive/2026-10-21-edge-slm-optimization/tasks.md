# Tasks

## 1. HNSW and hybrid retrieval

- [x] 1.1 Define a dense-index interface and implement `HNSWIndex` with add/update/delete/rebuild, persistence/version metadata, and exact-search fallback.
- [x] 1.2 Integrate HNSW candidate IDs with FTS5 lexical candidates and existing RRF scoring without changing public recall result fields.
- [x] 1.3 Ensure superseded/deleted facts are filtered and index rebuilds are atomic and repeatable.
- [ ] 1.4 Add 100k-fact benchmarks for p50/p95 latency, recall@5 against exact MVM, and a documented ~15 ms brute-force reference; target <5 ms and recall@5 >=0.95.
- [x] 1.5 Preserve current exact mode and test missing optional dependency, corrupted index, and restart/rebuild behavior.

## 2. Reranking and Matryoshka retrieval

- [x] 2.1 Implement optional `CrossEncoderReranker` over a bounded candidate set and expose pre/post scores.
- [x] 2.2 Add a fixed labeled query set and measure Precision@5 before/after reranking; target +20% relative precision.
- [x] 2.3 Implement `MatryoshkaAnchor` for validated 64-D and 384-D representations with versioned dual-index storage.
- [x] 2.4 Add migration/re-embedding and rollback behavior; prohibit silent truncation of incompatible existing embeddings.
- [x] 2.5 Measure 64-D first-stage latency and retrieval-quality loss against the 384-D baseline; target >=6x speed and <=5% relative quality loss.

## 3. Edge SLM catalog and recommendation

- [x] 3.1 Implement `EdgeSLMRegistry` with the requested 13 models, quantization/format metadata, RAM estimates, task tags, provenance, and optional unknown speed fields.
- [x] 3.2 Include the source-reported reference values for Llama 3.2 1B Q4_K_M, Qwen2.5 1.5B, SmolLM2 1.7B, and Gemma 3 1B with explicit provenance labels.
- [x] 3.3 Implement deterministic `recommend(ram_gb, min_tok_s, task)` that filters hard RAM/speed constraints and returns task-ranked candidates with confidence/provenance.
- [x] 3.4 Test low-RAM, throughput-constrained, reasoning, general-chat, unknown-device-speed, and no-match cases.
- [x] 3.5 Integrate recommended IDs/config with `GgufBackend` without downloading or loading a model merely to display recommendations.

## 4. Test-time compute

- [x] 4.1 Implement `TestTimeCompute` with diverse branching, candidate deduplication, verifier scoring, and strict depth/candidate/token/time budgets.
- [x] 4.2 Add a trace/result contract with selected candidate, verifier evidence/score, and consumed budget.
- [ ] 4.3 Add fixed reasoning-task evaluation against an explicitly identified 8B reference model at the same wall-clock or token budget.
- [ ] 4.4 Verify a 1B+search/verifier configuration has statistically higher task score than the chosen 8B baseline on the declared benchmark before claiming the target.
- [x] 4.5 Ensure unselected/unverified candidates are not persisted to knowledge without user approval.

## 5. Validation and compatibility

- [x] 5.1 Run existing knowledge-store, DroidEngine, GGUF, profile, and checkpoint tests.
- [ ] 5.2 Verify lexical exact-match behavior and the existing <5 ms at 100k knowledge-store SLA on documented hardware.
- [ ] 5.3 Run model catalog/recommendation and TTC benchmarks with provenance and configuration recorded.
- [x] 5.4 Document optional dependencies, exact fallback, profile flags, and index/embedding version migration.

## Archive status (2026-10-21, re-measured 2026-10-22)

Archived after implementation. Tasks `1.4`, `2.5`, `5.2` and `5.3` are checked
because their benchmarks ran and published results, but the numeric targets were
**not** met and are not claimed; tasks `4.3` and `4.4` (1B+TTC versus an
identified 8B reference) remain unchecked and blocked on unavailable weights.

Measured at commit `2c11003` (`scripts/bench_workloads.py`, mock embeddings,
100k facts, 384-D, 2 vCPU, torch 2.6.0+cu124 CPU build):

| Metric | Target | Measured |
| --- | --- | --- |
| Dense candidate p95, 100k | < 5 ms | **17.46 ms, exact fallback (miss)** |
| Dense candidate p95, 10k | < 5 ms | 2.38 ms (met) |
| recall@5 vs exact MVM, 20k | >= 0.95 | 1.0 |
| 64-D Matryoshka speedup, p50 | >= 6x | **5.96x (miss on this runner; 8.08x on the previous one)** |
| 64-D relative quality loss | <= 5% | 0.0% |
| Reranker Precision@5 gain | +20% | +57.6% (lexical fallback scorer) |
| Catalog recommendation p95 | routing overhead only | 0.09 ms over 13 models |
| TTC search p95 | within the declared budget | 0.08 ms, 6/8 candidates, 48/512 tokens |

`hnswlib` has no wheel in this environment, so the HNSW path reports
`mode == "exact-fallback"` and the timings above are brute-force numbers; no
cross-encoder weights were available either. Superseded rows stay `partial` or
`blocked` in the ledger, and `scripts/check_release_evidence.py
--require-complete` keeps failing on these rows plus the missing `grounding` and
`real-model` tiers.
