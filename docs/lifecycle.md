# Memory lifecycle semantics

This document describes the behaviour the code actually implements, for the
parts of the lifecycle that reached a testable state under
`openspec/changes/post-merge-completion`. Anything listed as *open* below is
still unchecked in that change's `tasks.md`.

## 1. Query classification and strategy selection

`src/model/memory_lifecycle.py::RetrievalPolicy` classifies with a fixed
precedence (highest first). Classification never depends on randomness:

| # | Class | Trigger |
|---|-------|---------|
| 1 | `ambiguous` | empty/whitespace, or an underspecified question (`what?`, `who?`, `which?`, `?`) |
| 2 | `temporal` | time words (`when`, `before`, `latest`, `history`, …) |
| 3 | `relational` | comparison/why/between/versus questions |
| 4 | `definitional` | `what is`, `define`, `definition`, `meaning` |
| 5 | `specific_fact` | everything else with more than two content words |

Punctuation is not a signal: removing the final `?` never changes a label.
Exploration (`epsilon_greedy`, seeded `random.Random`) affects only the
*strategy* choice, and its state (`seed`, `selection_count`,
`exploration_count`) is persisted and replayed on load, so a restarted profile
continues the same exploration stream.

Default strategies per class: `definitional → semantic_primary`,
`temporal → episodic_recency_weighted`, `specific_fact`/`ambiguous →
hybrid_rrf`.

`CLASSIFIER_VERSION = 2` is recorded in every persisted policy.

## 2. Strategy execution

The selected strategy changes which store answers (`KnowledgeStore.recall`):
`semantic_primary` consults `semantic_facts` and marks hits
`retrieval_store="semantic"`; `episodic_only` and `hybrid_rrf` read the
episodic facts table; `episodic_recency_weighted` multiplies the fused score by
`1/(1 + 0.5·age_days)`; `contextual_boosted` multiplies by
`1 + 0.1·decay_beta`. Every result and `last_retrieval_trace` records the
`requested_strategy` and the `executed_strategy` — an unknown label degrades to
`hybrid_rrf` and says so.

## 3. Health metrics

`KnowledgeStore.retrieval_health()` computes precision@k and MRR **only** from
`retrieval_feedback` rows written by `log_ranked_feedback(...)`, which require
both a ranked list and at least one judged-relevant fact. With no labelled rows
the report is `{"available": False, "reason": "no_labelled_feedback",
"samples": 0, ...}` and `get_memory_health()["retrieval_quality"]` is `None`.
The former `0.5`/`0.7` placeholders are gone. Approve/reject counts are reported
separately as `approval_rate` and `approval_samples`.

## 4. Forgetting

Per fact, `KnowledgeStore.accessibility(fact_id, now)` evaluates

```
A_i(t) = S_base · (1 + α · (t − t_last))^(−β_i)      α = 1 day⁻¹, S_base = max(1, access_count)
```

`β_i` starts at `DECAY_BETA_INITIAL = 1.0`, is stored in `facts.decay_beta`, and
each successful retrieval calls `reinforce()`, which multiplies β by
`(1 − δ)` with `δ = 0.2` (inside the required `[0.1, 0.3]`) and floors it at
`DECAY_BETA_MIN = 0.2 > 0`. β lives in SQLite, so accessibility after restart is
bit-for-bit the uninterrupted value.

## 5. Test-time training budget

`DroidEngine._consume_ttt_step()` is the single gate for every plastic update:
recall, `approve`, `reject` and `correct` all call it. It returns
`{"allowed": bool, "reason": "opt_out" | "exhausted" | "rate_limited" | None}`;
a denied step performs no update. `ttt_status()` exposes `used`, `max_steps`,
`attempts`, `skipped`, `rate`, `remaining` and `last_skip_reason`.
`correct()` erases the superseded association from plastic memory before
teaching the replacement.

## 6. Snapshots and rollback

`SnapshotManager.create_snapshot` writes one directory per version:

```
<profile>/snapshots/<version>/
  knowledge.db            # sqlite3 backup API: includes uncheckpointed WAL frames
  plastic.safetensors     # every plastic tensor, cloned
  plastic_meta.json       # non-tensor state (K_max, eviction_count, gates, …)
  retrieval_policy.json   # learned policy + exploration seed
  lifecycle_config.json   # adapter identity, schedule, TTT/sleep config
  snapshot.json           # manifest, written LAST, sha256 + size per artifact
```

Restore is staged: artifacts and checksums are verified, the database is opened
read-only and `PRAGMA integrity_check` must return `ok` with a fact count equal
to the manifest, tensor ranks/trailing dimensions are checked against the live
adapter, and the policy is validated. Only then is anything activated; a failure
during activation restores the captured live plastic state, policy and database,
so the previous profile stays usable.

`rollback(droid, version)` writes a **pre-rollback snapshot** first and returns
`{"restored", "version_id", "pre_rollback_version", "error"}`. Rolling back to
`pre_rollback_version` restores the state that was just replaced. The console,
`DroidEngine.rollback` and `DroidManager.rollback_version` all use that path.
Branches copy the artifact directory, so no two snapshots share a file.

Snapshots created while the engine still ran on an in-memory store are moved
next to the profile by `save_profile` (`SnapshotManager.relocate`), so they can
still be listed and restored after a restart.

## 7. Profile integrity

`save_profile` writes artifacts first and `config.json` last. `load_profile`
captures the live state, validates every referenced artifact (manifest, plastic
tensors, adapter sidecar, knowledge DB integrity, policy/snapshot JSON, schedule,
MAS files) and only then activates; any failure raises `InvalidProfileError`
and restores the previously active profile. Legacy manifests without
`plastic_adapter` load the documented default and record a versioned
`adapter_identity_default` migration.

## 8. Merge

`DroidEngine.merge_memory(other, policy)` validates the policy and the
adapter/dimension compatibility **before** any write. With
`require_manual` and detected conflicts it returns `status="manual_required"`
and `writes=0`. Successful merges report `writes` and `source_unchanged`, which
is a hash comparison of the source profile's tensors and facts.

## 9. Workers

`ContinuousStream` has exactly one thread per engine. `start()` is idempotent and
returns `False` when a worker is already alive; `stop(timeout)` signals, joins
and returns whether the thread exited; `pause()`/`resume()` are flag-only and
report whether a live worker saw them. `DroidEngine.shutdown()` stops the stream
and the auto-sleep worker with bounded joins, and `DroidManager.activate_droid`,
`delete_droid` and `shutdown` stop workers before switching or removing a
profile. The console delegates to these engine methods instead of constructing
its own worker.

## 10. Consolidation

`consolidate_memory(max_prune_ratio)` measures an objective *before* and
*after*: mean reconstruction cosine (when the adapter exposes `retrieve`) and
the Frobenius orthogonality loss of `S`. The plastic update runs through the
adapter's `consolidate()` when it reports support, otherwise a bounded
orthogonality step. Pruning is capped by `max_prune_ratio`, only touches
utilities below 0.5, and **supersedes** rows (`superseded=1`,
`valid_until`, `needs_review=1`) instead of deleting them, so the audit trail
survives. The report states `frozen_unchanged`, a hash of the anchor and the
generative decoder.

## Open (not implemented, still unchecked)

- Real KV prefill for the GGUF binding (3.4) and local-artifact GGUF/hookable
  decoder integration (3.7): no binding/model artifact in this sandbox.
- 100k-fact HNSW/exact and Matryoshka benchmarks (7.9); matched-budget
  1B+TTC vs 8B evaluation (7.10): no hardware/artifact.
- Historical product contracts in §8 and whole-product acceptance in §9.
