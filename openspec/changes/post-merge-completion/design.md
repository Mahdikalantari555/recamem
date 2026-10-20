# Design

## Context

See proposal.md for scope. Baseline is main `c246043`, after the merge of #5/#8/#9 through #10. `reports/2026-10-20-pr-integration.md` records 441 passed, one trained-checkpoint skip, no xfails, mock-anchor E2E success and 35 valid OpenSpec items. This is a tested integration baseline, not a completeness certificate.

Observed remaining gaps:

| Area | Evidence in baseline | Owning work |
|---|---|---|
| Adapter lifecycle | `PlasticAdapter` is an ABC without module inheritance or optional lifecycle defaults | Tasks 2 |
| Update compatibility | `teach()` passes new rate kwargs through an older generic adapter method and swallows errors | Tasks 2 |
| Profile identity | saver infers identity from class names; loader can swallow registry/load failures | Tasks 2 |
| Capacity metadata | adapter exports scalar counters and key registry; profile safetensor path filters non-tensors | Tasks 2 |
| Snapshots | `SnapshotManager` stores plastic state and fact count, not a DB checkpoint; branch shallow-copies tensors | Tasks 4 |
| Injection | hidden returns projected metadata, KV returns tokens/prompt, neither performs the promised backend operation; error path norms the invalid trace again | Tasks 3 |
| Context budgeting | prompt helper uses mixed character/token estimates and does not reserve query/output budget | Tasks 3 |
| Retrieval | short question-mark branch precedes ambiguity; health uses placeholder 0.5/0.7 or approval ratios rather than labeled precision/MRR | Tasks 4 |
| Decay contracts | main forgetting requirement says beta increases but its reinforcement scenario says decreases; idle sleep conflicts with newer opt-in proposal | Tasks 4/6; explicit delta resolves both |
| New features | no delta/sparse, autonomous sleep/MAS, HNSW/Matryoshka/catalog/TTC implementations in `src` | Tasks 5–7, existing active changes |
| Historical completeness | six September 29 archives retain 198 unchecked tasks; checked archives also need evidence | Tasks 1/8/9 |

The already-fixed negative decision confidence, legacy trace restoration, MCP width mismatch and associative state export are regressions to preserve, not outstanding implementation tasks.

## Goals / Non-Goals

**Goals:** close missing end-to-end behavior, preserve existing profiles, and make completion demonstrable. Partition code and evidence into small reviewable increments. All accepted requirements remain in scope unless a spec revision explicitly supersedes them.

**Non-goals:** no global refactor, automatic adapter migrations, network downloads in offline tests, research-target weakening, or inference training of frozen byte-RTU weights. A test file existing is not proof that its entire module contract is covered.

## Decisions

### 1. One coordination change, three feature owners

Keep `delta-plastic-memory`, `autonomous-sleep-consolidation` and `edge-slm-optimization` active. Implement their tasks under their existing definitions; link exact task IDs and resulting commits in the umbrella ledger. Their specs already define detailed algorithms and benchmark budgets, so duplicating them would create drifting requirements. This change owns cross-cutting repairs and the final acceptance gate. Archive dependencies independently only after their evidence is complete; archive this umbrella last.

`inventory.json` is a frozen baseline inventory, not a claim that each task is missing. It captures every archived checkbox, current requirement and all 65 dependency tasks. Each item needs a verified/missing/partial/blocked/superseded disposition in the implementation ledger. Existing code gets evidence rather than pointless reimplementation. Unsupported historical claims cannot disappear: they remain blockers or gain an explicit reviewed replacement.

### 2. Adapter protocol and persistence first

Use `PlasticAdapter(nn.Module, ABC)` with safe defaults and explicit calls to `nn.Module.state_dict` where needed to avoid the former MRO bug. Clone tensor snapshots and deep-copy metadata. Registry identity is attached at construction, never guessed from class names.

Add an optional scheduled-update capability with a separate adapter method/protocol. The default path calls the original signature once and reports scheduling unsupported. The scheduled path records the actually applied rate. Do not discover support by catching `TypeError` from an update: the adapter may already have mutated state before raising. Propagate failures or return an explicit failed result; do not log successful learning after swallowed update exceptions.

Persist tensor data in safetensors and scalar/version/schedule/config metadata in a versioned JSON sidecar named by the manifest. Include K_max, eviction count and key registry in both in-memory and profile round trips. Stage and validate an entire load before replacing the active engine. Keep manifests atomic and last; no fallback to random state on corrupt declared artifacts. Legacy manifests migrate only through explicit version handlers with fixtures.

### 3. Injection is a backend operation, not a label

Introduce a capability contract on generation backends: prompt encoding/context budget, optional scoped hidden injection, optional cache prefill/reset. The orchestrator reports requested/executed mode and reason for automatic fallback. Byte RTU remains prompt-only. GGUF advertises only operations actually supported by the installed binding; do not invent a hidden hook in llama.cpp.

Implement a real hidden-capable adapter for a locally supplied hookable PyTorch causal decoder, with a declared layer/hidden-size mapping and scoped hooks. A tiny deterministic decoder fixture proves the hook changes intermediate state/logits; an optional real-model test provides integration evidence. Implement GGUF cache prefill through supported binding APIs, with save/reset/finally cleanup and a real local-GGUF test. Explicit unsupported modes raise; auto mode falls back with provenance. Merely returning a projection or token list is rejected by tests.

Budget the complete prompt plus reserved generation using backend tokenization (UTF-8 bytes for byte RTU). Preserve the query verbatim, trim whole facts deterministically, and reject a query that cannot fit. Validate traces before any backend mutation; use local RNG generators for projections and never format invalid input by recomputing a failing norm.

### 4. Complete lifecycle paths instead of testing facades

Unify UI, engine and manager snapshots behind a durable implementation that captures the SQLite database using the backup API (including WAL content), tensors, metadata and retrieval policy. Stage rollback and atomically activate a validated profile; retain a pre-rollback snapshot. Branches must clone storage and tensors. Test teach → snapshot → mutate → rollback across fresh manager instances, not only dictionary equality.

Connect selected retrieval strategies to actual stores; add fixed classification fixtures and seeded exploration. Verify feedback changes policy and persists per profile. Derive precision@k/MRR from ranked query feedback with explicit sample count and unavailable status when labels are insufficient; retain legacy summary fields with documented provenance, not invented measurements.

Implement the existing power-law accessibility contract with persistent positive beta; reinforcement decreases beta, resolving the contradictory sentence. Consolidation uses the specified reconstruction/orthogonality objectives and auditable pruning. Use one profile-scoped lock for stream/sleep/teach/migration coordination and bounded event-based stop/join. TTT budgeting applies to recall AND feedback; opt-out freezes plastic state outside explicit teach or explicit maintenance, while the byte decoder stays frozen in every case.

### 5. Dependency implementation order

1. Adapter/persistence and lifecycle foundations.
2. Delta/sparse memory: residual gated Delta Rule, auditable erase, TopK sparse coder, registry integration, migration and the 20-task EWC comparison. Preserve the associative default.
3. Sleep: autonomous retrieval without evaluation labels, refractory state, MAS on eligible parameters only, reversible 10% masks, persisted decreasing-backprop schedule, manual command and opt-in single worker. Reuse complete snapshots and locks.
4. Edge features: dense-index abstraction, exact fallback plus HNSW, lexical RRF filtering, optional reranker, genuine compatible 64/384-D Matryoshka embeddings, versioned dual indexes, 13-model catalog with provenance, and strictly budgeted TTC with no unapproved candidate writes.
5. Whole-product lifecycle, model and performance verification.

Catalog reference figures are provenance-labelled source values, not locally measured throughput. Do not silently truncate current MiniLM embeddings for Matryoshka. TTC quality requires the specified matched-budget 8B comparison. Dependency tasks retain their target thresholds and measurement requirements.

### 6. Acceptance is a matrix, not one green test count

Produce an evidence ledger keyed by `spec-path#requirement` or `archive/change#task-id`, with code paths, test IDs, command, commit, environment, result, owner and replacement requirement where applicable. A guard rejects missing inventory entries, missing evidence files, unresolved completion claims and undocumented skipped gates. Extend surface coverage for every new module but do not treat that map as line/behavior coverage.

Offline tests use deterministic stubs and no downloads. Separate opt-in runs use local ONNX, GGUF and RTU artifacts with checksums and licenses recorded outside Git. Include Streamlit AppTest flows, MCP subprocess framing, package tampering/traversal checks, restart/migration tests, worker cancellation and failure injection. Benchmarks record raw per-run data, hardware, seeds, configurations and target comparisons. Missing artifacts produce blocked evidence; research target misses remain visible.

## Migration Plan

1. Preserve baseline fixtures and test report before schema changes.
2. Add backward-compatible readers, then versioned writers and snapshot metadata.
3. Explicitly migrate a copy of a profile; validate tensor, fact, policy and retrieval equality before activation.
4. Keep exact retrieval and associative adapter as defaults. Index, embedding and adapter migrations are user-selected.
5. Roll back through complete snapshots; never reuse a new-format file in the old profile directory without validation.
6. Retain old archive statements as historical notes; append evidence-backed reconciliation links rather than retroactively presenting old work as fully verified.

## Risks / Trade-offs

- **Large scope** → separate implementation milestones with independently passing gates; umbrella completion blocks on all dependencies.
- **Legacy third-party adapters** → capability negotiation, strict fixture coverage and clear migration errors; no retries after possible mutation.
- **Backend-specific hooks/cache APIs** → explicit capabilities, scoped cleanup, pinned tested bindings and honest unsupported errors.
- **Deadlocks and inconsistent snapshots** → documented lock order, SQLite backup, cancellation tests and staged activation.
- **Absent models or benchmark hardware** → evidence stays blocked; do not use mock quality or skip status as acceptance.
- **Conflicting archived targets** → link explicit superseding requirements; never resolve conflicts by weakening tests silently.
