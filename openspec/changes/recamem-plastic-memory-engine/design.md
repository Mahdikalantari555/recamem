# Design

## Context

See proposal.md for motivation and the capability list. The requirements live in
`specs/`. This document records how the work is split and the decisions that shape it.

Current state at HEAD (verified by reading code and running it, not from the plan
alone):

- `src/model/droid.py` is 3,089 lines and holds `DroidEngine`. `teach()` splits text
  with `_split_propositions`, a bare regex. The abbreviation-safe splitter exists in
  `src/model/propositions.py` and is not used by `teach()`.
- `clear_knowledge` is an alias of `clear_memory` (droid.py), so the plan's
  "missing method" defect is closed.
- The active plastic adapter in the Droid path is `DeltaMemory`
  (`src/model/delta_memory.py`), which exposes `update_scheduled`
  (`effective_lr` is passed through `plastic_adapter.apply_plastic_update`) and
  enforces `K_max` by attenuating the weakest key.
- `OnnxMiniLM` (`onnx_anchor.py`) and `GgufBackend` (`gguf_backend.py`) fall back to
  mock objects with a warning. The GGUF path substitutes `MockLlama`.
- `KnowledgeStore` opens SQLite with `check_same_thread=False` and no lock
  (`knowledge_store.py`).
- 912 tests are collected (`pytest --collect-only`).
- Real-model inputs are unavailable in this sandbox: Hugging Face is not reachable
  (TLS failures), and the MiniLM ONNX and GGUF weights are not present under
  `models/`. The gates that depend on them must report "blocked", not "passed".

## Goals / Non-Goals

**Goals:**

- Deliver the plan's phases 0–9 as one change, each phase gated by tests in tasks.md.
- Make every "works" claim traceable to a test or a labelled benchmark report.
- Keep the offline gate runnable without weights, network, or GPU.
- Preserve existing profiles and `.droid` archives through migration, never deletion.

**Non-Goals:**

- Training a model from scratch or shipping a new language model.
- Changing the post-merge-completion gates or archiving that change.
- A public WebUI deployment or cloud sync.
- Refactoring for its own sake. Module splits happen only where a phase needs them.

## Decisions

### D-1. Package layout: `recamem/` with a deprecation shim, not a parallel tree

New code lives in `recamem/` (engine, plasticity, retrieval, consolidation,
forgetting, episodic, anchor, backends, ingest, bench, cli). The old `src/model/*`
modules become thin re-exports that emit `DeprecationWarning` and are removed one
release later. `src/cli.py` keeps working as `python -m src.cli`.

- Alternative: keep `src/` and add `recamem` as a new top-level name. Rejected: two
  engines would drift, which is the scattered-methods problem the user asked to end.
- Alternative: hard rename with no shim. Rejected: breaks existing MCP clients and
  saved profiles without warning.

### D-2. Phase order: correctness before rename

Phase 0 (defect fixes plus behavioural tests) lands before any rename. Each defect
gets a regression test against the current code so the rename cannot hide it. Where
an audit item is already fixed, Phase 0 records the evidence and adds the regression
test, and makes no code change.

### D-3. Anchor is real by default and fails fast

The `MiniLMAnchor` loads the quantized ONNX MiniLM. If the model is missing it raises
`AnchorUnavailableError` at construction and names the path. Mock embeddings are
selected only with `--anchor mock` or `anchor="mock"` in code. Test fixtures that need
an anchor use the explicit mock, and the test names say so.

- Alternative: keep the warning-and-fallback. Rejected: the current behaviour already
  causes results that look real and are not, which the `droid/onnx-anchor` spec
  forbids.
- Consequence: the real-anchor latency and recall scenarios in `droid/onnx-anchor`
  are marked `real_model` and report "blocked" in this sandbox until weights exist.

### D-4. Two-tier plasticity

FAST tier: no-grad associative writes, with `effective_lr` recorded and applied to the
stored state, as the conversational-teaching spec already requires.
SLOW tier: replay-trained meta-parameters under a memory-aware-synapse (MAS) penalty.
Updates happen only inside a rollback gate: the candidate is accepted only if held-out
recall does not drop beyond a set tolerance; otherwise the previous state is restored.

- Alternative: a single gradient path for everything. Rejected: per-fact gradient steps
  are slow on the 2-CPU reference hardware and make forgetting harder to bound.

### D-5. Recall: two channels plus a learned scorer

The plastic channel gives a prior score. The episodic channel returns verbatim facts
with source. A logistic scorer over a few features (channel scores, FTS5 rank, recency,
access count, and the ANN similarity) is trained on the labelled set and on recorded
accept/reject feedback. Hardcoded boosts are removed.

- Alternative: keep tuned boosts and tune them more carefully. Rejected: not
  reproducible and not evaluated against a held-out set.
- ANN index: HNSW for large stores, exact search below a threshold. The exact search is
  the fallback and the recall baseline.
- 64-D first-stage path is optional and gated by the measured loss criterion in
  `memory/hybrid-retrieval`. It is not a default.

### D-6. Forgetting never destroys data

Eviction moves a fact to the `retired` tier, keeping its row, provenance, and audit
trail. Consolidate-then-forget: a fact is retired only after its content has been
absorbed into the plastic channel and a recall probe confirms it. Hard `DELETE` is
removed from the forgetting path. The decay exponent is fixed to decrease with
successful retrieval, matching the scenario text.

### D-7. Consolidation with a validation gate and a single writer

`sleep()` replays episodic items, proposes parameter updates, runs a held-out probe
set, and commits only when no probed domain regresses. Only one writer per profile
holds an exclusive lock file under the profile directory; a second writer fails with
a named error rather than blocking indefinitely.

### D-8. Identity and profile location

Profiles live under `~/.recamem/agents/<name>/`, overridable with `RECAMEM_HOME`.
Legacy `droids/<name>/` is read in place when present. `recamem migrate droids/`
copies each profile to the new home, writes a `migration.json` record with checksums,
and never deletes the source. Each profile's `config.json` carries `schema_version`.
A loader rejects an unknown newer version with a named error and reads older versions
through an explicit upgrade step.

### D-9. MCP names

Each tool is renamed `recamem_<old name>`, for example `recamem_recall_memory` and
`recamem_teach_fact`. The old names stay as aliases for one release and are listed in
`tools/list` with a `deprecated: true` annotation. The documented-list test checks the
union.

### D-10. Benchmarks: real when reachable, fixture when not, always labelled

LoCoMo, LongMemEval-S and DMR runners fetch their data once into a cache outside the
repository, with checksums recorded. The offline fixture is a small, committed,
license-compatible sample, and every report says `dataset_source=real|fixture`. No
score from a fixture is presented as a benchmark result. The apply phase verifies
which sources are reachable; sources that are not reachable are reported as blocked.

### D-11. Ingest

`recamem ingest <dir>` walks Markdown and text files. Markdown headings and the file
path are kept as provenance, and code fences are not segmented as prose. PDF support
uses `pypdf` as an optional extra; without it, PDF files are reported as skipped with
the reason. Re-ingesting an unchanged file is a no-op, keyed by content hash.
Segmentation uses the abbreviation-safe splitter and replaces the regex in
`_split_propositions`.

### D-12. Defect ledger

Phase 0 writes `reports/recamem-defect-ledger.md` with one row per audit item: status
(`fixed-evidence`, `fixed-now`, `not-reproduced`, `blocked`), the file and line at
HEAD, and the test that pins it. The defects listed in proposal.md are the initial rows.

## Risks / Trade-offs

- **Scope size** → 18 capabilities in one change. Mitigation: phases are independently
  testable, and tasks.md groups each phase so it can be reviewed and merged on its own.
- **Breaking renames for existing users** → shims and aliases for one release; migration
  copies, never deletes; a test covers loading an unmodified legacy profile.
- **Real-model gates cannot run here** → they report blocked with evidence (the
  existing pattern from `scripts/fetch_real_model.py`). They are not counted as passing.
- **Learned scorer overfits a small labelled set** → use a held-out split and report
  its size; below a minimum size the scorer is disabled and the plastic prior is used
  alone, with a reported reason.
- **Single-writer lock on platforms without `fcntl`** → the lock is advisory on Linux
  and macOS; Windows is not a supported target for the writer path in this change.
- **Rollback gate may reject useful updates** → the tolerance is configurable and the
  rejection reason is logged in the train log.
- **God-class split risk** → split only along the phase seams, with the shim keeping the
  public `DroidEngine` surface stable and covered by the existing tests.

## Migration Plan

1. Phase 0 lands with the ledger and regression tests on the current code. No behaviour
   change except the defect fixes, each covered by a test.
2. Phase 1 introduces `recamem/` and the shims. Old imports keep working.
3. Later phases each leave the offline gate green. A phase that cannot pass its gate is
   not merged.
4. Profiles: `recamem migrate droids/` is run by the user. The change never migrates
   silently at startup.
5. Rollback: each phase is a separate commit series on the branch. Reverting a phase is
   a revert of its commits. The `.droid` archive format is unchanged, so an archive
   exported after the change can be imported by the previous version.

## Open Questions

None that change the specs or the task breakdown. Dataset reachability (D-10) and the
exact hardware for the latency targets are checked during apply and reported with
evidence either way.
