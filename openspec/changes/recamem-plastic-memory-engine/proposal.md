# Proposal

## Why

The project began as a byte-level RTU language model and has grown into a personal
memory system (Droid profiles, teaching, recall, sleep, a memory console, MCP). Its
plan (`plans/09_recamem_plan.md`) asks for one coherent method, named **recamem**: a
small local memory engine that learns facts on the CPU, keeps them with bounded
forgetting, and answers from them with verbatim evidence. The current code mixes
several partial paths, has silent fallbacks that make results look real when they
are not, and has defects that contradict its own specs. This change implements the
plan end to end and fixes the verified defects, so that a user can point recamem at
a web search, a PDF, or about 100 Markdown files and get a working, honest,
specialist memory without training a model from scratch.

## What Changes

Phases follow `plans/09_recamem_plan.md`. Each phase is gated by its tests in `tasks.md`.

- **Correctness first (Phase 0).** Re-audit the defect list against HEAD, record each
  item as fixed (with evidence), still open, or not reproduced, then fix the open ones
  and add at least 15 behavioural tests before any rename.
- **Rebrand to `recamem` (Phase 1).** Package `recamem/`, CLI `recamem`, profile home
  `~/.recamem/agents/<name>/`, `recamem migrate droids/`. Old `droid`/`tmt-droid`
  names keep working through a deprecation shim for one release. **BREAKING** for
  the MCP tool names, which get `recamem_*` names with the old names as aliases.
- **Honest embedders (Phase 2).** The real quantized MiniLM anchor fails fast when
  its model is missing. The mock embedder runs only when explicitly selected
  (`--anchor mock`). `GgufBackend` no longer substitutes a mock on load failure.
  **BREAKING** for code that relied on silent fallback.
- **Two-tier plasticity (Phase 3).** FAST: no-grad associative writes with a schedule
  that actually changes stored state. SLOW: replay-trained meta-weights under a
  memory-aware-synapse penalty, applied only through a rollback gate. State is
  schema-versioned.
- **Consolidation with a validation gate (Phase 4).** Sleep replays episodic items,
  proposes changes, accepts them only if held-out recall does not regress, and runs
  under a single-writer lock.
- **Forgetting that never destroys (Phase 5).** Power-law accessibility, enforced
  `K_max`, consolidate-then-forget, and a soft `retired` tier instead of hard DELETE.
- **Two-channel recall (Phase 6).** A plastic prior channel and an episodic verbatim
  channel are fused by a small learned scorer. The hardcoded recall boosts are removed.
  The dense index is an ANN index with an exact fallback.
- **Benchmarks and reports (Phase 7).** LoCoMo, LongMemEval-S and DMR are run from a
  download-once cache, with an offline fixture fallback. Every report states its
  provenance and whether the run was real or fixture-based.
- **CLI, MCP, WebUI, CI (Phase 8).** New `ingest` command for a folder of Markdown,
  text and (optional) PDF files, `doctor`, `recall`, `migrate`, `rtu`. CI runs the
  offline gate. README and AGENTS.md describe only verified behaviour (Phase 9).
- **Ingestion fix.** Sentence splitting used by `teach` becomes abbreviation-safe.
- **Defect fixes (verified or to be re-verified in Phase 0):**
  - **D1** `DroidEngine._split_propositions` (`src/model/droid.py`) splits with a bare
    regex and skips the abbreviation-safe splitter in `src/model/propositions.py`.
    Reproduced: "The project lead is Dr. Mina Rostami." is stored as "...is Dr.".
  - **D2** Silent mock fallback: `OnnxMiniLM` (`src/model/onnx_anchor.py`, warnings
    only) and `GgufBackend` (`src/model/gguf_backend.py`, `MockLlama`). Reproduced by
    reading the code. This contradicts the `droid/onnx-anchor` spec.
  - **D3** `knowledge_store.py` opens SQLite with `check_same_thread=False` and no lock.
  - **D4** Bare `except:` clauses: about 87 occurrences under `src/`. Each is narrowed or
    re-raised with context.
  - **D5** Audit items to re-verify, not yet confirmed at HEAD: plastic updates under
    `no_grad` in the RTU path; `retrieve()` dead code; `consolidate_memory()` hard DELETE
    never scheduled; `save_profile` adapter-type detection reading the wrong attribute
    (the profile spec requires the adapter type to be recorded); duplicate
    `RTUMemoryBlock`; `error_vec` assignment; `README` test counts; `psutil` declared
    only in `requirements-optional.txt`.
  - **Closed without code change (evidence required):** `clear_knowledge()` is now an
    alias of `clear_memory()` (`droid.py`). `effective_lr` now reaches the adapter via
    `update_scheduled` (`plastic_adapter.py`). `K_max` is enforced in
    `delta_memory.py:_enforce_capacity`. Phase 0 keeps regression tests for each.
  - **D6** `DroidEngine` is a 3,089-line god class in `src/model/droid.py`. Split only
    along the phase seams above, not as a separate refactor.
- **Explicit out of scope:** model training from scratch, cloud services, a new LLM,
  hosting the WebUI publicly, and changing `post-merge-completion` gates. Real-model
  gates that need network or manually placed weights (MiniLM ONNX, GGUF, the 8B
  reference) stay blocked with their evidence, as in that change.

## Capabilities

### New Capabilities

- `plasticity/two-tier`: FAST no-grad writes and SLOW replay-trained meta-weights behind a rollback gate, with schema-versioned state.
- `memory/two-channel-recall`: fusion of the plastic prior and episodic verbatim channels with a learned scorer, reconsolidation on recall, and ANN plus exact fallback.
- `memory/ingestion`: abbreviation-safe segmentation, folder ingest of Markdown, text and optional PDF, stable source provenance, and idempotent re-ingest.
- `recamem/identity`: package, CLI, profile home, deprecation shim, migration from `droids/`, and the offline-by-default contract.
- `evaluation/recamem-benchmarks`: LoCoMo, LongMemEval-S and DMR runners, download-once cache, offline fixture fallback, and provenance-stamped reports.
- `verification/recamem-ci`: offline CI gate, regression tests for each defect, and the rule that real-model gates are reported as blocked rather than passed.

### Modified Capabilities

- `adaptation/memory-forgetting`: power-law accessibility, enforced `K_max`, consolidate-then-forget, soft `retired` tier, no hard DELETE. Resolves the contradictory wording of the decay exponent.
- `memory/sleep-consolidation`: replay with a validation gate, rollback on regression, single-writer lock.
- `droid/onnx-anchor`: fail-fast real anchor, explicit mock only, anchor mode reported by `doctor`.
- `model/gguf-backend`: load failure raises a named error; no silent mock substitution.
- `droid/profile-management`: `~/.recamem/agents/`, migration from `droids/`, schema version, adapter type read from the recorded config value.
- `droid/conversational-teaching`: `teach` uses the abbreviation-safe segmentation; the learning rate applied to every plastic update is the one recorded in the train log.
- `droid/memory-console`: Reset Memory calls the existing clear path; anchor mode and engine shown; rebrand.
- `memory/adaptive-retrieval`: strategies route to the two-channel retrieval path; policy persisted under the recamem profile.
- `knowledge-store`: single-writer lock for SQLite access; schema versioning; `retired` tier.
- `cli`: `recamem` subcommands (`chat`, `teach`, `ingest`, `recall`, `decide`, `serve`, `mcp`, `doctor`, `bench`, `migrate`, `package`, `rtu`) with the existing exit-code contract.
- `mcp-server`: `recamem_*` tool names with the old names as aliases; `tools/list` matches the documented list.
- `portable-brain`: the `.droid` archive keeps its extension, ZIP layout and SHA-256 contract; the manifest gains a format version, and older archives remain importable.
- `evaluation/lifelong-benchmarks`: adds backward-transfer metric.

## Impact

- **Code:** `src/` becomes `recamem/`. Modules named in the plan (`engine.py`, `plasticity.py`, `retrieval.py`, `consolidation.py`, `forgetting.py`, `episodic.py`, `anchor.py`, `backends.py`, `bench/`) replace parts of `src/model/droid.py`, `rtu.py`, `delta_memory.py`, `knowledge_store.py`, `onnx_anchor.py`, `gguf_backend.py`, and `propositions.py`. `src/model/` remains as a deprecation shim for one release.
- **APIs:** CLI and MCP names change (aliases kept). `DroidEngine` is kept as a shim.
- **Data:** existing `droids/` profiles and `.droid` archives are migrated, not deleted. Migration is idempotent and writes a backup first.
- **Dependencies:** `pyproject.toml` added. `pypdf` optional for PDF ingest. `psutil` declared in the base requirements if used by the core path.
- **Tests and CI:** offline gate runs without model weights. Real-anchor, GGUF and 8B tests are marked and reported blocked when weights are absent.
- **Docs:** README, AGENTS.md and `docs/testing.md` rewritten from verified behaviour.
- **Assumptions:** (a) "implement fully" means phases 0–9 of the plan with the blocked real-model gates reported honestly; (b) benchmark datasets are fetched from GitHub codeload or Hugging Face only where reachable, otherwise fixture-based and labelled; (c) the three archived 2026-10-21 changes are not reopened.
