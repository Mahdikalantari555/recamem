# Implementation Plan: Droid × Raw Byte RTU Fusion

**Branch**: `001-droid-rtu-fusion` (intended; not yet created — no branch hook exists) | **Date**: 2026-10-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-droid-rtu-fusion/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Add a third engine mode, "fused", that keeps the existing droid recall exactly
as it is and gives the generative path a grounded, frozen byte-level RTU
component. The new backend (`ByteRtuBackend`) is a drop-in peer of `GgufBackend`
on the engine's single backend slot: it primes a reset component with the
retrieved facts (query last), samples bytes under `frozen=True` (no autograd, no
optimizer step), and stops on a stop-head threshold or a byte cap. Retrieval
gains a bounded associative bonus from `PlasticAssociativeRTU.retrieve()` that
can only re-rank candidates already past the dense gate. Profiles persist the
component as one extra file (`rtu.safetensors`) plus four
manifest fields, and loading a declared-but-absent checkpoint fails loudly. The
WebUI gains the third radio option and a badge that reports the answer path and
component status. One blocker fix is borrowed from plan 09 Phase 0 because the
E2E verification gate cannot run without it: `clear_knowledge()` does not exist
on `DroidEngine` today, and both `scripts/verify_remote_sensing_e2e.py:18` and
`app.py:198` call it.

## Technical Context

**Language/Version**: Python 3.12 (project env: `ai` conda, `torch` 2.14 CPU, `safetensors` 0.8)

**Primary Dependencies**: PyTorch (CPU), safetensors, onnxruntime (INT8 MiniLM anchor, HF-cached), Streamlit (WebUI), pytest. No new third-party dependencies.

**Storage**: Per-profile directory `droids/<name>/`: SQLite (`knowledge.db`), safetensors (`memory.safetensors`; new `rtu.safetensors`), JSON manifests (`config.json`, `train_log.json`, `snapshots.json`, `retrieval_policy.json`).

**Testing**: pytest over `tests/` (15 files, 57 test functions) + `scripts/verify_remote_sensing_e2e.py`; new file `tests/test_droid_rtu_fusion.py`.

**Target Platform**: Local single-machine, CPU-only (engine pins `torch.device("cpu")`, `src/model/droid.py:307`).

**Project Type**: Single project — library `src/model/`, Streamlit app `app.py`, reference CLIs `main.py` / `main_mlx.py`.

**Performance Goals**: One fused generative answer completes in under 10 s (E3 gate / SC-008); measured ceiling recorded in `plans/10_droid_rtu_fusion.md`. Recall latency unchanged.

**Constraints**: Generative component frozen at inference (no autograd, no optimizer step — `frozen=True` path only, `src/model/rtu.py:261-265`); fused profile `dim=384` matches anchor/`h_trace`; injection strategy pinned to `prompt`; existing 57 tests green with zero contract changes; standalone raw mode byte-identical; no rename, no packaging, no god-class refactor (plan 09 scope).

**Scale/Scope**: One new module (~120 lines), small hooks in `src/model/droid.py` (constructor guard, recall bonus, save/load, one entry point), `src/model/droid_manager.py` (one flag), `app.py` (radio option + badge), one new test file (6 tests), doc updates in `README.md`, `AGENTS.md`, and the plan file.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` is an unfilled template (placeholder principle
names, no ratified content), so there are no project-level constitutional
principles to check against. Gates derive from `AGENTS.md` and the recorded plan
decisions:

1. **Environment gate**: all runs use the `ai` conda env; the full pytest suite
   runs before completion.
2. **Scope gate**: no rename, no packaging, no class decomposition (plan 09
   boundary); the single borrowed item is the `clear_knowledge` entry point
   (research D8), which adds behavior and removes none.
3. **Frozen-inference gate**: every generation call routes through
   `frozen=True`; the non-frozen training path stays unreachable from
   inference; a checksum test proves it.
4. **No-silent-degradation gate**: the only new fallback class is
   "declared-but-absent checkpoint raises"; no new silent mock or random-init
   path is introduced at profile load.
5. **Repo-quirk gates**: safetensors saves avoid shared-tensor duplication
   (`Model.save` already saves via `named_parameters()`, `rtu.py:329`); any
   in-place parameter write uses `torch.no_grad()`; the HF cache is reused, not
   re-downloaded.

**Post-design re-check**: the design adds one module and additive hooks only.
No gate is violated; nothing must be justified in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/001-droid-rtu-fusion/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   └── backend-protocol.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/model/
├── byte_rtu_backend.py      # NEW: frozen backend, generate() = GgufBackend protocol
├── droid.py                 # hooks: __init__ backend guard (:294-381), recall bonus (:504-611),
│                            #       clear_memory/clear_knowledge (new), save/load (:1217/:1309)
├── droid_manager.py         # create_droid(..., byte_rtu=True) (:45)
├── rtu.py                   # unchanged: Model (frozen path, reset, save/load), retrieve()
├── memory_injection.py      # unchanged: inject_prompt, select_strategy
├── knowledge_store.py       # unchanged: clear_all() already exists (:755)
└── gguf_backend.py          # unchanged

app.py                       # radio adds fused option (:71-75); badge in memory console
tests/
└── test_droid_rtu_fusion.py # NEW: 6 tests per spec (grounding, silence, frozen, roundtrip, bonus, UI-loop parity)
plans/10_droid_rtu_fusion.md # mark implemented; record E3 latency measurement
README.md                    # fused branch in architecture diagram
AGENTS.md                    # frozen-at-inference rule + file ownership
```

**Structure Decision**: Keep the existing single-project layout. The change is
one new module plus additive hooks; no new top-level directories, no moves, no
renames. Delivery order (for `/speckit-tasks`): A) backend + plumbing, B)
persistence, C) recall bonus, D) WebUI + docs, E) verification gates.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. No entries.
