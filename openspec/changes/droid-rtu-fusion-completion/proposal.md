# Proposal

## Why

Plan 10 (`plans/10_droid_rtu_fusion.md`) is partially landed. Spec Kit change
`specs/001-droid-rtu-fusion/` records T001–T011 as complete: the byte-RTU backend
exists (`src/model/byte_rtu_backend.py`), `DroidEngine` accepts `byte_rtu_backend=`,
the empty-recall guard returns the no-information fallback, and the answer path is
recorded on the engine. Three things are still missing, and each one is user-visible:

1. **Sixteen Spec Kit tasks remain (T012–T027).** The associative bonus is not in
   `recall()`; a fused profile is not round-tripped through `DroidManager`; the console
   has no third engine option and no answer-path badge; the docs still describe the
   fusion as unimplemented.
2. **The backend deviates from plan §A1 in three places.** `generate()` appends a
   trailing `"\n"` to the prompt that §A1 does not append; the sampling loop is bounded
   by `max_output_bytes` only and ignores the caller's `max_tokens`; and there is no
   public `reset()`, so a caller cannot clear the primed state or the prompt buffer
   between answers.
3. **There is no trained checkpoint.** `Model.load()` silently returns the random
   initialization when the file is missing (`src/model/rtu.py:355-357`), so a profile
   can declare a checkpoint, lose it, and keep answering — the exact silent degradation
   plan 09 Phase 2 exists to remove. The grounding test therefore skips today, and the
   fusion has never been validated against real weights.

## What Changes

- **Manifest.** Profiles gain `engine` (`droid` | `byte_rtu` | `hybrid`),
  `byte_rtu_checkpoint`, and `fusion_mode`. The manifest is written last, after every
  artifact it names, and the checkpoint lives inside `droids/<name>/` (1.1–1.4, 1.6).
- **Loud failure.** Loading a profile whose manifest declares a checkpoint that is
  absent raises a named error and never falls back to random weights; a missing or
  corrupt `config.json` is rejected with a named error instead of returning `False`
  (1.5, 1.16, 1.23).
- **Backend conformance.** Public `reset()`, no trailing newline, output bounded by
  `min(max_tokens, max_output_bytes)`, each covered by a test (1.7–1.10).
- **Ranking.** `recall()` computes the associative bonus from the plastic memory scores
  and ranks by the combined lexical + dense + associative + recency + confidence score
  (1.11, 1.12).
- **Profiles.** `DroidManager.create_droid(name, byte_rtu=True)`; the byte-RTU profile
  round-trips; a fused profile loads and saves with no loss of knowledge, RTU state, or
  config; two profiles of the same engine stay isolated (1.13–1.17).
- **Product surface.** A third engine option and an answer-path badge that reports
  `recall`, `fused`, or `llm`, validated manually and recorded (1.18–1.20).
- **Training.** `scripts/train_byte_rtu.py` trains the byte-RTU model on the project
  corpus with the existing losses, writes `rtu.safetensors` into the target profile,
  and the grounding test runs against it when `RTU_CHECKPOINT` is set (1.21, 1.22,
  1.24).
- **Documentation and gates.** README fusion section, AGENTS byte-RTU path and
  checkpoint location, plan 10 marked implemented with every box ticked, and gates
  E1/E2/E3 (1.25–1.30).

## Capabilities

### New Capabilities

None at the top level. One requirement is ADDED inside an existing capability:

- `model/dual-mode-generation` — ADDED "Fused RTU+Byte-RTU grounding path" (the fused
  answer path, its frozen-weight rule, and its grounding/fallback contract).

### Modified Capabilities

- `droid/profile-management` — MODIFIED "Named Droid Isolation" (engine field,
  checkpoint artifact, save order, named errors, strict isolation).
- `model/dual-mode-generation` — MODIFIED "Slow path — RTU+LLM synthesis" (the byte-RTU
  backend is a peer of the GGUF backend on the slow path).
- `model/memory-injection` — MODIFIED "Strategy selection" (a fused profile pins
  `prompt`; hidden/KV stubs are never selected).
- `droid/memory-console` — MODIFIED "Memory Console Tab - Fact Browser" and "Inline LLM
  Response Feedback" (engine/checkpoint status and the per-answer path badge).

## Impact

- **Code areas:** `src/model/droid.py`, `src/model/droid_manager.py`,
  `src/model/byte_rtu_backend.py`, `src/model/rtu.py`, `app.py`, `README.md`,
  `AGENTS.md`, `plans/10_droid_rtu_fusion.md`, `scripts/train_byte_rtu.py` (new),
  `tests/test_droid_rtu_fusion.py`.
- **Compatibility:** profiles without the new fields load exactly as before; `engine`
  is inferred (`byte_rtu` when a checkpoint is declared, else `droid`). `Model.load()`
  becomes strict, so every caller that previously relied on the silent no-op must check
  existence first — the backend and the profile loader are the only two callers.
- **Behavior change:** a byte-RTU profile that loses its checkpoint now fails to load.
  That is the intended contract, not a regression.
- **Ordering:** after `correctness-hardening` (the scorer is the base the associative
  bonus extends) and after `parallel-scan-training` (the E2 training run must finish in
  minutes, not hours). Before `surface-test-coverage` and
  `product-surface-completion`, which test and surface the new fields.
- **Spec Kit record:** `specs/001-droid-rtu-fusion/tasks.md` keeps T001–T011 as the
  historical record; T012–T027 fold into this change as tasks 1.11–1.30 (mapping is in
  the change `tasks.md`).

## Risks

- A strict `Model.load()` can break callers that passed an empty path and relied on
  random initialization. Mitigation: the backend only loads an existing path, and the
  profile loader raises a message that names the missing file.
- The associative bonus can re-order retrieval. Mitigation: only candidates already past
  the dense gate are re-ranked (plan 10 decision 6), the weight is documented, and the
  E2E script plus the fixture test guard the order.
- Training on the project corpus can produce a checkpoint whose output is worse than
  the random initialization. Mitigation: record the loss curve, keep the checkpoint out
  of the repo (profile directories are generated artifacts), and gate grounding on
  `RTU_CHECKPOINT` rather than pretending the repo ships weights.
- The manual WebUI validation (1.20) cannot be automated in this repository. It is a
  recorded checkpoint in `tasks.md`, not a passing test.
