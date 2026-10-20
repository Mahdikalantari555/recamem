# Proposal

## Why

Plan 09 (`plans/09_recamem_plan.md`) opens with a Phase 0 correctness pass. That pass
has not landed. Six defects in the current tree make the plastic-memory claims either
unverifiable or untrue:

1. **The learning-rate schedule is decorative.** `teach()` computes
   `effective_lr = base_lr / sqrt(1 + step_count / 20)` (`src/model/droid.py:455`),
   reports it in the result and the train log (`:524`), and never uses it. Every
   plastic write goes through `torch.no_grad()` (`src/model/rtu.py:571`, `:604`), so no
   optimizer step ever sees the value. The `droid/conversational-teaching` requirement
   "Automatic Hyperparameter Selection" is therefore unimplemented in the one place it
   claims to act.
2. **Dead and duplicated code hides the real path.** `PlasticAssociativeRTU.retrieve()`
   (`src/model/rtu.py:626`) is never called by `DroidEngine`; a second
   `RTUMemoryBlock` definition in `src/model/droid.py:164-203` shadows the imported one
   from `src/model/rtu.py`; `error_vec` is assigned twice in a row
   (`src/model/rtu.py:539`, `:544`), so the first value is thrown away. Reviewers cannot
   tell which definition is live.
3. **Capacity is reported, not enforced.** `K_max` (`src/model/droid.py:346`) feeds the
   saturation metric (`:796`) but nothing evicts at the bound. The associative state
   grows without a documented policy, and the health report cannot say whether
   anything was evicted.
4. **Recall ranking is hardcoded prose matching.** `recall()`
   (`src/model/droid.py:531-611`) adds fixed boosts: `0.25` for definitional text,
   `+0.2` for a "what is … acquisition … remote" string match, up to `+0.3` for term
   overlap and up to `+0.2` for "rare" overlap, then multiplies by
   `(1 + 0.2 * contextual)`. The weights have no named signals, no documented source,
   and no test that pins their order. Plan 09 Phase 6 and plan 10 A3 both need to
   extend this function, so it must stop being folklore first.
5. **Silent degradation is still the default.** `OnnxMiniLM` silently falls back to
   `_mock_embed` on any load or inference exception (`src/model/onnx_anchor.py:103-110`,
   `:241-244`), and `app.py:114` always shows "🟢 ONNX MiniLM (INT8, 22MB) Active". A
   user running the mock anchor is told the real one is active.
6. **Profile and store integrity are not asserted.** `save_profile()` writes
   `plastic_adapter` (`src/model/droid.py:1268-1281`) but `load_profile()` never reads
   it; it reads `adapter_kwargs` (`:1364`), a field nothing writes. `KnowledgeStore`
   opens SQLite with `check_same_thread=False` (`src/model/knowledge_store.py:62`) and
   has no writer lock, while `ContinuousStream` writes from a daemon thread.
   `requirements.txt` omits `psutil`, which `scripts/verify_reproduction.py:5` imports,
   and `README.md:122` advertises 43 tests while the suite collects 68.

## What Changes

- Apply `effective_lr` to every plastic update and record the applied rate in the train
  log (2.1, 2.2).
- Resolve the dead paths: call `retrieve()` from `recall()` if it carries the semantic
  signal, otherwise delete it (2.3); delete the shadowing `RTUMemoryBlock` (2.4); keep
  one `error_vec` assignment (2.5).
- Enforce `K_max` in the associative update with weakest-key eviction, and report the
  eviction count in memory health (2.6, 2.7).
- Replace the hardcoded recall boosts with a documented scorer whose every signal
  states a weight and a source, plus a fixture test that pins the documented order
  (2.8, 2.9).
- Make the mock anchor explicit: warn, expose `is_mock`, and drive the Streamlit badge
  from the flag (2.10, 2.11).
- Persist the adapter type as a written-and-read field (2.12).
- Add a single-writer lock shared by the main thread and the continuous-stream daemon,
  with a concurrent-write test (2.13, 2.14).
- Add the missing runtime dependencies and correct every README claim the code does not
  support (2.15–2.17).

No rename. No new capability. The plan 09 Phase 1 rename stays out of scope, so the
names `RTU`, `DroidEngine`, `DroidManager`, and `droids/` are unchanged.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `droid/conversational-teaching` — MODIFIED "Automatic Hyperparameter Selection"
  (the schedule must change the optimizer step, not only the report).
- `adaptation/plastic-memory-titans` — MODIFIED "Capacity bound enforcement"
  (documented eviction at `K_max`).
- `adaptation/memory-health` — MODIFIED "Retrieval quality metric"
  (health output reports capacity evictions alongside quality).
- `droid/profile-management` — MODIFIED "Named Droid Isolation"
  (adapter type is written on save and read on load).
- `droid/onnx-anchor` — MODIFIED "ONNX Inference Pipeline"
  (mock fallback is an explicit, observable state).
- `model/dual-mode-generation` — MODIFIED "Fast path — RTU factual recall"
  (documented, testable ranking scorer replaces hardcoded boosts).

## Impact

- **Code areas:** `src/model/droid.py`, `src/model/rtu.py`,
  `src/model/onnx_anchor.py`, `src/model/knowledge_store.py`,
  `src/model/memory_lifecycle.py` (continuous-stream writer), `app.py`,
  `requirements.txt`, `README.md`.
- **Tests:** new/extended coverage in `tests/test_droid_engine.py`,
  `tests/test_rtu.py`, `tests/test_onnx_anchor.py`, `tests/test_knowledge_store.py`,
  and a scorer-order fixture test.
- **Compatibility:** saved profiles keep loading. The adapter field is additive: a
  profile without it falls back to the current class-name inference. Ranked results
  keep their public fields (`similarity`, `contextual_alignment`,
  `original_similarity`); only the score composition changes.
- **Behavior change:** recall ordering can change where the old terms were dominating.
  The E2E fixture and the existing droid tests are the guard; any change to their
  expected top hit is a finding to report, not a test to edit.
- **Ordering:** this change lands before `droid-rtu-fusion-completion` because both
  edit `recall()` and the profile save/load path, and the fusion change must add its
  associative bonus on top of the documented scorer rather than on top of the
  hardcoded boosts.

## Risks

- Applying a learning rate that has never influenced an update can change plastic
  state trajectories. Scale the update conservatively, keep `base_lr` configurable,
  and assert the applied rate rather than a particular loss value.
- A new scorer can reorder retrieval in ways that break acceptance text. Pin the
  documented order with a synthetic fixture, not with real corpus facts, and run the
  remote-sensing E2E script before closing 2.9.
- `Model.load()` and profile loading are shared with the fusion change. This change
  touches only the adapter field; the missing-checkpoint error belongs to the fusion
  change so the two error paths do not diverge.
