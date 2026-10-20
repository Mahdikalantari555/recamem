# Tasks: Droid × Raw Byte RTU Fusion

**Input**: Design documents from `/specs/001-droid-rtu-fusion/`

> **Moved to OpenSpec.** T001–T011 stay here as the historical record. T012–T027 are
> tracked in the OpenSpec change `openspec/changes/droid-rtu-fusion-completion/` as
> tasks 1.11–1.30 (the mapping table is in that change's `tasks.md`). The T012–T027
> entries below stay unchecked on purpose; tick boxes in the OpenSpec change and in
> `plans/10_droid_rtu_fusion.md`, not here.

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Included. The feature specification and plan 10 explicitly name the test file `tests/test_droid_rtu_fusion.py` and its six tests, so test tasks are part of this list and are written before their implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Record the pre-change baseline so the zero-regression gate (SC-006) is measurable.

- [X] T001 Record the pre-change baseline: run `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests` in the repo root, note pass/fail counts and any pre-existing failures in `specs/001-droid-rtu-fusion/quickstart.md` (no code changes; read-only).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core engine hooks that MUST exist before any user story work.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Add `clear_memory()` to `DroidEngine` in `src/model/droid.py` — calls `self.knowledge.clear_all()` (already exists at `src/model/knowledge_store.py:755`), `self.memory.reset()`, and sets `self.step_count = 0`; add a `clear_knowledge = clear_memory` alias. Unblocks `scripts/verify_remote_sensing_e2e.py:18` and `app.py:198`, which both call `clear_knowledge()` on `DroidEngine` and currently raise `AttributeError` (research decision D8 — the only item borrowed from plan 09 Phase 0).
- [X] T003 Add the `byte_rtu_backend=` parameter to `DroidEngine.__init__` in `src/model/droid.py:294-381`: raise `ValueError` when both it and `gguf_backend` are supplied (FR-008); assign it to the single existing slot `self.gguf_backend` (contract `contracts/backend-protocol.md`); skip the gguf_config→`GgufBackend` auto-construction at `src/model/droid.py:368-379` when it is set; and default `self.gguf_config["injection_strategy"] = "prompt"` when it is set (research D3 — keeps `generate_with_memory` deterministic and avoids the `hidden` strategy's fixed random projection at `src/model/memory_injection.py:56`, which reseeds global RNG).

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 - Answers that are both grounded and fluent (Priority: P1) 🎯 MVP

**Goal**: Fused mode conditions a frozen byte-level RTU on retrieved taught facts (query last); empty retrieval returns the no-information fallback instead of free text; the extractive path stays byte-identical; the component is provably frozen.

**Independent Test**: Teach a paragraph, ask a `/slow` question on a fused engine, and check the answer contains taught content when a trained checkpoint is supplied (skips loudly otherwise); an empty fused engine returns the fallback; `/fast` output matches a non-fused engine byte-for-byte.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T004 [P] [US1] Write `test_byte_rtu_backend_matches_raw_ui_loop` in `tests/test_droid_rtu_fusion.py` — for the same primed state, backend `generate()` output equals the inline raw-UI loop at `app.py:364-382` (frozen calls, stop > 0.35, `errors="replace"` decode).
- [X] T005 [P] [US1] Write `test_no_free_generation_without_facts` in `tests/test_droid_rtu_fusion.py` — a fresh fused engine with empty knowledge returns the exact no-information fallback string on the `/slow` path and never generated text (FR-004).
- [X] T006 [P] [US1] Write `test_byte_rtu_backend_bounds_and_unicode` in `tests/test_droid_rtu_fusion.py` — empty prompt → `""`; non-ASCII prompt → valid `str`; priming with prompt bytes changes the output versus no priming (plan tasks A1.2/A1.3 — state carries context).
- [X] T007 [P] [US1] Write `test_fused_chat_is_grounded` in `tests/test_droid_rtu_fusion.py` — when a trained byte-RTU checkpoint is supplied via the `RTU_CHECKPOINT` environment variable, teach a remote-sensing paragraph and assert the `/slow` answer contains at least one taught fact substring (SC-001); when the variable is absent, call `pytest.skip()` with a message stating that no trained checkpoint exists in the repo (spec Assumptions record the random-init quality ceiling). Always assert the primed prompt/state plumbing regardless (the taught facts reach the priming bytes), so grounding is verified even without a trained checkpoint.
- [X] T011 [P] [US1] Write `test_rtu_weights_frozen_during_inference` in `tests/test_droid_rtu_fusion.py` — checksum every `m.*` parameter and every `state.*` / `decaytrace.*` / `embedtrace.*` buffer before and after several fused answers; identical (FR-003, SC-004 — proof of the frozen rule).

### Implementation for User Story 1

- [X] T008 [US1] Create `src/model/byte_rtu_backend.py` per plan §A1: `ByteRtuBackend(checkpoint="", dim=384, layers=8, spread=32, max_prompt_bytes=1024, max_output_bytes=256, stop_threshold=0.35)`; builds `Model(dim, layers, spread)` from `src/model/rtu.py`; calls `model.load(checkpoint)` only when the path exists (`Model.load` no-ops when missing, `src/model/rtu.py:356-357`); `generate(prompt, max_tokens=256, temperature=0.7, top_p=0.9)` resets state, primes all but the last byte with `frozen=True`, samples bytes until `stop > stop_threshold` or the output byte cap, and decodes with `errors="ignore"`; `temperature`/`top_p` accepted for protocol parity and ignored (the component owns its sampling temperature, `src/model/rtu.py:231`); `tokenize()` returns `list(text.encode("utf-8"))`.
- [X] T009 [US1] Add the free-generation guard in `generate_with_memory()` at `src/model/droid.py:1118-1121`: when recalled facts are empty and `isinstance(self.gguf_backend, ByteRtuBackend)`, return the existing no-information fallback text; extract the duplicated literal at `src/model/droid.py:676` and `:708` into one shared constant and reuse it at all sites (FR-004; decision 5 of plan 10).
- [X] T010 [US1] Record the answer path on the engine: set `self._last_answer_path` to `"extractive"`, `"fallback"`, or `"generative"` at the `chat()` return points in `src/model/droid.py:657-708` (default `"extractive"` before any chat), so the console badge can report the path per answer (FR-012); existing return strings unchanged.

**Checkpoint**: US1 is the MVP — teach + `/slow` (plumbing-verified, content-verified with a trained checkpoint), empty → fallback, fast path byte-identical, UI-loop parity, frozen proof.

---

## Phase 4: User Story 2 - Ranking uses plastic memory, not just surface text (Priority: P2)

**Goal**: Retrieval gains a bounded associative bonus from `PlasticAssociativeRTU.retrieve()` that only re-ranks candidates already past the dense gate; cold start is neutral.

**Independent Test**: Two facts with equal dense similarity re-order when the bonus is active; a fresh agent's ranking is identical with and without the bonus.

### Tests for User Story 2 ⚠️

- [ ] T012 [P] [US2] Write `test_associative_bonus_reranks_and_cold_start_is_neutral` in `tests/test_droid_rtu_fusion.py` — two candidates with equal dense similarity re-order when the plastic state favors one; an empty `S` (fresh agent) leaves the ranking identical (FR-010, FR-011, SC-009, SC-010).

### Implementation for User Story 2

- [ ] T013 [US2] Implement the A3 associative bonus in `recall()` at `src/model/droid.py:504-611`: after `q_emb` normalization (`:521`), compute `q_assoc = self.memory.retrieve(q_emb)` normalized, inside a guarded `try/except` that leaves `q_assoc = None` on any failure (bonus is optional, never fatal); per candidate that already passes the `sim >= 0.48` dense gate (`:564`), add `0.15 * max(0.0, dot(q_assoc, fact_emb))` to the final score next to the existing `contextual` term (`:569-580`); the dense gate stays authoritative (FR-010, decision 6 of plan 10; weight `0.15` is the starting point tuned against the E2E fixture).

**Checkpoint**: US2 works independently of US1 — it is testable on a plain (non-fused) engine.

---

## Phase 5: User Story 3 - A fused agent is an ordinary agent (Priority: P2)

**Goal**: Fused profiles save/load like any other: one extra file, four manifest fields, loud failure when a declared checkpoint is missing, and a single construction flag.

**Independent Test**: `create_droid(..., byte_rtu=True)` → save → load in a fresh manager → parameters and state buffers equal; deleting `rtu.safetensors` while config declares it raises; supplying both backends raises.

### Tests for User Story 3 ⚠️

- [ ] T014 [P] [US3] Write `test_fused_profile_roundtrip` in `tests/test_droid_rtu_fusion.py` — create via `DroidManager.create_droid(..., byte_rtu=True)`, teach, save, load into a fresh engine; assert every `m.*` parameter and the `state.0` / `decaytrace.0` / `embedtrace.0` buffers are equal; assert identical generation behavior via state tensors, never sampled text (research D10); delete `rtu.safetensors` from the profile dir and assert the next load raises a clear error rather than silently random-initializing (FR-006, FR-007, SC-005).
- [ ] T015 [P] [US3] Write `test_dual_backend_construction_raises` in `tests/test_droid_rtu_fusion.py` — `DroidEngine(gguf_backend=<any>, byte_rtu_backend=ByteRtuBackend())` raises `ValueError` (FR-008).

### Implementation for User Story 3

- [ ] T016 [US3] Add `byte_rtu: bool = False` to `DroidManager.create_droid()` in `src/model/droid_manager.py:45`: when true, construct `ByteRtuBackend` with default `dim=384, layers=8` and pass it to the `DroidEngine(...)` construction at `src/model/droid_manager.py:73` (FR-009); the strategy pin from T003 applies automatically.
- [ ] T017 [US3] Extend `save_profile()` at `src/model/droid.py:1217-1307`: when `isinstance(self.gguf_backend, ByteRtuBackend)`, call `self.gguf_backend.model.save(str(p / "rtu.safetensors"))` and add manifest fields `rtu_checkpoint` (`"rtu.safetensors"`), `injection_strategy` (`"prompt"`), `rtu_dim`, `rtu_layers` to the config written at `:1256-1257`; keep the save order knowledge → RTU checkpoint → `config.json` last; the GGUF path and non-fused profiles are untouched.
- [ ] T018 [US3] Extend `load_profile()` at `src/model/droid.py:1309-1417`: read `rtu_checkpoint` / `rtu_dim` / `rtu_layers` / `injection_strategy`; restore `self.gguf_config["injection_strategy"]` from the config (the `gguf_*`-prefix filter at `:1324` drops this key); after the existing memory load, re-attach `ByteRtuBackend(checkpoint=str(p / rtu_checkpoint), dim=rtu_dim, layers=rtu_layers)` when `rtu_checkpoint` is declared; raise a clear error (message names the missing path) when the declared file is absent — no silent random-init (FR-007); profiles without these keys load exactly as before.

**Checkpoint**: US3 round-trips fused profiles and fails loudly on a missing checkpoint.

---

## Phase 6: User Story 4 - Operators can see which path answered (Priority: P3)

**Goal**: The console gains a third engine option and a badge reporting the answer path and component status; the standalone raw mode is untouched.

**Independent Test**: Manual validation per `specs/001-droid-rtu-fusion/quickstart.md` scenario 5 (the repo has no automated Streamlit UI tests).

### Implementation for User Story 4

- [ ] T019 [US4] Add the third option `"Droid + Raw Byte RTU (Fused)"` to the engine-selection radio at `app.py:71-75`; fused mode reuses the existing droid chat handler unchanged (only the backend differs), so fast-path answers stay byte-identical (FR-005).
- [ ] T020 [US4] Add the badge in the sidebar / memory console of `app.py`: when a fused backend is attached, show RTU `dim`/`layers`, the checkpoint path (or "random init — no checkpoint"), and the per-answer path (`extractive` / `fallback` / `generative` from `engine._last_answer_path`, set by T010); the standalone raw-mode widgets (`app.py:206-215`) and byte loop (`app.py:364-382`) stay byte-identical (FR-012, FR-013).
- [ ] T021 [US4] Run the manual console validation in `specs/001-droid-rtu-fusion/quickstart.md` scenario 5 and record the result there (`streamlit run app.py`; three engine options; extractive answers identical to plain mode; badge reports path and component status).

**Checkpoint**: All user stories are independently functional.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T022 [P] Update `README.md`: add the fused branch to the architecture diagram and the expected quality ceiling (random-init component until a trained checkpoint exists).
- [ ] T023 Update `AGENTS.md`: record the rule "RTU weights are frozen at inference; all learning lives in plastic memory", plus the new file ownership (`src/model/byte_rtu_backend.py` owns the generative component; `droids/<name>/rtu.safetensors` is its checkpoint).
- [ ] T024 Update `plans/10_droid_rtu_fusion.md`: mark the phases implemented and record the measured E3 latency ceiling.
- [ ] T025 E1 verification gate: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests` — full suite green with zero changes to existing contracts (SC-006).
- [ ] T026 E2 verification gate: extend `scripts/verify_remote_sensing_e2e.py` additively (existing assertions untouched) so it also exercises a fused droid — attach the byte-RTU backend, confirm the extractive assertions still pass (FR-005) and the generative path returns a valid string or the fallback without raising — then run `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py` green (SC-007).
- [ ] T027 E3 verification gate: benchmark one fused `/slow` answer per `specs/001-droid-rtu-fusion/quickstart.md` scenario 4; if it exceeds 10 s, lower `max_prompt_bytes` in `src/model/byte_rtu_backend.py` and re-measure; record the measured ceiling in `plans/10_droid_rtu_fusion.md` (SC-008).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — run the baseline first.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational completion.
  - US1 (P1) → US3 (P2) → US4 (P3) form the main chain.
  - US2 (P2) is independent of US1 and can proceed in parallel with it (it is testable on a plain engine).
- **Polish (Phase 7)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational (needs T003). No dependency on other stories.
- **US2 (P2)**: Starts after Foundational. Independent of US1 — the associative bonus lives in `recall()` and is testable without any backend.
- **US3 (P2)**: Starts after US1 (persists the backend created in T008; needs T016's flag on top of it).
- **US4 (P3)**: Starts after US1 (badge reads `engine._last_answer_path` from T010 and the backend config); pairs naturally with US3's profile work but is independently testable.

### Within Each User Story

- Tests are written first and must fail before their implementation.
- Backend module (T008) before the engine guard (T009) and the path recorder (T010).
- US3: the manager flag (T016) and the save extension (T017) before the load extension (T018).

### Parallel Opportunities

- T001 runs alone first.
- T002 and T003 are both in `src/model/droid.py` but touch different functions; sequence them to avoid edit conflicts (T002 → T003).
- All US1 test tasks (T004–T007, T011) run in parallel — one new file, no conflicts.
- US1 and US2 can proceed in parallel (different files: `byte_rtu_backend.py` + `droid.py` guard vs the `recall()` bonus — coordinate the `droid.py` edit).
- T014 and T015 (US3 tests) run in parallel.
- T022 and T023 (docs) run in parallel.

---

## Parallel Example: User Story 1

```bash
# Write all US1 tests together (one new file tests/test_droid_rtu_fusion.py):
Task: "test_byte_rtu_backend_matches_raw_ui_loop (T004)"
Task: "test_no_free_generation_without_facts (T005)"
Task: "test_byte_rtu_backend_bounds_and_unicode (T006)"
Task: "test_fused_chat_is_grounded (T007)"
Task: "test_rtu_weights_frozen_during_inference (T011)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (baseline).
2. Complete Phase 2 (Foundational — clears the E2E blocker and the constructor guard).
3. Complete Phase 3 (US1): the backend, the fallback guard, the answer-path recorder, and the five US1 tests.
4. **STOP and VALIDATE**: US1 standalone — grounding plumbing, silence on empty retrieval, frozen proof, UI-loop parity.
5. The extractive behavior of the existing product is unchanged at this point.

### Incremental Delivery

1. Setup + Foundational → the E2E script stops raising `AttributeError`.
2. US1 → fused generative answers (MVP).
3. US2 → plastic-memory re-ranking (independent of US1).
4. US3 → fused profiles persist and fail loudly.
5. US4 → console visibility.
6. Polish → gates E1/E2/E3, docs, plan file marked implemented.

---

## Notes

- **[P] tasks** = different files, no dependencies.
- **[Story] label** maps the task to a spec user story for traceability.
- **Checkpoint-gated assertion**: no trained byte-RTU checkpoint exists in this repo, so T007's taught-substring assertion runs only when `RTU_CHECKPOINT` points at one and skips loudly otherwise. The plumbing assertion (facts reach priming) always runs. This matches the spec's Assumptions and the plan's documented quality ceiling.
- **T002 is borrowed from plan 09 Phase 0** (the `clear_knowledge` entry point) because the E2 verification gate (SC-007) cannot run without it. Nothing else from plan 09 is in scope.
- **Verification gates E1/E2/E3 are tasks, not afterthoughts** — the feature is not done until all three pass and the measured latency ceiling is written into `plans/10_droid_rtu_fusion.md`.
- Run every command with the `ai` conda environment and `PYTHONPATH=.` per `AGENTS.md`.
- Commit after each task or logical group (the user authorizes commits — ask before pushing or branching).
