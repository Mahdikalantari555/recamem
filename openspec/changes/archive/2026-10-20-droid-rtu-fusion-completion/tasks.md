# Tasks

Task IDs are the plan IDs (1.1–1.30). Group headers are descriptive so the validator
does not read them as a second numbering scheme.

## Spec Kit traceability

T001–T011 stay checked in `specs/001-droid-rtu-fusion/tasks.md` as the historical
record. T012–T027 fold into this change as follows:

| Spec Kit | This change | Note |
|---|---|---|
| T012 | 1.11 | bonus re-rank + cold-start test |
| T013 | 1.12 | bonus in `recall()` |
| T014 | 1.14, 1.15 | fused round trip; missing checkpoint raises |
| T015 | 1.13 | dual-backend construction raises, kept green |
| T016 | 1.13 | `create_droid(..., byte_rtu=True)` |
| T017 | 1.3, 1.4, 1.14 | checkpoint artifact and save order |
| T018 | 1.5, 1.6, 1.15, 1.16 | load path, named errors, isolation |
| T019 | 1.18 | third engine option |
| T020 | 1.19 | answer-path badge |
| T021 | 1.20 | manual console validation recorded here |
| T022 | 1.25 | README |
| T023 | 1.26 | AGENTS |
| T024 | 1.27 | plan 10 marked implemented |
| T025 | 1.28 | gate E1 |
| T026 | 1.29 | gate E2 (grounding, gated on `RTU_CHECKPOINT`) |
| T027 | 1.20 | latency ceiling recorded with the manual validation; the isolation gate is 1.30 |

## Manifest and persistence

- [x] 1.1 Add `engine` to the profile manifest. Values: `droid`, `byte_rtu`, `hybrid`. Write it on save, read it on load, and infer `byte_rtu` when a checkpoint is declared but the field is absent.
- [x] 1.2 Add `byte_rtu_checkpoint` and `fusion_mode` to the manifest. `byte_rtu_checkpoint` is the relative checkpoint filename; `fusion_mode` is `prompt_stuffing` or `off`. Keep writing the legacy `rtu_checkpoint` alias for one release and read `byte_rtu_checkpoint` → `rtu_checkpoint` → `None`.
- [x] 1.3 Write `rtu.safetensors` into `droids/<name>/` for byte-RTU and hybrid profiles. Save only through `ByteRtuBackend.model.save()`; the GGUF path and plain droid profiles write no checkpoint.
- [x] 1.4 Save in this order: `knowledge.json`, then `rtu.safetensors`, then `config.json` last. In this codebase the knowledge artifact is `knowledge.db` (the JSON writer is legacy), so the enforced rule is: every artifact the manifest names is written first, and the manifest is written last through `config.json.tmp` + `os.replace`.
- [x] 1.5 Raise a named error on load when the manifest names a checkpoint that is absent. Do not fall back to random weights. Use `MissingCheckpointError` and put the resolved path in the message.
- [x] 1.6 Keep every profile artifact inside `droids/<name>/`. Share no file between profiles. Reject an absolute checkpoint path or one containing `..` with `InvalidProfileError`, and test that two profiles never resolve to the same file.

## Backend conformance

- [x] 1.7 Add a public `reset()` to `ByteRtuBackend`. It clears the prompt buffer and the session cache. Call it on entry and in `finally` inside `generate()` so no primed state survives an answer.
- [x] 1.8 Remove the trailing `"\n"` that `ByteRtuBackend` appends to the prompt. The query stays last; the UI-loop parity test compares against the UI loop's own newline instead.
- [x] 1.9 Bound the backend output by `min(max_tokens, max_output_bytes)`. `max_tokens <= 0` returns `""`; `max_output_bytes` still caps a larger `max_tokens`.
- [x] 1.10 Add a test for each of the three items in 1.7–1.9. In `tests/test_droid_rtu_fusion.py`: `test_reset_clears_prompt_and_state`, `test_prompt_has_no_trailing_newline`, `test_output_bounded_by_min_of_tokens_and_bytes`.

## Recall ranking

- [x] 1.11 Compute the associative bonus inside `recall()` in `src/model/droid.py` from the associative memory scores. Do not hardcode boosts. Use `normalize(memory.retrieve(q_emb))` in a guarded block, apply `w_assoc * relu(dot(q_assoc, fact_emb))` to candidates that already passed the dense gate, and add the re-rank plus cold-start (empty `S`) tests from T012.
- [x] 1.12 Rank candidates by combined score: lexical + dense + associative bonus + recency + confidence. Raise the `associative`, `recency`, and `confidence` weights in the documented `RECALL_SCORE_WEIGHTS` table from the correctness change (0.15, 0.02, 0.05), keep the multiplicative contextual term, and assert the composition in the scorer test.

## Manager and profile round trip

- [x] 1.13 Add `DroidManager.create_droid(name, byte_rtu=True)`. Construct the backend with the profile's `rtu_dim/rtu_layers/rtu_spread`, attach it through `byte_rtu_backend=`, save the profile, and add `test_dual_backend_construction_raises` so the `ValueError` guard stays covered (T015).
- [x] 1.14 Persist the byte-RTU profile round trip through `DroidManager`. `create_droid(byte_rtu=True)` → teach → save → `get_droid` in a fresh manager must restore `engine`, the checkpoint field, and the backend; assert on parameters and buffers, never on sampled text.
- [x] 1.15 Load and save a fused profile with no loss of knowledge, RTU state, or config. Assert fact count, `step_count`, every `m.*` parameter, and `state.0`/`decaytrace.0`/`embedtrace.0` are equal after the round trip.
- [x] 1.16 Reject a profile whose `config.json` is missing or corrupt. Replace the `return False` path with `InvalidProfileError` naming the file and the reason; test missing, unreadable, and non-object JSON.
- [x] 1.17 Assert that a write in one profile does not change a second profile of the same engine. Write and save profile A, reload profile B, and assert B's state checksums and fact count are unchanged.

## Console

- [x] 1.18 Add the third engine option to the Streamlit engine selector in `app.py`. Label it "Droid + Raw Byte RTU (Fused)"; reuse the droid chat handler so fast-path answers stay byte-identical; leave the standalone raw-mode widgets and byte loop untouched.
- [x] 1.19 Add the answer-path badge. It names the path that produced the answer: recall, fused, or LLM. Read `engine._last_answer_path`, rename the T010 values at their source, keep a read-only alias for the old names, and show the engine, checkpoint filename, and anchor real-vs-mock state.
- [x] 1.20 Validate the fusion path manually in the WebUI. Record the result in the change `tasks.md`. Cover: three engine options, a taught remote-sensing paragraph, `/fast` unchanged versus plain droid mode, `/slow` returning grounded text or the fallback, and the badge naming the path. Record the measured `/slow` latency ceiling here (T027).
  - Recorded 2026-10-20 with `streamlit.testing.v1.AppTest` (headless; no human browser session): the selector renders the three options, the fused profile `manual-fused` loads with `engine=byte_rtu`, `/fast` returns the extractive recall text with path `recall` (0.15 s), `/slow` is produced by the frozen backend with path `fused` (0.36 s random-init; ≈0.6 s with the 384/8 trained checkpoint, 217-byte prompt, 256 output bytes), and the memory console caption shows `Engine: byte_rtu · Checkpoint: rtu.safetensors · Anchor: mock`. Latency ceiling (T027): **< 1 s on CPU**, far below the plan's 10 s gate.

## Training and checkpoint

- [x] 1.21 Add a training script that trains the byte-RTU model on the project corpus. Reuse the losses in `src/model/rtu.py`. Add `scripts/train_byte_rtu.py` with `--profile`, `--corpus`, `--steps`, `--seq-len`, `--dim`, `--layers`, `--spread`, `--seed`, and `--losses ce`; log the loss curve to the profile's `train_log.json`.
- [x] 1.22 Write the trained weights to `rtu.safetensors` in the target profile directory. Use `Model.save()` and update the manifest afterwards, in the save order from 1.4. Do not commit the weights; `droids/` is a generated artifact directory.
- [x] 1.23 Make `Model.load()` raise on a missing checkpoint. It currently returns the initialized model. Raise `MissingCheckpointError` for an empty or absent path, update the two production callers to check existence explicitly, and test both the raise and the explicit-random-init path.
- [x] 1.24 Keep the grounding assertion gated on the environment variable `RTU_CHECKPOINT`. The test skips when the variable is unset. The plumbing assertion (taught facts reach the primed bytes) always runs.

## Documentation and gates

- [x] 1.25 Update `README.md` with the fusion section and the engine options. Document `droid` / `byte_rtu` / `hybrid`, the `rtu.safetensors` location, the training script, the `RTU_CHECKPOINT` gate, and the expected quality ceiling of a small-corpus checkpoint.
- [x] 1.26 Update `AGENTS.md` with the byte-RTU engine path and the checkpoint location. Record the frozen-weights rule, the ownership of `src/model/byte_rtu_backend.py`, and that `droids/<name>/rtu.safetensors` is the only checkpoint location.
- [x] 1.27 Update `plans/10_droid_rtu_fusion.md`. Change the header from "plan (not implemented)" to "implemented". Tick each box as its task closes. Implementation record, latency, and quality ceiling added there.
- [x] 1.28 Gate E1: the profile round-trip test passes. Run `PYTHONPATH=. pytest tests/test_droid_rtu_fusion.py::test_fused_profile_roundtrip` and the full suite before ticking.
- [x] 1.29 Gate E2: the grounding test passes with `RTU_CHECKPOINT` set. Train the checkpoint per 1.21–1.22, export `RTU_CHECKPOINT=<profile>/rtu.safetensors`, and run `test_fused_chat_is_grounded`.
  - Passed 5/5 repeated runs with a 4000-step `dim=384 layers=8` checkpoint; the assertion checks a shared taught-fact substring (≥ 8 chars) because small-corpus output is not byte-perfect. The checkpoint is a build artifact and is not committed.
- [x] 1.30 Gate E3: the profile isolation test passes. Run `test_profiles_are_isolated` and record the two profile names used.
  - Passed; profiles `iso-a` and `iso-b` (same engine, distinct checkpoint and knowledge files, B unchanged after A is written).
