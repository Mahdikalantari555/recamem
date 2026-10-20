# Tasks

Task IDs are the canonical IDs from the plan 09 Phase 0 slice; group headers are
descriptive so the validator does not read them as a second numbering scheme.

## Plastic update and dead code

- [ ] 2.1 Apply `effective_lr` (`src/model/droid.py:455`) to every plastic update. The value must change the optimizer step, not only the reported value at `src/model/droid.py:524`. Pass the rate into `DroidEngine.teach()`'s `update_associative_memory()` call and scale the momentum write by `effective_lr / base_lr`; assert the state delta differs between `step_count=0` and a later step.
- [ ] 2.2 Record the applied learning rate in the train log. Add `applied_lr` to the teach result and to the `train_log.json` entry, keep `effective_lr` unchanged, and test that both keys are present and equal at `step_count=0`.
- [ ] 2.3 Remove `retrieve()` (`src/model/rtu.py:626-655`) if it stays unreferenced. Call it from `recall()` if it carries the semantic signal. Make the call guarded (`None` on failure) and expose it as the `associative` scorer signal at weight `0.0` in this change; verify zero behavior change on the existing fixture before closing.
- [ ] 2.4 Delete the second `RTUMemoryBlock` (`src/model/droid.py:164-203`). Keep the import at `src/model/droid.py:209`. Confirm `RTUMemoryBlock` resolves to `src.model.rtu.RTUMemoryBlock` at runtime and that no test imports the deleted class.
- [ ] 2.5 Remove the duplicate `error_vec` assignments (`src/model/rtu.py:539`, `:544`). Keep one. Keep one comment explaining why the normalized residual is used, and assert `compute_surprise()` still returns the same `(error_vec, surprise)` for a fixed input.

## Capacity and ranking

- [ ] 2.6 Enforce `K_max` (`src/model/droid.py:346`) in the associative memory update. Evict the weakest key when capacity is full. Use the key projection norm as the strength measure and the existing attenuation operator for removal; never evict below the bound.
- [ ] 2.7 Report the eviction count in the memory health metrics. Expose `memory.eviction_count` and add `evictions` to `get_memory_health()`; test that the count is zero below the bound and one per eviction at the bound.
- [ ] 2.8 Replace the hardcoded recall boosts (`src/model/droid.py:542-607`) with a documented scorer. Each signal states its weight and its source. Implement `score_candidate(signals, weights)` with the `RECALL_SCORE_WEIGHTS` table (`dense`, `lexical`, `definitional`, `associative`, `recency`, `confidence`, `contextual`) and keep the `dense >= 0.48` gate authoritative.
- [ ] 2.9 Add a test that asserts the scorer returns the documented order for a fixed fixture. `tests/test_recall_scoring.py` covers: definitional beats higher-dense non-definitional, equal-signal ties keep insertion order, and a below-gate fact never outranks a gated fact even with maximal lexical overlap.

## Anchor honesty

- [ ] 2.10 Make the mock anchor (`src/model/onnx_anchor.py:103-110`, `src/model/onnx_anchor.py:241-244`) explicit. Log a warning and expose an `is_mock` flag. Set `is_mock = False` in `__init__`, set it to `True` on every fallback path, keep `use_mock` as an alias, and test both the flag and the warning.
- [ ] 2.11 Make the Streamlit badge (`app.py:114`) read the flag instead of a hardcoded string. Show the success badge only when the anchor is real; show a warning that names the fallback reason otherwise.

## Profile and store integrity

- [ ] 2.12 Persist the adapter type in the profile. The loader at `src/model/droid.py:1360-1372` reads a field that does not exist. Write the field on save and read it on load. Save writes `plastic_adapter` plus the constructor kwargs; load reads `plastic_adapter` first, falls back to class-name inference, and reconstructs only when the live adapter differs.
- [ ] 2.13 Add a single-writer lock to `KnowledgeStore` (`src/model/knowledge_store.py:62`). The daemon thread in the continuous stream must use the same lock. Add `self._write_lock = threading.RLock()` and acquire it in every write method; do not add a second lock in `ContinuousStream`.
- [ ] 2.14 Add a test that writes from the main thread and the stream thread at the same time. Assert the final fact count, that no exception escaped either thread, and that the dense cache returns a consistent matrix after the concurrent writes.

## Dependencies and documentation

- [ ] 2.15 Add `psutil` and `numpy` to `requirements.txt`. `scripts/verify_reproduction.py` imports `psutil`. Pin lower bounds consistent with the existing file and confirm a fresh install can import both.
- [ ] 2.16 Correct the README test count. The README states 43 tests. The suite has 68. Replace the number with the collected count and state the command used to collect it.
- [ ] 2.17 Correct every other README claim that the code does not support. Review at minimum: "zero catastrophic collapse" (no forgetting measurement exists), "sub-15ms CPU" (not measured here), and "22 MB on disk" (cache-dependent). Remove or qualify each unsupported claim; do not add new performance numbers without a recorded measurement.
