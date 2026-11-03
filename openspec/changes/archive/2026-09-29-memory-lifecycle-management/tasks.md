# Tasks

## 1. Memory Lifecycle Core Module

- [x] 1.1 Create `src/model/memory_lifecycle.py` with `SnapshotManager`, `MergeEngine`, `SemanticStore`, `ContradictionResolver`, `RetrievalPolicy`, `ContinuousStream` classes
- [x] 1.2 Implement `SnapshotManager`: `create_snapshot()`, `restore_snapshot()`, `list_snapshots()`, `branch_from()` using safetensors + knowledge.db
- [x] 1.3 Implement `MergeEngine`: `merge_memories(store_a, store_b, policy)`, `fuse_plastic(rtu_a, rtu_b, weight_a, weight_b)`
- [x] 1.4 Implement `ContradictionResolver`: `detect_contradictions(facts)`, `resolve(contradictions, policy)` reusing SVO logic
- [x] 1.5 Implement `SemanticStore` as `KnowledgeStore` extension: `distill()`, `retrieve_semantic()`, `incremental_distill()`
- [x] 1.6 Implement `RetrievalPolicy`: epsilon-greedy bandit per query type, `select_strategy()`, `update()`, `persist()`, `load()`
- [x] 1.7 Implement `ContinuousStream`: background thread, rate limiter, novelty filter, pause/resume/status
- [x] 1.8 Verify: unit tests in `tests/test_memory_lifecycle.py` cover all classes

## 2. KnowledgeStore Schema Extensions

- [x] 2.1 Add `version_id TEXT`, `feedback TEXT`, `needs_review INTEGER DEFAULT 0` columns to `facts` table
- [x] 2.2 Create `semantic_facts` table: `id, text, source_cluster_ids, confidence, created_at, access_count`
- [x] 2.3 Create `feedback_log` table: `id, fact_id, action, correction_text, timestamp, strategy_used`
- [x] 2.4 Add migration in `__init__` for new columns/tables
- [x] 2.5 Update `recall()` to log strategy used for feedback correlation
- [x] 2.6 Verify: `tests/test_knowledge_store.py` covers migrations and new tables

## 3. DroidEngine Integration — Versioning & Snapshots

- [x] 3.1 Add `DroidEngine.create_snapshot(version_id)` returning snapshot dict with metadata
- [x] 3.2 Add `DroidEngine.rollback_to(version_id)` restoring plastic state and knowledge store
- [x] 3.3 Add `DroidEngine.branch_from(version_id, new_version_id)` creating divergent branch
- [x] 3.4 Add `DroidEngine.list_snapshots()` returning version metadata list
- [x] 3.5 Update `save_profile()` to include snapshot version in config
- [x] 3.6 Verify: `tests/test_droid_engine.py` snapshot create/restore/branch round-trip

## 4. DroidEngine Integration — Merge & Distillation

- [x] 4.1 Add `DroidEngine.merge_memory(other_droid, policy="require_manual")` returning new version ID
- [x] 4.2 Add `DroidEngine.distill_to_semantic()` clustering episodic facts, resolving contradictions, writing semantic store
- [x] 4.3 Add `DroidEngine.retrieve_semantic(query)` for definitional queries
- [x] 4.4 Add `DroidEngine.get_contradictions()` returning unresolved conflicts
- [x] 4.5 Verify: merge test with two droids having slot clash; distillation reduces fact count

## 5. DroidEngine Integration — Feedback Learning

- [x] 5.1 Add `DroidEngine.approve(fact_id)` incrementing access_count, boosting confidence, rewarding strategy
- [x] 5.2 Add `DroidEngine.reject(fact_id)` flagging needs_review, penalizing strategy, trying alternative next query
- [x] 5.3 Add `DroidEngine.correct(old_fact_id, new_text)` superseding old, inserting new with high novelty
- [x] 5.4 Add `DroidEngine.get_feedback_log()` returning audit trail
- [x] 5.5 Verify: feedback loop test showing strategy probability shifts

## 6. DroidEngine Integration — Continuous Stream & Test-Time Training

- [x] 6.1 Add `DroidEngine.start_continuous_stream(source, max_fpm=10)` launching background ingestion
- [x] 6.2 Add `DroidEngine.pause_stream()`, `resume_stream()`, `get_stream_status()`
- [x] 6.3 Add test-time training budget: `ttt_steps_this_session`, `max_ttt_steps=100`, LR decay
- [x] 6.4 In `recall()`: if budget allows, call `memory.update_associative_memory(retrieved_fact_emb, surprise=confidence)`
- [x] 6.5 In `approve()`/`reject()`/`correct()`: if budget allows, call `memory.update_associative_memory()` with feedback-derived surprise
- [x] 6.6 Add profile config `test_time_training: true/false` to opt out
- [x] 6.7 Verify: stream test absorbs chat log; TTT budget respected; opt-out freezes weights

## 7. DroidManager Extensions

- [x] 7.1 Add `DroidManager.merge_droids(name_a, name_b, new_name, policy)` creating merged profile
- [x] 7.2 Add `DroidManager.export_version(name, version_id, out_path)` exporting snapshot artifact
- [x] 7.3 Add `DroidManager.import_version(name, snapshot_path)` restoring from exported snapshot
- [x] 7.4 Verify: manager tests cover merge, export, import

## 8. Tests & Validation

- [x] 8.1 Create `tests/test_memory_lifecycle.py` with unit tests for all `memory_lifecycle.py` classes
- [x] 8.2 Extend `tests/test_droid_engine.py` with integration tests for versioning, merge, feedback, stream, TTT
- [x] 8.3 Extend `tests/test_knowledge_store.py` for semantic store, feedback log, migrations
- [x] 8.4 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [x] 8.5 Benchmark: merge 2×10k fact droids < 5s; distillation 10k→2k facts < 10s; stream 100 msg/min