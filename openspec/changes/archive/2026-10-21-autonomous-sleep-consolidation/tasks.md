# Tasks

## 1. Autonomous retrieval and sleep API

- [x] 1.1 Implement `SleepConsolidation` with `autonomous_retrieval()` driven only by the Droid's internal memory state and a decaying self-inhibition/refractory trace.
- [x] 1.2 Add bounded replay-step, time, confidence, cancellation, and duplicate-retrieval controls.
- [x] 1.3 Add `DroidEngine.sleep()` returning a structured report with status, duration, coverage, replay counts, pruning counts, and errors.
- [x] 1.4 Integrate sleep with existing `consolidate_memory()` without conflating fact pruning and parameter pruning.
- [x] 1.5 Add fixed correlated-pattern tests proving at least 95 of 100 stored patterns are recovered without passing an expected-pattern list to the replay method.

## 2. Memory Aware Synapses

- [x] 2.1 Implement MAS importance using the squared-L2 output objective and one backward pass per sampled batch.
- [x] 2.2 Restrict the eligible-parameter set to plastic-adapter weights; exclude the frozen anchor and buffers.
- [x] 2.3 Prune the lowest-importance 10% deterministically and save importance scores plus reversible masks in adapter state.
- [x] 2.4 Add tests for importance ranking, one-backward-pass behavior, exact prune count, checkpoint round trip, and no mutation outside the allowlist.
- [x] 2.5 Measure retrieval/retention quality before and after pruning; report rather than hide regressions.

## 3. Decreasing Backprop schedule

- [x] 3.1 Implement `DecreasingBackprop` with validated `base_lr`, `decay_factor`, and zero-based index using `lr_i = base_lr * decay_factor^i`.
- [x] 3.2 Persist and restore the schedule index and effective learning rate in Droid profile/checkpoint metadata.
- [x] 3.3 Add monotonicity, parameter-boundary, restart/resume, and optimizer-integration tests.
- [x] 3.4 Keep any ReLUDown activation transform isolated behind an explicit experimental option until its exact reference behavior is verified.

## 4. Command and worker lifecycle

- [x] 4.1 Add `/sleep` command dispatch and a CLI sleep action that invoke the same `DroidEngine.sleep()` API.
- [x] 4.2 Add an opt-in single auto-sleep worker per Droid with event-based stop/join and configurable idle/interval settings.
- [x] 4.3 Serialize sleep with teach/recall through a shared engine lock and test cancellation/profile shutdown.
- [x] 4.4 Ensure UI reruns and repeated start calls cannot create duplicate workers.

## 5. Validation

- [x] 5.1 Add lifecycle tests for an empty store, successful cycle, cancellation, concurrent teaching, and worker restart.
- [x] 5.2 Run the 100-correlated-pattern recovery benchmark and verify >=95% coverage without an external list.
- [x] 5.3 Run MAS pruning and DecreasingBackprop stability tests; document configurations and quality deltas.
- [x] 5.4 Run the full project test suite and checkpoint/profile compatibility tests.

## Archive status (2026-10-21)

Archived after implementation: all 22 tasks are checked with tests and evidence
in `reports/completion-evidence.json`.

Measured (commit `f31b00c`, mock embeddings, 384-D):

- 100 correlated patterns: all 100 recovered, coverage 1.0, 160 replay steps,
  `stop_reason=budget_exhausted`, expectation list scored only after replay.
- MAS pruning removes exactly `floor(0.1 * N)` elements, is reversible, and
  reports retrieval quality before/after with a warning on regression.
- Decreasing backprop advances once per committed cycle and survives profile
  reload; ReLUDown stays behind an explicit experimental flag.
