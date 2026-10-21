# Tasks

## 1. Benchmark measurement primitives

- [x] 1.1 Implement the report writer in `src/model/benchmarks.py`: a JSON document
  carrying the schema version, the commit hash, the command, the dependency
  versions, the configuration including seeds, the hardware description, the
  measurements, and the outcome. Verify in `tests/test_benchmarks.py` that a
  written report parses back and contains every one of those fields.
- [x] 1.2 Implement the outcome vocabulary and the skip path: an outcome of
  passed, failed, skipped, or blocked, and a skipped measurement carrying a
  reason with no value. Verify in `tests/test_benchmarks.py` that a skipped
  measurement contributes a reason and no value to the report.
- [x] 1.3 Implement the throughput measurement: a warm-up run, repeated timed
  runs, and the median rate reported with the token count, the elapsed wall clock,
  the path name, the batch size, the sequence length, the thread count, and the
  hardware. Verify in `tests/test_benchmarks.py` that the reported rate equals the
  median of the recorded runs within tolerance and that every run is present in
  the report.
- [x] 1.4 Implement the parameter count measurement from a constructed model and
  the resident memory measurement from a loaded profile. Verify in
  `tests/test_benchmarks.py` that the reported parameter count equals the number
  of trainable elements and that the memory report names the loaded artifacts.
- [x] 1.5 Implement the held-out perplexity measurement with a document-level
  split selected by a hash of the document identifier, plus the containment check
  that samples held-out 64-byte windows and refuses to report when they appear in
  the training corpus. Verify in `tests/test_benchmarks.py` that the split is
  deterministic across runs, that the two sides do not overlap, and that a
  contaminated corpus produces a refused measurement rather than a number.

## 2. Reference path comparison

- [x] 2.1 Implement the side-by-side comparison of the per-token path and the
  chunked path over the same corpus at the same shapes, reporting both timings,
  the ratio, and the hardware. Verify in `tests/test_benchmarks.py` that both
  timings appear with the ratio computed from them.
- [x] 2.2 Implement the instability guard: when the coefficient of variation
  across the repeated runs exceeds the documented threshold, report the comparison
  as skipped with the measured variance and no ratio. Verify in
  `tests/test_benchmarks.py` that an injected high-variance measurement produces a
  skipped comparison and no ratio, and that the documented threshold is not
  lowered by the test.
- [x] 2.3 Assert that the existing 10x factor threshold at sequence length 512 is
  unchanged by this change. Verify in `tests/test_benchmarks.py` that the
  threshold constant used by the harness equals the value the
  `training/parallel-scan` requirement specifies.
- [x] 2.4 Close the gap in the existing speed-factor guard. It currently skips
  only when the run-to-run coefficient of variation exceeds the threshold, which
  does not catch a machine that is uniformly loaded: the whole run is slow, the
  variation is low, and the factor assertion fails anyway. Add a load check to
  the guard and to the harness, and skip with the measured load as the reason.
  Verify in `tests/test_benchmarks.py` that a measurement taken while the machine
  is busy is reported as skipped rather than as a failed factor, and verify that
  the existing 10x threshold is not lowered to make the machine pass.

## 3. Grammar probe format and loader

- [x] 3.1 Define the probe record and the line-delimited JSON format with the
  fields `id`, `phenomenon`, `context`, `correct`, and `incorrect`. Implement the
  loader in `src/model/probes.py`. Verify in `tests/test_probes.py` that a valid
  file loads and that each probe exposes its phenomenon and continuations.
- [x] 3.2 Reject a malformed line with its line number and the reason while
  loading the rest of the file. Verify in `tests/test_probes.py` that a file with
  one broken line loads the remaining probes and reports the broken line number.
- [x] 3.3 Enforce the documented phenomenon set — subject-verb agreement, active
  voice, passive voice, question formation — and reject a probe that names an
  unknown phenomenon. Verify in `tests/test_probes.py` that the four phenomena are
  accepted and an unknown one is rejected with the value named.
- [x] 3.4 Add the versioned probe data file `evals/grammar_probes.jsonl` with at
  least eight probes per phenomenon. Verify in `tests/test_probes.py` that the
  shipped file loads, that every phenomenon is covered, and that the file's hash is
  stable across runs.

## 4. Probe scoring

- [x] 4.1 Implement the scoring of a probe as the mean log-probability per byte of
  the correct continuation minus the best incorrect continuation, on the frozen
  model. Verify in `tests/test_probes.py` that a model with a known preference
  scores the probe it prefers, and that the margin is reported.
- [x] 4.2 Score a probe set and report the probe count, the correct count, and the
  accuracy, with the margin distribution. Verify in `tests/test_probes.py` that the
  accuracy equals the correct count over the probe count and that the margins are
  present in the report.
- [x] 4.3 Report accuracy per phenomenon alongside the aggregate. Verify in
  `tests/test_probes.py` that a set where one phenomenon scores zero still shows
  that zero in the per-phenomenon breakdown and that the aggregate is reported
  beside it rather than instead of it.
- [x] 4.4 Score without mutating the model: reset the recurrent state before each
  probe, run frozen, and hash the parameters and buffers before and after the run.
  Verify in `tests/test_probes.py` that the before and after hashes match and that
  the report carries them.
- [x] 4.5 Tie the score to its inputs: record the checkpoint hash and the probe
  file hash in the report. Verify in `tests/test_probes.py` that both hashes are
  present and that the same checkpoint and probe file produce the same score
  twice.

## 5. Runnable harness

- [x] 5.1 Implement `scripts/bench_edge.py` with argument parsing for the profile,
  the checkpoint, the corpus, the token count, the probe file, and the report
  path. Verify by running it with `--help` that every documented flag is accepted
  and that it exits 0.
- [x] 5.2 Orchestrate a full run: measure throughput on both paths, the parameter
  count, the memory footprint, the held-out perplexity, and the probe score, then
  write one report. Verify by running the harness on a tiny model and corpus that
  the report contains every measurement and that the process exits 0.
- [x] 5.3 Make missing inputs skip their measurements instead of failing the run.
  Verify by running the harness without a checkpoint that the dependent
  measurements are reported as skipped with the missing checkpoint named and that
  no value appears for them.
- [x] 5.4 Add the live smoke path: the harness SHALL NOT touch the network, and the
  probe suite SHALL run from the repository file alone. Verify that a run with the
  network disabled still completes and reports the probe score.

## 6. CLI surface

- [x] 6.1 Add the `bench` subcommand to `src/cli.py`, printing one `key=value` pair
  per line for each measured metric and the report path. Verify in
  `tests/test_cli.py` that the output parses on `=` and contains
  `tokens_per_second=`, `params=`, and `rss_mb=` for a run that measured them.
- [x] 6.2 Report an untaken measurement as skipped with a reason and print no value
  for it. Verify in `tests/test_cli.py` that a run without a checkpoint prints the
  skip and its reason and prints no throughput value.
- [x] 6.3 Expose the probe scoring through the same subcommand and print the
  per-phenomenon accuracies next to the aggregate. Verify in `tests/test_cli.py`
  that the per-phenomenon lines are present alongside the aggregate and that the
  aggregate is not printed alone.

## 7. Documentation

- [x] 7.1 Document the harness in `docs/benchmarks.md`: the commands, every metric,
  the report schema, the thresholds, the skip rules, and the hardware conditions
  that must accompany any quoted number. Verify that every command in the document
  runs as written.
- [x] 7.2 Update `README.md` with the benchmark and probe commands, with an
  explicit statement that the probe suite is a project-authored regression suite
  and not an independent benchmark, and with the rule that any quoted figure names
  the machine and the commit it came from. Verify that the documented commands run.
