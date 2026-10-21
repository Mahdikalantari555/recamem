# Design

## Context

See `proposal.md` for motivation. The constraints that shape this design:

- Two measurement paths already exist and disagree on purpose. `Model.forward`
  advances one byte at a time and is the only path that supports the online
  learning machinery: the dummy-carrier gradient hook in `src/model/rtu.py:366`
  injects `dL/dstate` into the embedding weights and decay parameters per token.
  `Model.forward_chunk` uses the associative scan and is vectorized but is a
  forward-only primitive. Both numbers matter and they are not interchangeable.
- The speed factor between those two paths is already specified in
  `openspec/specs/training/parallel-scan/spec.md`: at least 10x at sequence length
  512, with the hardware recorded, and skipped rather than faked when the
  measurement is unstable. This change does not renegotiate that threshold.
- `docs/release-evidence.md` already defines the shape of an evidence record:
  commit hash, command, environment and dependency versions, configuration
  including seeds, tier, outcome, skip reasons, raw-result locations, artifact
  checksums, and hardware details. The report format here is shaped to satisfy
  that schema so the two do not drift apart.
- The project has no model weights in the repository, so every real measurement
  runs against a checkpoint that the operator supplies.

## Goals / Non-Goals

**Goals:**

- One command produces every number the project argues about, in one report.
- Every number is reproducible: the report states how to reproduce it.
- A missing or unstable measurement is visible as such, never as an absent key or
  a lower bar.
- Grammar competence is visible per phenomenon, not as one average.

**Non-Goals:**

- Comparing against another system's published numbers. The harness measures this
  system; it does not adjudicate claims made elsewhere.
- Adopting a third party's capability label. The report names metrics and probe
  sets; it does not say "GPT-1 level".
- Changing the model, the training loop, or the recall path.
- Building a benchmark server or a leaderboard.

## Decisions

### D1. One report format for every measurement

A single JSON document carries the schema version, the commit hash, the command,
the dependency versions, the configuration including seeds, the hardware, the
measurements, and the outcome. The report is written next to the stdout summary,
and its path is printed.

**Alternatives considered.** Ad-hoc output per script was rejected: the numbers
would not be comparable and the provenance would be missing exactly where it
matters. A separate reporting service was rejected: the project runs on one
CPU-only machine.

### D2. Measurement primitives in the library, the runnable harness in `scripts/`

`src/model/benchmarks.py` holds the measurement functions and the report writer, so
they are importable and unit-testable. `scripts/bench_edge.py` holds argument
parsing and the run orchestration, mirroring `scripts/train_byte_rtu.py` and
`scripts/bench_parallel_scan.py`.

**Alternatives considered.** Putting everything in the script was rejected: the
CLI and the tests would then share a module that is awkward to import. Putting
everything in the library was rejected: a library module that parses arguments is
harder to reuse from the CLI.

### D3. Throughput is the median of repeated runs after a warm-up

Each measurement warms up once, then repeats the timed run, and reports the
median rate together with every individual run so a reader can judge the spread.

**Alternatives considered.** A single timed run was rejected: allocator and
import warm-up distort it, and the spread is invisible. The mean was rejected as
outlier-sensitive. The minimum was rejected: it flatters a loaded machine.

### D4. Both paths are measured and reported side by side

The harness reports the per-token path and the chunked path separately and names
which is which.

**Alternatives considered.** Reporting only the chunked path was rejected: it is
the flattering number, and the per-token path is the one that supports online
learning. Reporting only the per-token path was rejected: it hides the speed the
project already claims from the scan work.

### D5. The perplexity split is by document and is verified disjoint

The corpus is split at the document level, with the held-out fraction selected by
a hash of the document identifier so the split is independent of file order and
reproducible. Before reporting, the harness checks that a sample of held-out
64-byte windows does not appear in the training corpus, and refuses to report a
perplexity when it does.

**Alternatives considered.** A byte-offset split was rejected: it leaks context
across the boundary and can place near-duplicate text on both sides. Trusting the
split rule without a containment check was rejected: the check is cheap and the
claim depends on it.

### D6. The probe format is line-delimited JSON

Each line is `{"id", "phenomenon", "context", "correct", "incorrect": [...]}`. A
malformed line is rejected with its line number and the reason, and the rest of
the file still loads.

**Alternatives considered.** A JSON array was rejected: one bad line invalidates
the whole file. CSV was rejected: the fields contain prose and punctuation that
would need quoting rules.

### D7. Scoring is a per-byte margin against the best incorrect continuation

A probe is correct when the mean log-probability per byte of the correct
continuation exceeds that of every incorrect continuation. The margin is reported
per probe, and the report includes the margin distribution.

**Alternatives considered.** Raw summed log-probability was rejected: it rewards
short continuations. Comparing against a random incorrect continuation was
rejected: it makes the task easy and it measures nothing. Token accuracy was
rejected: the model is byte-level.

### D8. Scoring mutates nothing and proves it

The model is scored frozen, the recurrent state is reset before each probe, and
the harness records a hash of the model's parameters and buffers before and after
the run and reports that they are unchanged.

**Alternatives considered.** Not resetting the state was rejected: the first
probe would condition the rest. Asserting immutability only in a test was
rejected: a recorded before-and-after hash in the report is the evidence that
travels with the number.

### D9. The probe set is a regression suite and is labeled as one

The probe file ships in the repository and is self-authored. The report and the
README state that it is a project-authored regression suite measuring the named
phenomena, and do not describe the result as an independent benchmark or as a
comparison with any published system.

**Alternatives considered.** Adopting a public multiple-choice benchmark was
deferred: those sets assume token-level models with a vocabulary, and this model
is byte-level. The harness accepts an arbitrary probe file and an arbitrary
perplexity corpus, so a public set can be added later without changing the
format.

### D10. Unstable or missing measurements are skipped with a reason

A missing checkpoint, an absent corpus, or a coefficient of variation above the
documented threshold produces a `skipped` measurement with the reason named, and
no value.

**Alternatives considered.** Reporting the best of several runs was rejected: it
converts noise into a result. Lowering the threshold was rejected explicitly by
the existing `training/parallel-scan` requirement, and this change does not
weaken it.

### D11. Determinism is scoped honestly

Probe scoring is deterministic: the model is frozen, no sampling occurs, and the
score is tied to the checkpoint hash and the probe-file hash. Throughput is not
deterministic, and the report says so by presenting the distribution rather than a
single number.

**Alternatives considered.** Reporting a single throughput number with a seed was
rejected: the seed does not make a wall-clock measurement reproducible, and
presenting it as reproducible would be dishonest.

### D12. The report carries a schema version

A `schema_version` field makes a later format change detectable rather than
silently incompatible.

**Alternatives considered.** No version field was rejected: the report is a
durable artifact that will outlive this change.

## Risks / Trade-offs

- **[A throughput number gets quoted without its conditions]** → The report and
  the stdout summary put the hardware, the shapes, and the thread count next to
  the rate, and the README requires the machine to be named with any quoted
  figure.
- **[A self-authored probe set flatters the model]** → The suite is labeled a
  regression suite, per-phenomenon accuracy is reported, and the margin
  distribution is published so a confidently-wrong model looks different from an
  unsure one.
- **[The probe set drifts into the training corpus]** → The probe file lives in
  the repository and is small; the report records its hash. It is not part of any
  training corpus, and the scale-run change keeps it out of the pretraining
  corpus.
- **[The report schema and the release-evidence schema drift apart]** → The
  report is shaped to the documented evidence-record schema from the start, and
  the design records that dependency explicitly.
- **[The benchmark suite slows CI]** → The test suite exercises the measurement
  code on a tiny model; the full-size runs are harness invocations, not CI jobs.
- **[A failed or unflattering measurement gets dropped]** → The outcome
  vocabulary includes failed and blocked, the report is written either way, and
  the evidence convention is that a skipped or failed record is never removed to
  obtain a green result.

## Migration Plan

1. The harness and the library module are additive. No existing behavior changes.
2. The probe file is a new data file. No existing artifact changes.
3. The `bench` subcommand is new; the `eval` subcommand is untouched.
4. Rollback is deletion of the new files. No data migration, no schema change.

## Open Questions

- Should the probe suite be extended with Persian phenomena, and should a Persian
  corpus be used for the held-out perplexity? Deferred: the probe format and the
  perplexity corpus are language-agnostic, and adding a language is a data and
  licensing decision rather than a format change.
- Should a public perplexity benchmark such as a standard text corpus be added
  alongside the project corpus? Deferred: the harness already accepts an arbitrary
  corpus, and adding one introduces a download and a license step.
- Should the report be committed to the repository per release, or kept outside
  it? Deferred: the repository currently git-ignores generated artifacts, and the
  choice depends on how the release-evidence ledger is finally stored.
