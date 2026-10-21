# Proposal

## Why

The project's central claims are quantitative — throughput on a CPU, a parameter
count, a quality level reached on a given corpus — and today none of them can be
produced by running a command. `scripts/bench_parallel_scan.py` times one
implementation against one reference on one machine and prints two columns;
`benchmark.py` evaluates CoLA and nothing else; there is no held-out split, no
grammar measurement, and no artifact that records what was measured, on what
hardware, with which commit. `docs/release-evidence.md` already defines an
evidence schema for exactly this purpose, but nothing produces the records.

The gap is not academic. Every external claim about this system — "it runs at N
tokens per second", "it understands active voice", "it matches GPT-1 on a test
suite" — is currently unfalsifiable because the measurement apparatus does not
exist. A harness that produces named, reproducible numbers is the difference
between a project and a claim about a project, and it is what lets a comparison
against anyone else's model be made honestly.

## What Changes

- Add a benchmark harness that measures the numbers the project actually argues
  about: generation and training throughput in tokens per second, the speed factor
  of the chunked path against the sequential reference, parameter count, resident
  memory, and byte-level perplexity on a held-out split.
- Require every measurement to carry its provenance — commit hash, exact command,
  dependency versions, configuration including seeds, hardware, and outcome — and
  forbid emitting a metric that was not measured.
- Add a grammar probe suite scored as a contrastive cloze task: for each probe the
  model scores the correct continuation against incorrect ones, and the margin is
  reported per grammar phenomenon rather than as one aggregate number.
- Report per-phenomenon accuracy so a single average cannot hide a failure in
  subject-verb agreement or passive voice.
- Make the whole suite hermetic and deterministic: versioned probe data in the
  repository, no network access, and a score that is tied to the exact checkpoint
  hash it was produced from.
- Expose the harness headlessly through a `bench` subcommand that prints the same
  parseable `key=value` output the `eval` subcommand already uses, and write a
  machine-readable report alongside it.

## Capabilities

### New Capabilities

- `evaluation/edge-performance`: reproducible measurement of the performance and
  size claims — throughput, the chunked-path speed factor, parameter count, memory
  footprint, and held-out perplexity — each with the provenance needed to reproduce
  it.
- `evaluation/grammar-probes`: a contrastive cloze probe suite that measures
  grammatical competence per phenomenon, deterministically, from a versioned data
  file.

### Modified Capabilities

- `cli`: a `bench` subcommand is added, with the parseable output contract the
  `eval` subcommand already uses and a written report artifact.

## Impact

- **Code areas:** new `src/model/benchmarks.py` (measurement primitives and report
  writer); new `scripts/bench_edge.py` (the runnable harness); new
  `src/model/probes.py` and `evals/grammar_probes.jsonl` (probe format, loader,
  scoring); `src/cli.py` (the `bench` subcommand); `src/model/rtu.py` (a read-only
  scoring entry point, if the frozen forward path cannot be reused as-is).
- **Tests:** new `tests/test_benchmarks.py` and `tests/test_probes.py`. Every test
  is hermetic and runs on a tiny model; the full-size numbers are produced by the
  harness, not by the test suite.
- **Dependencies:** none. Standard library for the report and the probe format;
  `torch` only for the measurements it already depends on.
- **Compatibility:** nothing existing changes behavior. The harness is additive.
- **Non-goals:** no new model architecture, no change to the recall scorer, no
  attempt to reproduce a third party's capability label, no GPU-specific code path.

## Risks

- A throughput number is meaningless without its conditions. Every reported figure
  carries the hardware, the thread count, the batch and sequence shapes, the
  checkpoint hash, and the command that produced it.
- The chunked-path speed factor is hardware-dependent. The ratio is reported with
  the hardware and is skipped with an explicit reason when run-to-run variance
  exceeds the documented threshold, rather than the threshold being lowered to make
  a machine pass.
- A grammar probe set written by the same person who wrote the model can flatter
  it. The probe data is versioned in the repository, the scoring rule is a margin
  against incorrect continuations rather than raw likelihood, and per-phenomenon
  accuracy is reported so the aggregate cannot hide a failure.
- Perplexity on a held-out split is only meaningful if the split is genuinely
  disjoint. The split rule and the corpus identity are recorded with the number,
  and the harness SHALL refuse to report a perplexity computed on data that was
  also used for training.
- An evaluation that fails is still evidence. The report records the outcome as
  passed, failed, skipped, or blocked, and a failed run SHALL NOT be discarded.
