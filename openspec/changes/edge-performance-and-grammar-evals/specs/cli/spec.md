# Spec Delta

## ADDED Requirements

### Requirement: Benchmark subcommand

The system SHALL provide a `bench` subcommand that measures the documented
performance and evaluation metrics for a profile, a checkpoint, or a model
configuration, and prints them in the parseable output format. A measurement that
could not be taken SHALL be reported as skipped with a reason, and no value SHALL
be printed for it.

#### Scenario: Benchmark subcommand measures the documented metrics

- **WHEN** user runs `python -m src.cli bench --profile droids/droid-rs --tokens 200000`
- **THEN** the CLI SHALL measure the documented throughput, parameter count, and memory footprint
- **THEN** stdout SHALL contain lines such as `tokens_per_second=`, `params=`, and `rss_mb=`
- **WHEN** a requested measurement cannot be taken
- **THEN** the CLI SHALL report that measurement as skipped with a reason
- **THEN** the CLI SHALL NOT print a value for it

#### Scenario: Benchmark writes a report

- **WHEN** the `bench` subcommand completes
- **THEN** the CLI SHALL write a machine-readable report containing the commit hash, the command, the dependency versions, the configuration including seeds, the hardware, the measurements, and the outcome
- **THEN** the CLI SHALL print the report path to stdout

#### Scenario: Benchmark can score the grammar probes

- **WHEN** user runs `python -m src.cli bench --probes evals/grammar_probes.jsonl --checkpoint droids/droid-rs/rtu.safetensors`
- **THEN** the CLI SHALL score the probe set against that checkpoint
- **THEN** stdout SHALL contain the per-phenomenon accuracies
- **THEN** the aggregate accuracy SHALL NOT be printed without the per-phenomenon breakdown
