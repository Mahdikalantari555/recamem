# evaluation/edge-performance Specification

## Purpose

Makes the project's performance and size claims measurable: throughput, the speed
factor of the chunked path, parameter count, memory footprint, and held-out
perplexity, each reported with the provenance needed to reproduce the number and
never reported at all when it was not measured.

## Requirements

### Requirement: Throughput measurement

The system SHALL measure throughput in tokens per second for a named path, and
SHALL report the token count, the elapsed wall clock, the path that was measured,
the batch size, the sequence length, the thread count, and the hardware
description alongside the rate. A rate SHALL NOT be reported for a run that
measured nothing.

#### Scenario: Generation throughput is reported with its conditions

- **WHEN** the harness measures generation throughput for a checkpoint
- **THEN** it SHALL report `tokens_per_second` together with the token count, the elapsed wall clock, the batch size, the sequence length, the thread count, and the hardware
- **THEN** the report SHALL state whether the sequential or the chunked path was measured

#### Scenario: Training throughput is reported with its conditions

- **WHEN** the harness measures training throughput for a training configuration
- **THEN** it SHALL report `training_tokens_per_second` with the same conditions
- **THEN** it SHALL report whether the per-token or the chunked training path was measured

### Requirement: Parameter and memory accounting

The system SHALL report the parameter count of a model configuration and the
resident memory footprint of a loaded profile, so that a claim about the size of
the model and its runtime footprint is checkable.

#### Scenario: Parameter count matches the model

- **WHEN** the harness reports the parameter count for a configuration
- **THEN** the count SHALL equal the number of trainable elements in the constructed model
- **WHEN** a configuration is given for a target size, such as a dimension and a layer count
- **THEN** the harness SHALL report the resulting count

#### Scenario: Resident memory is reported for a loaded profile

- **WHEN** the harness measures a loaded profile
- **THEN** it SHALL report the resident set size in megabytes
- **THEN** it SHALL report which artifacts were loaded

### Requirement: Held-out perplexity

The system SHALL compute byte-level perplexity over a held-out split that is
disjoint from the training corpus, and SHALL report the corpus identity, the byte
count of the split, and the split rule with the number. It SHALL NOT report a
perplexity computed over data that was also used for training.

#### Scenario: Perplexity is reported with its split

- **WHEN** the harness computes perplexity for a checkpoint
- **THEN** it SHALL report the perplexity, the corpus identity, the number of evaluated bytes, and the split rule
- **THEN** it SHALL verify that the evaluated bytes are not present in the training corpus and SHALL refuse to report the number when they are

### Requirement: Reference path comparison

The benchmark SHALL report the chunked path and the sequential reference path side
by side with the ratio between them and the hardware they ran on. When the
run-to-run variance of the measurement exceeds the documented threshold, the
harness SHALL report the comparison as skipped with the variance as the reason,
and SHALL NOT lower the threshold to produce a passing number.

#### Scenario: Ratio is reported with the hardware

- **WHEN** both paths are timed on the same corpus at the same sequence length
- **THEN** the report SHALL contain both timings, the ratio, and the hardware
- **THEN** the ratio SHALL be computed from the same corpus and shapes

#### Scenario: Unstable measurement is skipped, not lowered

- **WHEN** the coefficient of variation across repeated runs exceeds the documented threshold
- **THEN** the comparison SHALL be reported as skipped with the measured variance
- **THEN** no ratio SHALL be reported for that run

### Requirement: Benchmark report provenance

Every benchmark run SHALL emit a machine-readable report containing the commit
hash, the exact command, the dependency versions, the configuration including
seeds, the hardware description, the measurements, and the outcome. The report
SHALL contain everything needed to reproduce the run.

#### Scenario: Report carries the run's provenance

- **WHEN** a benchmark run completes
- **THEN** the report SHALL contain the commit hash, the command, the dependency versions, the configuration including seeds, the hardware, the measurements, and the outcome
- **THEN** the report SHALL name the checkpoint hash when a checkpoint was measured

#### Scenario: Outcome is one of the documented values

- **WHEN** a run finishes
- **THEN** the recorded outcome SHALL be one of passed, failed, skipped, or blocked
- **WHEN** a measurement was skipped
- **THEN** the report SHALL name the reason it was skipped

### Requirement: No unmeasured numbers

The harness SHALL NOT emit a metric that was not measured. A measurement that
could not be taken SHALL be reported as skipped with a reason, and the report
SHALL NOT contain a value for it.

#### Scenario: Missing checkpoint skips the dependent measurements

- **WHEN** a benchmark is requested without a checkpoint
- **THEN** the throughput and perplexity measurements SHALL be reported as skipped with the missing checkpoint named
- **THEN** no throughput or perplexity value SHALL appear in the report
