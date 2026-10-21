# Spec Delta

## Purpose

Defines the corpus preparation, split discipline, resumability, reporting, and
acceptance criteria for an edge-scale pretraining run of the byte-level recurrent
model on CPU-only hardware.

## ADDED Requirements

### Requirement: Deterministic corpus preparation

The system SHALL prepare a byte-level training corpus from documents with a
deterministic document order and a document-level split into a training set and a
held-out set. The split SHALL be reproducible from the seed and the document
identifiers alone, and preparing the same corpus twice with the same seed SHALL
produce byte-identical output.

#### Scenario: Corpus preparation is reproducible

- **WHEN** the same corpus is prepared twice with the same seed
- **THEN** the two preparations SHALL be byte-identical
- **THEN** the document order SHALL not depend on filesystem order

#### Scenario: Split is document-level

- **WHEN** the corpus is split
- **THEN** no document SHALL appear in both the training set and the held-out set
- **THEN** the split SHALL be reproducible from the seed and the document identifiers

### Requirement: Evaluation data is excluded from training

The held-out split, the grammar probe file, and any corpus reserved for
benchmarking SHALL NOT appear in the training bytes. The run SHALL record the
identity of every excluded set alongside the result.

#### Scenario: Excluded sets are recorded

- **WHEN** a scale run produces its report
- **THEN** the report SHALL name the held-out split, the probe file, and any reserved benchmark corpus
- **THEN** the report SHALL state that they were excluded from the training bytes

#### Scenario: Contamination is detected

- **WHEN** a document designated as held-out appears in the training bytes
- **THEN** the run SHALL fail the contamination check
- **THEN** the report SHALL name the overlapping document

### Requirement: Resumable training run

A training run SHALL checkpoint its weights, its optimizer state, its step count,
and its corpus position at a documented interval, and SHALL be able to resume from
the last checkpoint and continue on the same trajectory as an uninterrupted run.

#### Scenario: Interrupted run resumes

- **WHEN** a run is interrupted and then resumed from its last checkpoint
- **THEN** the resumed run SHALL continue from the recorded corpus position
- **THEN** the resumed run SHALL NOT reprocess bytes before that position

#### Scenario: Checkpoint contents

- **WHEN** a checkpoint is written
- **THEN** it SHALL contain the weights, the optimizer state, the step count, and the corpus position
- **THEN** it SHALL be written atomically, so an interrupted write leaves the previous checkpoint intact

### Requirement: Scale run reporting

A scale run SHALL report the bytes consumed, the elapsed wall clock, the final
training loss, the held-out perplexity, the parameter count, the hardware, and the
outcome. The report SHALL carry the provenance of the benchmark report format: the
commit hash, the exact command, the dependency versions, the configuration
including seeds, and the corpus identity.

#### Scenario: Run report is complete

- **WHEN** a scale run completes
- **THEN** the report SHALL contain the bytes consumed, the wall clock, the final training loss, the held-out perplexity, the parameter count, the hardware, and the outcome
- **THEN** the report SHALL contain the commit hash, the command, the configuration including seeds, and the corpus identity

#### Scenario: Interrupted run still reports

- **WHEN** a scale run is interrupted before completion
- **THEN** a report SHALL still be written
- **THEN** the outcome SHALL be blocked and the bytes consumed SHALL be reported

### Requirement: Scale run acceptance criteria

A scale run SHALL be reported as passed only when every one of the following
holds: the consumed byte count reached the documented target, the parameter count
matches the documented configuration, the final logged training loss is below the
initial logged training loss, the held-out perplexity is finite and was computed on
the excluded split, and the run completed on the documented hardware class. When any
criterion does not hold, the outcome SHALL be failed or blocked and the report
SHALL name the criterion that was not met.

#### Scenario: Passing run

- **WHEN** a run meets every acceptance criterion
- **THEN** the outcome SHALL be passed
- **THEN** every criterion SHALL be individually recorded as met

#### Scenario: Failing run names the criterion

- **WHEN** a run does not meet one acceptance criterion
- **THEN** the outcome SHALL be failed or blocked
- **THEN** the report SHALL name the criterion that was not met
- **THEN** the report SHALL NOT present the run as passed
