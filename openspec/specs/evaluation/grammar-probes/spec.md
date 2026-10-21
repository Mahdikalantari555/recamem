# evaluation/grammar-probes Specification

## Purpose

Measures grammatical competence with contrastive cloze probes: for each probe the
model scores the correct continuation against incorrect ones, and the result is
reported per grammar phenomenon so that one aggregate number cannot hide a
failure.

## Requirements

### Requirement: Contrastive cloze probe format

A grammar probe SHALL consist of a context and at least one incorrect
continuation, with exactly one correct continuation. Probes SHALL be stored as a
versioned data file in the repository, one probe per line, and each line SHALL
name the grammar phenomenon the probe targets.

#### Scenario: Probe file is loadable and well formed

- **WHEN** the probe set is loaded
- **THEN** every line SHALL parse into a context, one correct continuation, and at least one incorrect continuation
- **THEN** every probe SHALL name a phenomenon from the documented phenomenon set
- **WHEN** a line does not satisfy that shape
- **THEN** the loader SHALL reject it with the line number and the reason

### Requirement: Probe scoring by log-probability margin

The system SHALL score a probe by the model's mean log-probability per byte of
the correct continuation minus the highest mean log-probability per byte of any
incorrect continuation. It SHALL report the number of probes, the number correct,
and the accuracy, and SHALL score with the model frozen and no weight update
during scoring.

#### Scenario: Margin decides whether a probe is correct

- **WHEN** the correct continuation has the highest margin
- **THEN** the probe SHALL be counted as correct
- **WHEN** an incorrect continuation has a higher margin
- **THEN** the probe SHALL be counted as incorrect
- **WHEN** the model is frozen for scoring
- **THEN** no parameter SHALL change during the scoring run

#### Scenario: Accuracy is reported with the sample size

- **WHEN** the probe set is scored
- **THEN** the report SHALL state the number of probes, the number correct, and the accuracy
- **WHEN** no probe could be scored
- **THEN** the run SHALL be reported as skipped with the reason
- **THEN** no accuracy SHALL be reported

### Requirement: Per-phenomenon reporting

The system SHALL report accuracy per grammar phenomenon, not only in aggregate.
The probe set SHALL cover, at minimum, subject-verb agreement, active voice,
passive voice, and question formation.

#### Scenario: Each phenomenon is reported separately

- **WHEN** the probe set is scored
- **THEN** the report SHALL contain one accuracy per documented phenomenon
- **THEN** the aggregate accuracy SHALL be reported alongside the per-phenomenon values

#### Scenario: A failing phenomenon is visible

- **WHEN** one phenomenon scores zero and the others score high
- **THEN** the per-phenomenon report SHALL show the zero
- **THEN** the aggregate accuracy SHALL NOT be presented without the per-phenomenon breakdown

### Requirement: Deterministic and hermetic evaluation

The probe set SHALL require no network access and SHALL be small enough to run in
the test suite. The same checkpoint and the same probe file SHALL produce the same
scores on the same machine, and the score SHALL be reported with the hash of the
checkpoint it was produced from.

#### Scenario: Same inputs give the same score

- **WHEN** the probe set is scored twice against the same checkpoint
- **THEN** the two runs SHALL report the same accuracy and the same per-phenomenon values

#### Scenario: Score is tied to a checkpoint

- **WHEN** a probe score is reported
- **THEN** the report SHALL contain the checkpoint hash
- **THEN** the report SHALL contain the hash of the probe file

#### Scenario: Probes run without a network

- **WHEN** the probe suite runs
- **THEN** no network request SHALL be made
- **THEN** the run SHALL complete with the probe set in the repository alone
