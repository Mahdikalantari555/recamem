# companion-trainer Specification

## Purpose

Defines in-app training as a real, observable subprocess with provenance-backed
metrics and a strict definition of the "best" checkpoint.

## ADDED Requirements

### Requirement: Training jobs execute real work

A training job SHALL launch `scripts/train_lm.py` as a subprocess with explicit
arguments and environment. It SHALL record tokens seen, tok/s, loss, held-out
BPB, n-gram baselines, G0-G4 gate results and checkpoint paths parsed from
actual output. A job SHALL NOT report `done` or success without a subprocess
result.

#### Scenario: Job created

- **WHEN** `POST /api/v1/jobs` with `kind=train` is called
- **THEN** a real training process SHALL start and the job status SHALL reflect its state
- **THEN** progress SHALL be computed from tokens seen, not elapsed wall clock

#### Scenario: No fabricated progress

- **WHEN** the subprocess emits no metrics yet
- **THEN** the job SHALL report no progress value and no loss value
- **THEN** it SHALL NOT synthesize intermediate points for the chart

### Requirement: Cancellation and resume are real

Cancellation SHALL terminate the subprocess and leave no orphan process. Resume
SHALL continue from a checkpoint whose vocab size, architecture and seed match
the requested configuration; a mismatch SHALL be rejected before training
starts.

#### Scenario: Cancel

- **WHEN** `POST /api/v1/jobs/{id}/cancel` is called on a running job
- **THEN** the subprocess SHALL be terminated and the job SHALL be recorded as `cancelled`
- **THEN** a subsequent health check SHALL find no orphan training process

#### Scenario: Mismatched resume

- **WHEN** a resume checkpoint has a different vocab size
- **THEN** the run SHALL fail with a named reason and SHALL NOT start

### Requirement: Best checkpoint is provenance-backed

A checkpoint SHALL be eligible for selection only when it has a sha256 manifest,
a matching config, a recorded seed, dataset identity, commit and command, and a
measured held-out BPB with G0-G2 passed. The best checkpoint SHALL be the
eligible one with the lowest held-out BPB, with G3/G4 shown when measured. When
no checkpoint is eligible, the API SHALL return `none` with the reason.

#### Scenario: Random-init weights offered

- **WHEN** a checkpoint has no manifest or no measured BPB
- **THEN** it SHALL be marked ineligible and SHALL NOT be selectable as best

#### Scenario: Two eligible checkpoints

- **WHEN** two eligible checkpoints have different measured BPB values
- **THEN** the one with the lower BPB SHALL be returned with its provenance
- **THEN** the selection SHALL be reproducible from the recorded metrics

### Requirement: Training view reports gates honestly

The Training view SHALL display n-gram baselines, G0-G4 outcomes and generated
samples exactly as reported, and SHALL display `blocked` with the missing
artifact when a gate could not be evaluated.

#### Scenario: Gate not evaluated

- **WHEN** a run report lacks a G3 result
- **THEN** the UI SHALL show G3 as not evaluated rather than passed
