# recamem-ci Specification (Delta)

## ADDED Requirements

### Requirement: CI does not mask verification results

No CI job SHALL append `|| true` to a test, gate, or end-to-end verification
command, except for an explicitly labelled informational step whose output is
not used as evidence. A verification command that fails SHALL fail its job.

#### Scenario: Masked smoke test

- **WHEN** a workflow runs `scripts/verify_remote_sensing_e2e.py || true`
- **THEN** the CI contract test SHALL fail and SHALL name the workflow and line
- **THEN** the offline gate SHALL NOT treat that job as verification evidence

#### Scenario: Informational step is labelled

- **WHEN** a native-index build is marked informational and non-gating
- **THEN** its result SHALL be published as informational only
- **THEN** no acceptance row SHALL reference it as passing evidence

### Requirement: Documentation artifacts are machine-valid

The offline suite SHALL validate that every committed `.ipynb` parses as JSON
and declares `nbformat`, that backticked plan/document references resolve to
files, that no tracked Markdown contains stray heredoc or shell-echo lines, and
that agent-facing docs contain no machine-specific absolute interpreter paths.

#### Scenario: Invalid notebook

- **WHEN** a notebook contains a trailing heredoc marker after the JSON object
- **THEN** the documentation validity test SHALL fail naming the file
- **THEN** the offline gate SHALL fail until the notebook parses

#### Scenario: Stale plan reference

- **WHEN** a plan document references a filename that does not exist
- **THEN** the test SHALL fail naming the document and the missing target

### Requirement: Current full-suite result is recorded

The offline tier SHALL have a fresh evidence record produced from a clean
install of the declared core requirements, with the exact passed/failed/skipped
counts, commit and environment. A historical result SHALL NOT be cited as the
current state when a newer run exists.

#### Scenario: Failure count note disagrees

- **WHEN** a change task claims pre-existing failures that the fresh run does not reproduce
- **THEN** the discrepancy SHALL be triaged and recorded
- **THEN** the fresh result SHALL be the one referenced by the ledger
