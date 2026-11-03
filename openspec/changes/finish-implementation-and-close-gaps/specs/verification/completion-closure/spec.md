# completion-closure Specification

## Purpose

Defines the single auditable inventory of remaining work and the rules by which
a gap may be declared closed, so that "the implementation is finished" is a
measured statement rather than a collection of checked boxes.

## ADDED Requirements

### Requirement: Single machine-readable gap inventory

The project SHALL maintain `reports/completion-gaps.json` listing every known
unfinished item: unresolved ledger rows, unchecked tasks in active changes,
open historical tasks, unmet measured targets, open benchmark questions and
documentation defects. Each row SHALL carry an id, a source reference, an owner,
the applicable verification tier, the verification command, the artifact that
must exist, the test that must pass and a status.

#### Scenario: Gap is traceable to a source

- **WHEN** an inventory row is created
- **THEN** its `source` SHALL name a ledger row, a change task, a doc claim or a measured target
- **THEN** duplicate ids and rows without an owner SHALL be rejected by the guard

#### Scenario: Inventory is complete against the ledger

- **WHEN** `scripts/check_completion_gaps.py` runs
- **THEN** it SHALL compare the inventory against `--require-complete` unresolved rows, active change tasks and published target misses
- **THEN** a missing row SHALL fail the guard with the missing source named

### Requirement: Evidence-backed closure

A gap SHALL be closed only when its implementation exists, its named test passes,
its artifact exists in the repository and an evidence record with commit,
command, environment, configuration and outcome references it. Editing a status
field alone SHALL NOT close a gap.

#### Scenario: Status edited without evidence

- **WHEN** a row is marked `closed` but its artifact, test or evidence record is missing
- **THEN** the guard SHALL fail and SHALL name the missing element

#### Scenario: Genuine closure

- **WHEN** a row has a passing test, a present artifact and a passing evidence record
- **THEN** the guard SHALL accept the row
- **THEN** the guard SHALL report the number of open rows and exit 0 only when that number is zero

### Requirement: Gates are not weakened to obtain completion

The release guard, the offline gate, the benchmark outcome vocabulary and the
`blocked` semantics SHALL NOT be modified to accept fewer checks, fewer tiers or
unmeasured results. A diff that changes these guards' acceptance conditions
SHALL be rejected by the completion review.

#### Scenario: Threshold lowered in code

- **WHEN** a change lowers a tier requirement or removes an unresolved row without a replacement
- **THEN** the completion review SHALL fail
- **THEN** the row SHALL remain visible until implementation and evidence resolve it

#### Scenario: Blocked tier stays blocked

- **WHEN** a real-model or grounding artifact cannot be obtained
- **THEN** the tier SHALL be recorded as `blocked` with the missing artifact
- **THEN** completion SHALL NOT be declared and no release claim SHALL be made

### Requirement: Ordered acceptance tiers

Closure SHALL proceed in tier order: offline, concurrency-migration, benchmark,
grounding, real-model. A later tier SHALL NOT be recorded as passing on the
basis of an earlier tier's result.

#### Scenario: Mock anchor offline pass

- **WHEN** the offline suite passes with an explicitly selected mock anchor
- **THEN** only the `offline` tier SHALL pass
- **THEN** `real-model` and `grounding` SHALL remain unverified until their own artifacts are measured
