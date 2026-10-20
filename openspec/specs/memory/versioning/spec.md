# versioning Specification

## Purpose

Memory versioning with full state snapshots, branching, and point-in-time rollback for the Droid's plastic state and knowledge store.

## Requirements

### Requirement: Snapshot creation with version metadata
The system SHALL create a complete memory snapshot containing the plastic RTU state (associative matrices, trace, decay), knowledge store checkpoint, and metadata (version ID, timestamp, step count, fact count, parent version).

#### Scenario: Snapshot captures full state
- **WHEN** `create_snapshot("v1.0")` is called
- **THEN** the snapshot SHALL include plastic state, knowledge DB, and metadata
- **THEN** the snapshot SHALL be serializable to disk

#### Scenario: Version metadata recorded
- **WHEN** a snapshot is created
- **THEN** metadata SHALL include `version_id`, `timestamp`, `step_count`, `fact_count`, `parent_version`

### Requirement: Snapshot listing and inspection
The system SHALL list all available snapshots with their metadata and allow inspecting snapshot contents without restoring.

#### Scenario: List snapshots
- **WHEN** `list_snapshots()` is called
- **THEN** it SHALL return a list of version IDs with metadata sorted by timestamp

### Requirement: Point-in-time rollback
The system SHALL restore the Droid to a prior snapshot version, replacing plastic state and knowledge store atomically. Rollback SHALL be available from the memory console, SHALL require explicit user confirmation, and SHALL snapshot the current state before restoring so the rollback is itself reversible.

#### Scenario: Rollback restores exact prior state
- **WHEN** `rollback_to("v1.0")` is called
- **THEN** plastic state SHALL match snapshot exactly
- **THEN** knowledge store SHALL match snapshot exactly
- **THEN** current version SHALL become child of rolled-back version (branch)

#### Scenario: Console rollback is reversible
- **WHEN** user rolls back from the console
- **THEN** the console SHALL create a snapshot of the current state first
- **THEN** the restored state SHALL match the selected snapshot
- **THEN** reverting the rollback SHALL be possible by restoring the pre-rollback snapshot

#### Scenario: Unknown version
- **WHEN** a rollback names a version that does not exist
- **THEN** the operation SHALL fail with a clear error and SHALL NOT change any state

### Requirement: Branching from snapshot
The system SHALL support creating a new branch from any snapshot, allowing divergent memory evolution. Branching SHALL be available from the memory console and SHALL leave the source version unchanged.

#### Scenario: Branch creates independent lineage
- **WHEN** `branch_from("v1.0", "experiment-a")` is called
- **THEN** new version `"experiment-a"` SHALL start from `"v1.0"` state
- **THEN** subsequent changes SHALL not affect `"v1.0"` lineage

#### Scenario: Console branch is independent
- **WHEN** user branches from a snapshot in the console
- **THEN** the new version SHALL appear in the snapshot list with its parent recorded
- **THEN** writing to the branch SHALL NOT change the source snapshot's state or fact count
