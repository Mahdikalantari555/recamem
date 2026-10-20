# Spec Delta

## MODIFIED Requirements

### Requirement: Memory Console Tab - Fact CRUD
The system SHALL support Create, Read, Update, Delete operations on knowledge store facts. Superseding a fact SHALL record either the id of the fact that replaced it or an explicit no-successor value; it SHALL NOT record a sentinel such as `0`. Every failed action SHALL be reported with the exception it raised rather than swallowed.

#### Scenario: Edit fact text
- **WHEN** user clicks "Edit" on a fact and modifies the text
- **THEN** the system SHALL update the fact text in the knowledge store
- **THEN** the fact's timestamp SHALL update to current time
- **THEN** the fact SHALL be re-embedded with the ONNX anchor for semantic search

#### Scenario: Supersede fact
- **WHEN** user clicks "Supersede" on a fact
- **THEN** the system SHALL mark the fact as superseded (superseded=true, valid_until=now)
- **THEN** the fact SHALL be excluded from future recall queries
- **THEN** `superseded_by` SHALL be the id of the inserted replacement fact, or `NULL` when the user marks it superseded without a replacement
- **THEN** `superseded_by` SHALL NOT be `0`

#### Scenario: Delete fact
- **WHEN** user clicks "Delete" on a fact and confirms
- **THEN** the system SHALL remove the fact from the knowledge store entirely

#### Scenario: Failures are named, not swallowed
- **WHEN** an edit, supersede, delete, or reset action raises
- **THEN** the console SHALL name the exception type and show its message
- **THEN** the console SHALL NOT swallow the error with a bare `except:` and report success or `N/A` without a reason

### Requirement: Memory Console Tab - Fact Browser
The system SHALL provide a dedicated tab for browsing, searching, and managing all facts in the knowledge store, and SHALL provide the memory-lifecycle actions in that tab: snapshot, rollback, branch, merge with a conflict policy, and continuous-stream start/stop.

#### Scenario: Browse all facts
- **WHEN** user opens the Memory Console tab
- **THEN** the system SHALL display a searchable, paginated list of all active facts
- **THEN** each fact row SHALL show: fact text, source, timestamp, novelty score, and superseded status

#### Scenario: Search facts
- **WHEN** user enters a search query
- **THEN** the system SHALL filter facts by text match (case-insensitive substring)
- **THEN** results SHALL update in real-time without page reload

#### Scenario: View fact details
- **WHEN** user clicks a fact row
- **THEN** a detail view SHALL show: full text, source, timestamp, step, novelty, superseded status, valid_until, superseded_by

#### Scenario: Snapshot, rollback, and branch
- **WHEN** user creates a snapshot with a version id
- **THEN** the snapshot SHALL be listed with its metadata
- **WHEN** user rolls back to a snapshot
- **THEN** the console SHALL require confirmation, snapshot the current state first, and restore the selected version
- **WHEN** user branches from a snapshot
- **THEN** the new version SHALL start from that snapshot's state without modifying the source version

#### Scenario: Merge with a conflict policy
- **WHEN** user selects two profiles and a policy of `keep_newest`, `keep_highest_novelty`, `keep_both_as_alternatives`, or `require_manual`
- **THEN** the console SHALL show the detected conflicts before merging
- **THEN** the merge SHALL write to a new profile name and SHALL NOT overwrite either source profile

#### Scenario: Continuous stream controls
- **WHEN** user starts the continuous stream
- **THEN** the console SHALL show its status and stats and SHALL offer pause, resume, and stop
- **THEN** a console rerun SHALL NOT start a second worker for the same profile
