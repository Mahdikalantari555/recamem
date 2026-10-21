# memory-console Specification

## Purpose

Provides a persistent Memory Console for human-in-the-loop verification of LLM-synthesized responses and full CRUD management of the Droid's knowledge store facts.

## Requirements

### Requirement: Inline LLM Response Feedback

The system SHALL display an expandable feedback section after each LLM-synthesized
assistant response containing auto-extracted proposition checkboxes for selective
memory absorption. Every assistant answer SHALL carry an answer-path badge naming
how it was produced: `recall` for an extractive recall answer (including the
no-information fallback), `fused` for a frozen byte-RTU answer, `llm` for a GGUF or
external-LLM answer, or `researched` for an answer grounded in facts acquired from
an external source during the request.

#### Scenario: LLM response shows feedback section

- **WHEN** the assistant generates a response with source label indicating LLM synthesis (llm_grounded_memory or llm_general_knowledge)
- **THEN** an expandable "Memory Console" section SHALL appear below the response
- **THEN** the section SHALL contain checkboxes for each extracted proposition

#### Scenario: Proposition extraction from LLM response

- **WHEN** the feedback section is rendered
- **THEN** the system SHALL split the LLM response into sentences
- **THEN** sentences shorter than 30 characters SHALL be excluded
- **THEN** sentences matching conversational markers ("Based on...", "Here is...", "According to...", "In summary...", "To summarize...") SHALL be excluded
- **THEN** remaining sentences SHALL be presented as selectable propositions

#### Scenario: User accepts selected propositions

- **WHEN** user checks one or more proposition checkboxes and clicks "Accept Selected"
- **THEN** the system SHALL call `droid.teach()` with the concatenated selected propositions
- **THEN** the system SHALL display a confirmation with the number of facts absorbed
- **THEN** the feedback section SHALL collapse or show completion state

#### Scenario: Answer-path badge

- **WHEN** an assistant answer is rendered
- **THEN** a badge SHALL name the producing path as `recall`, `fused`, `llm`, or `researched`
- **THEN** the badge SHALL NOT be a hardcoded string that ignores `engine._last_answer_path`

#### Scenario: Researched answer shows its sources

- **WHEN** an answer whose answer path is `researched` is rendered
- **THEN** the badge SHALL show `researched`
- **THEN** the sources the answer was grounded in SHALL be listed with the answer

### Requirement: Memory Console Tab - Fact Browser
The system SHALL provide a dedicated tab for browsing, searching, and managing all facts in the knowledge store, and SHALL provide the memory-lifecycle actions in that tab: snapshot, rollback, branch, merge with a conflict policy, and continuous-stream start/stop. The tab SHALL display the active profile's engine and, when the engine uses a byte-RTU checkpoint, the checkpoint filename or an explicit "no checkpoint — extractive only" state. The tab SHALL display the anchor status from the anchor's `is_mock` flag.

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

#### Scenario: Engine and checkpoint status
- **WHEN** the active profile declares an `engine` of `droid`, `byte_rtu`, or `hybrid`
- **THEN** the console SHALL display that engine
- **THEN** for `byte_rtu` and `hybrid` it SHALL display the checkpoint filename, or state that no checkpoint is present
- **THEN** it SHALL display whether the anchor is real or mock, read from the flag rather than a constant

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

### Requirement: Memory Console Tab - LLM Teach Workflow
The system SHALL provide a workflow to paste raw text, distill it via LLM, review propositions, and teach selected ones to the Droid.

#### Scenario: LLM distillation of pasted text
- **WHEN** user pastes text into the LLM Teach input area and clicks "Distill"
- **WHEN** LLM Provider is enabled
- **THEN** the system SHALL call `teacher.distill_propositions()` on the text
- **THEN** the resulting propositions SHALL be displayed with checkboxes

#### Scenario: Teach selected distilled propositions
- **WHEN** user selects propositions from distillation results and clicks "Teach Selected"
- **THEN** the system SHALL call `droid.teach()` with the selected propositions
- **THEN** the system SHALL display absorption confirmation with fact count and memory norm

#### Scenario: Fallback without LLM Provider
- **WHEN** LLM Provider is not enabled
- **THEN** the system SHALL use local sentence splitting as fallback for proposition extraction
- **THEN** user can still review and teach the locally-split propositions
