# Spec Delta

## MODIFIED Requirements

### Requirement: Inline LLM Response Feedback
The system SHALL display an expandable feedback section after each LLM-synthesized assistant response containing auto-extracted proposition checkboxes for selective memory absorption. Every assistant answer SHALL carry an answer-path badge naming how it was produced: `recall` for an extractive recall answer (including the no-information fallback), `fused` for a frozen byte-RTU answer, or `llm` for a GGUF or external-LLM answer.

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
- **THEN** a badge SHALL name the producing path as `recall`, `fused`, or `llm`
- **THEN** the badge SHALL NOT be a hardcoded string that ignores `engine._last_answer_path`

### Requirement: Memory Console Tab - Fact Browser
The system SHALL provide a dedicated tab for browsing, searching, and managing all facts in the knowledge store. The tab SHALL display the active profile's engine and, when the engine uses a byte-RTU checkpoint, the checkpoint filename or an explicit "no checkpoint — extractive only" state. The tab SHALL display the anchor status from the anchor's `is_mock` flag.

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
