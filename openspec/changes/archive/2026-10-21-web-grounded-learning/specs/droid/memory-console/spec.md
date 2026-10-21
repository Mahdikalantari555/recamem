# Spec Delta

## MODIFIED Requirements

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
