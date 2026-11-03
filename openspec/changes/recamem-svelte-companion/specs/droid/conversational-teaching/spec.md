# conversational-teaching Specification (Delta)

## ADDED Requirements

### Requirement: Companion chat preserves teaching semantics

The companion chat loop SHALL route each turn through the same engine path as
`POST /api/v1/inference`, preserving `learn:` teaching, the abbreviation-safe
segmenter, `/sleep` consolidation, `/fast` and `/slow` modes, and the
no-facts fallback. Each turn SHALL display the recalled evidence rows and the
answer path returned by the engine.

#### Scenario: Teaching in chat

- **WHEN** the user sends `learn: Dr. Mina Rostami leads the project.`
- **THEN** the fact SHALL be stored through `teach()` with the shared segmenter
- **THEN** the turn record SHALL show the number of facts stored and the answer path

#### Scenario: No recalled facts

- **WHEN** recall returns no facts and no generation backend is available
- **THEN** the turn SHALL show the no-information message
- **THEN** it SHALL NOT show a generated answer or an invented memory row

### Requirement: Online state is visible between turns

The companion SHALL re-read measured state after each turn so that teaching,
feedback and sleep performed in chat change the displayed pet state and memory
inspector within the same session.

#### Scenario: Sleep command

- **WHEN** the user sends `/sleep` in chat
- **THEN** the engine SHALL run one consolidation cycle
- **THEN** the next state response SHALL reflect the recorded sleep event and its measured outcome
