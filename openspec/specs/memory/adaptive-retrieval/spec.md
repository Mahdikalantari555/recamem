# adaptive-retrieval Specification

## Purpose

Adaptive retrieval policy: the system selects retrieval strategy (episodic-only, semantic-only, hybrid RRF, contextual-boosted) based on query type and user feedback history, optimizing for precision or recall as needed.

## Requirements

### Requirement: Query classification for strategy selection
The system SHALL classify queries into `definitional`, `specific_fact`, `temporal`, `relational` and `ambiguous` with documented deterministic precedence. Empty or underspecified queries SHALL reach the ambiguous class regardless of question punctuation. A question mark alone MUST NOT force the specific-fact class.

#### Scenario: Definitional query uses semantic store
- **WHEN** query is classified as definitional
- **THEN** the default strategy SHALL be semantic-primary

#### Scenario: Temporal query uses episodic with recency
- **WHEN** query is classified as temporal
- **THEN** the default strategy SHALL be episodic-recency-weighted

#### Scenario: Ambiguous query with punctuation
- **WHEN** a query is empty, whitespace-only, or consists only of an underspecified question such as "what?"
- **THEN** it SHALL be classified as ambiguous
- **THEN** the classification SHALL NOT change merely by removing its final question mark

#### Scenario: Classification precedence is reproducible
- **WHEN** a fixed labeled query fixture exercises multiple matching categories
- **THEN** precedence SHALL produce the same labels and initial strategies across repeated runs
- **THEN** exploration randomness SHALL affect strategy exploration rather than query classification

### Requirement: Feedback-driven strategy adaptation
The system SHALL track user feedback (approve/reject) per strategy and query type, shifting probability toward strategies with higher approval rates.

#### Scenario: Strategy probability updates on feedback
- **WHEN** user approves result from `hybrid_rrf` strategy
- **THEN** `hybrid_rrf` probability SHALL increase for that query type
- **WHEN** user rejects result from `episodic_only`
- **THEN** `episodic_only` probability SHALL decrease

### Requirement: Strategy exploration
The system SHALL occasionally (epsilon-greedy, $\epsilon=0.1$) try alternative strategies to discover better policies.

#### Scenario: Exploration tries alternatives
- **WHEN** $\epsilon$-greedy triggers
- **THEN** a non-default strategy SHALL be selected
- **THEN** outcome SHALL update strategy probabilities

### Requirement: Policy persistence
The system SHALL persist strategy probabilities per Droid profile and restore on load.

#### Scenario: Policy survives restart
- **WHEN** Droid profile is saved and reloaded
- **THEN** strategy probabilities SHALL match pre-save values

### Requirement: Strategies execute the real recall path
Each retrieval strategy the policy can select SHALL execute the same two-channel recall path that the system uses for ordinary queries, with the channel weights taken from the learned scorer. A strategy that does not exist in code SHALL NOT be selectable.

#### Scenario: Selected strategy is the one executed
- **WHEN** the policy selects a strategy for a query
- **THEN** the recall executed for that query SHALL be recorded with that strategy name
- **THEN** feedback on the answer SHALL be attributed to that strategy and query type

#### Scenario: Policy persists under the recamem profile
- **WHEN** strategy probabilities change through feedback
- **THEN** they SHALL be saved in the profile directory and restored on load

### Requirement: Strategy selection affects actual retrieval
Selected semantic, temporal, episodic and hybrid strategies SHALL execute the corresponding retrieval behavior and expose the executed strategy in diagnostics. Feedback and exploration state SHALL be isolated per profile and persist across restart.

#### Scenario: Semantic strategy executes semantic lookup
- **WHEN** a definitional query selects semantic-primary retrieval and a semantic fact exists
- **THEN** retrieval SHALL consult semantic knowledge rather than silently run only episodic lookup

#### Scenario: Feedback survives restart without leaking profiles
- **WHEN** a strategy is rewarded in profile A and A is saved and loaded
- **THEN** A SHALL retain the learned strategy preference
- **THEN** profile B SHALL retain its independent policy
