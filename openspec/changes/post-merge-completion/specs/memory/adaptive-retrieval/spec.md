# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Strategy selection affects actual retrieval
Selected semantic, temporal, episodic and hybrid strategies SHALL execute the corresponding retrieval behavior and expose the executed strategy in diagnostics. Feedback and exploration state SHALL be isolated per profile and persist across restart.

#### Scenario: Semantic strategy executes semantic lookup
- **WHEN** a definitional query selects semantic-primary retrieval and a semantic fact exists
- **THEN** retrieval SHALL consult semantic knowledge rather than silently run only episodic lookup

#### Scenario: Feedback survives restart without leaking profiles
- **WHEN** a strategy is rewarded in profile A and A is saved and loaded
- **THEN** A SHALL retain the learned strategy preference
- **THEN** profile B SHALL retain its independent policy
