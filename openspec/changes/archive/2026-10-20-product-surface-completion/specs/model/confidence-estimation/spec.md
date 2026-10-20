# Spec Delta

## MODIFIED Requirements

### Requirement: Confidence thresholds for action
The system SHALL define thresholds: `high` (>0.8) = auto-execute, `medium` (0.5–0.8) = show with citation, `low` (<0.5) = defer to user. The memory console SHALL display the tier beside every answer, together with the numeric confidence and the basis used to compute it.

#### Scenario: Low confidence triggers clarification
- **WHEN** fused confidence < 0.5
- **THEN** response SHALL include "I'm uncertain; here's what I found..." with citations

#### Scenario: Tier is shown beside every answer
- **WHEN** any assistant answer is rendered
- **THEN** the console SHALL show `high`, `medium`, or `low` from `confidence_tier`
- **THEN** the tier SHALL be consistent with the numeric confidence displayed next to it

#### Scenario: Basis is named
- **WHEN** the answer comes from the extractive or fallback path and no generation confidence exists
- **THEN** the tier SHALL be labelled `retrieval-only`
- **THEN** the console SHALL NOT present a fused generation confidence that was never computed

#### Scenario: Tier boundaries
- **WHEN** confidence equals 0.8
- **THEN** the tier SHALL be `medium`
- **WHEN** confidence equals 0.5
- **THEN** the tier SHALL be `medium`
- **WHEN** confidence is below 0.5
- **THEN** the tier SHALL be `low`
