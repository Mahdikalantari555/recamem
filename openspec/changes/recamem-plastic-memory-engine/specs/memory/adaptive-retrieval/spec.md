# Spec Delta

## ADDED Requirements

### Requirement: Strategies execute the real recall path
Each retrieval strategy the policy can select SHALL execute the same two-channel recall path that the system uses for ordinary queries, with the channel weights taken from the learned scorer. A strategy that does not exist in code SHALL NOT be selectable.

#### Scenario: Selected strategy is the one executed
- **WHEN** the policy selects a strategy for a query
- **THEN** the recall executed for that query SHALL be recorded with that strategy name
- **THEN** feedback on the answer SHALL be attributed to that strategy and query type

#### Scenario: Policy persists under the recamem profile
- **WHEN** strategy probabilities change through feedback
- **THEN** they SHALL be saved in the profile directory and restored on load
