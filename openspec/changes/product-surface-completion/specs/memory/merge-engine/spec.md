# Spec Delta

## MODIFIED Requirements

### Requirement: Memory merge with conflict detection
The system SHALL merge two Droid memories by unioning their knowledge stores and fusing plastic states, detecting conflicts where both memories contain propositions about the same subject-predicate with different objects. The memory console SHALL offer the merge action and SHALL display the detected conflicts before the merge is applied.

#### Scenario: Merge detects slot clashes
- **WHEN** Droid A has "capital of France is Paris" and Droid B has "capital of France is Lyon"
- **THEN** `merge_memories(A, B)` SHALL detect the `capital_of` slot clash
- **THEN** the conflict SHALL be reported with both propositions

#### Scenario: Console shows conflicts first
- **WHEN** user selects two profiles in the memory console
- **THEN** the console SHALL list the detected conflicts before offering the merge
- **THEN** the merge SHALL require confirmation and SHALL write to a new profile name

### Requirement: Configurable conflict resolution policies
The system SHALL resolve merge conflicts using a configurable policy: `keep_newest` (by timestamp), `keep_highest_novelty`, `keep_both_as_alternatives`, or `require_manual`. The memory console SHALL expose the policy as a selector and SHALL pass the selected policy value to the merge engine unchanged.

#### Scenario: keep_newest selects later timestamp
- **WHEN** merging with `policy="keep_newest"`
- **THEN** the proposition with later timestamp SHALL be retained
- **THEN** the other SHALL be marked superseded

#### Scenario: keep_highest_novelty selects more surprising fact
- **WHEN** merging with `policy="keep_highest_novelty"`
- **THEN** the proposition with higher novelty score SHALL be retained

#### Scenario: keep_both stores alternatives
- **WHEN** merging with `policy="keep_both_as_alternatives"`
- **THEN** both propositions SHALL be stored with `alternative_of` linking

#### Scenario: Policy selector round trip
- **WHEN** user selects a policy in the console and confirms the merge
- **THEN** the engine SHALL receive exactly one of the four documented policy values
- **THEN** an unknown policy SHALL be rejected with a clear error before any write
