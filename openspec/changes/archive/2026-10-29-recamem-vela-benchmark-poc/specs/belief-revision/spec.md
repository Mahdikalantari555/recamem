# belief-revision Specification (Delta)

## ADDED Requirements

### Requirement: Policy-update supersession
When a taught proposition explicitly marks itself as a policy update or states that an earlier rule is superseded, and it shares the same soil code and growth phase with a prior active rule, the system SHALL mark the prior rule as superseded while retaining it in the audit trail.

#### Scenario: Amber phase 2 update
- **WHEN** the system has stored "For soil code Amber in phase 2, irrigation is required when moisture index below 36"
- **WHEN** it later teaches a proposition containing "policy update" and "Amber in phase 2" with a new threshold
- **THEN** the prior Amber phase 2 rule SHALL be marked superseded
- **THEN** the prior rule SHALL remain in the audit trail with superseded_by set

#### Scenario: Unrelated rule preserved
- **WHEN** a policy update targets soil code Amber phase 2
- **THEN** rules for other soil codes or phases SHALL NOT be superseded
