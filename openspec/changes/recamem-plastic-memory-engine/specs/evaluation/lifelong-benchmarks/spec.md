# Spec Delta

## ADDED Requirements

### Requirement: Backward transfer after sequential teaching
The lifelong benchmark SHALL teach a sequence of domains and, after each step, re-evaluate every earlier domain. It SHALL report the backward-transfer change for each earlier domain and the mean memory loss across domains, using the same question set at every step.

#### Scenario: Earlier domain forgetting is measured
- **WHEN** a third domain has been taught after two earlier domains
- **THEN** the report SHALL state the score for each earlier domain before and after the third teaching step
- **THEN** the report SHALL state the mean memory loss across earlier domains

#### Scenario: Retired facts are not counted as lost
- **WHEN** an earlier domain's facts were moved to the retired tier and remain answerable from the audit interface
- **THEN** the report SHALL state the loss separately for active-tier recall and audit-tier recall
