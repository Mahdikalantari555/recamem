# Autonomous incremental learning

## ADDED Requirements

### Requirement: Automatic CPU learning configuration
The system SHALL select a compatible model, tokenizer and bounded CPU training plan without requiring the user to choose architecture or token counts, and SHALL report when prerequisites prevent selection.

#### Scenario: No suitable model
- **WHEN** learning is enabled but no compatible model and data are available
- **THEN** the system reports a blocked state and does not silently use a mock or claim to be learning.

### Requirement: Durable background learning
The system SHALL process authorized learning events incrementally while the enabled local service is running, under configurable resource limits, with pause, recovery and observable status.

#### Scenario: Restart after interruption
- **WHEN** a training job is interrupted and the service restarts
- **THEN** it recovers from a consistent checkpoint or reports a failure without dropping or duplicating acknowledged intake.

### Requirement: Guarded model promotion
The system SHALL retain an active checkpoint while training a candidate and promote only after reproducible quality and retention gates pass, with rollback and attributable records.

#### Scenario: Candidate regresses
- **WHEN** the candidate fails the configured evaluation gates
- **THEN** the active model remains unchanged and the failure is visible.

### Requirement: Authorized data and network
The system SHALL train only on approved sources and SHALL NOT enable web access or execute retrieved instructions merely because background learning is enabled.

#### Scenario: Research not enabled
- **WHEN** autonomous web research is disabled
- **THEN** background learning performs no autonomous web research.
