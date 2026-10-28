# plasticity/two-tier Specification

## Purpose
Two-tier plastic memory for recamem: a FAST tier that writes new facts into stored state without gradients, and a SLOW tier that consolidates replayed experience into meta-parameters only when a rollback gate confirms that nothing regressed.

## Requirements

### Requirement: FAST writes change stored state through the schedule
The system SHALL apply the effective learning rate computed for each teaching call to every FAST write, and the change SHALL be visible in the stored plastic state. The system SHALL record both the requested and the applied learning rate for each write.

#### Scenario: Schedule changes the stored state
- **WHEN** the same proposition is written at a late step count and at an early step count with the same base learning rate
- **THEN** the stored plastic state after the two writes SHALL differ by the ratio of the two effective learning rates
- **THEN** the reported learning rate SHALL equal the rate that changed the state

#### Scenario: Applied rate is logged
- **WHEN** a teaching call completes
- **THEN** the train log SHALL contain the requested rate and the applied rate for each write

### Requirement: FAST writes run without gradient tracking
The system SHALL perform FAST writes without gradient tracking. Gradient computation SHALL occur only in the SLOW tier.

#### Scenario: No gradients in a FAST write
- **WHEN** a FAST write is performed
- **THEN** no tensor produced by the write SHALL require gradients
- **THEN** the write SHALL NOT allocate a backward graph

### Requirement: SLOW consolidation is gated by rollback
The system SHALL commit a SLOW replay update only when a held-out recall probe shows no domain regressing beyond the configured tolerance. Otherwise the system SHALL restore the previous meta-parameters and record the rejection reason in the train log.

#### Scenario: Accepted replay is committed
- **WHEN** a replay update leaves every probed domain within tolerance
- **THEN** the meta-parameters SHALL be replaced by the updated values
- **THEN** the train log SHALL record the accepted update and its probe scores

#### Scenario: Regressing replay is rolled back
- **WHEN** a replay update lowers held-out recall in any probed domain beyond the tolerance
- **THEN** the meta-parameters SHALL be restored to their values before the update
- **THEN** the train log SHALL record the rejection, the regressed domain, and the score drop

### Requirement: Memory-aware synapse penalty bounds drift
The system SHALL apply a memory-aware synapse importance penalty to SLOW updates so that parameters important to earlier facts move less than unimportant ones.

#### Scenario: Important parameters move less
- **WHEN** a SLOW update is computed with importance estimates for two parameters that differ
- **THEN** the parameter with higher importance SHALL change by no more than the parameter with lower importance under the same gradient

### Requirement: Plastic state is schema-versioned
The system SHALL store a schema version with the plastic state. A loader SHALL read older versions through an explicit upgrade step and SHALL reject a newer or unknown version with a named error.

#### Scenario: Newer state is rejected
- **WHEN** plastic state declares a schema version higher than the running code supports
- **THEN** loading SHALL raise a named error that gives both versions
- **THEN** the profile SHALL NOT be partially loaded

#### Scenario: Older state is upgraded
- **WHEN** plastic state declares an older supported schema version
- **THEN** loading SHALL apply the upgrade step and SHALL preserve every stored value
