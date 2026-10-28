# Spec Delta

## MODIFIED Requirements

### Requirement: Sleep API and report
`DroidEngine.sleep()` SHALL execute a bounded consolidation cycle and return a structured report distinguishing replay, fact retirement, parameter updates, duration, status, and warnings. A parameter update SHALL be committed only through the validation gate in the SLOW tier.

#### Scenario: Manual sleep
- **WHEN** a caller invokes `DroidEngine.sleep()` on a non-empty Droid
- **THEN** exactly one configured cycle SHALL run
- **THEN** the method SHALL return a report even when a phase has no eligible work
- **THEN** the report SHALL not claim recovered patterns or retirements that did not occur

#### Scenario: Empty memory
- **WHEN** sleep is requested on an empty Droid
- **THEN** the method SHALL return a successful no-op report with zero replay and retirement counts
- **THEN** it SHALL not raise an unhandled exception

#### Scenario: Regressing replay is not committed
- **WHEN** a sleep cycle's parameter update lowers held-out recall beyond tolerance in any probed domain
- **THEN** the update SHALL be rolled back
- **THEN** the report SHALL state the update was rejected and name the domain

## ADDED Requirements

### Requirement: Single writer per profile during sleep
The system SHALL allow only one writer to a profile at a time. A sleep cycle or a teaching call that cannot acquire the profile's writer lock SHALL fail with a named error and SHALL NOT wait indefinitely.

#### Scenario: Second writer is refused
- **WHEN** a sleep cycle starts while a teaching call holds the profile's writer lock
- **THEN** the sleep cycle SHALL fail with a named lock error
- **THEN** the profile's stored state SHALL be unchanged by the refused call
