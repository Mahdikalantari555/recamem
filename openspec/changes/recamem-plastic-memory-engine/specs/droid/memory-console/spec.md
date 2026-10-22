# Spec Delta

## ADDED Requirements

### Requirement: Reset memory uses the engine clear path
The memory console Reset action SHALL clear the knowledge store, the plastic state, and the step counters through the engine's single clear operation. If clearing fails, the console SHALL show the exception it raised and SHALL NOT report the memory as reset.

#### Scenario: Reset clears all learned state
- **WHEN** the user confirms Reset in the console
- **THEN** the active profile's facts, plastic state, and step counter SHALL be cleared
- **THEN** the console SHALL report the reset only after all three are cleared

#### Scenario: Reset failure is reported
- **WHEN** the clear operation raises an exception
- **THEN** the console SHALL display the exception message
- **THEN** the console SHALL NOT report success
