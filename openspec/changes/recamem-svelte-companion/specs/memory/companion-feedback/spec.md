# companion-feedback Specification

## Purpose

Defines game-like petting as real memory operations, with pet state derived only
from measured engine signals and an auditable before/after record.

## ADDED Requirements

### Requirement: Every companion action maps to a real engine call

The action `pet` and `treat` SHALL call `approve(fact_id)` on the last recalled
fact; `feed` SHALL call `teach(text)`; `scold` SHALL call `reject(fact_id)`;
`correct` SHALL call `correct(fact_id, text)`; `sleep` SHALL call `sleep()`;
`play` SHALL execute an inference turn. An action without a valid target fact
or non-empty text SHALL be rejected with a reason and SHALL NOT award a reward.

#### Scenario: Petting reinforces a fact

- **WHEN** the user pets after a turn that recalled fact 42
- **THEN** the engine's `approve(42)` SHALL be called
- **THEN** the response SHALL contain the before and after measured values and an event id

#### Scenario: Action with no target

- **WHEN** the user pets with no recalled fact in the current turn
- **THEN** the API SHALL return an error naming the missing target
- **THEN** no pet state value SHALL change

### Requirement: Pet state is derived from measured signals

Mood, energy, bond and knowledge SHALL be computed from active and retired fact
counts, retrieval hit rate, approve/reject counts, memory health, TTT status,
time since last sleep, training progress and selected checkpoint status. Each
value SHALL be returned with its source field and measurement timestamp, and the
Inspector SHALL show the raw numbers.

#### Scenario: Feedback changes the pet

- **WHEN** an approve event is recorded
- **THEN** the recomputed companion state SHALL reflect the new feedback count
- **THEN** the Inspector SHALL show the exact field that changed

#### Scenario: No invented signal

- **WHEN** an engine signal is unavailable
- **THEN** the corresponding pet attribute SHALL be reported as unavailable
- **THEN** the UI SHALL NOT substitute a local counter or a random value

### Requirement: Companion events are auditable

Every action SHALL append an event with timestamp, action type, target, engine
call, before/after values and outcome to the feedback log, and SHALL be
exportable through the existing feedback-log contract.

#### Scenario: Export

- **WHEN** the feedback log is requested after several companion actions
- **THEN** it SHALL contain each action with its engine call and measured deltas
