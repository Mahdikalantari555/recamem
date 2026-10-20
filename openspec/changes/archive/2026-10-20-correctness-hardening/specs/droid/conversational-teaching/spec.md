# Spec Delta

## MODIFIED Requirements

### Requirement: Automatic Hyperparameter Selection
The system SHALL automatically determine the learning rate, decay half-life, and consolidation iterations based on the ingested text length without user prompt. The system SHALL apply the resulting effective learning rate to every plastic update in the same `teach()` call, so the schedule changes the stored plastic state and not only the reported value. The system SHALL record the applied learning rate in the train log together with the computed value.

#### Scenario: Auto-scaled update
- **WHEN** a text of arbitrary length $N$ is provided for conversational teaching
- **THEN** the system SHALL apply an effective learning rate scaled by $1 / \sqrt{N}$ and log the chosen values

#### Scenario: Applied learning rate changes the plastic step
- **WHEN** the same proposition is taught at `step_count = 0` and at a later step where the effective rate is lower
- **THEN** the resulting associative-state delta SHALL be smaller in the later update
- **THEN** the update SHALL be scaled by `effective_lr / base_lr` rather than by a constant

#### Scenario: Applied learning rate is recorded
- **WHEN** a teach call completes
- **THEN** the teach result SHALL contain both `effective_lr` (computed) and `applied_lr` (used)
- **THEN** the `train_log.json` entry for that call SHALL contain the same `applied_lr`
- **THEN** `effective_lr` SHALL equal `applied_lr` at `step_count = 0`
