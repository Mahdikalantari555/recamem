## ADDED Requirements

### Requirement: CPU learning workspace
The Svelte companion SHALL expose dataset preparation, CPU model training, checkpoint and evaluation status using real backend results without requiring GGUF.

#### Scenario: Dataset is unavailable
- **WHEN** a selected substantive corpus is unavailable or unverified
- **THEN** the interface reports it as blocked and does not present fixture data as substantive training evidence.

### Requirement: Provenance-aware teaching
The interface SHALL distinguish retrieval-backed teaching from model weight updates and show the source and status of recalled information.

#### Scenario: User corrects a fact
- **WHEN** the user submits a correction
- **THEN** the interface displays the real operation result and the updated provenance rather than inventing a successful memory update.

### Requirement: Honest resource and capability status
The interface SHALL show measured CPU progress, checkpoint availability and evaluation results without claiming emergence, guaranteed understanding or unmeasured performance.

#### Scenario: No trained checkpoint
- **WHEN** no valid trained checkpoint is available
- **THEN** the interface shows an actionable unavailable state and never silently selects randomly initialized weights.
