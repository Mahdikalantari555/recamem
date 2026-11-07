## ADDED Requirements

### Requirement: Shared agent session
The system SHALL provide a unified agent session entry point for chat, teach and recall, preserving legacy results and errors while surfaces are migrated.

#### Scenario: Existing runner calls the coordinator
- **WHEN** a caller runs a chat, recall or teach request through the existing runner
- **THEN** the shared session SHALL execute the existing behavior without changing its response contract or silently enabling network access.

### Requirement: Evidence-led memory consolidation
The system SHALL retain provenance and support correction, supersession and safe retirement during memory consolidation. Replacing existing mechanisms SHALL require measured retrieval, retention and resource comparisons and a reversible migration.

#### Scenario: Candidate memory lacks provenance or fails retention
- **WHEN** a proposed unified memory cannot meet the provenance, correction or retention gates
- **THEN** existing stores SHALL remain available and the candidate SHALL NOT be promoted.

### Requirement: CPU language paths
The system SHALL evaluate scratch Muir/C4 and compatible pretrained CPU PyTorch options without requiring GGUF, and SHALL report model, license, tokenizer, checkpoint availability and measured resource use.

#### Scenario: Only smoke model available
- **WHEN** only a smoke-trained checkpoint is available
- **THEN** the system SHALL disclose its limitations and SHALL NOT advertise a capable pretrained assistant.
