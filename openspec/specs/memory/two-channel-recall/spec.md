# memory/two-channel-recall Specification

## Purpose
Recall for recamem combines a plastic prior channel with an episodic verbatim channel, fused by a learned scorer rather than hand-tuned boosts, so that every answer can be traced to a stored fact and its source, and so that unknown questions return nothing.

## Requirements

### Requirement: Recall uses two channels with source attribution
The system SHALL answer a recall query from two channels: a plastic prior score from the plastic channel and verbatim facts from the episodic channel. Every returned item SHALL carry its source reference and the score from each channel that contributed.

#### Scenario: Verbatim fact with source is returned
- **WHEN** a stored fact answers a query
- **THEN** the result SHALL contain the fact text verbatim
- **THEN** the result SHALL contain the source reference the fact was ingested with

#### Scenario: Plastic-only evidence is not presented as a fact
- **WHEN** only the plastic channel produces a score above the minimum and no episodic fact matches
- **THEN** the result SHALL be marked as plastic-only and SHALL NOT present any text as a stored fact

### Requirement: Fusion uses a learned scorer, not tuned boosts
The system SHALL combine channel features with a logistic scorer. The scorer SHALL be trained on a labelled split held out from its training data, and the training data and the held-out size SHALL be recorded. Below the minimum held-out size the scorer SHALL be disabled and the plastic prior SHALL be used alone, with the reason reported.

#### Scenario: Scorer is trained on labelled data
- **WHEN** a labelled query set of sufficient size is available
- **THEN** the scorer SHALL be fitted on the training split and evaluated on the held-out split
- **THEN** the report SHALL include the split sizes and the held-out ranking metrics

#### Scenario: Small labelled set disables the scorer
- **WHEN** the labelled set is below the minimum held-out size
- **THEN** the scorer SHALL NOT be used
- **THEN** the report SHALL state that the scorer was disabled and why

### Requirement: Unknown questions return an empty result
The system SHALL return an empty result when no item reaches the minimum fused score. It SHALL NOT return the nearest stored item in that case.

#### Scenario: Unrelated question
- **WHEN** a query shares no subject with any stored fact
- **THEN** the recall result SHALL be empty
- **THEN** the result SHALL NOT contain a fact from unrelated content

### Requirement: Successful recall reinforces the fact
The system SHALL record each successful recall as an access on the returned fact, and SHALL reinforce its retention according to the forgetting capability. The access SHALL be written to the audit log.

#### Scenario: Recall increases access count
- **WHEN** a fact is returned for a query and the user does not reject it
- **THEN** the fact's access count SHALL increase by one
- **THEN** the audit log SHALL record the query identifier and the fact identifier

### Requirement: Dense candidates use an ANN index with an exact baseline
The system SHALL use an approximate nearest-neighbour index for dense candidates above a configured store size, and an exact search below it. The exact search SHALL be the reference for measuring the index's recall.

#### Scenario: Index recall is measured against the exact baseline
- **WHEN** the benchmark runs with the index enabled
- **THEN** the report SHALL include recall at 5 relative to the exact search on the same store
- **THEN** a report SHALL NOT present index latency without that recall figure
