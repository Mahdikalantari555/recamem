## ADDED Requirements

### Requirement: Reproducible CPU datasets
The system SHALL provide a self-authored or suitably licensed offline smoke corpus and a reproducible, rights-verified acquisition procedure for a larger language corpus, with pinned provenance and document-level held-out splits. Smoke data SHALL NOT be presented as evidence of broad language understanding.

#### Scenario: Corpus unavailable
- **WHEN** the substantive dataset cannot be acquired or its usage terms cannot be verified
- **THEN** training requiring that dataset is reported blocked with the reason rather than silently substituting fixture data.

### Requirement: Native CPU training and generation
The training and generation path SHALL run on CPU using native PyTorch checkpoints without GGUF, llama.cpp, accelerator hardware or a network service. Checkpoints SHALL include the architecture and tokenizer identity required for validated reload and resume.

#### Scenario: Invalid checkpoint
- **WHEN** the requested checkpoint is missing or incompatible with its architecture or tokenizer
- **THEN** loading fails explicitly without silently generating from random weights.

### Requirement: Measured language learning
The system SHALL report held-out loss, n-gram comparisons and separate novel-example evaluations with data provenance, hardware and training budget. It SHALL NOT assert general understanding or emergent behavior from smoke tests or unverified examples.

#### Scenario: Small training run
- **WHEN** a model trains on a small corpus
- **THEN** the report distinguishes measured task outcomes and domain limitations from any unverified capability claim.

### Requirement: Incremental teaching with provenance
The system SHALL support teaching facts through RecaMem's existing memory lifecycle and supplying relevant retrieved context to the native generator, with explicit provenance, correction and persistence. Memory-based learning SHALL be distinguished from changing model weights.

#### Scenario: Fact corrected
- **WHEN** a user teaches a fact and later supplies a correction
- **THEN** the correction is retained using lifecycle supersession, survives restart, and evaluation includes old and corrected questions with memory-enabled and memory-disabled controls.
