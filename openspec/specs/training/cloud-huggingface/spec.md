# training/cloud-huggingface Specification

## Purpose
Enables training RecaMem models on cloud GPUs (Colab) and synchronizing
checkpoints with HuggingFace Hub for download, versioning, and sharing.

## Requirements

### Requirement: HuggingFace authentication
The system SHALL authenticate to HuggingFace Hub using a token from the
`HF_TOKEN` environment variable or a prior `huggingface-cli login`. The
system SHALL fail fast with a clear error when no token is available for
push operations. Pull operations on public repos SHALL work without a token.

#### Scenario: Push without token fails clearly
- **WHEN** a push is attempted and no token is available
- **THEN** the system SHALL raise an authentication error naming `HF_TOKEN`
- **THEN** no partial upload SHALL occur

#### Scenario: Pull public repo without token
- **WHEN** a pull is attempted on a public repo without a token
- **THEN** the download SHALL succeed

### Requirement: Push checkpoint to hub
The system SHALL upload a local checkpoint directory to a HuggingFace repo,
creating the repo if it does not exist. The upload SHALL include model
weights, optimizer state, tokenizer, config, and a manifest with sha256
checksums. The upload SHALL be atomic at the repo level (commit).

#### Scenario: First push creates repo
- **WHEN** pushing to a non-existent repo
- **THEN** the repo SHALL be created
- **THEN** all checkpoint files SHALL be uploaded in one commit

#### Scenario: Subsequent push adds new checkpoint
- **WHEN** pushing a new checkpoint to an existing repo
- **THEN** the new checkpoint SHALL be added without deleting prior ones
- **THEN** a manifest entry SHALL record the checkpoint id and sha256

### Requirement: Pull checkpoint from hub
The system SHALL download a checkpoint from a HuggingFace repo to a local
directory, verifying sha256 checksums against the manifest. A checksum
mismatch SHALL abort the download and leave the local directory unchanged.

#### Scenario: Successful pull
- **WHEN** pulling a valid checkpoint
- **THEN** all files SHALL be downloaded
- **THEN** sha256 of each file SHALL match the manifest

#### Scenario: Checksum mismatch aborts
- **WHEN** a downloaded file's sha256 does not match the manifest
- **THEN** the download SHALL abort
- **THEN** no partial files SHALL remain in the target directory

### Requirement: Incremental training resume
The system SHALL support resuming training from a pulled checkpoint,
continuing from the recorded step and corpus position without reprocessing
prior bytes. The system SHALL reject a checkpoint whose vocab size or
architecture does not match the requested config.

#### Scenario: Resume continues trajectory
- **WHEN** training resumes from a checkpoint
- **THEN** the step counter SHALL continue from the checkpoint's recorded step
- **THEN** no bytes before the recorded corpus position SHALL be reprocessed

#### Scenario: Mismatched config rejected
- **WHEN** a checkpoint's vocab size differs from the requested config
- **THEN** the system SHALL raise an error and not start training

### Requirement: Cloud training entry point
The system SHALL provide a Colab-compatible entry point that accepts
config via CLI args or environment variables, trains for a specified token
budget, checkpoints periodically, and pushes to hub at the end (or on
request). The entry point SHALL work in a fresh Colab runtime with only
pip install.

#### Scenario: Fresh Colab run
- **WHEN** the entry point runs in a fresh environment with dependencies installed
- **THEN** it SHALL train for the requested token budget
- **THEN** it SHALL write checkpoints and a report

#### Scenario: Push on completion
- **WHEN** training completes and push is enabled
- **THEN** the final checkpoint SHALL be pushed to the configured hub repo

### Requirement: Notebook template
The system SHALL provide a Colab notebook template that installs
dependencies, clones the repo, runs training, and pushes results, with
cells for configuration and monitoring.

#### Scenario: Notebook runs end to end
- **WHEN** a user opens the notebook in Colab and runs all cells
- **THEN** training SHALL complete and results SHALL be pushed to hub

### Requirement: CLI integration
The system SHALL expose `recamem push`, `recamem pull`, and
`recamem train-cloud` commands with the same behavior as the scripts.

#### Scenario: CLI push
- **WHEN** `recamem push --repo <id> --dir <path>` is invoked
- **THEN** the directory SHALL be pushed to the hub repo

#### Scenario: CLI pull
- **WHEN** `recamem pull --repo <id> --dir <path>` is invoked
- **THEN** the checkpoint SHALL be downloaded and verified
