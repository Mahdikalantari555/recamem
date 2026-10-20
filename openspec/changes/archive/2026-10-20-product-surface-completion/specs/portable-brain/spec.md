# Spec Delta

## MODIFIED Requirements

### Requirement: ZIP archive format
The system SHALL package a Droid's complete state into a `.droid` ZIP archive containing `manifest.json`, `config.json`, `memory.safetensors`, and `knowledge.db`. The memory console SHALL provide export and import actions, and export SHALL write the archive outside the profile directory so it cannot overwrite the profile it reads.

#### Scenario: Archive creation
- **WHEN** a Droid profile is exported
- **THEN** the `.droid` archive SHALL contain the following members:
  - `manifest.json`: Metadata including base anchor version, dimensions, creation timestamp, and SHA-256 checksums
  - `config.json`: Profile settings, decay parameters, learning rate schedules
  - `memory.safetensors`: Plastic RTU trace buffers (`states`, `decay`, `proj_weight`)
  - `knowledge.db`: SQLite database containing FTS5 tables, embeddings, and belief states
  - `train_log.json`: Auditable history of all learning sessions

#### Scenario: Archive restoration
- **WHEN** a `.droid` archive is imported
- **THEN** the system SHALL verify SHA-256 checksums of all archive members
- **THEN** the system SHALL extract members to the target Droid profile directory
- **THEN** the system SHALL load the Droid state from the extracted files

#### Scenario: Console export and import
- **WHEN** user exports a profile from the console
- **THEN** the console SHALL show the archive members, sizes, and manifest digests
- **WHEN** user imports an archive from the console
- **THEN** the console SHALL show the same manifest table and require confirmation before extraction

### Requirement: Integrity verification
The system SHALL verify SHA-256 checksums of all archive members during import to detect corruption or tampering. Verification SHALL happen before extraction, and a mismatch SHALL abort the import without presenting a partial restore as successful.

#### Scenario: Checksum verification
- **WHEN** a `.droid` archive is imported
- **THEN** the system SHALL compute SHA-256 for each member and compare against `manifest.json`
- **THEN** a mismatch SHALL raise a clear integrity error and abort the import

#### Scenario: Console reports the mismatch
- **WHEN** a console import finds a digest mismatch
- **THEN** the console SHALL name the member, the expected digest, and the observed digest
- **THEN** the target profile SHALL remain unchanged
