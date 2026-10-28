# droid/profile-management Specification

## Purpose
Enables creating, selecting, saving, and switching between named Droid instances, each maintaining independent persistent memory states, configs, and audit logs.

## Requirements

### Requirement: Named Droid Isolation
The system SHALL support creating and switching between multiple agent profiles stored under the recamem home, `~/.recamem/agents/<name>/` by default, and loading legacy profiles under `droids/<name>/` in place. Each profile SHALL declare an `engine` of `droid`, `byte_rtu`, or `hybrid`, the relative path of its byte-RTU checkpoint when it owns one, a `fusion_mode`, the plastic adapter type it was constructed with, and a `schema_version`. Every artifact a profile requires SHALL live inside that profile's directory; no artifact SHALL be shared between profiles. The system SHALL write all artifacts before writing `config.json`, and SHALL write the manifest atomically. Loading a profile whose manifest declares an artifact that is absent, or whose `config.json` is missing or corrupt, SHALL raise a named error and SHALL NOT fall back to random initialization or an empty profile. A profile without a recorded adapter type SHALL load with the current default adapter.

#### Scenario: Switching active Droid
- **WHEN** user selects a different profile from the profile selector
- **THEN** the active memory state, logs, and metadata SHALL switch cleanly to the target profile without cross-contamination

#### Scenario: Memory round-trip persistence
- **WHEN** a profile absorbs knowledge and is saved
- **THEN** re-loading that profile SHALL restore its exact internal memory states and audit log history

#### Scenario: Fused profile round trip
- **WHEN** a profile is created with `create_droid(name, byte_rtu=True)`, taught, and saved
- **THEN** loading it SHALL restore `engine`, `byte_rtu_checkpoint`, `fusion_mode`, every plastic parameter, the RTU state buffers, `step_count`, and the knowledge fact count
- **THEN** the restored engine SHALL expose a working byte-RTU backend

#### Scenario: Manifest names a missing checkpoint
- **WHEN** `config.json` declares `byte_rtu_checkpoint` and that file is absent
- **THEN** loading SHALL raise `MissingCheckpointError` naming the resolved path
- **THEN** no random-initialized backend SHALL answer a query from that profile

#### Scenario: Missing or corrupt manifest
- **WHEN** `config.json` is absent, unreadable, or not a JSON object
- **THEN** loading SHALL raise `InvalidProfileError` naming the file and the reason
- **THEN** the loader SHALL NOT silently create a fresh profile in its place

#### Scenario: Artifacts stay inside the profile directory
- **WHEN** a profile declares a checkpoint path that is absolute or contains `..`
- **THEN** loading SHALL raise `InvalidProfileError`
- **WHEN** two profiles of the same engine are saved
- **THEN** no file path SHALL be shared between their directories
- **THEN** writing to one profile SHALL NOT change the other's state or fact count

#### Scenario: Save order
- **WHEN** a profile is saved
- **THEN** the knowledge store, the plastic state, and the byte-RTU checkpoint SHALL be written before `config.json`
- **THEN** a profile interrupted before the manifest write SHALL NOT claim an artifact that is absent

#### Scenario: Adapter type survives a round trip
- **WHEN** a Droid constructed with a non-default plastic adapter is saved and loaded into a fresh engine
- **THEN** the loaded engine's adapter class SHALL match the saved `plastic_adapter` value
- **THEN** the loader SHALL read the field that the saver writes, and SHALL NOT read a field that no saver writes

#### Scenario: Legacy profile without the adapter field
- **WHEN** a profile's `config.json` has no `plastic_adapter` field
- **THEN** loading SHALL succeed with the default adapter and SHALL NOT raise

### Requirement: Profile schema version and upgrade
The system SHALL read a profile's `schema_version` before loading any other field. An unknown newer version SHALL raise a named error. An older version SHALL be upgraded by an explicit step that preserves every stored value.

#### Scenario: Newer profile is refused
- **WHEN** a profile declares a schema version higher than the running code supports
- **THEN** loading SHALL raise a named error giving both versions
- **THEN** the profile files SHALL NOT be modified

### Requirement: Explicit adapter identity is authoritative
A profile SHALL persist the registered adapter identifier and validated constructor configuration, not infer identity from a class-name substring. An explicitly configured unknown or incompatible adapter MUST fail with a profile error. Only a legacy profile that omits adapter identity SHALL use the documented default.

#### Scenario: Custom registered adapter round trip
- **WHEN** a registered adapter with a class name unrelated to its registry identifier is saved and loaded
- **THEN** the same adapter identifier, constructor arguments and behavior SHALL be restored
- **THEN** re-saving the loaded profile SHALL retain those arguments

#### Scenario: Explicit unknown adapter
- **WHEN** a manifest names an adapter that is not registered
- **THEN** loading SHALL identify the invalid field and SHALL NOT construct the default adapter

#### Scenario: Legacy manifest omits adapter identity
- **WHEN** a supported legacy manifest has no adapter field
- **THEN** the documented default SHALL load with a versioned migration record

### Requirement: Profile state is complete and transactionally validated
Profiles SHALL preserve adapter metadata as well as tensors, retrieval policy, lifecycle configuration and schedule position. All referenced artifacts and dimensions MUST be validated before committing loaded state. Missing or corrupt artifacts SHALL fail explicitly, and saving SHALL write the manifest last without advertising incomplete state.

#### Scenario: Metadata survives persistence
- **WHEN** a profile with eviction history, non-default capacity and learned retrieval policy is saved and reloaded
- **THEN** those values and subsequent behavior SHALL match the pre-save reference

#### Scenario: Corrupt adapter artifact
- **WHEN** a required plastic-state artifact is malformed or has incompatible dimensions
- **THEN** load SHALL raise a named profile error and preserve the previously active profile
- **THEN** load SHALL NOT silently restore only a trace buffer or random weights

### Requirement: Explicit reversible migrations
Adapter and embedding migrations SHALL be opt-in, versioned and reversible from a complete pre-migration snapshot. Failed migration MUST leave the source profile usable. Existing profiles MUST NOT switch adapter, embedding dimension or retrieval index merely because a new default is available.

#### Scenario: Delta adapter migration is requested
- **WHEN** the user explicitly migrates an associative profile to a delta adapter
- **THEN** the source snapshot and migration configuration SHALL be retained
- **THEN** rollback SHALL restore original adapter state and knowledge without shared mutable artifacts
