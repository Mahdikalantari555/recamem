# Spec Delta

## ADDED Requirements

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
