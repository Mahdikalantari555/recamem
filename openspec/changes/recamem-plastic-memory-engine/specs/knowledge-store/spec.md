# Spec Delta

## ADDED Requirements

### Requirement: Single writer and serialized access
The knowledge store SHALL serialize writes to a profile through one connection guarded by a lock. Connections SHALL NOT be shared across threads without that guard. A concurrent writer SHALL receive a named error rather than corrupt the store.

#### Scenario: Concurrent writes from threads
- **WHEN** two threads write facts to the same profile at the same time
- **THEN** both writes SHALL complete in some order with no lost or partial rows
- **THEN** no exception SHALL originate from sharing a connection across threads

### Requirement: Retired tier and schema version
The knowledge store SHALL record each fact's tier as `active` or `retired`, and SHALL record its schema version. Retrieval SHALL read the active tier unless the caller explicitly requests otherwise.

#### Scenario: Tier is stored with each fact
- **WHEN** a fact is retired and later restored
- **THEN** its tier SHALL change only through those operations
- **THEN** the row SHALL keep its identifier, source, and history throughout
