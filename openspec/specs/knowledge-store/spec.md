# knowledge-store Specification

## Purpose

Hybrid persistent knowledge store combining SQLite-backed full-text indexing (FTS5 BM25) with pre-normalized dense vector matrix-vector multiplication (MVM), fused via Reciprocal Rank Fusion (RRF) to replace the O(N) linear scan over in-memory JSON facts.

## Requirements

### Requirement: Hybrid lexical-dense retrieval
The system SHALL retrieve facts using a fused lexical (BM25) and dense (cosine similarity) ranking score, computed in under 5 ms on CPU for up to 100,000 facts.

#### Scenario: Sub-5ms retrieval at scale
- **WHEN** the knowledge store contains 100,000 facts
- **WHEN** a query is issued for a term matching both lexical and semantic patterns
- **THEN** the top-5 ranked facts SHALL be returned in under 5 ms
- **THEN** the response SHALL include both lexical rank and dense similarity scores

#### Scenario: Exact term precision
- **WHEN** a query contains an exact alphanumeric code, acronym, or coordinate (e.g., "NDVI", "RS", "52.5°N")
- **THEN** the top retrieved fact SHALL match the exact term with lexical rank contribution

### Requirement: Pre-normalized dense vectors on insert
The system SHALL normalize all knowledge vectors to unit $L_2$ norm at insertion time so that queries allocate 0 KB of dynamic heap and use a single matrix-vector multiplication.

#### Scenario: Zero query-time allocation
- **WHEN** a query is executed against a store of 50,000 facts
- **THEN** the query execution path SHALL allocate 0 KB of dynamic heap memory
- **THEN** the query execution time SHALL be under 3 ms

### Requirement: Persistent SQLite storage
The system SHALL persist facts, embeddings, and metadata in a SQLite database using WAL journaling mode and an external-content FTS5 virtual table.

#### Scenario: WAL persistence and recovery
- **WHEN** the Droid profile is saved
- **THEN** the knowledge store SHALL write facts and vectors to a SQLite database file
- **WHEN** the profile is loaded after a clean shutdown
- **THEN** the database SHALL recover all facts and vectors without loss

### Requirement: Reciprocal Rank Fusion
The system SHALL fuse lexical and dense rankings using the standard Reciprocal Rank Fusion formula $score = \frac{1}{60 + rank_{lex}} + \frac{1}{60 + rank_{dense}}$ with a single SQL CTE.

#### Scenario: Fused ranking
- **WHEN** a query matches both lexical and dense candidates
- **THEN** the fused score SHALL be computed in-database using a single FULL OUTER JOIN over ranked CTEs
- **THEN** the final ranking SHALL reflect contributions from both lexical and dense signals

### Requirement: Retrieval latency replaces allocator accounting
Retrieval performance SHALL be verified by measured latency at 10k, 50k and 100k facts on identified hardware. A zero-allocation claim SHALL NOT be a requirement, because CPython allocator behaviour is not observable from the query path.

#### Scenario: Latency benchmark runs
- **WHEN** the workload benchmark is executed
- **THEN** it SHALL publish p50 and p95 retrieval latency per fact count with the environment
- **THEN** a miss against the < 5 ms target SHALL be published as a miss

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
