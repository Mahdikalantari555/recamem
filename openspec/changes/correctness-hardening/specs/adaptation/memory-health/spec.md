# Spec Delta

## MODIFIED Requirements

### Requirement: Retrieval quality metric
The system SHALL track retrieval precision@k and mean reciprocal rank (MRR) over recent queries, with ground truth from user feedback (approve/reject). The health report SHALL also expose the associative capacity state: the active key count, the configured $K_{\max}$, and the cumulative eviction count produced by the capacity bound, so a reader can tell whether quality loss coincides with eviction.

#### Scenario: Quality reflects user satisfaction
- **WHEN** user approves retrieved facts
- **THEN** `get_memory_health()["retrieval_quality"]` SHALL increase
- **WHEN** user rejects retrieved facts
- **THEN** `get_memory_health()["retrieval_quality"]` SHALL decrease

#### Scenario: Eviction count is reported
- **WHEN** the associative memory evicts a weakest key at capacity
- **THEN** `get_memory_health()["evictions"]` SHALL increase by one
- **THEN** `get_memory_health()` SHALL report the current key count and `K_max`
- **WHEN** no key has been evicted
- **THEN** `get_memory_health()["evictions"]` SHALL be `0`
