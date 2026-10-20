# Spec Delta

## MODIFIED Requirements

### Requirement: Capacity bound enforcement
The system SHALL enforce the configured associative-memory capacity bound $K_{\max}$ in the plastic update path. When the key register is below $K_{\max}$, no eviction SHALL occur. When an update arrives with the register at $K_{\max}$, the system SHALL evict the weakest stored key before writing the new association, using the key projection norm as the strength measure and the existing attenuation operator for removal. The system SHALL keep a monotonic count of evictions that the health report can read.

#### Scenario: Eviction at capacity
- **WHEN** the key register holds $K_{\max}$ keys and a new key is written
- **THEN** the key with the smallest projection norm SHALL be attenuated before the new association is stored
- **THEN** the eviction count SHALL increase by exactly one
- **THEN** the register size SHALL remain at $K_{\max}$

#### Scenario: No eviction below capacity
- **WHEN** the key register holds fewer than $K_{\max}$ keys
- **THEN** no existing association SHALL be evicted
- **THEN** the eviction count SHALL be unchanged

#### Scenario: Rejected update does not consume capacity
- **WHEN** an update is rejected for a non-finite key or value
- **THEN** the register SHALL be unchanged and no eviction SHALL occur
