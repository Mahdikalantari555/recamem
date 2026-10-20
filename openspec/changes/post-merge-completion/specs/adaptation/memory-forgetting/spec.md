# Spec Delta

## MODIFIED Requirements

### Requirement: Power-law retrieval availability decay
The system SHALL compute fact accessibility as $A_i(t) = S_{\text{base}}(i) \cdot (1 + \alpha (t - t_i^{\text{last}}))^{-\beta_i}$ with a positive decay exponent. Successful retrieval SHALL decrease the exponent, within configured positive bounds, so reinforced facts decay more slowly. Per-fact decay state SHALL persist across restart.

#### Scenario: Frequent retrieval increases stability
- **WHEN** a fact is successfully retrieved
- **THEN** its decay exponent SHALL decrease by factor $(1 - \delta_{\text{reinforce}})$ with $\delta_{\text{reinforce}} \in [0.1, 0.3]$, bounded by the configured positive minimum
- **THEN** the fact SHALL decay more slowly thereafter

#### Scenario: One-off facts decay rapidly
- **WHEN** a fact is never retrieved after initial encoding
- **THEN** its accessibility SHALL follow the base power-law decay

#### Scenario: Decay survives restart
- **WHEN** a reinforced fact is saved and loaded
- **THEN** accessibility at a fixed evaluation timestamp SHALL equal the uninterrupted reference value

### Requirement: Sleep consolidation during idle
When automatic sleep is explicitly enabled, the system SHALL run a bounded background consolidation cycle after the configured idle interval, defaulting to 30 seconds. The cycle SHALL sample high-utility facts, perform 3–5 reconstruction gradient steps on eligible plastic weights, and prune eligible absorbed episodic facts. Automatic sleep SHALL be disabled by default; manual consolidation SHALL remain available.

#### Scenario: Consolidation runs automatically on idle
- **WHEN** automatic sleep is enabled and the idle interval elapses
- **THEN** one consolidation cycle SHALL sample high-utility facts from the knowledge store
- **THEN** it SHALL execute gradient steps on eligible plastic parameters without updating the frozen anchor or inference decoder

#### Scenario: Absorbed facts pruned from episodic store
- **WHEN** a fact's reconstruction error falls below the configured threshold and it meets pruning safeguards
- **THEN** it SHALL be removed from active episodic retrieval while retaining audit history and a reversible pre-cycle snapshot

#### Scenario: Default deployment does not start sleep
- **WHEN** a profile omits automatic sleep configuration
- **THEN** no sleep worker SHALL start
- **THEN** manual consolidation SHALL still run on explicit request
