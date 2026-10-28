# memory-forgetting Specification

## Purpose

Ebbinghaus power-law forgetting with sleep consolidation: retrieval accessibility decays as a power function of elapsed time since last access; during idle cycles, salient facts are replayed into associative weights and low-utility episodic facts are pruned.

## Requirements

### Requirement: Power-law retrieval availability decay
The system SHALL compute fact accessibility as $A_i(t) = S_{\text{base}}(i) \cdot (1 + \alpha (t - t_i^{\text{last}}))^{-\beta_i}$ where $\beta_i$ decreases with each successful retrieval (spacing effect), so that a frequently retrieved fact decays more slowly. A larger $\beta_i$ means faster decay.

#### Scenario: Frequent retrieval increases stability
- **WHEN** a fact is successfully retrieved
- **THEN** its decay exponent $\beta_i$ SHALL decrease by factor $(1 - \delta_{\text{reinforce}})$ with $\delta_{\text{reinforce}} \in [0.1, 0.3]$
- **THEN** the fact SHALL decay more slowly thereafter

#### Scenario: One-off facts decay rapidly
- **WHEN** a fact is never retrieved after initial encoding
- **THEN** its accessibility SHALL follow the base power-law decay

### Requirement: Sleep consolidation during idle
The system SHALL run a background consolidation cycle when no queries have been received for $> 30$ seconds: sample high-utility facts, compute associative reconstruction loss, execute 3–5 gradient steps into plastic weights, and retire episodic facts whose content has been verified as absorbed. Retirement SHALL follow the rules in the consolidate-then-forget requirement.

#### Scenario: Consolidation runs automatically on idle
- **WHEN** `consolidate_memory()` is called after idle period
- **THEN** it SHALL sample high-utility facts from the knowledge store
- **THEN** it SHALL execute gradient steps on the SLOW tier parameters only

#### Scenario: Absorbed facts pruned from episodic store
- **WHEN** a fact's reconstruction error falls below threshold and a recall probe confirms the fact is answerable from plastic memory
- **THEN** the fact SHALL be moved to the retired tier
- **THEN** the fact row SHALL remain in the knowledge store with its provenance

### Requirement: Utility-based episodic pruning
The system SHALL compute fact utility as $\text{Utility}(i) = \text{AccessCount}_i \cdot \exp(-(t - t_i^{\text{last}}) / \tau_{\text{half}}) + \lambda \cdot s_i^{\text{novelty}}$. When the number of active facts exceeds $K_{\max}$, the system SHALL move the lowest-utility facts to the retired tier until the active count is at or below $K_{\max}$. Moving a fact to the retired tier SHALL NOT remove its row.

#### Scenario: Pruning triggers at capacity limit
- **WHEN** total active facts exceed $K_{\max} = 5000$
- **THEN** the forgetting step SHALL move the lowest-utility facts to the retired tier
- **THEN** the active count SHALL be at or below $K_{\max}$ after the step
- **THEN** remaining facts SHALL preserve audit trail metadata

#### Scenario: High-novelty facts resist pruning
- **WHEN** a fact has high novelty score $s_i$
- **THEN** its utility SHALL be boosted by $\lambda \cdot s_i$
- **THEN** it SHALL be retained preferentially

#### Scenario: Retired facts remain auditable
- **WHEN** a fact has been moved to the retired tier
- **THEN** its text, source, and history SHALL remain queryable through the audit interface
- **THEN** it SHALL NOT appear in ordinary recall results unless the caller asks for retired facts

### Requirement: Consolidate-then-forget
The system SHALL retire a fact only after its content has been absorbed into the plastic channel and a recall probe confirms the absorption. A fact that fails the probe SHALL remain active.

#### Scenario: Unabsorbed fact stays active
- **WHEN** a fact is selected for forgetting and the recall probe does not answer it
- **THEN** the fact SHALL remain active
- **THEN** the forgetting step SHALL record that the fact was kept and why

### Requirement: Soft retirement with no hard delete
The forgetting path SHALL NOT issue a hard delete for a fact row. A retired fact SHALL be restorable to the active tier by an explicit user action.

#### Scenario: Forgetting never deletes rows
- **WHEN** a forgetting or consolidation cycle completes
- **THEN** the number of fact rows in the store SHALL be unchanged except for rows added by the cycle
- **THEN** no statement that deletes a fact row SHALL execute in the forgetting path

#### Scenario: Retired fact is restored
- **WHEN** the user restores a retired fact
- **THEN** the fact SHALL return to the active tier with its original source and history
