# Spec Delta

## MODIFIED Requirements

### Requirement: Strategy selection
The system SHALL select injection strategy based on fact count, query type, and profile config: `prompt` for <5 facts, `hidden` for 5–20, `kv_cache` for >20 or when context window constrained. The profile config SHALL be authoritative when it pins a strategy. A profile that attaches a byte-RTU backend SHALL resolve to `prompt` regardless of fact count, because the `hidden` and `kv_cache` implementations are stubs and a byte-level component consumes bytes only.

#### Scenario: Auto strategy selection
- **WHEN** 3 facts retrieved
- **THEN** `prompt` strategy SHALL be used
- **WHEN** 15 facts retrieved
- **THEN** `hidden` strategy SHALL be used

#### Scenario: Fused profile pins prompt
- **WHEN** a profile attaches a byte-RTU backend and the config declares `injection_strategy: "prompt"`
- **THEN** `select_strategy()` SHALL return `prompt` for 3 facts, 15 facts, and more than 20 facts
- **THEN** `hidden` and `kv_cache` SHALL NOT be selected for that profile

#### Scenario: Strategy survives a profile round trip
- **WHEN** a profile is saved and loaded
- **THEN** `injection_strategy` SHALL be restored from `config.json`
- **THEN** the loader SHALL NOT drop the key when filtering GGUF-prefixed settings

#### Scenario: Prompt keeps the query last
- **WHEN** facts are injected with the `prompt` strategy
- **THEN** the rendered prompt SHALL end with the query, after the fact block
- **THEN** reordering that places the query before the facts SHALL be rejected
