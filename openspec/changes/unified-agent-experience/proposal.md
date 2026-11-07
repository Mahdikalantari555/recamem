# Proposal: One agent experience and evidence-led memory consolidation

## Why
The user wants a single agent experience and eventual consolidation of internal memory mechanisms, both native scratch and pretrained CPU model paths, no GGUF requirement, and a roughly 16 GB RAM target. The repository has multiple surfaces and partially independent memory and generation paths; consolidating them all at once would risk losing provenance, persistence or quality.

## What Changes
Introduce a versioned coordinator around the existing runner and progressively route CLI, API, MCP and Svelte through it without changing their existing contracts. Inventory existing mechanisms and compare alternatives before replacing or merging internal stores. Establish one logical memory interface with sourced facts, correction and explicit availability; physically consolidate only after evidence, migration and rollback gates. Explore both native Muir/C4 and a CPU-compatible pretrained PyTorch model; neither is assumed capable without measurements. Autonomous research remains opt-in; the user selected automatic saving of eligible attributed facts, with unverified status and explicit policy controls.

## Capabilities
- `agent/unified-session`: shared coordinator, memory contract, model choices and measured migration.

## Non-goals
No GGUF dependency; no promise that Muir replaces a database or that tiny scratch models understand language. No deletion of stores, profiles or API surfaces without migration and evidence. Do not claim one memory implementation is superior without a measured comparison.

## Impact
Runner/API/MCP/CLI/Svelte adapters, memory and model backends, tests and documentation. Coordinate with `cpu-muir-core`, `cpu-base-learning` and `autonomous-web-research`; 16 GB is a target, not a guarantee on all hardware.

## Continuous learning alignment

The shared experience must expose the same autonomous incremental-learning lifecycle on all surfaces without requiring architecture or tokenizer selection; see `autonomous-incremental-learning`.
