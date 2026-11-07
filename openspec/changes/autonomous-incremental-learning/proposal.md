# Proposal: independent, continuous incremental learning

## Why
The user wants an independent CPU model that configures its own training and keeps learning incrementally in the background, without asking the user to choose a model architecture, tokenizer or training token budget. The current manual smoke run is not that system.

## What Changes
Specify a single local learning service across chat, memory, CLI, API, MCP and GUI. It chooses a compatible model, tokenizer and bounded training plan from validated policies and available artifacts, preserves incremental knowledge and resumes work while the service is running. The user's desired always-on behavior must be visibly distinguished from operation while the application or machine is stopped; optional OS service installation needs explicit consent. User controls are goals, approved sources, privacy, resource limits, pause and rollback, not mandatory model internals. No fabricated autonomy, hidden network access, or unsupported quality claims.

## Capabilities
- `training/autonomous-incremental-learning`: durable intake, autonomous planning, supervised background training, evaluation, promotion and recovery.

## Impact
Extend existing coordinator, SQLite memory, training jobs and checkpoint code rather than a second memory store. Depends on native model/memory integration and meaningful evaluations in `cpu-base-learning` and `unified-agent-experience`. Web content is opt-in and subject to `autonomous-web-research` security gates. Scratch and compatible pretrained CPU models are both candidates; no new GGUF dependency. Preserve existing advanced APIs and explicit manual experiments for compatibility.
