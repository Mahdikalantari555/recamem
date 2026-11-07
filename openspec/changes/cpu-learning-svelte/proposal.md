# Proposal: CPU learning workspace for Svelte

## Why
The existing Svelte companion supports training and chat but does not yet give a clear CPU-native workflow for choosing Muir or C4, inspecting corpus provenance, evaluating language learning, and teaching memory-backed facts without GGUF.

## What Changes
Extend the existing `companion/` Svelte app and API rather than create a competing frontend. Provide CPU dataset, training, checkpoint, evaluation, and teaching workflows, with measured status and explicit limitations. Depend on the implemented CPU core and native training/teaching APIs; do not invent job results.

## Capabilities
- `surface/cpu-learning`: CPU learning workspace and evidence-led teaching interface.

## Non-goals
No promise of broad understanding, emergent abilities, fast training, or silent random-model and mock fallbacks. This proposal does not implement the UI.

## Impact
Svelte companion, existing API/job runner, native backend integration, tests and documentation. Preserve existing local-first security and current companion flows.

## Continuous learning alignment

The basic experience should not require architecture or token selection. Show truthful background learning status and resource/privacy controls as the autonomous learning service is implemented; see `autonomous-incremental-learning`.
