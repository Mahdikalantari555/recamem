# Proposal

## Why

PRs #5, #8 and #9 are now merged via #10, but passing offline contracts is not proof that every archived promise or proposed feature is implemented. The post-merge baseline (`c246043`) has three unimplemented feature changes, historical unchecked tasks, placeholder injection paths, and adapter/persistence behavior that needs stronger end-to-end guarantees.

## What Changes

- Complete the three existing feature changes as dependencies: `delta-plastic-memory`, `autonomous-sleep-consolidation`, and `edge-slm-optimization`. Their specs and 65 tasks remain authoritative; this umbrella does not silently archive or duplicate them.
- Complete adapter lifecycle defaults, learning-rate capability negotiation, strict adapter identity, full checkpoint metadata, and visible update/load errors. **BREAKING:** invalid explicitly configured adapters will fail rather than silently become the default.
- Replace hidden/KV injection metadata-only stubs with real backend operations where supported. Capability-aware selection must report prompt fallback honestly; explicit unsupported modes fail clearly, and byte RTU remains prompt-only.
- Make ambiguous retrieval queries reachable and deterministic in tests; close feedback, semantic retrieval, lifecycle and TTT integration gaps against existing specs.
- Reconcile every historical task and current requirement with code, regression tests and reproducible evidence. Finish missing behavior rather than merely ticking archived boxes; record explicit requirement revisions for incompatible historical promises.
- Add separate release gates for offline correctness, live ONNX/GGUF, trained grounding, compatibility, worker safety and benchmark claims.

## Capabilities

### New Capabilities

- `verification/release-evidence`: requirement/task traceability, dependency completion, reproducible integration and performance evidence, and archive readiness.

### Modified Capabilities

- `adaptation/plastic-adapter-interface`: optional lifecycle defaults, independent snapshots and capability-aware update scheduling.
- `droid/profile-management`: exact adapter identity, complete metadata, strict failure and migration behavior.
- `model/memory-injection`: real supported backend injection, capability negotiation, explicit fallback reporting and token-budget safety.
- `memory/adaptive-retrieval`: reachable ambiguity classification with deterministic precedence and profile isolation.
- `adaptation/memory-forgetting`: resolve contradictory decay-exponent wording and align idle consolidation with the explicit opt-in sleep-worker contract.

The delta/sparse, sleep/MAS and edge capabilities remain owned by their existing active delta specs. Other existing requirements are completion work, not silently rewritten contracts.

## Impact

Primary areas: `src/model/{plastic_adapter,rtu,droid,droid_manager,memory_injection,memory_lifecycle,gguf_backend,knowledge_store,onnx_anchor}.py`, CLI/MCP/WebUI, new feature modules specified by the three dependent changes, tests, benchmark scripts, optional dependency packaging and documentation. Checkpoint evolution requires versioned migrations and rollback; default adapters, exact retrieval and frozen byte-RTU inference remain compatible.

## Non-goals

This proposal does not implement features, certify mock tests as model-quality evidence, commit model weights, replace the RTU recurrence, or turn research targets into guarantees. Existing archives remain historical records until evidence-backed reconciliation. No automatic profile migration, background sleep, or model download is enabled by default.
