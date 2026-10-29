# Proposal

## Why

RecaMem has no measured evidence that its plastic memory improves over ordinary retrieval on a controlled, synthetic reasoning task. The Vela text-only learning benchmark (`recamem_text_learning_benchmark.md`) supplies a small rule system with an explicit update event, underspecified inputs, and compositional cases, which is ideal for a proof-of-concept (PoC) test of teaching, recall, belief revision, and retention.

## What Changes

- Add a runnable PoC harness `scripts/run_vela_benchmark.py` that executes conditions A (base model, no memory), B (base model + ordinary retrieval), C (RecaMem + RTU), and E (oracle rules) using the local MiniCPM GGUF model, and optionally D (RTU frozen) as a best-effort isolation of plastic updates.
- Add a deterministic decision extractor that maps model output to one of `IRRIGATE`, `DO_NOT_IRRIGATE`, `INSUFFICIENT_INFORMATION`, `CONFLICT_REQUIRES_CLARIFICATION`.
- Fix belief-revision supersede detection so Vela-style policy updates (same soil code + phase, changed threshold) mark the obsolete rule as superseded while preserving it in the audit trail. Add a domain-aware update helper in `recamem/engine.py` triggered by explicit "policy update" / "superseded" markers.
- Write a JSON report under `reports/` with per-item decisions, per-category accuracy, latency, and retrieved evidence, following the provenance rules in `evaluation/recamem-benchmarks`.
- Add tests in `tests/test_vela_benchmark.py` covering scoring and update handling on a tiny fixture.

## Capabilities

### New Capabilities
- `evaluation/vela-benchmark-poc`: Run the Vela text-only learning benchmark as a PoC against RecaMem and report accuracy by category.

### Modified Capabilities
- `belief-revision`: Extend contradiction detection so explicit policy updates with a shared soil-code + phase supersede the prior threshold rule.

## Impact

- **Affected code**: `recamem/engine.py` (update helper in `_teach_impl`), new `scripts/run_vela_benchmark.py`, new `tests/test_vela_benchmark.py`.
- **Affected specs**: new `openspec/specs/evaluation/vela-benchmark-poc/spec.md`; modified `openspec/specs/belief-revision/spec.md`.
- **Dependencies**: local MiniCPM GGUF at `D:/AI/.lmstudio/models/OpenBMB/MiniCPM5-2B-Q4_K_M/MiniCPM5-2B-Q4_K_M.gguf`, ONNX anchor, CPU.
- **No breaking changes**: additions only; supersede helper is additive.
