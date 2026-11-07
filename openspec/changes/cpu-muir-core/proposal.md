# Proposal: CPU Muir sequence core

## Why
Muir (arXiv:2611.08376v1) offers a gated-delta recurrence with sensitivity-allocated gates and matched-filter readout that is worth testing against RecaMem's selected C4. The paper's TPU and large-model results are not evidence of CPU or small-model gains.

## What Changes
Add an opt-in PyTorch CPU Muir candidate conforming to SequenceCore, starting with a correct sequential reference and matched-budget evaluation. Keep the selected C4, its checkpoints, and the runtime memory engine unchanged. A performant parallel implementation is a later, independently verified optimization. No GGUF, llama.cpp, GPU, JAX, or TPU dependency is required.

## Capabilities
- `model/muir-core`: Muir recurrence, stateful execution, correctness and comparison.

## Non-goals
No claim of paper-level quality or speed, no automatic replacement of the frozen generator or the episodic memory engine, and no conversion of existing C4 weights.

## Impact
Candidate registry, parameter matcher, core tests and benchmark tooling. CPU base learning is separately proposed by `cpu-base-learning` and can begin with C4 while this core is incomplete.
