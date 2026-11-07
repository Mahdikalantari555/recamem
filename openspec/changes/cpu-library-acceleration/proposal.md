# Proposal: Evidence-led CPU libraries and optional native acceleration

## Why
The user wants to reuse existing libraries and consider Rust while unifying the CPU agent and enabling research. A broad rewrite or dependency choice without workload measurements would risk new packaging, security and correctness problems, and the available sandbox has no Rust toolchain and less RAM than the user's approximate 16 GB target.

## What Changes
- Survey orchestration, retrieval, inference, extraction and Rust integration candidates using upstream documentation and security advisories.
- Add reproducible CPU model profiling for C4 and Muir, including forward, incremental step, gradients, memory observations, provenance and streaming parity.
- Gate any optional library or Rust prototype on reproducible comparisons against the existing exact/reference implementation and preserve the Python-only core install.
- Record any unavailable toolchain, models, datasets or security controls as blocked rather than pretending a candidate was tested.

## Capabilities
- `performance/cpu-library-evaluation`: profiling, quality/correctness gates and optional acceleration evaluation.

## Non-goals
No mandatory Rust, GGUF, GPU, external service, unmeasured speedup, or migration of canonical memory. No deployment of an unsandboxed browser or arbitrary agent code. The initial CPU harness is a synthetic microbenchmark, not a language or general-memory benchmark.

## Impact
Benchmark scripts, tests and documentation; later optional dependencies and native extensions require a separate evidence-backed decision. The existing unification and web OpenSpec changes remain authoritative for functionality and safety.

## Continuous learning alignment

Acceleration candidates must support, or explicitly report incompatibility with, automatic configuration, incremental training, evaluation and safe checkpoint promotion in `autonomous-incremental-learning`.
