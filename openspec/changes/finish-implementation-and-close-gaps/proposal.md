# Proposal

## Why

RecaMem is broad and unusually honest about what it has not proven, but it is not
releasable. Verified on 2026-11-03 in this checkout:

- `scripts/offline_gate.py` passes 6/6 and `scripts/validate_openspec.py` passes,
  but `scripts/check_release_evidence.py --require-complete` exits 1: 3 `partial`
  and 2 `blocked` rows, plus no passing evidence for the `grounding` and
  `real-model` acceptance tiers (489 rows: 473 verified, 11 superseded).
- The last recorded offline result is 917 passed / 1 skipped, while
  `recamem-language-learning/tasks.md#7.2` records "10 pre-existing failures".
  There is no current full-suite result.
- No real model artifacts exist: no ONNX MiniLM, no GGUF, no
  `llama-cpp-python` build. `recamem doctor` exits 1. Tasks 3.4, 3.7, 7.10,
  8.1, 9.2, 9.4 and 9.5 of `2026-10-28-post-merge-completion` remain open.
- `docs/release-matrix.md` publishes unmet targets: 50k retrieval p95 8.87 ms,
  100k retrieval p95 17.46 ms and lexical p95 16.89 ms against a <5 ms target;
  Matryoshka 5.96x against >=6x; delta 20-task memory loss 0.1198 against
  <0.10; merge 2x10k facts 7.4 s against <=5 s; surprise-gate insert reduction
  0.0 against >=0.4.
- `docs/vela-poc-results.md` reports C (RecaMem + RTU) 3/12 against B (ordinary
  retrieval) 5/12 and E (oracle) 6/12. The central scientific claim is unproven,
  and the RTU contribution cannot be isolated because injection is prompt-only.
- Two active changes are unfinished. `recamem-language-learning` has 14
  unchecked tasks and the C4 core still sits at BPB 6.8 (no G2).
  `recamem-core-architecture-tournament` is fully checked, but its exported
  config disagrees with its own selection report.
- Concrete, reproducible repository defects: `notebooks/recamem_colab.ipynb`
  is invalid JSON (trailing `EOF`), `GLOSSARY.md` ends with stray shell text,
  `plans/README.md` links to a nonexistent `08_phase9_gate.md` (the file is
  `05_phase9_gate.md`), `configs/recamem_selected_core.json` declares
  `provisional_core` / `M3` / `interleaved` while
  `reports/recamem-core-selection.md` declares `verified_winner` / `M4` /
  `summed`, `AGENTS.md` and `docs/release-evidence.md` hardcode
  developer-machine absolute interpreter paths, and `.github/workflows/ci.yml` masks the
  remote-sensing E2E smoke with `|| true`.

The repository's own rule is "an archived checkbox is not verification". This
change converts that rule into a closed, evidence-backed state rather than a
growing list of caveats. It is deliberately a *closure* change: it fixes what is
left, verifies it, and leaves the release gate strict enough to reject a future
false completion.

## What Changes

- Add a machine-readable completion inventory at `reports/completion-gaps.json`
  and a guard `scripts/check_completion_gaps.py`. Every gap carries an id, a
  source (ledger row, spec task, doc claim or measured target), an owner, the
  verification command, and the artifact that must exist to close it. The guard
  fails when a gap is marked closed without its artifact and test.
- Fix all offline correctness, documentation and packaging defects listed
  above, each with a regression test that fails on the defective behaviour.
- Run the `real-model` and `grounding` acceptance tiers on hardware with model
  access, following `docs/testing.md`; record evidence with artifact sha256,
  binding/version, seed, configuration, command, commit and hardware. Anything
  not run stays `blocked` and is reported as such.
- Close or honestly re-baseline the seven unmet performance targets. A
  re-baseline requires an approved spec revision and a published measurement;
  it never silently lowers a threshold or deletes a case.
- Complete and integrate the two active research changes: language-learning
  debug ladder and G0-G4 gates, 1M/3M/5M token runs, config-to-report parity,
  then archive both changes with their own acceptance criteria satisfied.
- Make the RTU contribution measurable in the Vela benchmark (hidden injection
  or a documented `blocked` reason), run conditions C and D on a
  matched budget, and publish the comparison including a negative result.
- Compile the Tauri shell or report the exact missing toolchain, finish the UI
  pages currently labelled "not available yet", and make `recamem doctor` exit
  0 with a real anchor.
- Resolve the five unresolved ledger rows with real evidence, then make
  `--require-complete` exit 0 **without modifying the guard to accept less**.

## Capabilities

### New Capabilities
- `verification/completion-closure`: A single auditable inventory of remaining work, a closure protocol that requires implementation, a test and evidence, and an explicit ban on weakening gates to obtain a green result.

### Modified Capabilities
- `verification/release-evidence`: Require passing evidence in all five acceptance tiers before completion is declared, and keep unresolved rows visible until they are resolved.
- `verification/recamem-ci`: Forbid masked failures in CI and add machine checks for documentation artifact validity and portable command paths.
- `verification/surface-coverage`: Extend the shipped-surface contract to the desktop shell, the remaining UI pages and `recamem doctor`.
- `evaluation/edge-performance`: Require published misses to be either met or re-baselined through an approved requirement revision with provenance.
- `evaluation/vela-benchmark-poc`: Require a controlled C-versus-D comparison that isolates plastic updates and publishes negative results.
- `model/gguf-backend`: Require the manual real-model tier to be executed against real weights once, with a recorded artifact checksum and no mock substitution.
- `droid/onnx-anchor`: Require `doctor` and every evidence record to agree on the real anchor, with a real-anchor quality measurement.

## Impact

- **Affected code**: `recamem/engine.py`, `recamem/anchor.py`, `recamem/gguf.py`,
  `recamem/episodic.py`, `recamem/hybrid_retrieval.py`,
  `recamem/consolidation.py`, `recamem/api.py`, `recamem/cli.py`,
  `src/model/recamem_core.py`, `scripts/train_lm.py`,
  `scripts/run_core_tournament.py`, `scripts/run_vela_benchmark.py`,
  `scripts/bench_*.py`, new `scripts/check_completion_gaps.py`.
- **Affected specs**: the capabilities listed above; `configs/recamem_selected_core.json`
  becomes a generated artifact with a parity test against the tournament ledger.
- **Dependencies**: CPU runner for offline tiers; a machine with Hugging Face
  access, a C++ toolchain and disk space for ~1 GB of weights for the real-model
  tier; a GPU or many CPU hours for the 1M/3M/5M language runs.
- **Sequencing**: blocked by `recamem-language-learning` and
  `recamem-core-architecture-tournament`. The companion app proposed in
  `recamem-svelte-companion` is a separate, additive change and is out of scope.
- **Non-goals**: no new product surface, no committed model weights, no
  redefinition of "passed", no deletion of unresolved ledger rows, no
  restating a `blocked` tier as a pass.
