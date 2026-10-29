# Proposal

## Why

The selected core (C4 gated-delta, 349,920 params) passes all mechanistic gates (MQAR-4/16 at 1.000) but has not learned language: BPB 6.8 vs random baseline 8.0. The model only captured simple byte frequencies. The root cause is insufficient training data and steps (160–240 steps vs. needed 20M–100M tokens). Without a structured plan, long training runs risk wasting days of CPU compute on misconfigured experiments.

## What Changes

- Add a language-learning training methodology with a debug ladder (Phase 0) that catches code/data bugs before expensive runs.
- Add n-gram baselines (unigram through 4-gram) as measurable gates for "language learned."
- Add token-budget targets and time estimates for 1M/3M/5M/17M param models on CPU.
- Add success gates (G0–G4) that must pass before proceeding to larger models or architecture comparisons.
- Add a scale-up protocol (3M → 5M → 17M) with checkpoint/resume every 1,000 steps.
- Add code changes: `--lm-tokens` (replaces `--lm-steps`), checkpoint/resume, n-gram baselines in Stage 0, config printing in reports.
- Add new scripts: `prepare_tinystories.py`, `train_lm.py`, `eval_ngram.py`, `generate_samples.py`.
- Optional: BitNet ternary weights and linear-attention cumsum (after confirming fp32 model learns language).

## Capabilities

### New Capabilities
- `training/language-learning`: Debug ladder, n-gram baselines, token budget, success gates (G0–G4), scale-up protocol, and code changes for making the C4 core learn coherent English on CPU.

### Modified Capabilities
- `training/edge-pretraining`: Add language-learning acceptance criteria (BPB below n-gram baselines, readable generation) to the existing scale-run acceptance criteria.

## Impact

- **Affected code**: `scripts/run_core_tournament.py` (add `--lm-tokens`, `--checkpoint-every`, n-gram baselines), new scripts in `scripts/`, `src/model/rtu.py` (optional BitLinear/linear-attention).
- **Affected specs**: `openspec/specs/training/edge-pretraining/spec.md` (modified), new `openspec/specs/training/language-learning/spec.md`.
- **Dependencies**: TinyStories dataset (~470M tokens), Python ≥3.10, torch, tiktoken/sentencepiece for BPE.
- **No breaking changes**: All additions are backward-compatible with existing tournament and memory engine.