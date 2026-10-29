# Design

## Context

Current state: C4 core (349,920 params) passes all mechanistic gates (MQAR-4/16 at 1.000) but BPB is 6.8 (random baseline 8.0). Training used 160–240 steps on fixture data. The model needs 20M–100M tokens to learn language. Training infrastructure exists (`scripts/run_core_tournament.py`, `src/model/rtu.py`) but lacks token-based training, checkpoint/resume, and n-gram baselines.

See `plans/training-plan.md` for the full training methodology and `docs/راهنمای آموزش یادگیری واقعی زبان برای RecaMem.md` for the research basis.

## Goals / Non-Goals

**Goals:**
- Enable token-based LM training (not step-based) with checkpoint/resume.
- Add n-gram baselines as measurable language gates.
- Provide standalone training scripts separate from the tournament.
- Define clear success gates (G0–G4) that prevent wasted compute.

**Non-Goals:**
- Do not modify the core architecture (C4) — it passes all mechanistic gates.
- Do not implement BitNet ternary or linear-attention cumsum yet (Phase 2.3/2.4, optional).
- Do not train the 17M model yet (requires GPU or multi-day CPU).
- Do not add decision/router heads until LM reaches G2.

## Decisions

### D1: Standalone training script vs. tournament extension
**Decision**: Create `scripts/train_lm.py` as a standalone script, separate from `run_core_tournament.py`.
**Rationale**: The tournament is for architecture comparison. Language learning needs a simpler, faster loop. Mixing them slows iteration.
**Alternative**: Extend the tournament with `--lm-tokens`. Rejected because the tournament's multi-arm overhead is unnecessary for single-model training.

### D2: Tokenizer vocab size
**Decision**: Use V=1024 or V=2048 for sub-5M models (not 4096).
**Rationale**: With d=192, V=4096 wastes 786K params on embedding alone (78% of 1M budget). V=1024 uses 196K (20%).
**Alternative**: Keep V=4096 for consistency. Rejected because embedding-dominated models cannot learn language.

### D3: TinyStories as primary dataset
**Decision**: Use TinyStories (full train split, ~470M tokens) as the primary dataset.
**Rationale**: Proven to enable coherent generation with <10M params. Open-source. Well-studied baselines.
**Alternative**: General web text (FineWeb). Rejected for Phase 0–3 because general text requires much larger models. TinyStories provides a controlled environment for validating the training pipeline.

### D4: Checkpoint interval
**Decision**: Checkpoint every 1,000 steps (model + optimizer + step count + corpus position).
**Rationale**: Multi-hour runs need resumability. 1,000 steps ≈ 15–30 min on CPU. Atomic write (temp file + os.replace) prevents corruption.
**Alternative**: Every 500 steps. Rejected because checkpoint I/O overhead is non-trivial on CPU.

### D5: N-gram baselines as gates
**Decision**: Compute unigram/bigra/trigram/4-gram byte-level BPB on the same val set. These become G1/G2/G3 thresholds.
**Rationale**: Without baselines, "BPB 6.8" is meaningless. The guide doc's diagnosis depends on knowing where the model sits relative to simple statistics.

### D6: Tied embeddings
**Decision**: Use tied embeddings (same matrix for input and output).
**Rationale**: Halves embedding params. Standard practice for small models. No downside for language modeling.

### D7: No BitNet/linear-attention in Phase 0–3
**Decision**: Train in fp32 (or bf16) with standard attention first. Add ternary weights and linear attention only after confirming the model learns language.
**Rationale**: Changing too many variables at once makes debugging impossible. First prove the pipeline works, then optimize.

## Risks / Trade-offs

- **Risk**: CPU training may be too slow for 5M+ models. → Mitigation: time estimates in plan; use free GPU if needed.
- **Risk**: TinyStories results may not generalize to general text. → Mitigation: this is expected; plan includes Cosmopedia scale-up later.
- **Risk**: N-gram baselines may be too easy/hard. → Mitigation: adjust thresholds after first run with real numbers.
- **Risk**: Checkpoint I/O on CPU may bottleneck. → Mitigation: async write or reduce frequency.
- **Trade-off**: Standalone script duplicates some tournament code. → Accepted for faster iteration.

## Migration Plan

1. Create new scripts (`prepare_tinystories.py`, `train_lm.py`, `eval_ngram.py`, `generate_samples.py`).
2. Modify `run_core_tournament.py` to add `--lm-tokens`, `--checkpoint-every`, n-gram baselines in Stage 0, config printing.
3. Run Phase 0 (debug ladder) — ~1 hour.
4. If G0–G2 pass, proceed to Phase 3 (full training).
5. No rollback needed — all changes are additive. Existing tournament behavior is preserved.