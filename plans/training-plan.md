# RecaMem Language Learning — Training Plan

Date: 2026-10-28. Based on `docs/راهنمای آموزش یادگیری واقعی زبان برای RecaMem.md` and deep research.

## 0. Goal

Make the selected core (C4 gated-delta, 349,920 params) actually learn language.
Current state: BPB 6.8 (random baseline 8.0). Model only learned simple byte frequencies.

Success = pass G2: BPB below trigram baseline + readable generation (15/20 samples).

## 1. Phase 0 — Debug Ladder (before any long training)

Run these in order. Stop if any fails.

### 0a. Overfit single batch
- Take one fixed batch of 8 sequences × 128 tokens from training data.
- Train 300 steps on it only. Loss must reach ~0.
- If not: bug in model, loss, label shift (next-token), or LR.

### 0b. N-gram baselines on val set
- Compute unigram, bigram, trigram, 4-gram byte-level BPB with simple smoothing on the same val split.
- Record all four. These are the gates (G1–G3).

### 0c. Reference Transformer (C6) on 5M tokens
- Train C6 (linear_kv, softmax attention) on 5M tokens, same data, same tokenizer.
- If C6 does not beat trigram → data or eval code bug, not architecture.

### 0d. Token ladder
- Train C4 on 1M, 5M, 20M, 50M tokens (from scratch or continued with correct schedule).
- Plot BPB vs tokens (log scale). Must be monotonically decreasing and roughly linear.
- If it flattens → LR or batch size problem.

### 0e. Generation test
- Generate 20 stories at temperature 0.8. Read by eye: grammar, repetition, endings.
- Compute distinct-2 and 4-gram repetition rate.

### 0f. Memory re-test
- Run MQAR-4 on the LM-trained model. Must stay >0.75 (G4).

## 2. Phase 1 — Data Preparation

### 2.1 Dataset
- **Primary**: TinyStories (train split, full — not fixture). ~470M tokens with Llama tokenizer.
- **Secondary** (later): Cosmopedia (28B tokens) for generalization.
- **Custom**: Generate "textbook" data using an open teacher model (Llama-3-8B or Qwen-2.5-7B) to replicate Microsoft's Phi approach. This is the closed-source "way" — we replicate it with open models.

### 2.2 Tokenizer
- **V=1024 or 2048** for sub-5M models (not 4096). Embedding table = V×d; with d=64, V=4096 wastes 262K of 350K params.
- Tied embeddings (same matrix for input and output).
- BPE with byte fallback.

### 2.3 Preprocessing
1. Download TinyStories train + val.
2. Tokenize once with chosen vocab size.
3. Store as `uint16` memmap files: `train.bin`, `val.bin`.
4. Pack sequences with EOS between stories. No padding.
5. Fixed val split: ~1% (5,000 stories) with hash for reproducibility.

### 2.4 Code changes needed in tournament
- Replace `--lm-steps` with `--lm-tokens` in `scripts/run_core_tournament.py`.
- Log: tokens seen, tok/s, current LR at every step.
- Add n-gram baselines to Stage 0.
- Add checkpoint/resume every 1,000 steps.
- Print model config (d, layers, V, embedding params, non-embedding params) in every report.

## 3. Phase 2 — Model Configuration

### 3.1 Target: 1M params (first success target)

| Param | Value |
|-------|-------|
| Core | C4 (gated-delta + conv4) |
| d_model | 192 |
| n_layers | 6–8 |
| n_heads | 4 |
| d_ff | ~768 |
| Vocab | 1024 or 2048 |
| Context | 256 |
| Total params | ~1M (target: 1M–3M) |
| Non-embedding params | ~700K (most params should NOT be in embedding) |

### 3.2 Init
- Normal, std ≈ 0.02
- Output projections scaled by 1/sqrt(2L)
- Gate (forgetting alpha) init so sigmoid ≈ 0.9–0.99
- No bias in linear layers (BitNet style)

### 3.3 Optional: ternary weights (BitNet)
- Replace `nn.Linear` with `BitLinear` (weights {-1,0,+1}).
- Reduces memory 16x, enables CPU-only inference.
- Do this AFTER confirming the fp32 model learns language.

### 3.4 Optional: linear attention cumsum (FlashLM style)
- Replace O(N²) attention with O(N) cumsum.
- Fused projections (single d→3k matmul).
- Do this AFTER confirming the standard attention model learns language.

## 4. Phase 3 — Training

### 4.1 Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| betas | (0.9, 0.95) |
| weight decay | 0.1 (not on norm/bias) |
| gradient clip | 1.0 |
| LR peak | 1e-3 to 3e-3 |
| warmup | 2–5% of total steps |
| schedule | cosine or linear decay to 10% or 0 |
| dropout | 0 (none) |
| batch | 16k–64k tokens/step (gradient accumulation) |
| precision | bf16 if CPU supports AMX/AVX512-BF16, else fp32 |
| threads | `torch.set_num_threads(physical_cores)` |

### 4.2 Token budget
| Model size | Tokens target | Est. time (CPU) |
|-----------|--------------|-----------------|
| 1M | 20M | 0.5–1.7 h |
| 3M | 60M | 5–15 h |
| 5M | 100M | 14–42 h |
| 17M | 340M | 7–20 days (GPU recommended) |

### 4.3 Training loop
1. Load memmap data.
2. Every 200 steps: compute val loss on fixed val set.
3. Every 1,000 steps: checkpoint (model + optimizer + step count).
4. Every 1,000 steps: generate 5 samples at temperature 0.8, log to file.
5. On resume: load checkpoint, continue with correct LR for current step.

### 4.4 Evaluation
- BPB = (sum NLL in nats / ln2) / UTF-8 byte count of eval text.
- Val loss every 200 steps.
- 20 generated stories every 5,000 steps (manual review).

## 5. Phase 4 — Success Gates

| Gate | Condition | Meaning |
|------|-----------|---------|
| G0 | Overfit single batch succeeds | Code is correct |
| G1 | BPB 25% below unigram | Learned common chars/words |
| G2 | BPB below trigram + readable gen (15/20) | Language begins |
| G3 | BPB below 4-gram + coherent gen | Basic grammar |
| G4 | MQAR-16 >0.75 on LM-trained model | Recall preserved |

Only proceed to cores/tokenizer/interference comparison after G2 passes.

## 6. Phase 5 — Scale-Up (after G2)

1. **3M model**: Same recipe, 60M tokens. ~5–15 hours.
2. **5M model**: Same recipe, 100M tokens. ~14–42 hours.
3. **17M model**: Requires GPU (free cloud) or multi-day CPU. 340M tokens.
4. **Decision/router heads**: Add after LM reaches G2. Use curriculum sampling.
5. **JEPA objective**: Keep off until LM is solid.

## 7. Code Changes Required

### In `scripts/run_core_tournament.py`
- Add `--lm-tokens` argument (replaces `--lm-steps`).
- Add `--checkpoint-every` (default 1000).
- Add `--resume` support (already exists for scaleup).
- Add n-gram baseline computation to Stage 0.
- Print model config (d, layers, V, embed params, non-embed params) in report.

### New files needed
- `scripts/prepare_tinystories.py` — download, tokenize, memmap.
- `scripts/train_lm.py` — standalone LM training (not tournament). For Phase 0–3.
- `scripts/eval_ngram.py` — n-gram baselines.
- `scripts/generate_samples.py` — generation test.

### In `src/model/rtu.py`
- Add `BitLinear` option (ternary weights) — optional, Phase 2.3.
- Add linear attention cumsum option — optional, Phase 2.4.

## 8. Risk & Honest Notes

- Even with correct training, 1M–5M models only write simple stories (TinyStories vocab). No general knowledge, no Q&A, no Persian.
- TinyStories results don't generalize to general text. Need much larger model + tokens for that.
- Gate thresholds are estimates. Adjust after first full run with real numbers.
- Time estimates are analytical. Actual CPU speed may be 2x better or worse.
- If CPU is too slow for 17M, one occasional free GPU run is reasonable. Not mandatory.

## 9. Immediate Next Steps

1. Create `scripts/prepare_tinystories.py` (download + tokenize + memmap).
2. Create `scripts/train_lm.py` (standalone C4 training with checkpoint/resume).
3. Create `scripts/eval_ngram.py` (baselines).
4. Run Phase 0 (debug ladder) — ~1 hour.
5. If G0–G2 pass on 1M model, proceed to 3M and 5M.
6. Update `human.md` memory block with training plan summary.<|end|><|start|><|start|>assistant to=journal_write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="body">
Created comprehensive training plan for RecaMem language learning. Key points: (1) Phase 0 debug ladder (overfit, n-gram baselines, reference transformer, token ladder, generation test, memory re-test). (2) Phase 1 data prep (TinyStories, V=1024/2048, memmap uint16, tied embeddings). (3) Phase 2 model config (1M params, C4 gated-delta, d=192, 6-8 layers, optional BitNet ternary + linear attention). (4) Phase 3 training (AdamW, LR 1e-3-3e-3, cosine schedule, no dropout, 16k-64k tokens/step, checkpoint every 1000). (5) Phase 4 gates G0-G4. (6) Phase 5 scale-up (3M, 5M, 17M). (7) Code changes needed: --lm-tokens, checkpoint/resume, n-gram baselines, standalone train_lm.py. Plan saved to plans/training-plan.md.