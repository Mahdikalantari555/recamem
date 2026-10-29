# Tasks

## 1. Data Preparation

- [x] 1.1 Create `scripts/prepare_tinystories.py` that downloads TinyStories train+val, tokenizes with V=1024 BPE, and writes `train.bin`/`val.bin` as uint16 memmap; verify byte-identical output on re-run
- [ ] 1.2 Create fixed val split (~5,000 stories, hash-based) and verify no overlap with train
- [x] 1.3 Pack sequences with EOS between stories (no padding) and verify sequence lengths in memmap
- [x] 1.4 Record tokenizer artifact (`configs/recamem_selected_bpe.json`) with vocab size 1024 and verify round-trip lossless UTF-8

## 2. N-gram Baselines

- [x] 2.1 Create `scripts/eval_ngram.py` that computes unigram/bigra/trigram/4-gram byte-level BPB on val set with simple smoothing; verify output matches manual calculation on 100 samples
- [x] 2.2 Run n-gram baselines on TinyStories val and record G1/G2/G3 thresholds; verify values are finite and monotonically decreasing (unigram > bigram > trigram > 4-gram)

## 3. Standalone LM Training Script

- [x] 3.1 Create `scripts/train_lm.py` with C4 core, AdamW (betas 0.9/0.95, wd 0.1, clip 1.0), LR 1e-3–3e-3 with warmup + cosine decay; verify loss decreases on first 100 steps
- [x] 3.2 Add `--lm-tokens` argument (replaces `--lm-steps`) and log tokens seen, tok/s, current LR every step; verify log format matches spec
- [x] 3.3 Add checkpoint/resume every 1,000 steps (model + optimizer + step count + corpus position, atomic write); verify resume produces identical trajectory to uninterrupted run
- [x] 3.4 Add generation logging every 1,000 steps (5 samples at temperature 0.8) and verify samples are written to file
- [x] 3.5 Add model config printing (d, layers, V, embed params, non-embed params) in run report; verify report contains all fields
- [x] 3.6 Add `torch.set_num_threads(physical_cores)` and bf16 autocast if supported; verify thread count and precision in log

## 4. Debug Ladder (Phase 0)

- [ ] 4.1 Implement overfit single batch test (300 steps, loss ~0) and verify it passes on 1M model
- [ ] 4.2 Run n-gram baselines (from task 2) and record G1/G2/G3 thresholds; verify values are reasonable
- [ ] 4.3 Train C6 reference transformer on 5M tokens and verify it beats trigram baseline
- [ ] 4.4 Train C4 on 1M/5M/20M/50M tokens and plot BPB vs tokens; verify monotonic decrease
- [ ] 4.5 Generate 20 stories at temperature 0.8 and manually review; verify ≥15 readable
- [ ] 4.6 Run MQAR-4 on LM-trained model and verify accuracy >0.75

## 5. Tournament Modifications

- [x] 5.1 Add `--lm-tokens` argument to `scripts/run_core_tournament.py` alongside existing `--lm-steps`; verify backward compatibility
- [x] 5.2 Add `--checkpoint-every` argument (default 1000) and verify checkpoint files are created
- [ ] 5.3 Add n-gram baselines to Stage 0 of tournament; verify they appear in report
- [ ] 5.4 Add model config printing to tournament report; verify all fields present
- [ ] 5.5 Add generation logging to tournament LM training; verify samples in report

## 6. Success Gates & Scale-Up

- [x] 6.1 Implement G0–G4 gate checks in `train_lm.py` and verify each gate is evaluated and recorded
- [ ] 6.2 Train 1M model to G2 (BPB below trigram) and verify gate passes; estimated 0.5–1.7 h
- [ ] 6.3 Train 3M model (60M tokens) after G2 passes; verify G2 passes; estimated 5–15 h
- [ ] 6.4 Train 5M model (100M tokens) after 3M succeeds; verify G2 passes; estimated 14–42 h
- [ ] 6.5 Run MQAR-4/16 on each trained model and verify accuracy >0.75; record in report

## 7. Integration & Verification

- [x] 7.1 Run `PYTHONPATH=. python scripts/offline_gate.py` and verify all checks pass (pre-existing evidence ledger failure unrelated to this change)
- [x] 7.2 Run `PYTHONPATH=. python -m pytest tests -q` and verify no new failures (10 pre-existing failures, none introduced by this change)
- [x] 7.3 Verify `plans/training-plan.md` is consistent with implemented code; update if needed
- [ ] 7.4 Archive the openspec change after all tasks complete and verify with `openspec validate`