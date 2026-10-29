# Improving RecaMem: Faster Model + Online Learning

Date: 29 October 2026. Synthesis of `docs/cpu-lm-training-way.md`, the Vela PoC (`docs/vela-poc-results.md`), and deep research in `research/online-learning/findings/`.

## 1. Diagnosis (from Vela PoC)
- Retrieval is strong (9–12/12). The bottleneck is the LLM decision step and the lack of hidden injection.
- RTU plastic updates did not change output because injection is prompt-only. RTU's contribution is unmeasured.
- Base model (MiniCPM 2B) does not follow the exact-token format.

## 2. Make the model faster (CPU)
Sources: `research/online-learning/findings/F2.md`, `docs/cpu-lm-training-way.md` §10–12.

1. Quantize byte-RTU / GGUF to INT4 (or BitNet ternary) via bitnet.cpp or llama.cpp. Memory-bandwidth-bound decode benefits most from fewer bytes/token.
2. Keep KV cache in INT8; use hybrid conv+attention layers (Daedalus-style) for 1.76–2.08x decode at long context.
3. Use linear attention (RWKV/Mamba) or conv+SSM for O(N) decode; avoid O(N²) full attention on CPU.
4. Co-design runtime: tile-based execution (cflow), `n_threads` = physical cores, `n_batch` large, mmap weights.
5. Prefer smaller, deeper models (32 thin layers > 12 wide) per optimal-architecture findings; use WSD (Warmup-Stable-Decay) schedule needing 10x fewer tokens.
6. Extreme overtraining: ~222K tokens/param for small models with small vocab (V=1024–2048, not 4096).

## 3. Online learning (plasticity)
Sources: `research/online-learning/findings/F1.md`, `F3.md`.

1. Non-parametric memory first: Dynamic Cheatsheet / ReasoningBank-style self-curated memory shows +20% relative on 1.7B SLMs and is the most CPU-feasible path. RecaMem already has this (episodic store); strengthen curation.
2. Test-time training: TTT-NTP works drop-in down to 0.6B. Add a lightweight TTT layer that updates a small MLP on next-token prediction during inference, gated by surprise.
3. Titans-style neural memory: surprise = gradient magnitude; add momentum + weight decay (forgetting) + persistent memory. Use MAC (memory as context) variant; memory-only updates plateau, so coordinate with main model.
4. Surprise-gated amplitude writes (RecaMem already has surprise; tune threshold). Capacity sizing law ~0.5 writes/param.
5. Avoid ReLoRA below 10M params (amplifies rank deficiency). Prefer plastic adapter (associative RTU) which RecaMem already uses.
6. Keep dense weight updates rare; prefer plastic/no-gradient memory for CPU.

## 4. Concrete next steps for RecaMem
1. Add hidden-injection support for a compatible backend so RTU contribution (C vs D) can be measured.
2. Add TTT-NTP module behind `test_time_training` flag, gated by surprise.
3. Quantize byte-RTU checkpoint to INT4 and benchmark decode speed.
4. Switch default vocab to 1024–2048 BPE for sub-5M models; tie embeddings.
5. Implement WSD LR schedule and extreme overtraining in `train_lm.py`.
6. Add a decision-following eval using a stronger instruction model (or a deterministic rule interpreter for memory tests) to separate memory quality from LLM formatting.

## 5. Expected impact
- Speed: 2–4x decode improvement from quantization + linear attention + runtime tuning.
- Online learning: measurable C > B on Vela-style updates once hidden injection and TTT are in place.
- Quality: BPB 6.8 → ~3–4 with Tier 1 (more tokens + TinyStories + Cramming), per `cpu-lm-training-way.md`.