# Change Proposal: Smart Online-Learning Model via Gated Delta Fast Weights + Local GGUF

- **Change ID**: `recamem-smart-online-learning`
- **Target**: `recamem`
- **Status**: Proposed

## Why
Standard pre-trained language models (MiniCPM, Qwen, Llama) are static after weights are finalized. Retraining or fine-tuning them on streaming conversational text requires full backpropagation with AdamW, which is too slow on CPU and rapidly causes catastrophic forgetting of general language fluency.

Conversely, standalone plastic memory blocks can learn facts in milliseconds, but lack native conversational fluency, deep grammatical nuance, and broad world knowledge.

## What
Build a unified, hybrid continuous-learning agent that pairs:
1. **Slow Generative Tier**: The local GGUF model (`MiniCPM-V-4_7-1B-Q8_0.gguf`) with weights permanently frozen, providing fluent language syntax, reasoning, and conversational comprehension.
2. **Fast Plastic Memory Tier**: A high-speed `GatedDeltaMatrixCore` (C4 architecture) running online rank-1 Householder delta updates in CPU cache ($>100\text{k tokens/sec}$).
3. **Surprise Gating Engine**: Evaluates prediction error against existing associative state so routine filler tokens are ignored while novel facts trigger immediate plastic storage.
4. **Context Injection Pipeline**: Dynamically primes recalled associative memory vectors into the GGUF backend context during `/slow` synthesis and streaming chat.

## Success Criteria
- [ ] Ingests new text at $>50,000\text{ tokens/sec}$ on CPU via plastic delta updates.
- [ ] Correctly answers questions about recently learned facts using the frozen local GGUF model without hallucinations.
- [ ] Preserves 100% of base model grammar and fluency across repeated online learning sessions (zero weight drift on slow tier).
- [ ] Passes all offline gate verification checks (`scripts/offline_gate.py`).
