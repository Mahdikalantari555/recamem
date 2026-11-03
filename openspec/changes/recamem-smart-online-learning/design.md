# Design: Smart Online-Learning Engine (Hybrid SLM + Gated Delta Core)

## Context
RecaMem contains both:
- A high-speed plastic associative memory engine (`PlasticAssociativeRTU` and candidate core `C4: GatedDeltaMatrixCore`).
- A GGUF backend adapter (`GgufBackend` using `llama-cpp-python`) validated against local MiniCPM-1B weights.

Currently, these two components are only loosely connected through static prompt prefixing. This design specifies an end-to-end continuous learning loop with surprise-gated plastic updates and associative context injection.

## Architecture

### 1. The Online Plastic Step
For each incoming token/proposition embedding $e_t \in \mathbb{R}^d$:
1. **Prediction**: Compute top-down expectation $\hat{e}_t = S_{t-1} k(e_t)$.
2. **Surprise**: Compute cosine prediction error:
   $$\text{surprise}_t = 1.0 - \frac{\langle \hat{e}_t, v(e_t) \rangle}{\|\hat{e}_t\| \|v(e_t)\|}$$
3. **Selective Update**:
   - If $\text{surprise}_t < \text{threshold}$ (default $0.4$): skip update.
   - If $\text{surprise}_t \ge \text{threshold}$: update fast matrix state using the Gated Delta Rule:
     $$S_t = S_{t-1} (I - \alpha_t k_t k_t^T) + \eta_t (v_t - S_{t-1} k_t) k_t^T$$
   - Write fact into episodic SQLite store for auditability and consolidation.

### 2. Dual-Path Chat Generation
- **Path 1 (Fast Recall, $< 5\text{ ms}$)**: Direct associative retrieval from $S_t$ and SQLite knowledge base. Returns exact propositions if similarity $\ge 0.60$.
- **Path 2 (Slow Generative Synthesis, $< 800\text{ ms}$)**:
  1. Retrieve top-$K$ grounded propositions from associative memory.
  2. Inject them into the GGUF backend context using structured prompt blocks.
  3. Execute greedy or low-temperature generation with `MiniCPM-V-4_7-1B`.

## Security & Secrets
- Uses local GGUF weights exclusively.
- No network requests, zero external API keys required.
