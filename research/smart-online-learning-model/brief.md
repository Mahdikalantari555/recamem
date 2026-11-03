# Research Brief: Smart Online-Learning Neural Architecture

- **Topic**: Designing a smart, fluent language model architecture capable of real-time online learning (fast weight adaptation / TTT / continuous factual and conversational absorption) without catastrophic forgetting, runnable on standard local hardware (CPU + optional local GGUF).
- **Date**: 2026-11-03
- **Depth**: standard (4 parallel angles, 1 follow-up round budget)
- **Scope Boundaries**:
  - In scope: Test-Time Training (TTT / Sun et al. 2024–2025), Titans (Google 2024–2025), RWKV-7 with vector-valued delta rules, RecaMem C4 Gated Delta Matrix integration with pre-trained SLMs (MiniCPM-1B / LFM / Qwen-1.5B), continuous factual updating, surprise-gated Hebbian retention, mitigating catastrophic forgetting.
  - Out of scope: Cloud-only closed APIs, multi-million dollar cluster pre-training from scratch.
- **Goal**: Formulate the architectural blueprint and concrete OpenSpec change proposal for RecaMem.

## Angles
1. **Hybrid Architecture Patterns**: How pre-trained SLMs (MiniCPM/Qwen/Llama) can be coupled with online linear recurrent / delta-rule layers (TTT-Linear, Titans neural memory, DeltaNet) at inference and decode time.
2. **Online Plasticity & Surprise Gating**: How the model decides *what* to memorize vs. what to discard during online interaction (surprise metrics, prediction error gradients, weight decay, memory compaction).
3. **Catastrophic Forgetting & Retrieval Anchoring**: Mechanisms to prevent online weights from corrupting pre-trained language fluency (orthogonal projections, multi-tier fast/slow states, episodic grounding).
4. **Implementation & OpenSpec Proposal Specification**: Concrete software design in RecaMem (`recamem/`) linking the local GGUF backend (`MiniCPM-V-4_7-1B`) to dynamic associative state matrices with streaming CLI / companion UI integration.
