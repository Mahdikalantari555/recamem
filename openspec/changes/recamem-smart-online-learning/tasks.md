# Tasks: Smart Online-Learning Model

## 1. Fast-Weight Surprise Engine
- [x] 1.1 Expose continuous streaming update API in `recamem/plasticity.py` using `GatedDeltaMatrixCore` (C4).
- [x] 1.2 Implement cosine surprise calculation with dynamic $\alpha_t$ (erasure) and $\eta_t$ (plasticity rate).
- [x] 1.3 Add test `tests/test_online_surprise_gating.py` verifying novel facts trigger updates while repeated text produces zero parameter drift.

## 2. GGUF Integration & Memory Injection
- [x] 2.1 Enhance `recamem/gguf.py` to support dynamic session context priming with the local `MiniCPM-V-4_7-1B` model.
- [x] 2.2 Wire `DroidEngine.chat()` to automatically route between associative fast recall and GGUF-synthesized slow path based on query complexity.
- [x] 2.3 Add integration test `tests/test_smart_online_chat.py` verifying real-time learning followed by fluent generative answers.

## 3. Verification & Benchmarks
- [x] 3.1 Benchmark CPU ingestion speed: target $>50,000\text{ tokens/second}$ for plastic state updates.
- [x] 3.2 Verify zero catastrophic forgetting of base model reasoning on benchmark prompts.
- [x] 3.3 Run `PYTHONPATH=. python scripts/offline_gate.py` and ensure 100% pass.
