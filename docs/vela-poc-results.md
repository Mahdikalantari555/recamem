# Vela PoC Results

Date: 29 October 2026. Proof-of-concept run of the Vela text-only learning benchmark on RecaMem using MiniCPM 2B (Q4_K_M) on CPU.

## Setup
- Model: MiniCPM 2B Q4_K_M (GGUF, CPU, 16 threads)
- Anchor: ONNX MiniLM (real)
- Conditions: A (base, no memory), B (base + ordinary retrieval), C (RecaMem + RTU), D (RTU frozen), E (oracle rules)
- Items: V1–V12 (see `recamem_text_learning_benchmark.md`)
- Report: `reports/vela-poc-final.json`

## Results

| Cond | Decision acc | Retrieval hits |
|---|---|---|
| A (base, no mem) | 1/12 | 0/12 |
| B (base + retrieval) | 5/12 | 10/12 |
| C (RecaMem + RTU) | 3/12 | 9/12 |
| D (RTU frozen) | 3/12 | 10/12 |
| E (oracle rules) | 6/12 | 12/12 |

## Interpretation (per benchmark's stated criteria)
- **C ≈ B on retrieval** (9 vs 10/12): no evidence yet that RTU adds value beyond ordinary retrieval on this task.
- **E is higher than B/C** (6/12 vs 3–5/12): retrieval/memory selection is not the bottleneck; the base model's decision procedure is.
- **D ≈ C**: RTU plastic updates do not change output because injection is prompt-only with this backend (hidden injection unavailable).
- Retrieval itself works well (9–12/12 hits). The LLM is the bottleneck, not memory.

## What worked
- Teaching, recall, and supersede pipeline runs end-to-end.
- Policy-update supersede marks the obsolete Amber phase-2 rule (threshold 36) as superseded while retaining it in the audit trail.
- Conditions run from clean state; no answer-key leakage into prompts.

## What did not work
- MiniCPM 2B does not emit the exact required decision tokens; extraction heuristics were needed and decision accuracy stayed low.
- RTU's own contribution could not be isolated (no hidden injection).

## Open questions
- Does a more instruction-following model raise decision accuracy under condition E?
- Does hidden injection (requires compatible backend) show C > D?
- Do held-out generated Vela rule sets confirm retrieval advantage?

## Provenance
- Commit: see report `commit` field
- Command: `PYTHONPATH=. python scripts/run_vela_benchmark.py`
- Model: `/mnt/d/AI/.lmstudio/models/OpenBMB/MiniCPM5-2B-Q4_K_M/MiniCPM5-2B-Q4_K_M.gguf`
- Anchor: ONNX MiniLM (real)
- RTU: plastic updates enabled in C, skipped in D