# Quickstart Validation: Droid × Raw Byte RTU Fusion

Run these scenarios after implementation to prove the feature works end to end.
All commands run from the repository root. The environment is the `ai` conda
environment per `AGENTS.md`.

## Prerequisites

- `PYTHONPATH=.` is set (or rely on `pytest.ini`, which does it for pytest).
- The HF cache with the INT8 MiniLM anchor exists, or the tests that need a real
  anchor download it once (first run is slower).
- No GGUF model is needed: the fused backend is the byte-level component that
  ships in-repo.

## Pre-change baseline (task T001)

Measured on 2026-10-11 before any implementation edit:

- `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests` → **1 failed, 62
  passed** (63 tests total).
- The single failure is pre-existing and unrelated to this feature:
  `tests/test_rtu.py::test_model_forward_accepts_ce_only` (the raw `Model`'s
  `ce_only` step does not record `last_loss` as the test expects). Fixing it is
  out of scope; the E1 gate accepts this one known failure, everything else must
  stay green.

## Scenario 1 — New unit tests (grounding, silence, frozen proof, persistence)

```bash
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests/test_droid_rtu_fusion.py -v
```

Expected: all tests pass, including:

- `test_byte_rtu_backend_matches_raw_ui_loop` — backend output equals the inline
  raw-UI loop (`app.py:364-382`) for the same primed state.
- `test_fused_chat_is_grounded` — after teaching a paragraph, a fused `/slow`
  answer contains a taught fact.
- `test_no_free_generation_without_facts` — an empty fused agent returns the
  fallback string, never generated text.
- `test_rtu_weights_frozen_during_inference` — checksums of every parameter and
  buffer are identical before and after fused answers.
- `test_fused_profile_roundtrip` — save/load keeps the checkpoint equal; a
  declared-but-missing checkpoint raises.
- `test_associative_bonus_reranks_and_cold_start_is_neutral`.

## Scenario 2 — Full existing suite (regression gate)

```bash
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests
```

Expected: the full pre-existing suite passes with zero changes to existing
contracts (SC-006).

## Scenario 3 — Remote-sensing end-to-end with the fused backend

```bash
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py
```

Expected: the script runs end to end (this requires the `clear_knowledge`
entry point from research decision D8, without which it raises
`AttributeError` at line 18) and all assertions pass (SC-007).

## Scenario 4 — Latency ceiling measurement (E3 gate)

Time one fused `/slow` answer, e.g.:

```bash
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python - <<'EOF'
import time
from src.model.droid_manager import DroidManager
mgr = DroidManager(base_dir="droids-fused-demo")
d = mgr.create_droid("fused-demo", byte_rtu=True)
d.teach("Remote sensing is the acquisition of information about an object or phenomenon without making physical contact with the object.")
t0 = time.perf_counter()
reply = d.chat("/slow What is remote sensing?")
dt = time.perf_counter() - t0
print(f"elapsed={dt:.2f}s reply_head={reply[:120]!r}")
assert "acquisition" in reply.lower(), "fused answer must contain taught content"
EOF
```

Expected: the answer contains taught content, and `elapsed` is recorded. If it
exceeds 10 s, lower `max_prompt_bytes` and write the measured ceiling into
`plans/10_droid_rtu_fusion.md` before marking the plan implemented (SC-008).

## Scenario 5 — Console spot check (manual)

Start the WebUI with
`/home/asus/miniforge3/envs/ai/bin/streamlit run app.py`. Confirm:

1. A third engine option ("Droid + Raw Byte RTU (Fused)") appears beside the
   existing two, and the standalone raw mode's controls are unchanged.
2. In fused mode, the extractive path answers exactly as the plain droid mode.
3. The memory console badge reports generative vs extractive per answer and
   shows the component configuration and checkpoint status.
