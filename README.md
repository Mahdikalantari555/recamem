# RecaMem

RecaMem is a local-first plastic memory engine for foundation models.

Foundation models stay frozen and provide language; RecaMem supplies adaptive long-term memory, online learning, memory consolidation, forgetting, and trace-based recall. At its core, a Recurrent Trace Unit (RTU) continuously updates a persistent memory state from interactions, feedback, and learned knowledge, so future reasoning is influenced by past experience rather than simple text retrieval.

Every claim is measured, not asserted. See `plans/09_recamem_plan.md`, `reports/recamem-core-selection.md`, and `docs/`.

## Status

Phase: core architecture selected (`C4` gated-delta matrix + BPE-4096 + chunked BPTT + D3 readout + R4 router + M4 memory arm, unified backbone). Package rename `src` → `recamem` and two-tier plasticity are in progress. See `plans/09_recamem_plan.md`.

## Quick start

```bash
# Python environment
conda activate ai   # or use /home/asus/miniforge3/envs/ai/bin/python

# Run tests
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests

# Run the embedded micro-model CLI (optional)
PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python main.py

# Launch WebUI
/home/asus/miniforge3/envs/ai/bin/streamlit run app.py
```

## Architecture (selected core)

- `src/model/recamem_core.py` — `build_recamem_core` (selected core loader)
- `src/model/cores/` — candidate cores, objectives, routers, readouts, memory arms, tokenizer
- `src/model/tournament/` — architecture tournament (funnel, stats, trainer, workloads)
- `src/model/rtu.py` — RTU block, Model, custom AdamW, losses
- `src/model/droid.py` — memory engine (to become `MemoryEngine`)
- `configs/recamem_selected_core.json` — selected core manifest

## Benchmarks

- `scripts/run_core_tournament.py` — full architecture tournament
- `reports/recamem-core-selection.md` — selection report
- `reports/evidence/recamem-core-tournament.json` — evidence ledger

## License

MIT — see `LICENSE.md`.
