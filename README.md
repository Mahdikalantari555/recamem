# RecaMem

RecaMem is a local-first plastic memory engine for foundation models.

Foundation models stay frozen and provide language; RecaMem supplies adaptive
long-term memory, online learning, consolidation, bounded forgetting and
trace-based recall. A Recurrent Trace Unit (RTU) continuously updates a
persistent memory state from interactions, feedback and learned knowledge, so
future reasoning is influenced by past experience rather than simple retrieval.

Every claim is measured, not asserted. See `plans/09_recamem_plan.md`,
`reports/recamem-core-selection.md` and `docs/`.

## Install

```bash
pip install -e .               # core: torch, numpy, safetensors, psutil
pip install -e ".[pdf]"        # + PDF ingestion (pypdf)
pip install -e ".[onnx]"       # + real ONNX MiniLM anchor
pip install -e ".[gguf]"       # + local GGUF generation (llama.cpp binding)
pip install -e ".[web]"        # + REST/web surface (fastapi, uvicorn)
pip install -e ".[dev]"        # everything the test tiers use
```

## Quick start

```bash
recamem ingest ./notes --name agent-notes --save
recamem recall "What is the calibration constant?" --name agent-notes
recamem chat --name agent-notes
recamem doctor
```

The real anchor and a GGUF backend are optional and never mocked silently: with
no model on disk, `recamem doctor` reports the anchor as `unavailable` and names
the expected path. Mock embeddings are used only when explicitly selected
(`--anchor mock`).

## Layout

- `recamem/engine.py` — `DroidEngine` (also exported as `MemoryEngine`)
- `recamem/plasticity.py` — FAST/SLOW tiers, schema-versioned state
- `recamem/retrieval.py` — two-channel recall and the learned fusion scorer
- `recamem/episodic.py` — SQLite knowledge store with `active`/`retired` tiers
- `recamem/forgetting.py` — utility, capacity retirement, consolidate-then-forget
- `recamem/consolidation.py` — sleep replay and MAS
- `recamem/anchor.py`, `recamem/gguf.py`, `recamem/byte_rtu.py` — backends
- `recamem/ingest.py`, `recamem/segmenter.py` — ingestion and segmentation
- `recamem/bench/` — metrics, continual benchmarks and the LoCoMo /
  LongMemEval-S / DMR runners
- `recamem/cli.py` — the `recamem` command
- `src/model/rtu.py`, `src/model/cores/`, `src/model/tournament/` — training and
  research tree (selected core: C4 gated-delta matrix + BPE-4096 + chunked BPTT
  + D3 readout + R4 router + M4 memory arm)

The old `src/model/*` engine imports and `python -m src.cli` still work through
deprecation shims for one release; see `docs/migration.md`.

## Surfaces

```bash
recamem run --prompt "What is the calibration constant?" --name demo   # headless JSON
recamem serve --rest            # versioned loopback API on 127.0.0.1:8000
npm --prefix web install && npm --prefix web run build
recamem serve --web             # API + built frontend
recamem ui                      # Tauri desktop shell, browser fallback
```

The runner and the API need core dependencies only; the desktop app needs Node to
build the frontend and Rust/Tauri for the shell. `docs/surfaces.md` documents the
contracts, the loopback-only rules and what is blocked without a Rust toolchain.

## Verify

```bash
python -m pytest tests -q                  # offline contracts
python scripts/offline_gate.py             # ledger, MCP/CLI parity, packaging
python -m pytest --collect-only -q tests   # the authoritative test count
```

Tiers and blocked semantics are documented in `docs/testing.md`. Real-model
gates (ONNX anchor, local GGUF, an 8B reference) are reported blocked with their
missing path rather than passed.

## Documentation

`docs/migration.md`, `docs/plasticity.md`, `docs/recall.md`,
`docs/forgetting.md`, `docs/sleep.md`, `docs/ingest.md`, `docs/benchmarks.md`,
`docs/surfaces.md`, `docs/testing.md`, `docs/release-evidence.md`,
`docs/ui-design.md`. The Phase 0 defect ledger is
`reports/recamem-defect-ledger.md`.

## License

MIT — see `LICENSE.md`.

## Cloud Training
See `docs/cloud-training.md`. Train on Colab and sync via HuggingFace Hub:
`python scripts/colab_train.py --lm-tokens 20000000 --push-to-hub yourname/recamem-ckpt`
