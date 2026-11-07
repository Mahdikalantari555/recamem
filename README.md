# RecaMem

RecaMem is a local-first plastic memory engine for foundation models.

Foundation models stay frozen and provide language; RecaMem supplies adaptive
long-term memory, online learning, consolidation, bounded forgetting and
trace-based recall. A Recurrent Trace Unit (RTU) continuously updates a
persistent memory state from interactions, feedback and learned knowledge, so
future reasoning is influenced by past experience rather than simple retrieval.

Every claim is measured, not asserted. See `plans/09_recamem_plan.md`,
`reports/recamem-core-selection.md` and `docs/`.

**Measured status (2026-11-07).** The `offline` and `benchmark` tiers pass on a
2 vCPU CPU-only runner (counts live in `reports/evidence/`, not in this file). `real-model` and `grounding`
are **blocked**: no ONNX MiniLM weights, no GGUF artifact, no trained
`RTU_CHECKPOINT`, so `recamem doctor` exits 1 and
`scripts/check_release_evidence.py --require-complete` exits 1 by design. The
Vela PoC measured **no advantage** for memory over ordinary retrieval
(`docs/vela-poc-results-v2.md`); several latency/size targets are still missed
and are published as `partial` in `docs/release-matrix.md`. No release claim
exceeds the evidence.

## Install

```bash
pip install -e .               # core: torch, numpy, safetensors, psutil
pip install -e ".[pdf]"        # + PDF ingestion (pypdf)
pip install -e ".[onnx]"       # + real ONNX MiniLM anchor
pip install -e ".[gguf]"       # + local GGUF generation (llama.cpp binding)
pip install -e ".[web]"        # + REST/web surface (fastapi, uvicorn)
pip install -e ".[dev]"        # everything the test tiers use
```

## Run Learning Studio on CPU

From a checkout, Python 3.10 or newer is required. A virtual environment keeps
RecaMem's dependencies separate from your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .             # CPU use; no GGUF or llama.cpp required
recamem companion --no-browser
```

Open **http://127.0.0.1:8000/**. The default server is loopback-only; stop it
with Ctrl+C. The committed Svelte build is served by the same Python process,
so Node is not needed just to run it. To rebuild after changing the UI, install
Node.js and run `npm --prefix companion ci && npm --prefix companion run build`,
then start the companion again.

In the Learning Studio, press **Start training**; the basic view selects a
fixed C4 smoke configuration automatically. The default run prepares the included self-authored smoke corpus, trains a tiny
model on the CPU and shows real progress; open **Evaluation** to inspect the
held-out loss and try a generated sample. The corpus and run are *only a
pipeline test*: they do not produce a useful language model or demonstrate
language understanding. Muir is an experimental sequential reference, not the
paper's optimized accelerator kernel. Model weights are not included.

**Memory & companion** uses the existing RecaMem engine and normally needs the
real ONNX anchor configured and its model files installed; missing weights are
reported as unavailable. `pip install -e ".[onnx]"` installs the optional ONNX
runtime dependencies, not the weights. See `docs/real-model-runbook.md` and run
`recamem doctor` for expected paths. For testing only, mock embeddings must be
explicitly selected; they are not a real language or embedding model. The CPU
native checkpoint sampler and normal memory-backed chat are separate paths; an
experimental evaluation button can include recalled evidence. Training a
checkpoint does not automatically attach it to the companion or update its
memory. See `docs/cpu-muir-learning.md` for CLI training, checkpoints and current
limitations.

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
recamem companion              # one-command Svelte companion (API + built UI)
recamem ui                      # Tauri desktop shell, browser fallback
```

The runner and the API need core dependencies only; the desktop app needs Node to
build the frontend and Rust/Tauri for the shell. `docs/surfaces.md` documents the
contracts, the loopback-only rules and what is blocked without a Rust toolchain.

## Verify

```bash
python -m pytest tests -q                  # offline contracts
python scripts/offline_gate.py             # ledger, MCP/CLI parity, packaging, gap inventory
python scripts/check_completion_gaps.py --schema-only
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

## Companion app

`recamem companion` serves the committed `companion/dist` build next to the
versioned API, without Node at runtime. Learning Studio offers CPU smoke
training and evaluation; the companion chat and memory features require a
working configured anchor and backend. The advanced training view runs
`scripts/train_lm.py` as a cancellable job and includes a provenance-backed
checkpoint picker. See `docs/surfaces.md` and `docs/cpu-muir-learning.md`.

```bash
recamem companion --dry-run
npm --prefix companion install && npm --prefix companion run build
```

### Direction: independent background learning

The intended direction is a CPU-local model that automatically selects compatible
model and tokenizer settings and learns incrementally in the background from
approved sources, with resource limits, status, evaluation and rollback. This is
**not implemented yet**: the Learning Studio currently runs a manually started,
bounded smoke test, not continuous learning. It does not learn while the app or
machine is stopped. Web research remains separately opt-in and unimplemented.
The plan and remaining tasks are in
`openspec/changes/autonomous-incremental-learning/`.

The initial background-learning control plane persists a per-profile preference
and shows **blocked** or **paused** in the Studio. A consent and reference-only
intake ledger exists as a Python primitive, but is not connected to teaching or
a training worker. Its saved thread setting is not yet enforced. No queued
record should be interpreted as a trained model update.
