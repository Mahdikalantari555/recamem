# Repository Instructions for AI Agents (RecaMem)

## Environment & Toolchain
- **Package**: engine code lives in `recamem/`; `src/model/*` are deprecation
  shims for one release (`docs/migration.md`).
- **Python Environment**: any Python >= 3.10 with the declared dependencies:
  - `pip install -e ".[dev]"` (core + every optional integration)
  - Always prepend `PYTHONPATH=.` when invoking python directly from project root.

## Verification & Commands
- **Run all tests**:
  ```bash
  PYTHONPATH=. python -m pytest tests -q
  ```
- **Run the offline gate** (ledger, MCP/CLI parity, packaging, blocked semantics):
  ```bash
  PYTHONPATH=. python scripts/offline_gate.py
  ```
- **Run a single test file**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests/test_recamem_core.py
  ```
- **Run End-to-End Verification**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py
  ```
- **Launch WebUI**:
  ```bash
  /home/asus/miniforge3/envs/ai/bin/streamlit run app.py
  ```
- **Run core tournament**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/run_core_tournament.py
  ```

## Architecture & Code Boundaries
- `src/model/rtu.py`: RTU block, Model class, custom AdamW, loss functions (`unbiased=False` on variance loss). Supports `ce_only` flag. Owns `MissingCheckpointError(FileNotFoundError)` and `InvalidProfileError(ValueError)`; `Model.load()` raises on empty/absent path. Constructing `Model(...)` is the explicit random-init path.
- `src/model/byte_rtu_backend.py`: `ByteRtuBackend`, frozen generative peer of `GgufBackend`. Owns prompt encoding (verbatim, query last, no appended newline), public `reset()`, and `min(max_tokens, max_output_bytes)` bound. Only module allowed to prime `Model` for inference.
- `src/model/onnx_anchor.py`: INT8 quantized `all-MiniLM-L6-v2` via ONNX Runtime (<25 MB). Cached in `~/.cache/huggingface`. No silent mock fallback — mock is explicit opt-in only.
- `recamem/engine.py`: memory engine (`DroidEngine` / `MemoryEngine`). Couples the anchor to plastic memory blocks, owns the canonical answer paths, profile manifest/persistence (artifacts first, `config.json` last via `os.replace`), the writer lock and schema version.
- `recamem/plasticity.py`: FAST/SLOW tiers, MAS penalty, rollback gate, schema versions.
- `recamem/retrieval.py`: two-channel recall, logistic fusion scorer, ANN recall measurement.
- `recamem/episodic.py`: SQLite store with `active`/`retired` tiers and ingest provenance.
- `recamem/forgetting.py`: utility, capacity retirement, consolidate-then-forget (no hard delete).
- `recamem/anchor.py`, `recamem/gguf.py`, `recamem/byte_rtu.py`: fail-fast backends; mock is explicit only.
- `recamem/ingest.py`, `recamem/segmenter.py`: folder ingestion and abbreviation-safe segmentation.
- `recamem/bench/`: metrics, continual benchmarks, LoCoMo/LongMemEval-S/DMR runners.
- `recamem/cli.py`: the `recamem` command; `src/cli.py` delegates with a deprecation notice.
- `src/model/droid_manager.py`: multi-profile manager for named agents in `droids/<name>/`.
- `src/model/teacher.py`: stdlib `urllib` client for OpenAI-compatible endpoints.
- `src/model/recamem_core.py`: `build_recamem_core` — loads the selected core from `configs/recamem_selected_core.json`.
- `src/model/cores/`: candidate cores, objectives, routers, readouts, memory arms, tokenizer, calibration.
- `src/model/tournament/`: architecture tournament (funnel, stats, trainer, workloads).
- `src/model/decision_head.py`: system-1 decision readout.
- `app.py`: Streamlit control station (Droid Engine vs. Raw Byte RTU).
- `main.py` / `main_mlx.py`: PyTorch and MLX reference implementations of the embedded micro-model.

## Droid × Byte-RTU Fusion Rules
- RTU weights frozen at inference. Fused slow path calls `Model.forward(frozen=True)` only.
- One checkpoint location: `droids/<name>/rtu.safetensors`. Manifest field `byte_rtu_checkpoint`; `rtu_checkpoint` legacy alias. Absolute path or `..` rejected with `InvalidProfileError`.
- Fail loud, never degrade: missing checkpoint → `MissingCheckpointError`; missing/unreadable/non-object `config.json` → `InvalidProfileError`. No fallback to random init or empty profile.
- Engines: `droid` | `byte_rtu` | `hybrid` (`fusion_mode`: `prompt_stuffing` | `off`). Byte-RTU profile pins `injection_strategy: "prompt"`.
- Training is a build step: `scripts/train_byte_rtu.py`. `droids/` git-ignored; weights never committed.
- Empty retrieval, empty generation: no recalled facts → `NO_INFORMATION_MESSAGE`, never samples.

## Critical Implementation Quirks & Gotchas
- PyTorch in-place parameter edits: wrap in `with torch.no_grad():` or write `.data`.
- Safetensors shared memory: do not save duplicate refs to `blocks.i.states` and `state.i`. Extract via `named_parameters()`, not `state_dict()`.
- Hugging Face cache: ONNX weights/tokenizers in `~/.cache/huggingface`. Do not re-download.
- OpenSpec: managed via `/home/asus/.local/share/pnpm/bin/openspec`. Specs in `openspec/specs/`, archived in `openspec/changes/archive/`.
- No silent mock anchor fallback. Mock is explicit opt-in (`--anchor mock`, `RECAMEM_ANCHOR=mock`).
- No hard-delete of facts without consolidate-then-forget.

## RecaMem Rules (ALWAYS)
- Never re-add a silent mock fallback.
- Never hard-delete a fact without consolidate-then-forget.
- Every benchmark number needs provenance (config, seed, anchor mode, backend, ablations).
- Prefer editing existing skills/modules over creating duplicates; merge when overlap >= 80%.
