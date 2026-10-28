# Repository Instructions for AI Agents (RecaMem)

## Environment & Toolchain
- **Python Environment**: ALWAYS use the `ai` conda environment:
  - Python: `/home/asus/miniforge3/envs/ai/bin/python`
  - Pytest: `/home/asus/miniforge3/envs/ai/bin/pytest`
  - Streamlit: `/home/asus/miniforge3/envs/ai/bin/streamlit`
  - Always prepend `PYTHONPATH=.` when invoking python directly from project root.

## Verification & Commands
- **Run all tests**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests
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
- `src/model/droid.py`: memory engine (`DroidEngine` → becoming `MemoryEngine`). Couples ONNX anchor to plastic RTU memory blocks. Owns recall scorer, canonical answer paths (`recall`/`fused`/`llm`), profile manifest/persistence (artifacts first, `config.json` last via `os.replace`).
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
