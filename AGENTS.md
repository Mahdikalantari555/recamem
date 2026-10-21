# Repository Instructions for AI Agents

## Environment & Toolchain
- **Python Environment**: ALWAYS use the `ai` conda environment for all executions:
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
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests/test_droid_engine.py
  ```
- **Run End-to-End Verification**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py
  ```
- **Launch WebUI**:
  ```bash
  /home/asus/miniforge3/envs/ai/bin/streamlit run app.py
  ```

## Architecture & Code Boundaries
- `src/model/rtu.py`: 1:1 PyTorch port of the MLX RTU block, Model class, custom AdamW, and loss functions (`unbiased=False` on variance loss). Supports a `ce_only` flag for crossentropy-only training (see `docs/rtu.md`). Owns `MissingCheckpointError(FileNotFoundError)` and `InvalidProfileError(ValueError)`; `Model.load()` raises on an empty/absent path instead of silently returning random weights. Constructing `Model(...)` *is* the explicit random-init path.
- `src/model/byte_rtu_backend.py`: `ByteRtuBackend`, the frozen generative peer of `GgufBackend`. Owns prompt encoding (verbatim, query last, no appended newline), the public `reset()`, and the `min(max_tokens, max_output_bytes)` output bound. It is the only module allowed to prime `Model` for inference.
- `src/model/onnx_anchor.py`: INT8 quantized `all-MiniLM-L6-v2` loaded via ONNX Runtime (< 25 MB). Models and tokenizer cached in `~/.cache/huggingface`.
- `src/model/droid.py`: `DroidEngine` coupling ONNX anchor to plastic RTU memory blocks with automatic learning rate and zero catastrophic collapse. Owns the recall scorer (`RECALL_SCORE_WEIGHTS`, dense gate 0.48, associative bonus applied only to candidates past the gate), the canonical answer paths (`recall` / `fused` / `llm`), and the profile manifest/persistence rules (artifacts first, `config.json` last via `os.replace`).
- `src/model/droid_manager.py`: Multi-profile manager for named droids in `droids/<name>/` (`config.json`, `memory.safetensors`, `knowledge.json`, `train_log.json`).
- `src/model/teacher.py`: Stdlib `urllib` client for OpenAI-compatible endpoints to distill unstructured text into propositions.
- `app.py`: Streamlit control station with dual modes (Droid Engine vs. Raw Byte RTU).
- `main_mlx.py`: MLX reference implementation. **Not** byte-identical to upstream — it intentionally retains trace-buffer checkpoint persistence that upstream removed.

## Droid × Byte-RTU Fusion Rules

- **RTU weights are frozen at inference.** The fused slow path only calls
  `Model.forward(frozen=True)` (no autograd, no optimizer step). If anyone wants the
  RTU to learn, it runs on a copy guarded by a snapshot — never on the inference
  decoder.
- **One checkpoint location:** `droids/<name>/rtu.safetensors`. No profile shares a
  file with another profile. The manifest field is `byte_rtu_checkpoint` (relative
  filename); `rtu_checkpoint` is a one-release legacy alias. An absolute path or a
  path containing `..` is rejected with `InvalidProfileError`.
- **Fail loud, never degrade:** a manifest that names an absent checkpoint raises
  `MissingCheckpointError`; a missing/unreadable/non-object `config.json` raises
  `InvalidProfileError`. There is no fallback to random initialisation or an empty
  profile.
- **Engines:** `droid` | `byte_rtu` | `hybrid` (`fusion_mode`: `prompt_stuffing` |
  `off`). A byte-RTU profile pins `injection_strategy: "prompt"`; the `hidden` and
  `kv_cache` implementations are stubs.
- **Training is a build step:** `scripts/train_byte_rtu.py` writes the checkpoint and
  then the manifest. `droids/` is git-ignored; weights are never committed. The
  grounding test runs only when `RTU_CHECKPOINT` is set.
- **Empty retrieval, empty generation:** with no recalled facts the fused path
  returns `NO_INFORMATION_MESSAGE` and never samples; an empty fused generation is
  also reported as the fallback rather than an empty answer.

## Upstream / Fork Policy
- `origin` = `Mahdikalantari555/test-model-thing-torch` (this fork).
- `upstream` = `jrz97619761/test-model-thing` — the parent fork. Note the README's clone URL is wrong (has a spurious `-torch`).
- `main` is **55 commits ahead** of `upstream/main`; `upstream/main` is **5 commits** past the divergence point `0a47782` (as of upstream tip `9867867`). A straight merge conflicts on `README.md` and `main.py`.
- Pull from upstream **selectively**, never as a blanket merge. Files in this fork are not the same as upstream's (`main.py` is the PyTorch port; the MLX original lives in `main_mlx.py`).
- Known-good procedure: cherry-pick upstream commits, rewrite paths (`main.py` → `main_mlx.py`), then verify with `git diff --no-index <(git show upstream/main:main.py) main_mlx.py` so the only diff is an intended divergence.
- Do **not** adopt upstream's removal of `state.i`/`decaytrace.i`/`embedtrace.i` from `save`/`load` — `tests/test_checkpoint.py` and `docs/rtu.md` depend on that persistence.
- Upstream commits after `0a47782` and their status here: `2c846b7` (drop trace persistence) **rejected** on purpose; `8eb4f94` (`--ce-only`) ported in `780accb`; `5aa3fef` (`VISION.md`) ported in `94c42cf`; `348203a`/`9867867` (README prose) ported in the fork's own README wording.

## Critical Implementation Quirks & Gotchas
- **PyTorch In-Place Parameter Edits**: When updating `nn.Parameter` values (e.g. `decay`) from disk or buffers, wrap in `with torch.no_grad():` or write to `.data` to prevent autograd leaf Variable runtime errors.
- **Safetensors Shared Memory**: Do NOT save duplicate references to `blocks.i.states` and `state.i`. Extract model weights via `named_parameters()` rather than `state_dict()`.
- **Hugging Face Cache**: Quantized ONNX weights and tokenizers reside in `~/.cache/huggingface`. Do not re-download if present.
- **OpenSpec**: Managed via `/home/asus/.local/share/pnpm/bin/openspec`. Specs reside in `openspec/specs/` and archived changes in `openspec/changes/archive/`.
