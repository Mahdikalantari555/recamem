# Real-model runbook (manual acceptance tier)

The `real-model` and `grounding` acceptance tiers cannot run in CI or in a
sandbox without model weights. This runbook is the documented manual procedure;
a record produced by it is the only thing that counts as real-model evidence.
See `docs/testing.md`, `docs/release-evidence.md` and
`reports/2026-10-22-real-model-verification.md` for the constraints.

## 0. Preconditions

- A machine with outbound access to `huggingface.co` (the sandbox used for the
  2026-10-22 attempt could not complete a TLS handshake).
- Python >= 3.10, a C++ toolchain and CMake for `llama-cpp-python`.
- About 2 GB free RAM for a 1B Q4_K_M GGUF, more for larger models.
- `HF_TOKEN` in the environment for private repos or pushes. Never write the
  token into a report, a job parameter or a log.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt
.venv/bin/pip install -r requirements-optional.txt   # ONNX, llama-cpp-python, HF Hub
export PYTHONPATH=.
```

## 1. Fetch and identify the artifacts

```bash
.venv/bin/python scripts/fetch_real_model.py
```

The script writes `models/<name>/...` (git-ignored) and a machine-readable
record under `reports/evidence/<date>-*.json` with either `status: downloaded`
(bytes and sha256) or `status: blocked` (the exact exception). A blocked record
is not a pass: the process exits 1. Weights are never committed.

Real ONNX anchor weights (`Xenova/all-MiniLM-L6-v2`, INT8) are cached in
`~/.cache/huggingface` by `OnnxMiniLM`. Record the ONNX Runtime version and the
model directory. Confirm the anchor is real:

```bash
.venv/bin/python -c "from recamem.anchor import OnnxMiniLM; a=OnnxMiniLM(); print(a.is_mock)"
# must print False
```

If it prints `True`, stop: mock embeddings were selected and nothing below may
be recorded as real evidence.

## 2. Verify no silent mock substitution

Before recording results, confirm `GgufBackend` raises `GgufLoadError` when the
path is wrong and that `generate()` does not return a `[Mock generation ...]`
string. The 2026-10-22 report found a silent mock fallback; it must stay fixed.

```bash
.venv/bin/python -m pytest tests -q -k "onnx or gguf"
```

## 3. Exercise the real-model tier

```bash
.venv/bin/python -m pytest tests -q -k "real_model"
.venv/bin/python scripts/verify_remote_sensing_e2e.py
.venv/bin/python -m recamem doctor
```

`doctor` must exit 0, report the real anchor and name the loaded generation
backend. Generation, streaming and KV prefill are exercised through the
documented capability matrix in `docs/backend-contracts.md`.

## 4. Grounding tier

Grounding needs a trained RTU checkpoint. Supply one with `RTU_CHECKPOINT` (or
the profile manifest), then run the end-to-end script and the cross-feature
suite. If no trained checkpoint exists, record `blocked` with its exact name;
generation quality cannot be asserted from random weights.

## 5. Write the evidence record

Each record under `reports/evidence/` must contain:

| Field | Value |
| --- | --- |
| `outcome` | `passed`, `failed`, `skipped` or `blocked` |
| `tier` | `real-model` or `grounding` |
| `commit` | full 40-character commit hash |
| `command` | the exact command, including flags |
| `environment` | Python, torch, ONNX Runtime, `llama-cpp-python`, `huggingface_hub` versions |
| `configuration` | model path, sha256, seed, context, sampling parameters |
| `hardware` | CPU model, core count, RAM, accelerator if any |
| `result` | raw output locations; no derived claim without the measurement |
| `skip_reason` | required when blocked or skipped |

A `blocked` record never certifies a `verified` row in
`reports/completion-evidence.json`, and it never closes a row in
`reports/completion-gaps.json`.
