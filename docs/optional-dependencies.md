# Core versus optional dependencies

The offline contract suite must run on a core-only install. Optional
integrations degrade to an explicit fallback and never to fake success.

## Install

```bash
# core: droid engine, plastic memory, offline tests
pip install -r requirements-core.txt

# everything (development)
pip install -r requirements.txt

# a single integration
pip install -r requirements-optional.txt   # or pick the packages you need
```

## What each optional package unlocks

| Package | Enables | Without it |
| --- | --- | --- |
| `onnxruntime`, `onnx`, `transformers`, `huggingface_hub` | real MiniLM embeddings | `OnnxMiniLM.is_mock = True`; every retrieval number is mock evidence |
| `llama-cpp-python` | GGUF generation and KV prefill | `GgufBackend` is prompt-only; `MockLlama` never claims prefill |
| `hnswlib` | approximate dense index | `HnswIndex.mode == "exact-fallback"` (exact MVM search) |
| `sentence-transformers` | cross-encoder reranking | `CrossEncoderReranker.scorer == "fallback-lexical"` |
| `streamlit`, `pandas` | console | console tests are skipped |
| `psutil` | process metrics in benchmarks | metrics reported as unavailable |

Hookable-decoder hidden injection needs **no extra package**: it is a PyTorch
forward hook on a decoder you already have in-process.

## Verification

```bash
# core-only: imports and the offline suite work with no optional package
PYTHONPATH=. python -m pytest tests -q

# optional smoke checks (skipped when the package is absent)
PYTHONPATH=. python -m pytest tests/test_gguf_capabilities.py -q
```

`tests/test_optional_dependencies.py` blocks the optional imports in a
subprocess and asserts that `src.model.droid` and the core modules still import,
and that the requirement files keep the split honest.
