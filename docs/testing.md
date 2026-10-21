# Testing: offline, model-dependent and benchmark tiers

The suite is tiered so that a machine without model weights can still verify
every contract. `scripts/check_release_evidence.py` uses the same tier names.

| Tier | Needs | Command |
|------|-------|---------|
| `offline` | Python + core requirements | `pytest tests -q` |
| `real-model` | ONNX MiniLM / local GGUF weights | `pytest tests -q -k "onnx or gguf"` after installing the optional extras |
| `grounding` | a generative backend plus the profile it grounds | `python scripts/verify_remote_sensing_e2e.py` |
| `concurrency-migration` | nothing extra (threads only) | `pytest tests -q -k "stream or migration or snapshot"` |
| `benchmark` | dedicated hardware and datasets | `python scripts/bench_hybrid_retrieval.py`, `benchmark.py` |

## Offline core install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt   # torch, numpy, safetensors, pytest
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

`requirements-core.txt` is the contract: every test above passes without
`onnxruntime`, `hnswlib`, `sentence-transformers`, `llama-cpp-python`,
`streamlit` or `psutil`. When an optional package is missing, the code reports
the capability instead of silently degrading:

* `OnnxMiniLM.is_mock == True` and embeddings are deterministic mock vectors;
* `HnswIndex.mode == "exact-fallback"`;
* `CrossEncoderReranker.scorer == "fallback-lexical"`;
* `MockLlama` never claims KV prefill support;
* the console (`streamlit`) is not importable, so console tests check `app.py`
  structurally and exercise the same engine APIs directly.

See `docs/optional-dependencies.md` for the per-package fallback table.

## Optional integrations

```bash
.venv/bin/pip install -r requirements-optional.txt   # ONNX, HNSW, reranker, GGUF, console
pytest tests -q                                       # same suite, real implementations where available
pytest tests -q -k "optional_dependencies"            # probes the fallback contract
```

## Evidence ledger and OpenSpec structure

```bash
python scripts/check_release_evidence.py               # schema + inventory coverage (allowed to be incomplete)
python scripts/check_release_evidence.py --require-complete   # release gate; exits 1 until §9 is closed
python scripts/validate_openspec.py                    # change/task/spec structure
```

`--require-complete` failing is expected while tasks 3.4/3.7/7.9/7.10 and
sections 1.2/1.3/8/9 stay open.

## Fixtures

`tests/test_migration_fixtures.py` builds legacy/default/custom profiles in a
temporary directory and compares their layout against
`tests/fixtures/profile_checksums.json` (no model weights are committed). After
an intentional change to the written profile layout:

```bash
UPDATE_FIXTURE_MANIFEST=1 PYTHONPATH=. pytest tests/test_migration_fixtures.py
```

## CI

`.github/workflows/offline-contracts.yml` runs, on every push and pull request:
the offline suite, the evidence-ledger schema/coverage guard and the OpenSpec
structure validator. Model-dependent tiers are not part of that job because no
weights exist in CI.
