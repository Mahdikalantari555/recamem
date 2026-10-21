# Test-Model-Thing (TMT) — PyTorch & Droid Lifelong Engine

[Original YouTube Demo](https://youtu.be/9UERVVwpNew) | [Original MLX Repo](https://github.com/jrz97619761/test-model-thing)

This repository provides a 1:1, contract-verified **PyTorch** implementation of Test-Model-Thing (TMT) alongside the **Droid Lifelong Engine** — a lightweight continual learning system that combines a frozen ONNX semantic anchor with plastic Recurrent Trace Units (RTUs) for conversational domain teaching while retaining prior domain facts in sequential-teaching tests.

---

## Key Features

1. **Pure PyTorch Implementation**:
   - 100% parity with original MLX math (variance loss `unbiased=False`, custom AdamW optimizer, dual gradient hooks, decay trace updates).
   - Runs everywhere: Linux, macOS, WSL2, and Windows (CPU or CUDA).
   - Preserves original MLX scripts as `main_mlx.py` and `benchmark_mlx.py`.

2. **Droid Lifelong Learning Engine**:
   - **Semantic Anchor**: Quantized INT8 `all-MiniLM-L6-v2` via ONNX Runtime (model files cached in `~/.cache/huggingface`; disk usage varies by environment and quantization).
   - **Plastic RTU Memory**: Recurrent trace unit accumulating proposition vectors with automatic learning rate decay $\eta = \eta_0 / \sqrt{1 + \text{steps}/20}$.
   - **Retention in Sequential Teaching**: Lifelong-learning tests teach multiple domains and verify prior facts remain recallable; no general forgetting metric is claimed beyond those tests.
   - **In-Chat Conversational Teaching**: Teach the model simply by chatting, sending text paragraphs, or using `learn: <text>`.

3. **Interactive Streamlit WebUI**:
   - Three engine modes: anchored Droid, **Droid + Raw Byte RTU (Fused)**, and raw byte RTU.
   - Live chat with a per-answer **answer-path badge** (`recall`, `fused`, `llm`), instant domain paragraph ingestion, transparent audit logs, and RTU memory/decay visualization.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/jrz97619761/test-model-thing-torch.git
cd test-model-thing-torch

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

### 2. Launch WebUI (Streamlit)

```bash
streamlit run app.py
```
Open `http://localhost:8501` to chat, switch Droids, teach new domains, and inspect internal states.

### 3. CLI Usage

#### PyTorch CLI
```bash
# Train on raw bytes
python main.py <checkpoint.safetensors> train

# Interactive chat
python main.py <checkpoint.safetensors> chat

# Inference only (frozen weights)
python main.py <checkpoint.safetensors> chat --frozen

# Benchmark with CoLA
python benchmark.py <checkpoint.safetensors> <epochs> <slice>
```

#### Droid Lifelong Engine in Python
```python
from src.model.droid_manager import DroidManager

mgr = DroidManager(base_dir="droids")
droid = mgr.get_droid("droid-geospatial")

# Teach a domain paragraph (absorbed in < 30ms on CPU)
droid.teach(
    "Remote sensing is the acquisition of information about an object or phenomenon "
    "without making physical contact with the object, in contrast to on-site observation."
)

# Recall accurately in chat
reply = droid.chat("What is remote sensing?")
print(reply)
# -> "Remote sensing is the acquisition of information about an object..."

# Save profile
mgr.save_droid("droid-geospatial")
```


---

## Droid × Raw Byte RTU Fusion

The fused engine gives the droid a generative voice without giving up grounding: a
**frozen** byte-level RTU is primed with the facts returned by `recall()` (query last)
and produces the slow-path answer. Weights never train at inference; every call routes
through `Model.forward(frozen=True)` under `torch.no_grad()`, and no optimizer step is
taken.

### Engines and the profile manifest

Each profile declares its engine in `droids/<name>/config.json`:

| `engine` | Meaning | Generative backend |
|---|---|---|
| `droid` | anchored recall only (default) | none, or GGUF if a GGUF config is present |
| `byte_rtu` | droid + frozen byte RTU | `ByteRtuBackend` |
| `hybrid` | carries both a byte-RTU checkpoint and a GGUF config | `fusion_mode` decides (`prompt_stuffing` → byte RTU, `off` → GGUF) |

The manifest also records `byte_rtu_checkpoint` (relative filename),
`fusion_mode` (`prompt_stuffing` / `off`), `rtu_dim`, `rtu_layers`, and
`rtu_spread`. `rtu_checkpoint` is written for one release as a legacy read-only
alias. Profiles without these fields load exactly as before; `engine` is inferred as
`byte_rtu` when a checkpoint is declared, else `droid`.

Save order is **artifacts first, manifest last**: `knowledge.db`, then
`memory.safetensors`, then `rtu.safetensors`, then `config.json` (written through
`config.json.tmp` + `os.replace`). A missing or corrupt manifest raises
`InvalidProfileError`; a manifest that names an absent checkpoint raises
`MissingCheckpointError`. Neither falls back to random weights.

### Checkpoint location and training

`droids/<name>/rtu.safetensors` is the **only** checkpoint location; no artifact is
shared between profiles. `droids/` is a generated artifact directory and is
git-ignored — training weights are never committed.

```bash
# Train a checkpoint for a profile (build step)
PYTHONPATH=. python scripts/train_byte_rtu.py \
    --profile droids/droid-rs --corpus README.md plans docs \
    --steps 4000 --seq-len 512 --dim 384 --layers 8 --spread 32 --seed 0

# The script writes <profile>/rtu.safetensors first, then updates config.json.
```

`--losses full` (default) trains the variance, prediction, cross-entropy, and stop
heads; stop targets are annotated at blank-line paragraph boundaries. `--losses ce`
trains cross-entropy only and leaves the stop head untrained (the model then stops
almost immediately when sampled).

Create a fused profile in Python:

```python
from src.model.droid_manager import DroidManager

mgr = DroidManager(base_dir="droids")
fused = mgr.create_droid("droid-rs", byte_rtu=True, rtu_dim=384, rtu_layers=8)
fused.teach("Remote sensing is the acquisition of information about an object...")
mgr.save_droid("droid-rs")
```

### Answer paths

`engine.answer_path` (and the per-answer console badge) is one of:

- `recall` — extractive answer from `recall()`, including the no-information fallback
  (returned when no facts match; the fused path never free-generates without facts);
- `fused` — `ByteRtuBackend.generate()` produced the text;
- `llm` — a GGUF or external LLM provider produced the text.

`legacy_answer_path` maps these to the old `extractive` / `generative` names for
external callers.

### Grounding gate and quality ceiling

The grounding test is gated on an environment variable, because the repository ships
no weights:

```bash
RTU_CHECKPOINT=droids/droid-rs/rtu.safetensors \
    PYTHONPATH=. pytest tests/test_droid_rtu_fusion.py::test_fused_chat_is_grounded
```

Without `RTU_CHECKPOINT` the test still asserts the plumbing (taught facts reach the
primed bytes) and then skips the grounding assertion. With it, the answer must share a
taught-fact substring. **Expected quality ceiling:** a small-corpus byte RTU is a
surface-realisation toy, not a fluent model. It echoes and garbles bytes from what it
memorised; the fast path stays extractive so recall quality is unaffected, and the slow
path is opt-in per query.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Input Text / Query        |
                      +--------------+--------------+
                                     |
                                     v
               +-------------------------------------------+
               |  ONNX MiniLM Anchor (INT8)                |
               |  (Frozen Base Semantic Feature Extractor) |
               +---------------------+---------------------+
                                     |
                          Embedding z in R^384
                                     |
                                     v
               +-------------------------------------------+
               |  Plastic RTU Memory Block (dim=384)       |
               |  h_t = decay * h_{t-1} + z_t              |
               |  x = LayerNorm + Linear + SiLU            |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  Episodic Knowledge Bank & Dual Recall    |
               |  -> Grounded domain synthesis             |
               |  -> Prior facts retained in tests            |
               +---------------------+---------------------+
                                     |
                     recalled facts, query last (prompt stuffing)
                                     |
                                     v
               +-------------------------------------------+
               |  Frozen Byte RTU (fused engine, optional) |
               |  primed bytes -> sampled bytes -> answer  |
               |  no autograd, no optimizer step           |
               +-------------------------------------------+
```

---

## Benchmarks and grammar probes

One command produces the numbers this project argues about — throughput,
parameter count, resident memory, held-out perplexity, the chunked-path speed
factor, and grammar-probe accuracy — in one JSON report with its provenance:

```bash
PYTHONPATH=. python scripts/bench_edge.py \
    --checkpoint droids/droid-rs/rtu.safetensors \
    --corpus README.md VISION.md --tokens 2000 \
    --probes evals/grammar_probes.jsonl \
    --report reports/bench-edge.json
```

```bash
# The same metrics through the headless CLI, as key=value lines
PYTHONPATH=. python -m src.cli bench \
    --checkpoint droids/droid-rs/rtu.safetensors \
    --corpus README.md,VISION.md --tokens 2000 \
    --probes evals/grammar_probes.jsonl
```

The probe suite (`evals/grammar_probes.jsonl`) is a **project-authored regression
suite**, not an independent benchmark: it scores the correct continuation against
distractors and reports accuracy per phenomenon (`subject_verb_agreement`,
`active_voice`, `passive_voice`, `question_formation`). It is small, hermetic, and
deterministic, and it is a regression signal rather than a capability certificate.

A measurement that could not be taken is printed as `<metric>_skipped=<reason>`
and never as a value. **Any quoted figure must name the machine and the commit it
came from** — both are in the report. See [`docs/benchmarks.md`](docs/benchmarks.md)
for the report schema, the thresholds, and the skip rules.

---

## Researching a topic (web-grounded learning)

A Droid can look things up instead of only learning what it is handed. Research
is opt-in through the profile manifest:

```json
{
  "research": {
    "enabled": true,
    "provider": "duckduckgo",
    "confidence_threshold": 0.55,
    "max_pages_per_search": 3,
    "max_queries_per_minute": 3
  }
}
```

```bash
# Search, fetch, distill and teach a topic; then report what was learned
PYTHONPATH=. python -m src.cli research "what is a lidar bathymetry system" --name droid-alpha
# -> topic=... profile=... searched=true facts_learned=... sources=https://...

# The same loop through MCP: research_topic(query, droid_name)
# A guarded live smoke check (never runs in the test suite):
TMT_ALLOW_NETWORK=1 PYTHONPATH=. python scripts/smoke_web_research.py "lidar bathymetry"
```

Every fact learned this way records its source URL, the query that found it, and
the retrieval timestamp; recall reports the origin (`user_text`, `chat`,
`continuous_stream`, or `web`), and an answer built from web-sourced facts is
reported as `researched` with its sources rather than as a memory answer. With
no provider configured, the system reports that external acquisition is
unavailable and never fails. See [`docs/research.md`](docs/research.md) for the
configuration keys, the bounds, and the safety limits.

---

## Verification & Tests

Run the complete test suite (contract, unit, fusion, and lifelong learning tests).
Obtain the current test count with `python -m pytest --collect-only -q`:

```bash
pytest tests
```

The fusion tests live in `tests/test_droid_rtu_fusion.py`; one grounding test skips
unless `RTU_CHECKPOINT` is set (see the fusion section above).

Run end-to-end domain teaching verification (Remote Sensing acquisition & recall):

```bash
python scripts/verify_remote_sensing_e2e.py
```

---

## License

MIT
