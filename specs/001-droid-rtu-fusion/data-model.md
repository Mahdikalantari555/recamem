# Phase 1 Data Model: Droid × Raw Byte RTU Fusion

Entities come from the feature spec; field names below map to the concrete
profile layout the implementation touches. No new database tables are added.

## Fused Agent Profile (extended)

A profile is one directory `droids/<name>/` holding the agent's saved state. This
feature adds exactly one file and three manifest fields. Everything else is
unchanged.

**Manifest** (`config.json` — the manifest is written last; it records what
exists):

| Field | Type | Meaning | Validation rule |
|---|---|---|---|
| `rtu_checkpoint` | string | File name of the generative checkpoint, relative to the profile dir (planned: `rtu.safetensors`) | If present, the file MUST exist on load, else the load raises (FR-007) |
| `injection_strategy` | string | Conditioning strategy; fused profiles pin `"prompt"` (FR-001, D3) | Must be one of `prompt`, `hidden`, `kv_cache`, `auto`; anything else falls through to auto-selection |
| `rtu_dim` | int | Generative component width (384) | Must equal the anchor embedding width so the associative bonus is dimensionally well-defined (D7) |
| `rtu_layers` | int | Generative component depth (8) | Positive int; standalone raw-mode widgets are independent |
| (existing fields) | — | `name`, `dim`, `step_count`, `base_lr`, `plastic_adapter`, `gguf_*`, … | Unchanged |

**Files in the profile directory after this feature**:

| File | Owner | Change |
|---|---|---|
| `config.json` | engine | + 4 fields above |
| `rtu.safetensors` | generative backend | new; `m.*` params, `state.i` / `decaytrace.i` / `embedtrace.i` buffers, `o.i.*` optimizer state, `step_count` (schema owned by `Model.save`/`Model.load`) |
| `memory.safetensors` | plastic memory | unchanged |
| `knowledge.db` (+ legacy `knowledge.json`) | knowledge store | unchanged |
| `train_log.json`, `snapshots.json`, `retrieval_policy.json` | engine | unchanged |

**State transitions**:

1. Create fused agent (construction flag) → random-init generative component →
   profile saved with manifest fields + checkpoint file.
2. Teach → plastic memory + knowledge store change; generative component
   unchanged (frozen, FR-003).
3. Answer on generative path → retrieval → conditioning → generation; any state
   changed is plastic memory only.
4. Save → knowledge, then checkpoint, then manifest.
5. Load → manifest read; knowledge loaded; memory loaded; generative checkpoint
   attached; missing declared checkpoint = hard error.

## Answer Path (new label)

The route an answer took. Carried per answer for the console badge (FR-012).

| Value | Meaning |
|---|---|
| `extractive` | Verbatim taught facts, byte-identical to the non-fused fast path (FR-005) |
| `generative` | Fluent prose conditioned on retrieved facts; only reachable when retrieval returned facts (FR-004) |

Rules: the extractive path exists in both plain and fused mode with identical
output. The generative path must not exist when retrieval is empty — the
fallback message replaces it.

## Generative Component State (frozen evidence)

Tensor groups whose checksums prove the frozen rule (FR-003, SC-004):

- Parameters: all `m.*`-prefixed weights in the checkpoint schema (encoder
  embedding, per-layer `decay`/`norm`/`weights`, decoder `decode`/`stop`).
- Buffers per layer: `state.i`, `decaytrace.i`, `embedtrace.i`.
- Scalar: `step_count` (frozen inference never advances it).

Priming and generation mutate these buffers' *live* copies inside a call, but a
full `reset()` precedes every generation, so checksums are compared outside
inference, before and after whole answers.

## Relationships

```text
DroidEngine 1──1 PlasticAssociativeRTU        (memory; S, momentum, h_trace; retrieve())
DroidEngine 1──1 KnowledgeStore               (facts + embeddings)
DroidEngine 1──1 ByteRtuBackend (optional)    (fused mode; owns one Model)
ByteRtuBackend 1──1 Model                     (frozen generative component; reset/save/load)
DroidManager 1──* DroidEngine                 (profiles under droids/<name>/)
```
