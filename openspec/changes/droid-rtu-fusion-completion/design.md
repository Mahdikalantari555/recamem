# Design

## Context

Plan 10 adds a frozen byte-level RTU as a generative peer of `GgufBackend`. T001–T011
landed the backend and the engine hook; the rest is open. Current state, verified:

| Seam | Location | State |
|---|---|---|
| Backend `generate()` | `src/model/byte_rtu_backend.py:31-60` | Appends `"\n"`; loop uses `max_output_bytes` only; no public `reset()` |
| Model load | `src/model/rtu.py:355-357` | Returns silently when the path is missing |
| Recall | `src/model/droid.py:531-611` | No associative term; hardcoded boosts (replaced by the correctness change) |
| Profile save | `src/model/droid.py:1254-1344` | Writes `config.json` **first**, then memory, knowledge, log |
| Profile load | `src/model/droid.py:1346-1454` | Returns `False` on missing files; no checkpoint handling |
| Manager | `src/model/droid_manager.py:45-73` | `create_droid()` has no `byte_rtu` argument |
| Console | `app.py:71-75`, `:114` | Two engine modes; hardcoded anchor badge |
| Answer path | `src/model/droid.py` T010 sites | `extractive` / `fallback` / `generative` |

The correctness change (2) lands first and replaces the hardcoded boosts with a named
scorer whose `associative`, `recency`, and `confidence` weights are zero-valued slots.
This change raises those weights and adds the bonus computation; it does not introduce a
second scorer.

## Goals / Non-Goals

**Goals**

- Every profile declares its engine and the artifacts it needs; the manifest is the last
  write and the first read.
- A missing or corrupt artifact produces a named error, never a silent random
  initialization.
- The backend conforms to plan §A1 exactly, with a test per deviation.
- The fused profile round-trips through `DroidManager` and stays isolated from its
  siblings.
- The console exposes the third engine and reports which path produced each answer.
- A trained checkpoint exists as a build step with a script, and the grounding test uses
  it when `RTU_CHECKPOINT` is set.

**Non-Goals**

- No rename, no `recamem` packaging, no `pyproject.toml`.
- No byte-level `teach_bytes()` into plastic memory (plan 10 pattern D).
- No training of the RTU on exported droid facts (pattern B), no ANN index.
- No new embedder; the anchor stays ONNX MiniLM with an explicit mock flag.
- No change to the standalone raw-mode widgets or byte loop in `app.py`.

## Decisions

### Decision 1 — Manifest schema, with one path field and one legacy alias

`droids/<name>/config.json` gains these fields; every one is optional on read:

| Field | Values | Default when absent | Owner |
|---|---|---|---|
| `engine` | `droid`, `byte_rtu`, `hybrid` | inferred: `byte_rtu` if a checkpoint is declared, else `droid` | engine |
| `byte_rtu_checkpoint` | relative filename (`rtu.safetensors`) or `null` | `null` | backend |
| `fusion_mode` | `prompt_stuffing`, `off` | `prompt_stuffing` for byte/hybrid, `off` for droid | engine |
| `rtu_dim`, `rtu_layers`, `rtu_spread` | ints | `384`, `8`, `32` | backend |
| `rtu_checkpoint` | legacy alias written by T017 | — | compatibility only |

`rtu_checkpoint` and `byte_rtu_checkpoint` previously named the same artifact. This
change makes `byte_rtu_checkpoint` authoritative, keeps writing `rtu_checkpoint` for one
release, and reads `byte_rtu_checkpoint` → `rtu_checkpoint` → `None`. There is never a
second copy of the file, so the two fields cannot disagree.

`engine` describes what produces generative text; grounding always comes from
`recall()`. `hybrid` means the profile carries both a byte-RTU checkpoint and a
GGUF config, and `fusion_mode` decides which backend answers the slow path.

### Decision 2 — Save order: artifacts first, manifest last, atomically

`save_profile()` becomes:

1. knowledge store export (`knowledge.db` via `KnowledgeStore.export_to`; a legacy
   `knowledge.json` is written only for profiles that already had one),
2. `memory.safetensors` (existing plastic state),
3. `rtu.safetensors` (`ByteRtuBackend.model.save`) when the profile owns a checkpoint,
4. `config.json` last, via `config.json.tmp` + `os.replace`.

The plan text says "`knowledge.json`, then `rtu.safetensors`, then `config.json`". The
code's knowledge artifact is `knowledge.db` (the JSON writer is legacy), so this change
follows the *rule* — every artifact the manifest names is written before the manifest —
and records the naming difference here rather than adding a second knowledge file to
every profile.

Writing the manifest last means a partially written profile never claims an artifact
that is not on disk. `os.replace` makes the manifest update atomic on POSIX.

### Decision 3 — Named errors, defined where both sides can import them

`MissingCheckpointError(FileNotFoundError)` and `InvalidProfileError(ValueError)` are
defined in `src/model/rtu.py`, which imports nothing from the engine or the backend.
`Model.load()` raises `MissingCheckpointError` for a missing/empty path;
`load_profile()` raises it when the manifest declares a checkpoint that is absent, with
the resolved path in the message; `load_profile()` raises `InvalidProfileError` when
`config.json` is missing, unreadable, or not a JSON object.

Placement rationale: `droid.py` imports `byte_rtu_backend`, which imports `rtu`; putting
the exceptions in `rtu` avoids a cycle and keeps one class identity for `except`
clauses in all three modules.

### Decision 4 — Backend conformance: reset, prompt, bounds

- `reset()` clears `self._prompt_bytes` (the retained prompt buffer) and the model's
  session state. `generate()` calls it on entry and in `finally`, so no primed state
  survives an answer.
- The prompt is encoded without the trailing `"\n"`. The query stays last (plan 10
  decision 4); the newline was an artifact of the raw-UI loop, which appends it for the
  interactive terminal case. The parity test therefore compares against the UI loop's
  bytes *including* its newline, not against the backend's.
- The sampling loop runs `min(max_tokens, max_output_bytes)` times. `max_tokens <= 0`
  returns `""`.
- Tests (1.10): `reset()` clears the buffer and the state; no trailing newline in the
  primed bytes; a `max_tokens` of 4 stops at 4 bytes even when `max_output_bytes` is
  256, and `max_output_bytes` still caps a larger `max_tokens`.

### Decision 5 — Combined score: five named signals, gate first

`recall()` computes, per candidate that already passed the `dense >= 0.48` gate:

```
score = w_dense·dense + w_lexical·lexical + w_assoc·relu(dot(q_assoc, fact_emb))
      + w_recency·recency + w_conf·confidence
score *= (1 + w_contextual·max(0, contextual))
```

Weights live in the single `RECALL_SCORE_WEIGHTS` table introduced by the correctness
change. This change raises `associative` from `0.0` to `0.15`, `recency` from `0.0` to
`0.02`, and `confidence` from `0.0` to `0.05`. `q_assoc` is
`normalize(memory.retrieve(q_emb))`, computed once per recall inside a guarded block;
on failure the bonus is zero and the recall proceeds. The bonus is `relu`-clipped, so a
negative association cannot push a candidate down below a fact that passed the gate on
its own merit.

Cold start: an empty `S` makes `retrieve()` return `0.1 · h_trace`, which is near zero
for a fresh profile; the no-regression test asserts identical ordering with and without
the bonus when `S` is empty.

### Decision 6 — Manager and isolation

`DroidManager.create_droid(name, byte_rtu=False, ...)` constructs a `ByteRtuBackend`
with the profile's `rtu_dim/rtu_layers/rtu_spread`, attaches it through the existing
`byte_rtu_backend=` slot, sets `engine="byte_rtu"` (or `hybrid` when a GGUF config is
also present), and saves the profile. The dual-backend guard already raises `ValueError`
in `DroidEngine.__init__`; 1.13 keeps that test green.

Round trip: `create_droid(byte_rtu=True)` → teach → save → `DroidManager` in a fresh
directory-free instance → `get_droid` → every `m.*` parameter, `state.*`,
`decaytrace.*`, `embedtrace.*`, `step_count`, and knowledge fact count match. Text is
never compared (sampling is stochastic).

Isolation: the manifest's checkpoint path must resolve inside the profile directory. An
absolute path or a path containing `..` raises `InvalidProfileError`. The isolation test
writes to profile A, then re-loads profile B of the same engine and asserts B's state
checksums and fact count are unchanged.

### Decision 7 — Console: third engine and the answer-path badge

`app.py` engine selection gains `"Droid + Raw Byte RTU (Fused)"`. Fused mode reuses the
droid chat handler; only the backend differs. Fast-path answers stay extractive, so they
are byte-identical to plain droid mode.

The badge reads `engine._last_answer_path`, whose canonical values become
`recall` | `fused` | `llm`:

- `recall` — extractive answer from `recall()` (includes the no-information fallback);
- `fused` — `ByteRtuBackend.generate()` produced the text;
- `llm` — `GgufBackend.generate()` (or the external LLM provider) produced the text.

The T010 sites set `extractive`/`fallback`/`generative`; this change renames them at the
source and keeps a read-only alias property that maps the old names for any external
caller.

The badge also shows the profile engine, the checkpoint filename (or "no checkpoint —
extractive only"), and the anchor real-vs-mock state from the correctness change.

### Decision 8 — Training script is a build step, not a shipped weight

`scripts/train_byte_rtu.py`:

```
python scripts/train_byte_rtu.py --profile droids/droid-rs --corpus plans docs README.md \
    --steps 2000 --seq-len 512 --dim 384 --layers 8 --spread 32 --seed 0
```

It reads the corpus as bytes, slices fixed-length sequences, calls
`Model.forward(..., ce_only=True)` (or the full loss set — see below), reuses
`compute_losses` from `src/model/rtu.py`, logs loss every N steps to
`<profile>/train_log.json`, and finishes with `model.save(<profile>/rtu.safetensors)`
followed by the manifest update. `ce_only` is chosen because the corpus is
unstructured bytes with no `end` annotations; the stop loss cannot be trained without
them. The default `--losses ce` is documented in the script's help text.

Weights are never committed: `droids/` is a generated artifact directory. The grounding
test reads `RTU_CHECKPOINT` and skips loudly when it is unset.

### Decision 9 — Verification gates

- **E1 (1.28):** the profile round-trip test passes
  (`tests/test_droid_rtu_fusion.py::test_fused_profile_roundtrip`).
- **E2 (1.29):** the grounding test passes with `RTU_CHECKPOINT` set: teach a
  remote-sensing paragraph, ask `/slow`, assert at least one taught fact substring
  appears in the answer.
- **E3 (1.30):** the isolation test passes
  (`tests/test_droid_rtu_fusion.py::test_profiles_are_isolated`).
- The full suite (`PYTHONPATH=. pytest tests`) runs before any gate is ticked.

## Risks / Trade-offs

- **Strict load breaks loose callers.** `Model.load()` no longer tolerates a missing
  file. The two production callers are updated in the same change; any third caller
  fails loudly, which is the point.
- **Double path fields.** Writing `rtu_checkpoint` and `byte_rtu_checkpoint` for one
  release is redundant. The alternative — silently dropping the old field — breaks the
  profiles T017 already wrote. The alias is documented and read-only.
- **Recency and confidence weights are new signal.** They are small and clamped, but
  they can reorder near-ties. The fixture test and the E2E script are the guard; if the
  E2E top hit moves, that is a finding to report before the weights ship.
- **Training quality is unknown until measured.** A small corpus with random init may
  emit garbled bytes. The fast path stays extractive, the slow path is opt-in per query,
  and the README states the quality ceiling instead of implying fluency.
- **Manual validation cannot be automated.** 1.20 is a recorded human check; the
  automatic gates cover the backend, the profile, and the ranking instead.
- **Checkpoint size.** `dim=384, layers=8` is ~1.2M parameters per layer for the
  embedding trace buffers; a profile checkpoint is a few MB, not a few hundred KB. It
  stays inside the profile directory and is excluded from the package archive only if
  it exceeds the documented cap; the default `.droid` export includes it.
