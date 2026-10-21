# Edge-scale pretraining run

This document describes `scripts/train_edge_scale.py`, the runnable protocol for
an edge-scale pretraining run of the byte-level recurrent model on CPU-only
hardware, and the `training/edge-pretraining` capability behind it
(`src/model/edge_pretraining.py`). The chunked training pass itself is documented
in [`docs/rtu.md`](rtu.md).

## Command

A short run over a small corpus, which is the form that runs in a test:

```bash
PYTHONPATH=. python scripts/train_edge_scale.py \
  --corpus docs/rtu.md docs/benchmarks.md \
  --dim 64 --layers 2 --spread 4 --vocab-size 256 \
  --chunk-size 32 --chunked --seed 0 \
  --byte-target 256 --max-chunks 8 \
  --report reports/edge-pretraining.json
```

The headline run uses the default configuration and a real byte target:

```bash
PYTHONPATH=. python scripts/train_edge_scale.py \
  --corpus docs plans README.md openspec/specs \
  --chunk-size 128 --chunked --seed 0 \
  --checkpoint-interval 1000000 --checkpoint droids/edge-run/rtu.safetensors \
  --byte-target 25000000 \
  --report reports/edge-pretraining.json
```

`--help` lists every flag. The exit code is `0` when the reported outcome is
`passed` (or `skipped` for a run that measured nothing), and `1` for `failed` or
`blocked`. The report is written in every case, atomically.

Flags:

| Flag | Meaning |
| --- | --- |
| `--corpus` | files or directories to prepare (required) |
| `--dim`, `--layers`, `--spread`, `--vocab-size` | model configuration |
| `--chunk-size` | bytes per chunk; one optimizer step per chunk |
| `--chunked` / `--per-token` | training path (chunked by default) |
| `--seed` | seed for the document order, the split and the run state |
| `--held-out-fraction` | fraction of documents held out (default 0.2) |
| `--probe` | grammar probe files, excluded from training (default `evals/grammar_probes.jsonl`) |
| `--reserved` | benchmark corpora, excluded from training |
| `--byte-target` | bytes consumed before the run counts as complete |
| `--checkpoint-interval` | bytes between checkpoints (`0` disables) |
| `--checkpoint` | checkpoint path |
| `--resume` | resume from the checkpoint's recorded corpus position |
| `--max-chunks` | stop after this many chunks (a short run) |
| `--verify-equivalence` | run the per-token equivalence check on this environment |
| `--report` | report path (default `reports/edge-pretraining.json`) |
| `--hardware-class` | the documented hardware class the run completed on |

The headline configuration is the default: `dim=1024`, `layers=24`,
`spread=32`, `vocab_size=256`, which is 25,765,121 parameters
(`edge_pretraining.parameter_count`). Use the smaller configuration for a short
run; the parameter count is checked against the configuration, not assumed.

## Corpus preparation

`prepare_corpus(sources, seed=..., held_out_fraction=..., probe_files=...,
reserved_sources=...)` returns one mapping with everything the run needs:

* `documents` — every document's bytes, keyed by its identifier (the path
  relative to `--root`, or the file name);
* `order` — documents sorted by `sha256(seed:document_id)`, never by filesystem
  order;
* `train_ids` / `held_out_ids` — the document-level split, decided per document
  by the hash of its identifier, so a document is entirely on one side and the
  split is reproducible from the seed and the identifiers alone;
* `train_bytes` — the excluded sets removed, the remaining documents joined with
  `\n` in hash order;
* `excluded_sets` — held-out split, grammar probe file, reserved benchmark
  corpus;
* `corpus_id`, `train_sha256`, byte counts.

Two preparations of the same corpus with the same seed are byte-identical, and
creating the files in a different order changes nothing.

## Exclusions and contamination

The held-out split, the grammar probe file and any `--reserved` corpus are never
part of `train_bytes`, and the report records their identity next to the result.
Before training starts, `check_contamination` compares each excluded document
against the training identifiers and bytes. An excluded document that reappears
in the training set — by identifier or by content — fails the check and is named
in the report (`corpus.contaminated_documents`, outcome `failed`). A run whose
contamination check fails reports no held-out perplexity rather than a number
computed on contaminated data.

## Resumable checkpoints

`write_checkpoint(path, model, state)` writes two files: the weights at `path`
(safetensors) and the run state at `path.state.json`, carrying the corpus
position, the byte step count, the seed and the configuration. The weights carry
the optimizer state as well. The state is staged and moved into place only after
the weights land, and `Model.save` writes a temporary file and renames it, so an
interrupted write leaves the previous checkpoint intact and readable.

`--resume` continues exactly at the recorded corpus position: bytes before it are
never read again. Checkpoints are written every `--checkpoint-interval` bytes.

## Report

The report uses the benchmark report format (`src/model/benchmarks.py`):
`commit`, `command`, `dependencies`, `config` (including the seed), `hardware`,
`measurements`, `artifacts`, `outcome`, `schema_version`. Its measurements are:

| Measurement | Content |
| --- | --- |
| `bytes_consumed` | bytes trained on, with the target |
| `wall_clock_seconds` | elapsed time and the training path |
| `initial_training_loss` / `final_training_loss` | first and last logged chunk loss |
| `held_out_perplexity` | measured on the excluded held-out split, or skipped with the reason |
| `parameter_count` | `Model.count()` and the configured expectation |
| `hardware_class` | the class the run completed on, and the documented class |

The document also carries `corpus` (identity, train digest, split rule, excluded
sets, contamination check), `run` (positions, chunks, bytes, interrupt flag,
wall clock), `acceptance` (one record per criterion), `acceptance_met`,
`unmet_criteria` and `equivalence`.

## Acceptance criteria

A run is reported as `passed` only when every criterion below holds. Otherwise it
is `failed` (a criterion did not hold) or `blocked` (it was interrupted), and
`unmet_criteria` names each one that did not hold:

1. `byte_target` — the consumed byte count reached `--byte-target`;
2. `parameter_count` — the model's parameter count matches the configuration;
3. `loss_decreased` — the final logged loss is below the initial logged loss;
4. `held_out_perplexity` — finite, measured, and computed on the excluded split;
5. `hardware_class` — the run completed on the documented hardware class;
6. `completed` — the run was not interrupted;
7. `contamination` — no excluded document appeared in the training set.

An interrupted run still writes a report, with the outcome `blocked`, the bytes
consumed and the wall clock.

## Equivalence boundary

A chunked run reports what is equivalent to the per-token path and what is not:
the losses, the accumulated gradients and the trace buffers are equivalent; the
parameter trajectory is not, because the chunked path takes one optimizer step
per chunk. With `--verify-equivalence` the check runs on this environment and the
report carries its result. Without it, the report sets `claim: false` and names
the missing verification instead of claiming the equivalence.
