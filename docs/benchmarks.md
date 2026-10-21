# Benchmarks and grammar probes

Every quantitative claim this project makes is produced by one command and one
report. The rule is simple: **a number that was not measured is reported as a
skip with a reason, never as a value, and never as a lower bar.**

## Commands

```sh
# Everything, against a checkpoint, with a report written beside the summary
PYTHONPATH=. python scripts/bench_edge.py \
    --checkpoint droids/droid-rs/rtu.safetensors \
    --corpus README.md VISION.md \
    --tokens 2000 \
    --probes evals/grammar_probes.jsonl \
    --report reports/bench-edge.json

# The same metrics through the headless CLI, as key=value lines
PYTHONPATH=. python -m src.cli bench \
    --checkpoint droids/droid-rs/rtu.safetensors \
    --corpus README.md,VISION.md \
    --tokens 2000 --probes evals/grammar_probes.jsonl

# Only the parameter count of a configuration (no checkpoint needed)
PYTHONPATH=. python -m src.cli bench --dim 512 --layers 16

# The chunked forward against the sequential forward, with the guard applied
PYTHONPATH=. python scripts/bench_parallel_scan.py --lengths 512 --runs 5
```

`bench --help` and `scripts/bench_edge.py --help` list every flag. The CLI prints
one `key=value` line per metric and always prints the report path.

## Metrics

| key | meaning | conditions reported |
| --- | --- | --- |
| `tokens_per_second` | generation throughput on the sequential per-token path | token count, elapsed wall clock, path, batch size, sequence length, thread count, hardware, every timed run, coefficient of variation |
| `training_tokens_per_second` | training throughput on the per-token path | same conditions, `path=per_token` |
| `params` | trainable elements of the constructed model | dimension, layer count, vocabulary |
| `rss_mb` | resident set size of the process after loading the profile | the artifacts that were loaded |
| `held_out_perplexity` | byte-level perplexity on a hash-selected held-out split | corpus id, evaluated bytes, split rule, seed, containment check |
| `chunked_speed_factor` | sequential seconds / chunked seconds | both timings, ratio, hardware, sequence length, threads, CV, load |
| `grammar_accuracy.<phenomenon>` | probe accuracy per phenomenon | probe file hash, checkpoint hash, margins |
| `grammar_accuracy` | aggregate probe accuracy (always printed with the per-phenomenon lines) | same |

Throughput is the **median of the repeated timed runs after one warm-up**. Every
individual run is in the report so the spread is visible; a mean hides an
outlier and a minimum flatters a loaded machine.

## Report schema

The report is JSON, written atomically, with:

```json
{
  "schema_version": 1,
  "commit": "<full 40-character hash>",
  "command": "<the exact command line>",
  "dependencies": {"python": "...", "torch": "...", "numpy": "...", "...": "..."},
  "config": {"seed": 0, "runs": 3, "tokens": 2000, "model": {"...": "..."}, "corpus": ["..."]},
  "hardware": "machine | processor | cpu_count | threads | python | platform",
  "measurements": {"<name>": {"status": "passed", "value": 0.0, "...conditions": "..."}},
  "outcome": "passed"
}
```

A measurement that was not taken is written as:

```json
{"status": "skipped", "reason": "no checkpoint supplied (use --checkpoint or --profile)"}
```

and carries **no** `value` key. Outcomes are the closed set `passed`, `failed`,
`skipped`, `blocked`. An all-skipped run is `skipped`, not `passed`. A failed or
unflattering measurement is still written; nothing is discarded to obtain a green
result.

## Thresholds and skip rules

| constant | value | meaning |
| --- | --- | --- |
| `SPEED_FACTOR_THRESHOLD` | 10.0 | the `training/parallel-scan` factor at sequence length 512 |
| `SPEED_FACTOR_SEQUENCE_LENGTH` | 512 | the sequence length the factor is stated at |
| `VARIANCE_THRESHOLD` | 0.10 | coefficient of variation above which a comparison is skipped |
| `LOAD_THRESHOLD` | 0.40 | one-minute load average / CPU count above which a comparison is skipped |
| `PERPLEXITY_WINDOW_BYTES` | 64 | window length of the containment check |

Both guards are mandatory. Variance alone cannot catch a machine that is
*uniformly* busy: the whole run is slow, the variation is low, and the factor
fails for a reason that has nothing to do with the code. The load guard skips
that run with the measured load instead. `scripts/bench_parallel_scan.py` accepts
`--max-load` and `--max-cv` only to *tighten* the guards; the threshold constants
are not lowered to make a machine pass.

## Held-out split

Documents are ordered by `sha256(f"{seed}:{document_id}")` and the first
`round(held_out_fraction * document_count)` are held out. The split is
independent of filesystem order and reproducible from the seed alone, and no
document can appear on both sides. Before a perplexity is reported, sampled
64-byte windows from the held-out documents are searched for in the training
bytes; anything found is a refusal (a skip naming the documents), not a number.

## Grammar probes

`evals/grammar_probes.jsonl` is a **project-authored regression suite**, not an
independent benchmark and not a capability certificate. It exists to catch
regressions in four constructions — `subject_verb_agreement`, `active_voice`,
`passive_voice`, `question_formation` — with at least eight probes each.

Each line is one JSON object:

```json
{"id": "sva-01", "phenomenon": "subject_verb_agreement",
 "context": "The dog ", "correct": "barks loudly.",
 "incorrect": ["bark loudly.", "barking loudly."]}
```

Scoring is a contrastive cloze margin: the model's mean log-probability per byte
of the correct continuation minus the best incorrect continuation's. The margin,
not raw likelihood, decides the probe, so a model that likes everything equally
cannot look competent. Scoring is frozen (no optimizer step; the recurrent
buffers are saved and restored) and the report carries the parameter hash before
and after, the checkpoint hash, and the probe-file hash.

A malformed line is rejected with its line number and reason while the rest of
the file loads. An unknown phenomenon is rejected with the value named.

## Quoting a number

Any figure quoted from this harness must name **the machine and the commit**. A
bare "N tokens per second" is not evidence; the report it came from is. Do not
commit model weights or generated reports that contain machine-local paths as
release evidence without the ledger row that cites them.
