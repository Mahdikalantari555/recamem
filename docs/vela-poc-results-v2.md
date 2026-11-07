# Vela PoC Results — v2 (disposition)

Date: 2026-11-07. Status report on the Vela text-only learning benchmark PoC.
The original report, `docs/vela-poc-results.md` (29 October 2026), remains
unchanged and is the record of the run itself; this document records what the
project now claims, what it does not claim, and why.

## Finding

**No memory advantage over ordinary retrieval was measured.** On the shipped
12-item set the original run recorded:

| Condition | Decision accuracy | Retrieval hits |
| --- | --- | --- |
| A — base, no memory | 1/12 | 0/12 |
| B — base + ordinary retrieval | 5/12 | 10/12 |
| C — RecaMem + RTU | 3/12 | 9/12 |
| D — RTU frozen | 3/12 | 10/12 |
| E — oracle rules | 6/12 | 12/12 |

C did not beat B, and E did not beat the model's own ceiling. The bottleneck in
that run was the base model's decision procedure plus a prompt-only injection
path, not memory selection. That is a negative result and it is published as
one.

## Why C-versus-D cannot be read as a plastic-memory effect

Condition D froze plastic updates, but the only injection path available with
the shipped GGUF backend is prompt stuffing. With prompt-only injection, a
frozen C and a plastic C are identical in what the model sees, so the D ≈ C
result in the original report measures the absence of an injection channel, not
the absence of a memory effect.

`scripts/run_vela_benchmark.py` now probes the backend before running any
condition (`--probe-only`). The probe uses `recamem.backend_contracts.capabilities_of`
and writes an explicit record:

* hidden injection supported → conditions C and D run on the matched context
  budget and the report may state a paired comparison;
* hidden injection unsupported → the report is written with `outcome: blocked`,
  the backend id and `missing_capability: hidden_injection`, and the claim field
  reads `none`. No C-versus-D score is published as evidence.

The probe result for this environment is recorded in
`reports/evidence/2026-11-07-vela-capability-probe.json`: `blocked`, because no
GGUF artifact exists here, and the only shipped real backend advertises
prompt-only injection.

## Provenance

* Original run: `reports/vela-poc-final.json`, MiniCPM 2B Q4_K_M (GGUF, CPU), 16
  threads, ONNX MiniLM anchor. The artifact path in that record was
  machine-local (`/mnt/d/AI/...`, host not this one); the file is identified by
  its report and checksum is **not** recorded, so the weights cannot be
  re-verified from this repository. That is a provenance gap, not a passing
  result.
* Harness: `scripts/run_vela_benchmark.py`. The default model path is now empty;
  a model must be named by `--model` or `RECAMEM_GGUF`, so no machine-local path
  is ever published as an identity.
* Items: V1–V12 in `recamem_text_learning_benchmark.md`; no item was added or
  removed for this report.

## Still open (and why)

| Item | State | Missing |
| --- | --- | --- |
| C vs D on matched budget with repeats and a confidence interval | blocked | a backend with hidden injection plus a named model artifact |
| Re-run with a more instruction-following model | blocked | model weights and an inference budget |
| Held-out generated Vela rule sets | open | rule-set generator and a second run |
| Extraction heuristics removal | open | a model that emits the decision tokens directly |

None of these is claimed. The negative finding above is the only conclusion the
repository supports, and it is the one published.
