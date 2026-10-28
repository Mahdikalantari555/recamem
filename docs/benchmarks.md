# Benchmarks

Three long-conversation memory benchmarks have runners: LoCoMo,
LongMemEval-S and DMR (`recamem/bench/runners.py`).

```python
from recamem.bench import runners
report = runners.run("locomo", engine, seed=0, size=50)
for line in runners.describe(report):
    print(line)
```

## Dataset policy

* Each dataset is fetched **once** into `~/.cache/recamem/benchmarks` (override
  with `RECAMEM_CACHE`), with the SHA-256 recorded in `checksums.json`. A cached
  file is reused only while its checksum matches; a mismatch forces a refetch.
* A payload that is not a JSON question file (a release tarball, for example) is
  reported as blocked rather than cached.
* When the dataset cannot be fetched, the committed fixture in
  `evals/benchmarks/` may run. Its report states `dataset_source=fixture` and is
  labelled "not a LoCoMo, LongMemEval-S or DMR score".
* When neither is available the report is `status=blocked` with the failed
  source and **no metric** is printed.

## Deterministic subsets and provenance

`deterministic_subset(questions, seed, size)` shuffles with a seeded RNG and
returns the selected questions in their original order, so two runs with the
same seed evaluate the same questions in the same order.

Every report carries `provenance`: commit, command, seed, subset size,
dependency versions and hardware.

## Real-model gates

A benchmark that needs a generation model or the real anchor reports `blocked`
with the missing path. A mock anchor is never substituted silently; the offline
tier runs with an explicitly selected mock and says so.
