# Real-model verification report — `post-merge-completion`

Date: 2026-10-22 (UTC). Branch: `arena/6b82bf5b-test-model-thing-torch`, based on `764f554`.
Scope: download a real MiniCPM model from Hugging Face and run the open real-model tasks
(3.4, 3.7, 7.10, 8.1, 9.2) against it, then archive the change.

## Outcome

| Item | Result |
| --- | --- |
| Real MiniCPM model downloaded from Hugging Face | **Blocked.** The sandbox cannot complete a TLS connection to `huggingface.co`. No MiniCPM weights exist on disk. |
| Real-model tasks 3.4, 3.7, 7.10, 8.1, 9.2 | **Not completed.** All five are blocked on the missing artifacts (table below). Their checkboxes stay unchecked. |
| Heavy files kept out of Git | **Done.** `.gitignore` now excludes model weights and `models/`. Nothing matching those patterns is tracked. |
| Report | This file. The machine-readable record is `reports/evidence/2026-10-22-minicpm5-1b-download.json`. |
| Archive `post-merge-completion` | **Not performed.** See "Archive decision". |

The model was not obtained, so no real-model result is reported here. Nothing in this report
should be read as a passing real-model, grounding or benchmark result.

## Target model

- Family: MiniCPM (OpenBMB). The sandbox has 4 GB RAM and 2 vCPUs, so the 8B models are too large to run here.
- Chosen artifact: `openbmb/MiniCPM5-1B-GGUF`, file `MiniCPM5-1B-Q4_K_M.gguf` (657 MB, Q4_K_M).
- Source for the repo ID and filename: the OpenBMB MiniCPM deployment guide
  (`docs/deployment/llama_cpp.md`, "Released GGUF artifacts" table), which lists the file and size
  and gives the `huggingface-cli download openbmb/MiniCPM5-2B-GGUF ...` pattern. I could not open
  the Hugging Face model page from here, so the repo ID and file list come from that guide and
  were not checked against the Hub. Confirm both on the Hub before relying on the download.

## Download attempts (all failed at the network layer)

Environment: Linux 6.1, 2 vCPU, 4 GB RAM, Python 3.11.2 (system), repo venv `.venv`
(git-ignored) with `huggingface_hub 2.2.0` and `pytest 9.1.1`. torch is not installed.
The conda environment named in `AGENTS.md` (`/home/asus/miniforge3/envs/ai`) does not exist here.

| # | Time (UTC) | Command | Result |
| --- | --- | --- | --- |
| 1 | 06:40 | `curl https://huggingface.co/api/models/openbmb/MiniCPM4-0.5B` | `SSL_ERROR_SYSCALL` (exit 35), no HTTP response |
| 2 | 06:40 | `curl -v https://huggingface.co/` | DNS resolves; TLS ClientHello is sent; connection closes during handshake |
| 3 | 06:40 | same probe on `cdn-lfs.hf.co`, `hf-mirror.com`, `modelscope.cn`, `objects.githubusercontent.com` | same TLS failure |
| 4 | 06:40 | `hf_hub_download("openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf")` | `LocalEntryNotFoundError` after 5 retries (about 23 s); every retry: `TLS/SSL connection has been closed (EOF)` |
| 5 | 06:41 | `.venv/bin/python scripts/fetch_real_model.py` | `status: blocked`; exit 1; record written to `reports/evidence/2026-10-22-minicpm5-1b-download.json` |

Reachable from the same sandbox, for comparison: `api.github.com` (HTTP 200), `pypi.org`
(HTTP 200) and `pypi.org/pypi/llama-cpp-python/json` (HTTP 200). The sandbox's outbound policy
allows only `github.com`, `codeload.github.com`, `api.github.com`, `registry.npmjs.org`,
`pypi.org` and `files.pythonhosted.org`. Hugging Face is not on that list, and it is the only
host that serves these weights.

I did not work around this restriction. I did not try proxies, third-party mirrors, or model
weights redistributed on PyPI or GitHub. Those copies would have unknown provenance, and
switching sources is a decision for you, not for this run.

## Task-by-task status

| Task | What it needs | Status | Missing artifact or blocker |
| --- | --- | --- | --- |
| 3.4 Real KV prefill for the GGUF binding | `llama-cpp-python` plus a local GGUF | Blocked | No GGUF file. The binding is also not installed; it is a source build (cmake and a C++ toolchain), which was not attempted. |
| 3.7 Local GGUF streaming/cache and hookable-decoder tests, with checksums | Local GGUF plus a hookable PyTorch decoder | Blocked | No GGUF file, no hookable decoder artifact, no torch in this environment. |
| 7.10 Matched-budget 1B+TTC vs an identified 8B reference | Two models, budgets and a statistical comparison | Blocked | Both the 1B and the 8B reference weights. The 8B model needs far more RAM than this sandbox has. The reference model is not yet identified in the spec. |
| 8.1 GGUF manager download/config, CLI generate and streaming | HF download and a local GGUF round trip | Blocked | Hugging Face access and a GGUF file. |
| 9.2 Real ONNX, local GGUF and trained `RTU_CHECKPOINT` integration | ONNX MiniLM, GGUF, trained RTU checkpoint | Blocked | Hugging Face access for `Xenova/all-MiniLM-L6-v2` and the GGUF file; no RTU checkpoint supplied. |

## Findings that affect the real-model evidence

1. **`GgufBackend` falls back silently to a mock.** `src/model/gguf_backend.py` (`load_model`)
   replaces the model with `MockLlama` on an `ImportError` or any load exception, and
   `generate()` returns `[Mock generation for: ...]` when `model is None`. A real-model run that
   fails to load can therefore look like it ran. This must be made to fail loudly before any GGUF
   result is recorded as evidence. I did not change it in this run, because it is a code change
   with its own tests.
2. **`OnnxMiniLM` also falls back to mock embeddings** (`src/model/onnx_anchor.py`), with only a
   `UserWarning`. Real-anchor quality stays unverified until the model is present, as the
   release-evidence spec requires.
3. **`reports/` is git-ignored** (`/reports/` in `.gitignore`), but the existing reports are
   tracked. New report files must be added with `git add -f`.

## Heavy-file policy (done)

Added to `.gitignore`: `/models/`, `*.gguf`, `*.ggml`, `*.onnx`, `*.onnx.data`, `*.pt`, `*.pth`
and `*.bin`. Verified that `git ls-files` contains no file matching any of these. `.venv/` was
already ignored. The `.git` directory is 1.2 MB. The empty `models/` directory created by the
failed download was removed.

`scripts/fetch_real_model.py` writes weights to `models/<name>/` (ignored). It records size and
SHA-256 for a successful download and the exact exception for a failed one. The token from
`HF_TOKEN` is never written to evidence; a test checks this.

## Verification run in this environment

| Command | Result |
| --- | --- |
| `python scripts/validate_openspec.py` | exit 0: "structure only; checked boxes are not evidence" |
| `python scripts/check_release_evidence.py` | exit 0: 489 entries, schema and coverage valid, "NOT a release-completion claim" |
| `python scripts/check_release_evidence.py --require-complete` | exit 1: 5 unresolved items (2 blocked, 3 partial) and, as the gate reports, "release acceptance tier has no passing evidence: grounding" and "... real-model" |
| `pytest tests/test_fetch_real_model.py tests/test_release_evidence.py` | 37 passed |
| `pytest tests/test_migration_fixtures.py` | not run; it imports torch, which is not installed here |

The full offline suite was not re-run here, because torch is not installed. Its last recorded
result is 916 passed and 1 skipped at commit `2c11003` (see `tasks.md`, task 9.1). Nothing in
this change touches model code; it adds `scripts/fetch_real_model.py`, its tests, and
`.gitignore` rules.

## Runbook: complete the real-model tier on a machine with Hugging Face access

```bash
cd test-model-thing-torch
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt huggingface_hub
.venv/bin/python scripts/fetch_real_model.py          # -> models/minicpm5-1b/MiniCPM5-1B-Q4_K_M.gguf + evidence
# Real ONNX anchor (downloads Xenova/all-MiniLM-L6-v2 into ~/.cache/huggingface)
.venv/bin/pip install -r requirements-optional.txt    # needs a C++ toolchain for llama-cpp-python
PYTHONPATH=. .venv/bin/pytest tests -q -k "onnx or gguf"
```

Before accepting the results, check that the GGUF tests run against the real model rather than
`MockLlama` (finding 1). Then record the binding version, the file SHA-256 from the evidence file,
the seed and the outcome in the ledger, and leave any tier that did not run as **blocked**.

For task 7.10 you must name the 8B reference model and its budget first. Task 9.2 also needs a
trained `RTU_CHECKPOINT`, which is a separate artifact that the repo's `scripts/train_byte_rtu.py`
can produce but that has not been produced here.

## Archive decision

`post-merge-completion` is **not** archived, for three reasons:

1. Your condition was to archive after the tasks succeed. They did not: five tasks are blocked.
2. The release-evidence spec says a change with incomplete dependency acceptance "SHALL NOT be
   archived as fully implemented". Here, task 9.4 and task 9.5 are open, and task 7.10 comes from
   an archived dependency that its own archive note records as blocked.
3. `tasks.md` states that this umbrella "stays active" until sections 1.2, 1.3, 1.5, 3.4, 3.7,
   7.9, 7.10, 8.x and 9.x close. Archiving now would record those as done.

Archiving is still possible if you explicitly approve it as a requirement revision. That would
keep the open tasks and this report in the archive as blocked, not passed.

## Next steps

1. Allow `huggingface.co` (and `cdn-lfs.hf.co`) in the sandbox egress, or place the GGUF file
   in `models/minicpm5-1b/` yourself. I will then run `scripts/fetch_real_model.py` (or verify the
   file), record its SHA-256, and run the real-model tier.
2. Choose the 8B reference model and its budget for task 7.10, or accept that 7.10 stays blocked.
3. Make `GgufBackend` and `OnnxMiniLM` fail loudly when a requested real model does not load
   (finding 1 and 2). This should be done before any real-model result is recorded.
