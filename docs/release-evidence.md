# Completion evidence ledger

`reports/completion-evidence.json` covers the frozen post-merge inventory:
310 archived tasks, 114 baseline requirements and 65 dependent feature tasks.
An archived checkbox is **not** verification. Initial rows are `blocked` pending
historical reconciliation or `missing` for the unimplemented dependencies. Empty
implementation/test/evidence arrays on those rows are intentional, not fabricated
proof. Each row has an owning task for follow-up.

## Check schema and coverage

```sh
PYTHONPATH=. python scripts/check_release_evidence.py
```

Exit 0 means the inventory is fully represented and all claims have structurally
valid references. It does **not** mean the release is complete.

```sh
PYTHONPATH=. python scripts/check_release_evidence.py --require-complete
```

The release gate rejects all unresolved rows and requires passing evidence in
all five tiers: offline, real-model, grounding, concurrency-migration, benchmark.
Exit codes: 0 valid, 1 invalid/incomplete, 2 unreadable/malformed input.
This gate is expected to fail during implementation. Do not weaken it to merge.

## Completion gap inventory

`reports/completion-gaps.json` lists every item still open, with its source,
owner, verification command, artifact, test and status. A row may be marked
`closed` only when its artifact exists, its pytest node id resolves and a
passing evidence record references it.

```sh
PYTHONPATH=. python scripts/check_completion_gaps.py --schema-only   # structure/coverage, CI
PYTHONPATH=. python scripts/check_completion_gaps.py                 # exit 1 until every gap is closed
```

The default mode is expected to fail during implementation. It is a second,
independent guard: it does **not** replace `--require-complete` above and it
must not be weakened to merge.

## Row schema

- `id`: exact `source-path#task-id` or `source-path#requirement-name` from inventory.
- `status`: verified, missing, partial, blocked, superseded.
- `owner`: existing `tasks.md#task-id` responsible for the row.
- `reason`: human-readable status rationale.
- `implementation`: repository-relative implementation/documentation paths.
- `tests`: pytest node IDs; source AST validation checks named tests/classes exist.
- `evidence`: repository-relative JSON evidence files, not a bare test-count claim.
- `required_tiers`: nonempty list of the applicable tiers.
- `replacement`: required on superseded rows; an existing spec requirement reference.

Verified rows require nonempty implementation, tests and evidence, with passing
results for every required tier. Superseded rows require a real replacement and
an approved evidence record. Skipped/blocked evidence requires a reason and can
never certify a verified row. Paths must exist inside the repository; traversal
and escaping symlinks are rejected. Duplicate, missing and unknown IDs fail.

## Evidence record

Each JSON evidence record includes full 40-character commit hash, command,
environment/dependency versions, configuration (including seeds when applicable),
tier and outcome. Include `skip_reason` when skipped or blocked. Include raw-result
locations, artifact checksums and hardware details for model/benchmark work.
Allowed outcomes are passed, failed, skipped, blocked and approved (spec revision).
No model weights should be committed.

The guard verifies schema and references; it cannot prove that an experiment was
run or its scientific conclusions are sound. Review the actual command output,
commit and measurements before certifying evidence. Full historical reconciliation
(tasks 1.2/1.3) and the final cross-feature gates remain required. Update the ledger
and guard together if an approved inventory revision adds requirements; never delete
unresolved entries merely to obtain a green release gate.

## Current records (2026-11-03)

- `reports/evidence/2026-11-03-offline-baseline.json` — fresh full-suite
  baseline with the exact failing test ids (outcome `failed`, recorded rather
  than hidden).
- `reports/evidence/2026-11-03-offline-closure.json` — the offline fixes in
  `finish-implementation-and-close-gaps` with their contract tests.

Neither record claims a `real-model`, `grounding` or `benchmark` tier result.
`--require-complete` still exits 1 until those tiers have passing evidence.
