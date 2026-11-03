# Design

## Context

The project has three independent notions of "done" that currently disagree:

1. *Structural*: `offline_gate.py` and `validate_openspec.py` are green.
2. *Ledger*: `check_release_evidence.py` is green (schema/coverage only), while
   `--require-complete` is red by design.
3. *Scientific*: the selection report says `verified_winner`, the Vela PoC says
   memory does not beat retrieval, and the language-learning change says the
   core has not learned language (BPB 6.8).

Closing the project means reconciling all three with measured evidence, not
choosing the cheapest green signal. Relevant sources: `docs/release-evidence.md`,
`docs/release-matrix.md`, `docs/vela-poc-results.md`,
`reports/completion-evidence.json`, `reports/recamem-core-selection.md`,
`plans/09_recamem_plan.md` and the two active change directories.

## Goals / Non-Goals

**Goals:**
- One authoritative, machine-readable list of everything left, with a guard
  that cannot be satisfied by editing a status field.
- A green offline suite with a current, provenance-stamped result.
- Passing evidence in all five acceptance tiers, or an explicit `blocked` record
  naming the missing artifact.
- Config, docs and reports that agree with each other and with the code.
- A measurable statement about whether the RTU adds value beyond retrieval.
- A release gate that exits 0 because work was done, not because it was relaxed.

**Non-Goals:**
- No new user-facing product (see `recamem-svelte-companion`).
- No architecture redesign, no new core candidate, no new benchmark family.
- No committing weights, checkpoints, or generated reports into Git beyond the
  repository's existing evidence conventions.
- No claiming general intelligence or production readiness from a PoC.

## Decisions

### D1: Inventory is a data file, not a checklist in prose
**Decision**: `reports/completion-gaps.json` is the single source of truth; each
gap has `id`, `source`, `owner`, `verify_command`, `artifact`, `test`, `tier` and
`status`. `scripts/check_completion_gaps.py` validates that a `closed` gap has a
real artifact, a named test and a passing evidence record.
**Rationale**: The existing ledger proves that prose checkboxes drift. A guard
over a data file can be run in CI.
**Alternative**: Add more rows to `completion-evidence.json`. Rejected: that
ledger maps historical requirements, not the current gap list; mixing them
would let a status edit masquerade as a fix.

### D2: Never weaken a gate
**Decision**: `check_release_evidence.py` keeps its row schema, tier list and
`--require-complete` semantics. Only implementation, tests and evidence records
change.
**Rationale**: `docs/release-evidence.md` explicitly says the gate is expected
to fail during implementation and must not be weakened to merge.
**Alternative**: Allow an `accepted-miss` status. Rejected: that is exactly the
shortcut the ledger was built to prevent. Misses are published as `partial`.

### D3: Tier isolation, with an honest blocked fallback
**Decision**: Offline, concurrency-migration and benchmark tiers run in CI. The
real-model and grounding tiers run on a machine with weights, via a documented
runbook; each produces a JSON evidence record with sha256, versions, seed,
command, commit and hardware. If weights cannot be obtained, the row stays
`blocked` and completion is declared **not reached**, not "done with caveats".
**Rationale**: `reports/2026-10-22-real-model-verification.md` already shows
that a blocked download is not a result.
**Alternative**: Substitute deterministic mocks. Rejected: silently mocked
real-model tests are an explicit anti-rule in `AGENTS.md`.

### D4: Missed performance targets are met or formally re-baselined
**Decision**: For each of the seven misses in `docs/release-matrix.md`, first
attempt an implementation fix (HNSW index build, FTS-first lexical path,
fused-scan pruning, surprise-gate calibration, batched merge). Only when a
target is shown to be unattainable on the declared hardware may it be changed,
and only through an approved requirement revision that records the old target,
the new target, the measurement and the reason.
**Rationale**: The ledger requires "target misses MUST be published without
changing thresholds".
**Alternative**: Drop the slow cases. Rejected: deleting a case to pass is
fabrication.

### D5: Core config is generated, not hand-maintained
**Decision**: `configs/recamem_selected_core.json` is regenerated from
`reports/evidence/recamem-core-tournament.json` plus
`reports/recamem-core-selection.md`; a parity test asserts
`selection_status`, `memory_arm_id`, `multitask_batching_mode`, calibration
values and `dataset_source` all match. The 17M shape keeps `trained: false`
until a real run exists.
**Rationale**: Today the config says `provisional_core` / `M3` / `interleaved`
while the report says `verified_winner` / `M4` / `summed`. Downstream
`build_recamem_core` reads the config, so the disagreement is load-bearing.
**Alternative**: Patch the three fields by hand. Rejected: it will drift again.

### D6: RTU isolation before any superiority claim
**Decision**: Extend `scripts/run_vela_benchmark.py` with a real hidden-injection
capability check (`docs/backend-contracts.md`), run C versus D on the same
model, seed, context budget and item set, and report a paired comparison with
a confidence interval. If no backend supports hidden injection, record
`blocked` with the missing capability and publish the existing C<=B result as a
negative finding.
**Rationale**: `docs/vela-poc-results.md` says D approx C because injection was
prompt-only; without isolation no claim about plasticity is meaningful.
**Alternative**: Change the benchmark prompt to favour C. Rejected: answer-key
leakage and prompt tuning would invalidate the control.

### D7: Language learning is a dependency, not a duplicate
**Decision**: This change verifies, integrates and archives
`recamem-language-learning` and `recamem-core-architecture-tournament`; it does
not re-specify G0-G4, token budgets or the tournament funnel. Its own tasks
reference those task ids.
**Rationale**: Two changes describing the same training work would create
conflicting spec deltas on `training/edge-pretraining`.
**Alternative**: Copy the tasks here. Rejected: duplication and drift.

### D8: Documentation and CI hygiene are testable defects
**Decision**: Add one parity test that parses every `.ipynb`, checks
`plans/README.md` links resolve, rejects stray shell heredoc text in Markdown,
rejects machine-specific absolute paths in agent-facing docs, and asserts no
CI step masks a verification command with `|| true`.
**Rationale**: The Colab notebook is currently unopenable and
`docs/cloud-training.md` points users to it. Masked CI failures would hide the
very regressions this change fixes.

## Risks / Trade-offs

- **Risk**: Real weights may be unreachable in the sandbox. → Mitigation: the
  runbook is explicit; the outcome is a documented `blocked`, and completion is
  not declared. No proxy or redistributed weight source is used without the
  user's decision.
- **Risk**: 1M/3M/5M CPU training can take days. → Mitigation: checkpoint/resume
  from `scripts/train_lm.py`, and gates G0-G2 before scale-up; a failed gate
  stops the ladder instead of consuming compute.
- **Risk**: Re-baselining a target can be abused. → Mitigation: requires an
  approved spec revision in the ledger and a published measurement, with the
  old target retained in the record.
- **Risk**: Fixing config parity may change runtime behaviour for code that
  reads `memory_arm_id`. → Mitigation: a regression test pins the loader
  contract, and `docs/migration.md` records the change.
- **Trade-off**: A single closure change is wide. → Accepted: the user asked for
  everything left to be fixed; the inventory keeps the width auditable.

## Migration Plan

1. Freeze the baseline: full test run, gate results, inventory file (no code
   changes yet).
2. Offline correctness and documentation fixes with pinning tests.
3. Config/ledger parity and the notebook/CI/doc fixes.
4. Performance work, measured and published.
5. Research dependencies (language learning, tournament) completed and archived.
6. Real-model and grounding tiers on a machine with weights; evidence records.
7. Ledger rows resolved; `--require-complete` green; release docs updated.
8. Archive this change only after every task has an evidence record.

No destructive migration: profile schema v2, loopback-only API, fail-fast
anchor semantics and deprecation shims stay intact. Shim removal is deferred to
a separate release decision per `docs/migration.md`.

## Verification / Definition of Done

- `PYTHONPATH=. pytest tests -q` green with a recorded count, commit and
  environment.
- `PYTHONPATH=. python scripts/offline_gate.py` exit 0.
- `PYTHONPATH=. python scripts/check_completion_gaps.py` exit 0 with every row
  `closed` and every artifact present.
- `PYTHONPATH=. python scripts/check_release_evidence.py --require-complete`
  exit 0 with passing evidence in all five tiers.
- `PYTHONPATH=. python scripts/validate_openspec.py` exit 0.
- `recamem doctor` exit 0 with `anchor=real` and a named GGUF backend.
- `docs/release-matrix.md` contains no unmet target unless it is marked
  partial with an approved re-baseline reference.

## Open Questions

- Which 8B reference model and budget satisfy task 7.10's matched comparison?
- Is a GPU available for the 3M/5M/17M runs, or are CPU estimates the only
  feasible path?
- Should the deprecation shims under `src/model/` be removed in this release or
  in the next one?
- Which backend will provide hidden injection for the Vela C-versus-D study?
