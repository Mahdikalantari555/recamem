# Post-merge completion — foundation implementation, not release completion

## Scope and status

The proposal was brought from the earlier planning branch onto the active branch
`arena/67a51f42-test-model-thing-torch`. Only tasks **1.1, 2.1, 2.2, 2.3 and 2.4**
are complete. **65/70 umbrella tasks remain**, including all three feature
dependencies, historical evidence reconciliation, full profile persistence,
real injection, durable DB snapshots, and model/benchmark acceptance.

Implemented:

- Machine-readable ledger covering all 489 baseline inventory entries. A schema
  and reference guard catches missing/duplicate IDs, nonexistent tests/evidence,
  unsupported verified claims and incomplete release tiers. Initial unreviewed
  rows remain blocked; unimplemented dependency rows remain missing.
- Module-compatible adapter ABC and honest optional lifecycle defaults.
- Independent plastic snapshot/branch/restore tensors and metadata; a pristine
  associative snapshot also clears keys created after the snapshot.
- Explicit scheduled-update capability, original-signature adapter compatibility
  and requested/applied rate reporting. Unsupported scheduling reports null.
- Exactly-once updates, no TypeError-based retry, propagated named update errors
  with causes, and no success log for a failed teaching operation.

Tests and API documentation are in `tests/test_adapter_lifecycle.py`,
`tests/test_release_evidence.py`, `docs/adapter-contract.md`, and
`docs/release-evidence.md`. State may be partially mutated by an adapter before
it fails; this work does not pretend teaching is transactional. In-memory plastic
snapshot independence does not imply durable knowledge-DB rollback is complete.

## Verification

See `reports/evidence/post-merge-foundation.json` for the exact implementation
commit, environment and command.

- Full offline suite: **492 passed, 1 skipped**, 51 explicit mock-anchor warnings.
- Skipped: trained byte-RTU grounding, because RTU_CHECKPOINT/artifact is absent.
- Remote-sensing E2E: passed with mock anchor.
- OpenSpec: **36 items valid**; this change also passes strict validation.
- Evidence schema/coverage: **489 entries valid**.
- `--require-complete`: **fails intentionally**, not merge-ready.

The workstation Conda environment was absent; a disposable virtualenv was
provisioned at the repository-prescribed executable path. No runtime environment,
model artifact or generated profile is committed.

## Blockers and next work

The remaining code is not implemented. Continue tasks 1.2–1.5 and 2.5 onward;
do not mark their checkboxes complete because the current regression suite passes.

Full acceptance also requires local ONNX/tokenizer, GGUF, hookable decoder, trained
RTU and the matched-budget 1B/8B evaluation artifacts, with hashes and configuration.
They are not present. Model-host downloads are unavailable under this sandbox's
network allowlist. Supply those artifacts or run the explicit integration and
benchmark gates in a model-enabled environment and bring back reproducible evidence.
Research targets must be measured, not presumed satisfied.

The proposal is deliberately not archived and the implementation is deliberately
not merged into main. A draft PR preserves reviewable progress without presenting
this foundation as the completed 70-task umbrella.
