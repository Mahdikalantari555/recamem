# Delta plastic memory: selection, migration and rollback

This page documents how a Droid chooses its plastic adapter, how existing
profiles keep working after the delta adapters were added, and how to roll a
profile back if a migration is undesirable.

## Registry

`src.model.plastic_adapter` exposes four registered adapters:

| Name | Class | Notes |
| --- | --- | --- |
| `associative_rtu` | `PlasticAssociativeRTU` | Default; unchanged behavior for existing profiles. |
| `delta_rtu` | `DeltaPlasticAdapter` | Gated residual Delta Rule with Householder erase. |
| `sparse_delta` | `SparseDeltaAdapter` | 384→2048 TopK-128 codes on top of `DeltaPlasticAdapter`. |
| `fly_model` | `FlyModelAdapter` | Delta state with a configurable sparse/associative front end. |

Adapter identity is persisted in the profile manifest as `adapter` (also
mirrored in `adapter_meta.json`). The stored value is the **registered name**,
never a class name inferred at load time.

## Choosing an adapter for a new profile

```python
engine = DroidEngine(name="delta", plastic_adapter="delta_rtu")
```

An explicit unknown name raises instead of silently falling back to the
default:

```python
DroidEngine(name="typo", plastic_adapter="delta_rut")   # -> ValueError
```

Legacy manifests without an `adapter` field load as `associative_rtu` and are
re-saved with the identity recorded (`test_legacy_manifest_without_identity_records_migration`).

## Migrating an existing profile

There is no implicit swap. To migrate, create a new profile and copy the
knowledge base, then teach the facts again under the new adapter:

```python
old = manager.get_droid("legacy")
new = manager.create_droid("legacy-delta", plastic_adapter="delta_rtu")
with open(old.knowledge.db_path, "rb") as source, open(new.knowledge.db_path, "wb") as target:
    target.write(source.read())
```

The plastic state is **not** copied: delta state has a different geometry
(`S` per head plus sparse codes) and cannot be reinterpreted as an associative
matrix. Re-teaching is the supported migration.

### Dimension incompatibility

`load_profile()` validates dimensions and state versions before activating
anything. A checkpoint whose geometry does not match the manifest raises
`InvalidProfileError` (or the adapter's version error) and leaves the previously
active state untouched, so a failed migration never corrupts a profile
(`test_incompatible_profile_dimension_fails_without_corruption`).

## Rollback

Because migration writes a new profile directory, rollback is deletion of the
new profile; the original directory is never modified. For in-place experiments,
snapshot first (`manager.save_droid(...)`, `engine.snapshot(...)`) and restore
the snapshot, which carries the original manifest and adapter state.

## State versions

`DELTA_STATE_VERSION = 1` and `SPARSE_STATE_VERSION = 1`. Loading a newer state
version than the running code is rejected rather than partially applied. Old
`associative_rtu` checkpoints (no version field) load unchanged; see
`test_legacy_profile_still_loads_associative_adapter`.

## Measured migration cost

Not measured on real checkpoints. The evidence ledger records this as a
miss rather than claiming a migration benchmark.
