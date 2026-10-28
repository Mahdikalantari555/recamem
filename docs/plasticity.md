# Two-tier plasticity

## FAST tier

A teaching call embeds the proposition and writes it into the plastic adapter
without building a backward graph. The write uses the learning rate from the
schedule:

```
effective_lr = base_lr / sqrt(1 + step_count / 20)
```

`recamem.plasticity.fast_write(engine, x, base_lr=..., step_count=...)` returns
`{"requested_lr", "applied_lr", "step", "grad_tracking", "timestamp"}` and
appends the record to `engine.write_log`. The applied rate is the one that
changed stored state: `DeltaPlasticAdapter.update_associative_memory` scales the
write gate, and the persisted `h_trace` absorbs the same gain, so two writes at
different step counts differ by the ratio of their effective rates.

Parameters:

| name | default | meaning |
| ---- | ------- | ------- |
| `base_lr` | `0.05` | requested learning rate (engine `base_lr`) |
| `schedule` | inverse-sqrt | `effective_lr` above |
| `scale` | `20.0` | denominator scale in the schedule |

FAST writes never attach a graph: `grad_tracking` is `False`, meaning no tensor
produced by the write has a `grad_fn` (the adapter's `nn.Parameter`s keep their
own `requires_grad`, which is not a graph).

## SLOW tier

`recamem.plasticity.SlowTier` holds bounded meta-parameters and replays stored
experience through them:

* `replay_step(items)` takes one optimiser step on a MSE objective plus the
  memory-aware-synapse penalty.
* `gated_update(items, probe)` snapshots the parameters, runs the step, calls
  `probe()` (a per-domain held-out recall map), and commits only when no domain
  drops more than `tolerance` (default `0.05`). Otherwise it restores the
  snapshot and records `{"accepted": False, "rejected_domain": ..., "reason": ...}`.
* `MemoryAwareSynapses` keeps per-parameter importance and anchors; parameters
  with higher importance move less under the same gradient.

Every commit/rejection is appended to `SlowTier.train_log`.

## Schema version

Plastic state carries `schema_version` (`PLASTIC_SCHEMA_VERSION = 2`). A state
declaring a newer version raises `UnsupportedPlasticSchema` naming both
versions; an older state is upgraded by `upgrade_state`, which preserves every
stored value and adds the schedule/meta/log fields.
