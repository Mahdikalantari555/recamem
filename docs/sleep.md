# Sleep consolidation

`DroidEngine.sleep()` runs one bounded consolidation cycle and returns a
structured report. It never claims work that did not happen.

```bash
recamem sleep --name agent-notes        # CLI
/sleep                                  # inside `recamem chat`
```

## Report

| key | meaning |
| --- | ------- |
| `status` | `completed`, `no_op_empty`, `cancelled` or `partial_failure` |
| `replay` | proposed replay steps, coverage, duplicates, warnings |
| `fact_pruning` | facts retired by the forgetting path, counts before/after |
| `parameter_pruning` | MAS parameter selection (disabled by default) |
| `updates` | `{proposed, committed, rejected, rejected_domain, reason}` |
| `warnings`, `errors` | per-phase problems, named |
| `duration_ms`, `cycles`, `learning_rate` | timing, persisted cycle count, schedule |

An empty profile returns `status=no_op_empty` with zero counts and no exception.
A replay that produces steps but does not commit reports the rejection and the
domain (`updates.rejected_domain`) instead of a success.

## Single writer

Teaching and sleep take `ProfileWriterLock` — an exclusive `flock` on
`<profile>/.recamem-writer.lock`, acquired non-blocking. A second writer gets
`ProfileBusyError` naming the profile and the holder's pid; it does not wait,
and its refused call leaves stored state unchanged. The lock is reentrant within
one thread, so sleep can replay through `teach` without deadlocking.

## Two tiers

Replay parameter updates go through the SLOW tier's rollback gate
(`recamem/plasticity.py`): commit only when a held-out probe shows no domain
regressing beyond tolerance, otherwise restore the previous meta-parameters and
record the reason.
