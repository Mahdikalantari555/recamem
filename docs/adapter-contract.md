# Plastic adapter extension contract

`PlasticAdapter` is an abstract `torch.nn.Module`. Call `super().__init__()` before
registering parameters or buffers. Implement update, retrieve, predict_next,
compute_surprise, state_dict and load_state_dict. A custom serializer can call
`torch.nn.Module.state_dict(self)` explicitly; do not call an abstract stub via
`super().state_dict()`.

Optional defaults:

- `consolidate()` returns `status: unsupported, updated: false`; it does not claim
  an optimization occurred.
- `get_health()` returns `status: unsupported, metrics: {}`; unavailable is not zero.
- `create_snapshot()` independently clones tensors and deep-copies metadata.
- `restore_snapshot(snapshot)` supplies an independent copy even to loaders that
  retain their input. Engine/manager in-memory snapshot and branch paths likewise
  own their plastic state. Full durable database rollback is separate unfinished
  work (post-merge tasks 4.4–4.5); these helpers do not certify that behavior.

## Scheduled updates

Legacy adapters need no new signature. Teaching calls their historical
`update_associative_memory(x, surprise_factor=...)` (or `update(x)`) exactly once.
The result retains `effective_lr` as the requested rate and adds `requested_lr`,
`applied_lrs` and `scheduling_supported`. `applied_lr` is null when unsupported or
not uniform across updates; unsupported adapters must not claim a rate was applied.

An adapter explicitly opts in by implementing:

```python
def update_scheduled(self, x, *, surprise_factor, effective_lr, base_lr):
    output = ...  # perform exactly one update with the requested rate
    return output, effective_lr  # report the rate actually applied
```

The associative RTU implements this capability. Original-signature and legacy
RTU adapters retain their original update path. No execution-based signature
probing or retry is permitted: an internal TypeError can occur after mutation.

`PlasticUpdateError` wraps a failed update with its original cause. Teaching
stops, inserts no fact for that failed proposition and does not log success.
The adapter may already be partially mutated and earlier propositions may have
succeeded; this is **not** transactional teaching. Inspect the error and explicitly
restore a prior snapshot before retrying if necessary. Invalid requested rates
are rejected before the update. Successful teaching never implies successful
learning-rate scheduling on an unsupported adapter.
