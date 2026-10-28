# Forgetting that never destroys

## Utility and decay

Accessibility follows the power law

```
A_i(t) = S_base * (1 + alpha * (t - t_last))^(-beta_i)
```

with `beta_i` stored per fact (`decay_beta`) and decreased by
`(1 - delta_reinforce)` on every successful retrieval, so frequently retrieved
facts decay more slowly (`KnowledgeStore.reinforce`).

Retirement uses utility:

```
utility(i) = access_count_i * exp(-(t - t_last_i) / tau_half) + lambda * novelty_i
```

`tau_half` defaults to 7 days, `lambda` to 1.0 (`recamem.forgetting`).

## Capacity and retirement

`enforce_capacity(store, K_max)` moves the lowest-utility facts to the
`retired` tier until `active_count <= K_max`. A retired fact keeps its row,
text, source, provenance and audit trail; it is simply excluded from ordinary
recall and from `active_count`.

`consolidate_then_forget(engine, ...)` retires a fact only when the recall probe
(by default the plastic channel's `recall_score` for that fact) is at or above
`probe_threshold`. Facts below it stay active and the report names the reason.

## No hard delete

Nothing in `recamem/forgetting.py` issues a `DELETE`. The `retired` tier is
reversible:

```python
engine.knowledge.retire_fact(fact_id, reason="capacity")
engine.knowledge.restore_fact(fact_id)     # same id, source and history
```

`tests/test_forgetting_tiers.py` asserts that a full forgetting cycle executes
no `DELETE FROM facts` statement and that the row count is unchanged.
