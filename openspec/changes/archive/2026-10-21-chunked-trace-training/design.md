# Design

## Context

See `proposal.md` for motivation. The technical constraints that shape this design
are all in `src/model/rtu.py:366` (`Model._forward`), the per-token training step:

1. The residual `x` is reset to `enc` at every token, so each token's graph is
   local and the per-token backward never walks a chain across time.
2. The three trace buffers are plain buffers with `requires_grad=False`. They are
   advanced by hand after the backward pass: `states` takes the new state,
   `decaytrace` takes `d ⊙ g_{t-1} + d ⊙ (1-d) ⊙ s_{t-1}`, and `embedtrace` takes
   `d ⊙ u_{t-1} + onehot(c_t)`.
3. The gradient hook injects the state gradient into the parameters by hand: the
   embedding weight gradient is **incremented** by `dlds_t ⊗ (d ⊙ u_{t-1})`, and
   the decay gradient is **replaced** by `dlds_t ⊙ g_t`, where `dlds_t` is the
   gradient with respect to the dummy carrier at token `t`.

Points 2 and 3 are what make the model learn beyond the truncated
back-propagation window, and they are what makes training a Python loop over
bytes.

## Goals / Non-Goals

**Goals:**

- One training chunk processed without a Python loop over its tokens.
- The chunked path's losses, accumulated gradients, and buffer values match the
  per-token path exactly enough to trust it as the same model.
- A run that can be resumed, and a report that says what was and was not
  equivalent.

**Non-Goals:**

- Byte-identical parameter trajectories between the two paths. They cannot be
  identical when one takes an optimizer step per byte and the other per chunk.
- Any change to the recurrence, the decay parameterization, the loss terms, or
  the online-learning semantics.
- A GPU kernel.

## Decisions

### D1. The chunked path is a forward planner plus a closed-form trace update

`Model.forward_chunk` already produces the per-layer state trajectory with the
associative scan. The training chunk reuses that trajectory and adds the vectorized
trace advance and hook accumulation. The per-token path is untouched.

### D2. The closed forms

With `d ∈ (0,1)^D` the per-dimension decay, constant across a chunk because the
parameter does not move mid-chunk, and the chunk's tokens `t = 1..L`:

```
s_t = d ⊙ s_{t-1} + e_t                       (state, from the scan)
u_t = d ⊙ u_{t-1} + o_t,   o_t[v,d] = [v == c_t]     (embedtrace)
g_t = d ⊙ g_{t-1} + d ⊙ (1-d) ⊙ s_{t-1}             (decaytrace)
```

Chunk-final values:

```
s_L = d^L ⊙ s_0 + Σ_t d^{L-t} ⊙ e_t
u_L = d^L ⊙ u_0 + Σ_t d^{L-t} ⊙ o_t
g_L = d^L ⊙ g_0 + Σ_t d^{L-t} ⊙ (d ⊙ (1-d)) ⊙ s_{t-1}
```

The state term is the scan's own output. The `embedtrace` term is a scatter: for
each byte value `v` and dimension `d` it is `Σ_{t : c_t = v} d^{L-t}`, which is one
`index_add_` of an `(L, D)` weight matrix into a `(V, D)` zero matrix — `L·D`
work, not `L·V·D`. The `decaytrace` term is a weighted sum of the scanned states,
`L·D` work. The chunk-final weights `d^{L-t}` are formed from a cumulative sum of
`log d`, so a decay of 0.999 over 512 steps is computed, not truncated.

### D3. One reverse scan supplies both hook contributions

The per-token hook needs, for each `t`:

```
embedding contribution:  dlds_t ⊗ (d ⊙ u_{t-1})
decay contribution:       dlds_t ⊙ g_t
```

Substituting the closed forms for `u_{t-1}` and `g_t` and swapping the sums gives
two terms: one that depends only on the buffer's initial value, and one that
depends on the bytes. Define

```
A[d] = Σ_t dlds_t[d] · d^t[d]                       (a D-vector)
T_s[d] = Σ_{t > s} dlds_t[d] · d^{t-s}[d]            (an L-by-D matrix)
```

Then

```
embedding contribution = A ⊙ u_0  +  scatter_{s}(T_s into row c_s)
decay contribution     = A ⊙ g_0  +  Σ_s T_s ⊙ (d ⊙ (1-d)) ⊙ s_{s-1}
```

`T` is a suffix sum with decay weights, which is a prefix scan of the reversed
sequence: running the same associative scan on `dlds` reversed with the same
decays yields `H_j = Σ_{t >= L-j} d^{t-(L-j)} ⊙ dlds_t`, and `T_s = H_{L-s} - dlds_s`.
One extra scan, `L·D` work per layer, and both hook contributions fall out.

The embedding contribution is a scatter into a `(V, D)` matrix, exactly like the
`embedtrace` update, so the two share one implementation.

### D4. The chunk loss is the sum of the per-token losses

Each token's loss is computed with the planned state detached, so each token's
graph is local and the gradient with respect to that token's dummy is that token's
own `dL_t/dstate_t` — the same quantity the per-token path reads from
`dummies[i].grad`. The chunk's loss is the **sum** of the per-token losses, so
the accumulated gradient equals the per-token path's accumulated gradient over the
same chunk.

The batched loss terms are the per-token terms evaluated per token: the variance
term is taken across the dimension axis per token and averaged over tokens; the
cross-entropy term is the mean over tokens of the per-token cross-entropy; the
prediction and stop terms are the means of their per-token values.

### D5. The equivalence is asserted on the accumulated gradient, before scaling

The reference oracle is the per-token path with the optimizer step replaced by
gradient accumulation: each token's gradient is captured after its backward and
summed. The chunked path's accumulated gradient is compared against that sum. The
`1/L` scaling of D6 is applied afterwards, to the value handed to the optimizer,
so the comparison is between like and like.

### D6. One optimizer step per chunk, with a `1/L` gradient scale and a byte-indexed schedule

The chunked path takes one optimizer step per chunk. A gradient summed over `L`
bytes is `L` times a single byte's gradient, so the accumulated gradient is
divided by `L` before the step; the update magnitude stays on the per-byte scale.
The learning-rate schedule in `Model.lr_at_step` is re-expressed as a function of
the number of bytes consumed, and both paths evaluate it from that function: the
per-token path at every byte, the chunked path once per chunk at the bytes
consumed at the chunk boundary.

**Alternatives considered.** Taking `L` optimizer steps inside the chunk was
rejected: it is the per-byte loop this change exists to remove. Not scaling the
gradient was rejected: the step would be `L` times too large. Changing the
schedule to be chunk-indexed was rejected: it would silently change the meaning
of the existing `bound` parameter and make the two paths incomparable.

### D7. The equivalence boundary is stated in the output

The chunked path is not parameter-identical to the per-token path, and every
report says so: the loss, the gradients, and the buffers are equivalent; the
parameter trajectory is not. A run whose equivalence has not been verified on the
target environment does not claim it.

### D8. The corpus is prepared from documents, with a hash-ordered split

Documents are ordered by the hash of their identifier and their seed, not by
filesystem order, so the corpus is reproducible across machines. The split is
document-level: the held-out fraction is selected by the hash of each document
identifier, so a document is entirely in one side and the split does not depend on
how the corpus was assembled.

### D9. Evaluation data is excluded and the exclusion is checked

The held-out split, the grammar probe file, and any reserved benchmark corpus are
excluded from the training bytes. Before the run starts, the harness checks the
held-out documents against the training set and fails the contamination check
naming the overlapping document if any appears in both.

### D10. Checkpoints are atomic and carry the corpus position

A checkpoint holds the weights, the optimizer state, the step count in bytes, and
the corpus position. It is written to a temporary file and moved into place, so an
interrupted write leaves the previous checkpoint intact. A resumed run continues
from the recorded position and does not reprocess earlier bytes.

### D11. The run report reuses the benchmark report format

The scale run's report uses the same shape as the benchmark harness's report —
commit hash, command, dependency versions, configuration including seeds, hardware,
measurements, outcome — so a training result and a performance result can be read
the same way. That format is defined by the `edge-performance-and-grammar-evals`
change; this change depends on it for the report shape and for the held-out
perplexity measurement, and the chunked pass and its equivalence tests do not
depend on it.

### D12. The edge-scale configuration is `dim=1024`, `layers=24`

`Model.count()` gives `vocab·dim + layers·(dim² + 3·dim) + vocab·dim + vocab + dim
+ 1`. At `dim=1024`, `layers=24`, `vocab=256` that is about 25.9 million
parameters, which is the size the project claims to be about. At `dim=384`,
`layers=8` it is about 1.4 million, which is the size of the profile checkpoints
that exist today.

## Risks / Trade-offs

- **[The closed-form trace update diverges from the loop]** → The chunked update
  is tested against the per-token path at short chunks, chunks longer than the
  chunk size, batches, all-zero and all-one byte sequences, and the real decay
  initialization, at a stated tolerance. The oracle is the loop, never an
  analytic ideal.
- **[A chunked run is not comparable to a per-token run]** → The report states
  the equivalence boundary, the step count is reported in bytes, and the
  equivalence check is a prerequisite for claiming the chunked path in a report.
- **[The `1/L` scale interacts with the optimizer's bias correction]** → The
  custom `CustomAdamW` uses `bias_correction=False`, so the scale is the only
  factor between the summed and the scaled gradient. This is checked explicitly
  against the reference in the equivalence test rather than assumed.
- **[Decay powers underflow over long chunks]** → The weights come from a
  cumulative log-decay, and the result is checked against the per-token
  reference. An underflowed weight is a correct weight when the decay is small
  enough that the contribution is genuinely negligible; the test uses a real
  decay, not a toy one, so a wrong exponent is caught.
- **[The corpus is self-graded]** → The corpus identity and the excluded sets are
  recorded with every result, so the number can be re-derived or challenged. A
  public corpus is a data decision, not a format decision, and the harness accepts
  an arbitrary corpus.
- **[A multi-day run is interrupted]** → The run is resumable, writes a blocked
  outcome with the bytes consumed when it stops, and reports the wall clock
  against the target rather than hiding it.

## Migration Plan

1. The chunked training path is additive and selected by an explicit flag. The
   default stays the per-token path.
2. `lr_at_step` keeps its current meaning for the per-token path; the byte-indexed
   form is used by both and asserted to agree at the same byte count.
3. No checkpoint format changes: the chunked path writes the same
   `rtu.safetensors` shape, and the corpus position lives in the run's own state,
   not in the checkpoint.
4. Rollback is removing the flag. No data migration.

## Open Questions

- Should the chunked path become the default once the equivalence test is green,
  or stay behind the flag permanently? Deferred: the existing parallel-scan
  requirement already keeps the flag off until the equivalence test passes, and
  flipping a default belongs in its own change.
- Should a run be checkpointed per chunk or per interval? Deferred: an interval
  is simpler and the chunk size is a parameter; the choice is a configuration
  detail, not a behavioral one.
