# Proposal

## Why

`training/parallel-scan` made the *forward* pass fast: the recurrence is now
evaluated with an associative scan instead of a Python loop over tokens, and the
chunked path is specified to be at least 10x the sequential path at sequence length
512. Training, however, is still a Python loop over every byte of the corpus, and
it cannot stop being one. The reason is in `src/model/rtu.py:366`: after each
token's backward pass, three trace buffers are advanced (`states`, `decaytrace`,
`embedtrace`) and the gradient-hook contribution is injected into the embedding
weights and the decay parameters by hand. Those updates are per-token, in Python,
and they are the mechanism that carries gradient beyond the truncated
back-propagation window — the thing that makes online learning work.

So the scan gave a fast forward and left the slow part exactly where it was. At
roughly 100-200 bytes per second, a 25-million-byte corpus is a two-day run, and
the headline claim of this project — a small model, trained on a CPU, on data
nobody has a GPU for — cannot be demonstrated at that rate. This change vectorizes
the trace updates and the gradient-hook injection over a chunk, and then runs the
corpus.

## What Changes

- Vectorize the per-chunk trace-buffer advance. The three buffers are affine
  recurrences of the same family as the state, so their chunk-final values follow
  in closed form; the decay-weighted byte counts in `embedtrace` are computed with
  a scatter over the chunk's byte values rather than a loop.
- Vectorize the gradient-hook injection. The per-token state gradient is retained
  for the whole chunk, one reverse decay-weighted scan recovers the suffix sums
  that both hook contributions need, and the two contributions are accumulated
  once per chunk.
- Define the equivalence precisely and test it: the chunked path produces the same
  per-token losses, the same accumulated gradients, and the same buffer values as
  the per-token path, and the per-token path is retained as the named reference.
- Be explicit about what is *not* equivalent: the chunked path takes one optimizer
  step per chunk instead of one per byte, so its parameter trajectory differs. The
  accumulated gradient is scaled by the inverse of the chunk length to keep the
  update magnitude on the per-byte scale, and the learning-rate schedule becomes a
  function of bytes consumed so both paths read the same schedule.
- Add a corpus and run protocol for an edge-scale pretraining run: a deterministic
  document-level split, explicit exclusion of the held-out split and the
  evaluation probe data from the training bytes, resumable checkpoints, and a
  report that states the bytes consumed, the wall clock, the losses, the
  parameter count, and the hardware.
- Report the outcome honestly: passed, failed, or blocked, with the criterion
  that was not met named, and never a number for a measurement that was not taken.

## Capabilities

### New Capabilities

- `training/edge-pretraining`: the corpus preparation, split, exclusion, resumability,
  reporting, and acceptance criteria for an edge-scale pretraining run.

### Modified Capabilities

- `training/parallel-scan`: adds the chunked trace-buffer training pass, the
  chunked loss and gradient equivalence, the optimizer-step granularity, and the
  rule that a chunked run reports which quantities are equivalent and which are
  not.

## Impact

- **Code areas:** `src/model/rtu.py` (a chunked training pass beside the existing
  per-token path, plus the vectorized trace update and hook accumulation);
  `src/model/parallel_scan.py` (the reverse decay-weighted scan used for the
  suffix sums); new `scripts/train_edge_scale.py` (corpus preparation and the
  runnable protocol); `scripts/train_byte_rtu.py` (unchanged default behaviour).
- **Tests:** new `tests/test_chunked_training.py`, extending the existing
  `tests/test_parallel_scan.py` with the chunked-training equivalence cases.
- **Dependencies:** none.
- **Compatibility:** the per-token training path is unchanged and remains the
  default. The chunked path is selected by an explicit flag, as the existing
  parallel-scan integration already is.
- **Ordering:** this change benefits from the measurement harness and the held-out
  split discipline of the `edge-performance-and-grammar-evals` change, and it
  should land after it. The chunked pass and its equivalence tests do not depend
  on that harness and can land independently.
- **Non-goals:** no change to the recurrence, the decay parameterization, the
  loss terms, the recall path, or the online-learning semantics. No GPU kernel and
  no attempt to make the chunked path produce byte-identical parameters.

## Risks

- **The chunked path is not the same optimizer trajectory as the per-token path.**
  One step per chunk is a different training regime, and a chunked run is not
  directly comparable to a per-token run at the same step count. Mitigation: the
  specification states the equivalence precisely and separately from the
  trajectory, the gradient is scaled to the per-byte magnitude, and every report
  says which quantities are equivalent and which are not.
- **A closed-form trace update is wrong in a way a loop is not.** Mitigation: the
  chunked update is tested against the per-token path for short chunks, chunks
  longer than the chunk size, batches, all-zero and all-one byte sequences, and
  the real decay initialization, at a stated tolerance.
- **The decay powers underflow over long chunks.** Mitigation: the chunk-final
  weights are computed from a cumulative log-decay and the update is verified
  against the per-token reference rather than against an analytic ideal, so a
  decay of 0.999 over 512 steps is caught by the test rather than by inspection.
- **A 25-million-byte corpus on a CPU is a multi-day run.** Mitigation: the run is
  resumable, reports its bytes consumed and wall clock as it goes, writes a
  blocked outcome with the bytes consumed when it is interrupted, and the budget
  is stated as a target rather than a promise.
- **A self-graded corpus flatters the model.** Mitigation: the split rule and the
  excluded sets are recorded with the run, the perplexity is computed on the
  excluded split, and the report carries the corpus identity so the number can be
  re-derived or challenged.
