# Tasks

## 1. Chunked trace-buffer update

- [ ] 1.1 Implement the chunk-final `embedtrace` update as a scatter of the
  decay-weighted byte counts into the `(vocab, dim)` buffer: form the chunk-final
  weights from a cumulative log-decay and accumulate them into the rows of the
  chunk's byte values with one indexed addition. Verify in
  `tests/test_chunked_training.py` that the result matches the per-token path for
  a chunk of random bytes within the documented tolerance.
- [ ] 1.2 Implement the chunk-final `decaytrace` update as the closed-form weighted
  sum of the scanned states. Verify in `tests/test_chunked_training.py` that it
  matches the per-token path within the documented tolerance, including a chunk
  longer than the chunk size and a batch.
- [ ] 1.3 Implement the chunk-final `states` update from the scan output and apply
  all three buffers once per chunk. Verify in `tests/test_chunked_training.py` that
  the three buffers after a chunk equal the per-token path's values within the
  documented tolerance for the real decay initialization, an all-zero byte
  sequence, and an all-one byte sequence.
- [ ] 1.4 Assert that the chunked update does not loop over tokens in Python. Add a
  counter to the chunked path and assert in `tests/test_chunked_training.py` that
  the count does not grow with the chunk length, and that the per-token path's
  count does.

## 2. Vectorized gradient-hook accumulation

- [ ] 2.1 Implement the reverse decay-weighted scan of the per-token state
  gradients, and derive the suffix-sum matrix that both hook contributions need.
  Verify in `tests/test_chunked_training.py` that the derived suffix sums match a
  direct per-token computation within the documented tolerance.
- [ ] 2.2 Implement the accumulated embedding-weight gradient contribution: the
  buffer-initial term and the scattered byte term. Verify in
  `tests/test_chunked_training.py` that the applied contribution equals the sum
  of the per-token contributions within the documented tolerance.
- [ ] 2.3 Implement the accumulated decay gradient as the replacement value the
  per-token path produces, summed over the chunk. Verify in
  `tests/test_chunked_training.py` that it equals the sum of the per-token values
  within the documented tolerance, and that it replaces rather than adds to the
  autograd gradient as the per-token path does.

## 3. Chunked training pass

- [ ] 3.1 Implement the batched per-token loss: the variance term across the
  dimension axis per token, the cross-entropy, the prediction term, and the stop
  term, each reduced over tokens. Verify in `tests/test_chunked_training.py` that
  each batched term equals the per-token term for the same token.
- [ ] 3.2 Implement the chunked training pass: the scan for the state trajectory,
  the per-token loss with detached planned states, the per-token dummy carriers,
  one backward over the summed chunk loss, and the vectorized trace and hook
  updates. Verify in `tests/test_chunked_training.py` that the per-token loss
  values equal the per-token path's values within the documented tolerance.
- [ ] 3.3 Implement the reference oracle: the per-token path with the optimizer
  step replaced by gradient accumulation, capturing the gradient after each
  token's backward and summing it. Verify in `tests/test_chunked_training.py`
  that the oracle's accumulated gradient equals the chunked path's accumulated
  gradient within the documented tolerance.
- [ ] 3.4 Apply the `1/L` gradient scale before the optimizer step, and re-express
  the learning-rate schedule as a function of the bytes consumed so both paths
  read the same schedule. Verify in `tests/test_chunked_training.py` that the two
  paths agree on the learning rate at the same byte count and that the optimizer
  receives the accumulated gradient divided by the chunk length.
- [ ] 3.5 Take exactly one optimizer step per chunk and advance the step counter by
  the chunk length in bytes. Verify in `tests/test_chunked_training.py` that the
  optimizer step count is one per chunk and that the byte counter advanced by the
  chunk length.

## 4. Equivalence test suite

- [ ] 4.1 Cover the chunked path against the per-token oracle for a short chunk, a
  chunk longer than the chunk size, a batch, all-zero bytes, all-one bytes, and
  the real decay initialization. Verify in `tests/test_chunked_training.py` that
  every case passes at the documented tolerance.
- [ ] 4.2 Assert that the per-token path still exists and is still the oracle.
  Verify in `tests/test_chunked_training.py` that the reference implementation is
  referenced by the equivalence test and is not the chunked path itself.
- [ ] 4.3 Record the equivalence boundary in the chunked run's output: the loss,
  the gradients, and the buffers are equivalent, and the parameter trajectory is
  not. Verify in `tests/test_chunked_training.py` that the output names both the
  equivalent and the non-equivalent quantities.
- [ ] 4.4 Assert that an unverified chunked path does not claim the equivalence.
  Verify in `tests/test_chunked_training.py` that a run whose equivalence check
  was skipped names the missing verification instead of claiming the equivalence.

## 5. Corpus preparation

- [ ] 5.1 Implement document-level corpus preparation with a deterministic
  document order derived from the document identifier hash and the seed, not from
  filesystem order. Verify in a new test that preparing the same corpus twice
  with the same seed produces byte-identical output and that reordering the files
  on disk does not change the result.
- [ ] 5.2 Implement the document-level split, selected per document by the hash of
  its identifier. Verify in the same test that no document appears on both sides
  and that the split is reproducible from the seed alone.
- [ ] 5.3 Implement the exclusion of the held-out split, the grammar probe file,
  and any reserved benchmark corpus from the training bytes, and the contamination
  check that fails when an excluded document appears in the training set. Verify in
  the same test that the contamination check fails and names the overlapping
  document.

## 6. Resumable run

- [ ] 6.1 Implement the run state: the corpus position, the step count in bytes,
  the seeds, and the configuration. Verify in a new test that a run interrupted
  after a chunk resumes from the recorded position and does not reprocess the
  bytes before it.
- [ ] 6.2 Implement atomic checkpoint writing that carries the weights, the
  optimizer state, the step count, and the corpus position. Verify in the same
  test that an interrupted write leaves the previous checkpoint intact and
  readable.

## 7. Scale run script and reporting

- [ ] 7.1 Implement `scripts/train_edge_scale.py` with the corpus, the model
  configuration, the chunk size, the chunked flag, the seed, the checkpoint
  interval, and the report path as arguments. Verify by running it with `--help`
  that every documented flag is accepted and that it exits 0.
- [ ] 7.2 Orchestrate a scale run: prepare the corpus, train with the chunked
  path, checkpoint on the interval, and write the report in the benchmark report
  format. Verify by running a short run on a small corpus that the report
  contains the bytes consumed, the wall clock, the final training loss, the
  held-out perplexity, the parameter count, the hardware, and the outcome, and
  that the process exits 0.
- [ ] 7.3 Implement the acceptance check: passed only when the byte target is
  reached, the parameter count matches the configuration, the final logged loss
  is below the initial logged loss, the held-out perplexity is finite and was
  computed on the excluded split, and the run completed on the documented
  hardware class; otherwise failed or blocked with the criterion named. Verify in
  the new test that a run missing a criterion reports the criterion and is not
  reported as passed.
- [ ] 7.4 Write a blocked report when the run is interrupted, carrying the bytes
  consumed and the wall clock. Verify in the new test that an interrupted run
  still leaves a report with the blocked outcome and the bytes consumed.
- [ ] 7.5 Add the equivalence boundary to the run report, and omit the equivalence
  claim when the equivalence check was not run on the target environment. Verify in
  the new test that the report names the equivalent and the non-equivalent
  quantities and names the missing verification when the check was skipped.

## 8. Documentation

- [ ] 8.1 Document the chunked training path in `docs/rtu.md`: the closed forms
  for the three buffers, the suffix-sum derivation of the hook contributions, the
  optimizer-step granularity, the `1/L` scale, and the equivalence boundary.
  Verify that every formula in the document matches the implementation.
- [ ] 8.2 Document the scale run in `docs/benchmarks.md` or a new
  `docs/edge-pretraining.md`: the corpus preparation, the split rule, the
  excluded sets, the resumable checkpoints, the acceptance criteria, and the
  command. Verify that every command in the document runs as written.
