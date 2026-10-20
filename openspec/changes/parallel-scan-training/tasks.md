# Tasks

Task IDs match the plan slice. Group headers are descriptive so the validator does not
read them as a second numbering scheme.

## Scan implementation

- [ ] 4.1 Implement the associative prefix scan for `h_t = decay ⊙ h_{t-1} + e_t` over a chunk of the sequence. Add `src/model/parallel_scan.py` with `associative_scan(decays, inputs, h0=None, chunk=128)` accepting `(B, L, D)` tensors and returning `(B, L, D)` states plus the final state.
- [ ] 4.2 Use a closed-form associative operator. Do not loop over tokens in Python. Compose `(a1, b1) ∘ (a2, b2) = (a1 ⊙ a2, a2 ⊙ b1 + b2)` with `ceil(log2 L)` vectorized Hillis–Steele levels; assert by inspection that the only Python loop is over levels and chunks.
- [ ] 4.3 Keep the batch dimension parallel. Do not loop over the batch. All batch elements share the same tensor operations; add a `B = 3` case to the tests so a batch-shape bug cannot hide.

## Reference and equivalence

- [ ] 4.4 Add a test that compares the chunked result with the sequential loop. Use `rtol=1e-4`. Cover a random decay from the layer initialization (`spread=32`), a sequence longer than the chunk so a boundary is crossed, `B > 1`, an all-zero input, and an all-one input.
- [ ] 4.5 Keep the sequential loop as the reference. Keep it for tests. Commit `sequential_scan(decays, inputs, h0=None)` in `src/model/parallel_scan.py`, documented as the test oracle and used by no production path.

## Measurement

- [ ] 4.6 Measure the wall clock of the chunked path against the sequential path on a fixed corpus. Report the factor. Add `scripts/bench_parallel_scan.py` with a deterministic corpus slice, `dim=384`, `L ∈ {128, 512, 1024}`, three runs per path, and print `path,L,seconds,factor`. Record the numbers and the hardware in `plans/10_droid_rtu_fusion.md`.
- [ ] 4.7 Assert that the factor is at least 10 at sequence length 512. Put the assertion in `tests/test_parallel_scan.py`, skip with an explicit message when the run-to-run coefficient of variation exceeds 20%, and never lower the threshold to make a machine pass.

## Training integration

- [ ] 4.8 Add the chunked path to the training loop. Select it with a flag. Add `--parallel-scan` to `main.py`, default off, and route the state-trajectory computation for a chunk through `associative_scan` while leaving output and trace-buffer code unchanged.
- [ ] 4.9 Keep the flag default off until the equivalence test passes. Assert the default value in `tests/test_parallel_scan.py`; flip the default only in a separate change once 4.4 is green on the target environment.
