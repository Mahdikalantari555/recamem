# Proposal

## Why

`openspec/specs/training/parallel-scan/spec.md` specifies a chunked recurrent forward:
compute the linear recurrence over a sequence chunk with an associative scan instead of
a per-token loop, matching the sequential result within `rtol=1e-4`, with the batch
dimension parallel. No code implements it today. The only recurrence path is
`Layer.forward` under `Model.forward`, which advances one token at a time in Python
(`src/model/rtu.py:136-142`, `:236-285`), and the training loop in `main.py` drives it
token by token.

The absence is now on the critical path. Plan 10's E2 gate trains a byte-RTU checkpoint
on the project corpus (change `droid-rtu-fusion-completion`, tasks 1.21–1.22, 1.29).
That run is a Python loop over every byte of the corpus; the same loop shrinks to
`O(log L)` vectorized composition levels per chunk with this change. Landing the scan
first turns the gate into a run of minutes.

## What Changes

- Add an associative prefix scan for `h_t = decay ⊙ h_{t-1} + e_t` over a sequence
  chunk, using the closed-form composition of the affine recurrence
  (4.1, 4.2).
- Keep the batch dimension parallel: all batch elements are scanned in the same tensor
  operation (4.3).
- Keep the sequential loop as the named reference implementation and use it only in
  tests (4.4, 4.5).
- Add an equivalence test against the sequential loop at `rtol=1e-4` (4.4).
- Measure the wall-clock factor of the chunked path against the sequential path on a
  fixed corpus, report the factor, and assert at least `10x` at sequence length 512
  (4.6, 4.7).
- Wire the chunked path into the training loop behind a flag that defaults off until
  the equivalence test passes (4.8, 4.9).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `training/parallel-scan` — MODIFIED "Chunked Recurrent Forward" (chunk-boundary
  continuity, batch parallelism, measured speed factor, and the reference path are made
  explicit).

## Impact

- **Code areas:** new `src/model/parallel_scan.py`; `src/model/rtu.py` (a chunked
  state-trajectory helper that calls the scan); `main.py` (training-loop flag).
- **Tests:** new `tests/test_parallel_scan.py`; the equivalence test is the gate.
- **Compatibility:** the default training path is unchanged while the flag is off. The
  sequential implementation is retained, not deleted, so existing numerical tests keep
  their reference.
- **Non-goals:** no change to `Model.forward`'s signature, no change to the custom
  optimizer or trace-buffer commit path, no new training loop, no GPU-specific kernel.
- **Ordering:** this change lands after `correctness-hardening` and before
  `droid-rtu-fusion-completion`, whose E2 gate benefits from the speed-up.

## Risks

- A closed-form scan in float32 can drift when `decay` is near 1 and the chunk is long.
  The equivalence test uses the real `decay` initialization and a long chunk, not a
  toy decay of 0.5.
- A speed target is hardware-dependent. The factor is measured on a fixed corpus, with
  the hardware and the tensor shapes recorded next to the number; the assertion is
  skipped rather than faked when the environment cannot reproduce it.
- Wiring the scan into training changes the gradient path if done carelessly. This
  change scans the state trajectory only and leaves the existing autograd wiring for
  outputs and trace buffers intact.
