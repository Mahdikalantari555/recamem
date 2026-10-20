# Design

## Context

The RTU layer is a first-order linear recurrence with a per-dimension decay:

```
h_t = decay ⊙ h_{t-1} + e_t          # Layer.forward, src/model/rtu.py:136-142
```

`decay = sigmoid(layer.decay)` is a learned vector of length `dim`; `e_t` is the
encoder output for token `t`. `Model.forward` advances it one token at a time
(`src/model/rtu.py:236-285`) and the training loop in `main.py` calls it once per byte.
The existing spec (`openspec/specs/training/parallel-scan/spec.md`) already defines the
required behavior and tolerance; nothing implements it.

Scope note: the recurrence is linear in `h`, but the layer output
`x + silu(weights(norm(state)))` and the decoder are per-token nonlinear. Only the state
trajectory is scanned; the per-token nonlinearity is applied to the scanned states as a
single vectorized operation over `(B, L, D)`, which is elementwise and therefore still
has no token loop.

## Goals / Non-Goals

**Goals**

- One closed-form associative scan for `h_t = decay ⊙ h_{t-1} + e_t`.
- Batch-parallel execution with no Python loop over tokens or batch elements.
- Numerical equivalence with the sequential reference at `rtol=1e-4`.
- A measured, reported speed factor with a 10x assertion at `L=512`.
- A flag-gated training-loop integration that stays off until equivalence passes.

**Non-Goals**

- No new training loop, no optimizer change, no checkpoint format change.
- No gradient rewrite: the scan covers the forward state trajectory; the existing
  autograd path that computes `dummies`, trace buffers, and optimizer steps is kept.
- No GPU kernel, no `torch.compile`, no fused C++ extension.
- No change to `Model.forward(currb, nextb, end, frozen, ce_only)`.

## Decisions

### Decision 1 — Compose affine maps, do not iterate

Each step is the affine map `h ↦ a ⊙ h + b` with `a = decay` and `b = e_t`. Two
consecutive maps compose associatively:

```
(a1, b1) ∘ (a2, b2) = (a1 ⊙ a2, a2 ⊙ b1 + b2)
```

The inclusive prefix of this operator is the state trajectory. The scan is implemented
as a Hillis–Steele scan over `ceil(log2 L)` levels, each level a vectorized shift and
composition over the `(B, L, D)` tensors. The Python loop is over levels, not tokens, and
its length grows logarithmically with the sequence length.

Alternatives rejected:

- **Closed form with division** (`A_t = cumprod(a)`; `h_t = A_t ⊙ (h_0 + cumsum(b/A_t))`).
  Mathematically identical but divides by products of decays that underflow to zero for
  long sequences, producing `inf`/`NaN`. Rejected on numerical grounds.
- **`torch.cumsum` in log space.** Same underflow/overflow class of problem, plus a
  `log(0)` guard for exact zeros in `e_t`.
- **Blelloch scan.** Correct, but needs a padded power-of-two length and two passes;
  Hillis–Steele is simpler to verify against the reference and fast enough at the sizes
  this project trains on.

### Decision 2 — Chunking with a carried state

The sequence is processed in chunks of `C` tokens (default `C = 128`). Each chunk scans
with the incoming state as the initial condition: the first level adds `a_t ⊙ h_prev`
to `b_t`. Chunk boundaries are therefore exact, not approximate. A chunk of length 1
degenerates to the sequential step.

`h_prev` is carried per layer, per chunk, in the same dtype as the state buffer.

### Decision 3 — The sequential loop stays as the reference

`sequential_scan(decays, inputs, h0)` is a plain Python loop kept in the same module and
marked as the test reference. It is the oracle for the equivalence test and is used by
no production path. Deleting it would remove the only independent check on the scan.

### Decision 4 — Equivalence is asserted on states and outputs, not on wall clock

`tests/test_parallel_scan.py` compares `associative_scan` against `sequential_scan` at
`rtol=1e-4`, `atol=1e-6`, for:

- a random decay drawn from the layer's real initialization (`spread=32`),
- a chunk longer than the default `C` so at least one boundary is crossed,
- `B > 1` so a batch-shape bug cannot hide in a `B = 1` special case,
- an all-zero input (state must stay zero) and an all-one input.

Task 4.2 verifies the "no per-token loop" rule by reading the hot function and its call
graph: the only loops are over scan levels and chunk boundaries, and the sequential loop
lives in the separate, explicitly named reference function, so a later "optimization"
that reintroduces a token loop is visible in review.

### Decision 5 — The speed factor is measured and reported, never assumed

`scripts/bench_parallel_scan.py` builds a fixed corpus (a deterministic slice of bytes
from `README.md` + `plans/10_droid_rtu_fusion.md`), runs both paths at
`L ∈ {128, 512, 1024}` with `dim=384`, and prints `path, L, seconds, factor`. Task 4.7's
assertion (factor `>= 10` at `L=512`) lives in the test and skips with an explicit
message when the machine is too noisy to measure a stable baseline (coefficient of
variation of three runs above 20%). The measured numbers and the hardware are recorded
in the plan file at close, per the plan-10 acceptance rule.

### Decision 6 — Flag-gated training integration

`main.py` gains `--parallel-scan`, default **off**. When on, the training path computes
the state trajectory for a chunk with the scan and feeds the existing per-token output
and trace-buffer code from the scanned states. When off, the current path runs
unchanged. 4.9 keeps the default off until the equivalence test is green in the same
commit; flipping the default is a separate, deliberate change.

## Risks / Trade-offs

- **Float32 drift.** Composition multiplies `decay` values together; at `spread=32` the
  slowest half-life makes `decay` very close to 1, so long scans accumulate error. The
  tolerance is `1e-4` because the spec says so; if the test cannot hold it, the chunk
  length shrinks rather than the tolerance.
- **Memory.** Holding `(B, L, D)` pairs for the scan costs roughly `2 ×` the token
  tensor, which is fine at the sizes this project trains on but is a constraint worth
  recording before anyone raises `L`.
- **Benchmark flakiness.** Wall-clock assertions are noisy in shared CI. The skip rule
  above keeps the test honest instead of timing-dependent; the recorded factor in the
  plan file is the durable artifact.
- **Gradient coverage.** Because the scan covers the state trajectory only, the
  training path still relies on the existing autograd wiring. That is deliberate: the
  project's custom optimizer and trace-buffer semantics are contract-tested elsewhere
  and rewriting them is out of scope.
- **Flag rot.** A default-off flag can silently diverge. The equivalence test keeps the
  scan honest, and the flag state is asserted in `tests/test_parallel_scan.py` so a
  default flip is an explicit, reviewed line.
