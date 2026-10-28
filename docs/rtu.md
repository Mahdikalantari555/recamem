# Recurrent Trace Unit (RTU) Architecture & Memory Spec

## Overview

The Recurrent Trace Unit (RTU) is a recurrent memory cell designed for byte-level language modeling and streaming latent prediction without a conventional attention context window.

## Core State Equations

For layer $i$ at step $t$ given input representation $\text{enc}_t$ and residual vector $x_t$:

$$
\text{decay}_t = \sigma(\text{decay\_param}_i)
$$

$$
\text{state}_t = \text{decay}_t \odot \text{state}_{t-1} + \text{enc}_t + \text{dummy}_t
$$

$$
x_{t+1} = x_t + \text{SiLU}\Big(W_i \, \text{LayerNorm}(\text{state}_t)\Big)
$$

- $\text{decay\_param}$ is initialized across dimensions by spanning log half-lives across the layer spread.
- $\text{dummy}_t$ is an identity carrier allowing automatic differentiation to recover $d\mathcal{L} / d(\text{state}_t)$.

## Persistent Trace Buffers

Each layer maintains three persistent states:
1. `states` $(\text{dim})$: Hidden memory vector, updated via exponential moving average (EMA).
2. `decaytrace` $(\text{dim})$: Decayed state accumulator tracking state changes over time.
3. `embedtrace` $(\text{vocab\_size}, \text{dim})$: Decayed one-hot accumulator recording byte history.

Updates per step:
$$
\text{embedtrace}_t = \text{decay}_t \odot \text{embedtrace}_{t-1} + \mathbf{1}_{c_t}
$$
$$
\text{decaytrace}_t = \text{decay}_t \odot \text{decaytrace}_{t-1} + \text{decay}_t \odot (1 - \text{decay}_t) \odot \text{states}_{t-1}
$$

## Custom Gradient Hooks

To enable long-range memory updates beyond standard truncated backpropagation, custom gradient hooks adjust the encoder embedding weights and decay parameters directly:

1. **Embedding weights**:
$$
\nabla_{W_{\text{embed}}} \leftarrow \nabla_{W_{\text{embed}}} + \frac{\partial \mathcal{L}}{\partial \text{state}_i} \otimes (\text{embedtrace}_{i} \odot \text{decay}_i)
$$
*(Gradient adds to autodiff)*

2. **Decay parameters**:
$$
\nabla_{\text{decay}_i} \leftarrow \frac{\partial \mathcal{L}}{\partial \text{state}_i} \odot \text{decaytrace}_i
$$
*(Gradient replaces autodiff)*

## Detach & Persistence Semantics

In PyTorch, `states`, `decaytrace`, and `embedtrace` are registered buffers detached from the computational graph (`requires_grad=False`). They are updated in-place via `.copy_()` using detached values, preventing the optimizer from erroneously treating them as trainable weights while allowing weights and traces to round-trip through checkpoints.

## Training Loss Modes

The total loss spans four terms:

$$
\mathcal{L} = \underbrace{\text{var}}_{\text{anti-collapse}} + \underbrace{\text{pred mse}}_{\text{latent prediction}} + \underbrace{\text{ce}}_{\text{next-byte}} + \underbrace{\text{stop mse}}_{\text{stop token}}
$$

**Default mode** uses all four terms.

**Crossentropy-only mode** (`ce_only=True`, CLI flag `--ce-only`) reduces this to the single
crossentropy term:

$$
\mathcal{L}_{ce} = -\text{output}[n] + \log\sum\text{output}
$$

Ports the `--ce-only` crossentropy-only flag. The variance term is the anti-collapse
regularizer that keeps state scale near unit variance, so dropping it trades stability for a pure
language-modelling objective — do not combine `--ce-only` with aggressive learning rates.

When `nextb` is `None` in `ce_only` mode the loss is a constant `0`; `Model.forward` therefore only
calls `backward()` when the loss `requires_grad`, which keeps that path from raising.

## Divergence From Upstream

Upstream also stopped persisting the trace buffers (`state.i`, `decaytrace.i`, `embedtrace.i`) in
the same commit that added `--ce-only`. This fork **keeps** that persistence, because:

- `tests/test_checkpoint.py` asserts `decaytrace`/`embedtrace` round-trip through checkpoints.
- The spec above documents the round-trip as intended behaviour.

Consequently `main_mlx.py` is not byte-identical to upstream `main.py` — it is intentionally upstream
plus trace-buffer persistence.

## Chunked training pass (`chunked-trace-training`)

`Model.train_chunk(tokens, next_tokens=None, ends=None, *, chunk=128,
ce_only=False, verify_equivalence=False)` trains a `(B, L)` chunk with **one
forward, one backward and one optimizer step**, without a Python loop over the
tokens. `Model.per_token_gradient_oracle(...)` is the reference implementation
and is what the equivalence test compares against; the per-token path is not
removed and remains the default in `scripts/train_byte_rtu.py`.

### Closed forms for the three buffers

Per layer, let $d = \sigma(\text{decay}) \in (0,1)^D$ (constant across the chunk,
because the parameter does not move mid-chunk), $L$ the chunk length, $c_s$ the
byte at position $s$, $S_t$ the scanned state after token $t$, and $B$ the batch
size. The chunk-final values the implementation writes are

$$
\text{embedtrace}_L = \text{embedtrace}_0 \odot d^{L}
  + \frac{1}{B}\sum_{s=0}^{L-1} \mathbf{1}_{c_s}\, d^{\,L-1-s}
$$

$$
\text{decaytrace}_L = \text{mean}_b\Big(\text{scan}\big(d,\;
  d\odot(1-d)\odot S_{t-1}\big)_{L}\Big)
$$

$$
\text{states}_L = \text{mean}_b\big(S_{L-1}\big)
$$

The scatter form of the `embedtrace` update costs $L\cdot D$, not $L\cdot V\cdot D$;
the exponent vector is `exp(log d * (L-1-s))`, so a decay of 0.999 over 512 steps
is computed rather than truncated. The `(D,)` and `(V, D)` buffers have no batch
axis, so at $B>1$ they hold the batch mean; at $B=1$ the formulas are the exact
per-token recurrence.

### Suffix-sum derivation of the hook contributions

With $g_t = \partial \mathcal{L} / \partial S_t$ (the gradient of the per-token
dummy carrier, identical to the per-token path's `dummies[i].grad`), the per-token
hooks are

$$
\text{embedding hook} = \sum_t g_t \otimes \text{embedtrace}_{t-1}
\qquad
\text{decay hook} = \sum_t g_t \odot \text{decaytrace}_t
$$

Expand $\text{embedtrace}_{t-1} = d^{\,t}\odot \text{embedtrace}_0
+ \sum_{s<t}\mathbf{1}_{c_s}\odot d^{\,t-1-s}$ and swap the sums:

$$
\text{embedding hook}
= \text{embedtrace}_0 \odot \sum_t g_t \odot d^{\,t}
+ \sum_{s=0}^{L-1} \mathbf{1}_{c_s} \otimes R_{s+1}
$$

where $R$ is the reverse decay-weighted scan of the gradients,

$$
R_t = d \odot R_{t+1} + g_t, \qquad R_L = 0
$$

computed by `parallel_scan.reverse_decay_scan` (a suffix sum, implemented as the
associative scan of the reversed sequence). The decay hook sums the scanned
`decaytrace` trajectory: `blocks[i].decay.grad = (g * decaytrace_trajectory).sum()`,
which **replaces** (does not add to) the autograd gradient of the decay parameter,
exactly as the per-token path does.

### Optimizer-step granularity, the `1/L` scale, and the schedule

* Exactly one `optimizer.step()` per chunk.
* The gradient handed to the optimizer is the accumulated gradient divided by
  $L$ (`grad_scale = 1/(B*L)`), keeping the update on the per-byte scale.
* The step counter advances by `B*L` **bytes**.
* Both paths read the learning rate from `Model.lr_at_bytes(bytes_consumed)`
  (`lr_at_step` re-expressed): the per-token path at every byte, the chunked path
  once per chunk at the chunk boundary.

### Equivalence boundary

| Quantity | Equivalent to the per-token path |
| --- | --- |
| `per_token_losses` (summed over the batch) | yes, within `CHUNK_EQUIVALENCE_TOLERANCE = 1e-4` |
| `accumulated_gradients` (before the `1/L` scale) | yes, within the tolerance |
| `trace_buffers` (`states`, `decaytrace`, `embedtrace`) | yes, within the tolerance |
| `parameter_trajectory` | **no** — one optimizer step per chunk is a different regime |

`train_chunk(..., verify_equivalence=True)` runs the oracle beside the chunked
pass and reports `equivalent`, `not_equivalent`, `equivalence_verified`,
`equivalence_note` and a `verification` record (max deltas and the failing
quantities). When the check is not run, `equivalent` is empty and the note says
the check was not run: the output never claims an unmeasured equivalence.

```python
from src.model.rtu import Model

model = Model(dim=384, layers=8, vocab_size=256)
info = model.train_chunk(tokens, chunk=128, verify_equivalence=True)
assert info["verification"]["verified"], info["verification"]["problems"]
assert info["optimizer_steps"] == 1
assert info["step_count"] == info["tokens"]
```
