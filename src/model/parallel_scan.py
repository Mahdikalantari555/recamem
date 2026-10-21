"""Associative prefix scan for the RTU linear recurrence.

The RTU layer state follows the first-order linear recurrence

    h_t = decay ⊙ h_{t-1} + e_t

with a per-dimension ``decay`` in ``(0, 1)``. Each step is an affine map on the
state,

    h -> a ⊙ h + b        with a = decay, b = e_t,

and two consecutive steps compose associatively:

    (a2, b2) ∘ (a1, b1) = (a2 ⊙ a1, a2 ⊙ b1 + b2)

The inclusive prefix of that operator over a sequence is exactly the state
trajectory. :func:`associative_scan` evaluates it with a Hillis-Steele scan:
for a chunk of ``C`` tokens it performs ``ceil(log2 C)`` vectorized
shift-and-compose levels, and carries the chunk-final state into the next chunk
so chunk boundaries are exact rather than approximate. The only Python loops
are over scan levels and chunk boundaries; tokens and batch elements are always
tensor dimensions.

:func:`sequential_scan` is the token-by-token oracle used by the tests. It is
the reference these functions are verified against and is deliberately kept
out of every production path.

Gradients
---------
The scan is a forward primitive. When autograd is enabled and an input requires
a gradient, the composition uses out-of-place concatenations so the graph stays
valid. Otherwise (the normal evaluation/benchmark case) it uses in-place
updates on private clones, which is what makes the chunked path fast.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["associative_scan", "sequential_scan", "reverse_decay_scan"]


def _broadcast_decays(decays: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return ``decays`` broadcast to ``(B, L, D)`` without materializing it."""
    batch, length, dim = inputs.shape
    if not isinstance(decays, torch.Tensor):
        decays = torch.as_tensor(decays, dtype=inputs.dtype, device=inputs.device)
    else:
        decays = decays.to(dtype=inputs.dtype, device=inputs.device)

    if decays.dim() == 0:
        return decays.expand(batch, length, dim)
    if decays.dim() == 1:
        if decays.shape[0] != dim:
            raise ValueError(f"decay vector has length {decays.shape[0]}, expected {dim}")
        return decays.view(1, 1, dim).expand(batch, length, dim)
    if decays.dim() == 2:
        if decays.shape == (batch, dim):
            return decays.view(batch, 1, dim).expand(batch, length, dim)
        if decays.shape == (1, dim):
            return decays.view(1, 1, dim).expand(batch, length, dim)
        raise ValueError(f"2-D decays must be (B, D)={(batch, dim)} or (1, D), got {tuple(decays.shape)}")
    if decays.dim() == 3:
        if decays.shape == (batch, length, dim):
            return decays
        if decays.shape == (batch, 1, dim) or decays.shape == (1, length, dim) or decays.shape == (1, 1, dim):
            return decays.expand(batch, length, dim)
        raise ValueError(
            f"3-D decays must broadcast to (B, L, D)={(batch, length, dim)}, got {tuple(decays.shape)}"
        )
    raise ValueError(f"decays must be scalar, 1-D, 2-D or 3-D, got {decays.dim()}-D")


def _initial_state(h0: Optional[torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    """Return the carried state as ``(B, D)``."""
    batch, _, dim = inputs.shape
    if h0 is None:
        return inputs.new_zeros(batch, dim)
    h0 = h0.to(dtype=inputs.dtype, device=inputs.device)
    if h0.dim() == 1:
        if h0.shape[0] != dim:
            raise ValueError(f"h0 has length {h0.shape[0]}, expected {dim}")
        return h0.view(1, dim).expand(batch, dim)
    if h0.dim() == 2:
        if h0.shape == (batch, dim):
            return h0
        if h0.shape == (1, dim):
            return h0.expand(batch, dim)
        raise ValueError(f"h0 must be (B, D)={(batch, dim)} or (D,), got {tuple(h0.shape)}")
    raise ValueError(f"h0 must be 1-D or 2-D, got {h0.dim()}-D")


def _powers_for_chunk(a_chunk: torch.Tensor) -> torch.Tensor:
    """``decay ** (i + 1)`` for the first token of a time-invariant chunk."""
    batch, length, dim = a_chunk.shape
    return torch.cumprod(a_chunk[:, :1].expand(batch, length, dim), dim=1)


def _scan_chunk_inplace(a: torch.Tensor, b: torch.Tensor, h: torch.Tensor,
                        powers: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hillis-Steele scan of one chunk, mutating private ``a``/``b`` clones.

    When ``powers`` is supplied the decay is constant along the sequence, so
    the composed decay at level ``shift`` is exactly ``decay ** shift`` and the
    ``a`` tensor does not need to be carried: the level reads one row of
    ``powers``. Otherwise ``a`` is composed level by level.
    """
    length = b.shape[1]
    b[:, 0].add_(a[:, 0] * h)
    if powers is not None:
        # Ping-pong buffers let each level be one fused addcmul into memory
        # that does not overlap its inputs.
        other = torch.empty_like(b)
        shift = 1
        while shift < length:
            composed = powers[:, shift - 1:shift].expand(b.shape[0], length - shift, a.shape[2])
            other[:, :shift] = b[:, :shift]
            torch.addcmul(b[:, shift:], composed, b[:, :-shift], out=other[:, shift:])
            b, other = other, b
            shift *= 2
        return b
    shift = 1
    while shift < length:
        b[:, shift:].add_(a[:, shift:] * b[:, :-shift])
        a[:, shift:] = a[:, shift:] * a[:, :-shift]
        shift *= 2
    return b


def _scan_chunk_autograd(a: torch.Tensor, b: torch.Tensor, h: torch.Tensor,
                         powers: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hillis-Steele scan of one chunk using out-of-place composition."""
    length = b.shape[1]
    b = torch.cat((b[:, :1] + a[:, :1] * h.unsqueeze(1), b[:, 1:]), dim=1)
    shift = 1
    while shift < length:
        if powers is not None:
            composed = powers[:, shift - 1:shift].expand(b.shape[0], length - shift, a.shape[2])
        else:
            composed = a[:, shift:]
            a = torch.cat((a[:, :shift], a[:, shift:] * a[:, :-shift]), dim=1)
        merged_b = b[:, shift:] + composed * b[:, :-shift]
        b = torch.cat((b[:, :shift], merged_b), dim=1)
        shift *= 2
    return b


def associative_scan(
    decays: torch.Tensor,
    inputs: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    chunk: Optional[int] = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``h_t = decay ⊙ h_{t-1} + e_t`` with an associative scan.

    Parameters
    ----------
    decays:
        Per-dimension decay. Accepts a scalar, ``(D,)``, ``(B, D)``,
        ``(B, 1, D)`` or ``(B, L, D)`` tensor; it is broadcast over the
        missing axes.
    inputs:
        Encoder outputs ``e_t`` with shape ``(B, L, D)``.
    h0:
        Optional initial state, ``(D,)`` or ``(B, D)``. Defaults to zeros.
    chunk:
        Tokens per scan chunk. ``None`` or ``0`` scans the whole sequence in
        one chunk. Chunk boundaries are exact because the final state of one
        chunk is carried into the next.

    Returns
    -------
    (states, final_state)
        ``states`` has shape ``(B, L, D)`` and holds every intermediate state
        (inclusive prefix); ``final_state`` has shape ``(B, D)``.

    The implementation loops only over chunk boundaries and ``ceil(log2 C)``
    scan levels. It never loops over tokens or batch elements. For a chunk of
    ``C`` tokens the composition work is ``C * log2(C)``, so a full sequence of
    ``L`` tokens costs ``L * log2(C)`` instead of ``L`` Python-level steps.
    """
    if inputs.dim() != 3:
        raise ValueError(f"inputs must be a (B, L, D) tensor, got {inputs.dim()}-D")
    batch, length, dim = inputs.shape
    if chunk is None or chunk <= 0:
        chunk = length

    decay = _broadcast_decays(decays, inputs)
    state = _initial_state(h0, inputs)

    if length == 0:
        return inputs.clone(), state.clone()

    grad_path = torch.is_grad_enabled() and (
        inputs.requires_grad or decay.requires_grad or state.requires_grad
    )
    # A decay expanded over the sequence axis has stride 0 there; for such a
    # time-invariant decay the composed decay after `shift` levels is exactly
    # decay ** shift, which removes the per-level `a` composition.
    time_invariant = decay.stride(1) == 0

    pieces = []
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        a_chunk = decay[:, start:stop]
        b_chunk = inputs[:, start:stop]
        powers = _powers_for_chunk(a_chunk) if time_invariant else None
        if grad_path:
            scanned = _scan_chunk_autograd(a_chunk, b_chunk, state, powers)
        else:
            # Private contiguous buffers: the level loop mutates them.
            scanned = _scan_chunk_inplace(
                a_chunk if powers is not None else a_chunk.clone(),
                b_chunk.clone(),
                state,
                powers,
            )
        pieces.append(scanned)
        state = scanned[:, -1]

    states = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
    return states, state


def sequential_scan(
    decays: torch.Tensor,
    inputs: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token reference implementation of the RTU recurrence.

    This is the oracle ``tests/test_parallel_scan.py`` compares
    :func:`associative_scan` against. It is intentionally a plain Python loop
    and is not called by any production path: deleting it would delete the only
    independent check on the scan.
    """
    if inputs.dim() != 3:
        raise ValueError(f"inputs must be a (B, L, D) tensor, got {inputs.dim()}-D")
    batch, length, dim = inputs.shape
    decay = _broadcast_decays(decays, inputs)
    state = _initial_state(h0, inputs)

    pieces = []
    for t in range(length):
        state = decay[:, t] * state + inputs[:, t]
        pieces.append(state)

    if not pieces:
        return inputs.clone(), state
    return torch.stack(pieces, dim=1), state


def reverse_decay_scan(
    decays: torch.Tensor,
    inputs: torch.Tensor,
    h_last: Optional[torch.Tensor] = None,
    chunk: Optional[int] = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse decay-weighted suffix sums.

    Evaluates ``R_t = decay ⊙ R_{t+1} + v_t`` with ``R_{L-1} = decay ⊙ h_last + v_{L-1}``
    (equivalently ``h_last`` as the value carried past the end). This is the
    suffix-sum matrix the chunked gradient-hook accumulation needs: for a state
    gradient ``g_t`` at chunk position ``t``, ``R_{t+1}`` is exactly
    ``sum_{u>t} g_u ⊙ decay**(u-t-1)``, the weight the trace-buffer contribution
    of token ``t`` carries.

    The recurrence is the same affine family as the forward scan, so it is
    evaluated by reversing the sequence, running :func:`associative_scan`, and
    reversing the result. It is exact at chunk boundaries for the same reason.
    """
    if inputs.dim() != 3:
        raise ValueError(f"inputs must be a (B, L, D) tensor, got {inputs.dim()}-D")
    reversed_inputs = torch.flip(inputs, dims=[1])
    suffix, final = associative_scan(decays, reversed_inputs, h0=h_last, chunk=chunk)
    return torch.flip(suffix, dims=[1]), final


def _self_check(dim: int = 16, length: int = 40, chunk: int = 16, seed: int = 0) -> None:
    """Small executable check; ``python -m src.model.parallel_scan`` runs it."""
    generator = torch.Generator().manual_seed(seed)
    decays = torch.rand(dim, generator=generator) * 0.9 + 0.09
    inputs = torch.randn(2, length, dim, generator=generator)
    fast, fast_final = associative_scan(decays, inputs, chunk=chunk)
    slow, slow_final = sequential_scan(decays, inputs)
    torch.testing.assert_close(fast, slow, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(fast_final, slow_final, rtol=1e-4, atol=1e-6)
    suffix, _ = reverse_decay_scan(decays, inputs, chunk=chunk)
    expected = torch.zeros_like(inputs)
    running = torch.zeros_like(inputs[:, 0])
    for t in range(length - 1, -1, -1):
        running = decays * running + inputs[:, t]
        expected[:, t] = running
    torch.testing.assert_close(suffix, expected, rtol=1e-4, atol=1e-6)
    print(f"parallel_scan self-check ok: L={length} chunk={chunk} dim={dim}")


if __name__ == "__main__":  # pragma: no cover - manual check
    _self_check()
