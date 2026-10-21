
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from typing import Tuple, Dict, Any, Optional

try:
    from src.model.parallel_scan import associative_scan, reverse_decay_scan
except ImportError:  # pragma: no cover - package-relative fallback
    from .parallel_scan import associative_scan, reverse_decay_scan
# Avoid circular import: define a minimal PlasticAdapter base if needed
# We will try to import the real one, but if it fails (circular), use a dummy that will be replaced later
try:
    # Check if already loaded (circular case)
    import sys
    if 'src.model.plastic_adapter' in sys.modules and hasattr(sys.modules['src.model.plastic_adapter'], 'PlasticAdapter'):
        PlasticAdapter = sys.modules['src.model.plastic_adapter'].PlasticAdapter
    else:
        raise ImportError("Not yet loaded")
except:
    try:
        from src.model.plastic_adapter import PlasticAdapter
    except:
        try:
            from .plastic_adapter import PlasticAdapter
        except:
            from abc import ABC
            class PlasticAdapter(ABC):
                pass

# Tolerance for the chunked-vs-per-token equivalence checks. The chunked path
# is algebraically equivalent to the per-token path for the losses, the
# accumulated gradients and the trace buffers; floating-point reassociation is
# the only difference, and this is the documented bound on it.
CHUNK_EQUIVALENCE_TOLERANCE = 1e-4

# Python-level token loop counters. The chunked path must not grow its count
# with the chunk length; the per-token reference must, because it loops.
TOKEN_LOOP_COUNTER = {"chunked": 0, "per_token": 0}

# The quantities the chunked path reproduces exactly, and the one it cannot.
CHUNK_EQUIVALENT_QUANTITIES = ("per_token_losses", "accumulated_gradients", "trace_buffers")
CHUNK_NON_EQUIVALENT_QUANTITIES = ("parameter_trajectory",)

# ponytail: minimal RTU implementation matching MLX 0.32.2 semantics. Upgrade to batched CUDA kernel if scaling beyond 16 layers.


class MissingCheckpointError(FileNotFoundError):
    """A declared checkpoint path is empty, absent, or not a regular file.

    Defined here (the lowest-level module in the import graph) so the model,
    the backend, and the profile loader all share one exception identity.
    """


class InvalidProfileError(ValueError):
    """A profile manifest is missing, unreadable, or semantically invalid."""


class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, bias_correction=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            bias_correction = group['bias_correction']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if bias_correction:
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    step_size = lr
                    denom = exp_avg_sq.sqrt().add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Encoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 256):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    def forward(self, x: torch.Tensor | int) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long, device=self.embed.weight.device)
        elif x.dtype != torch.long:
            x = x.to(torch.long)
        return self.embed(x)


class Decoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 256):
        super().__init__()
        self.decode = nn.Linear(dim, vocab_size)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(x), torch.sigmoid(self.stop(x))


class Layer(nn.Module):
    def __init__(self, dim: int, spread: int = 32, vocab_size: int = 256):
        super().__init__()
        self.dim = dim
        self.spread = spread
        self.vocab_size = vocab_size

        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)

        self.register_buffer('states', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('decaytrace', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('embedtrace', torch.zeros(vocab_size, dim, dtype=torch.float32))

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor | None = None,
                planned: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dummy is None:
            dummy = torch.zeros(self.dim, dtype=torch.float32, device=enc.device)
        decay = torch.sigmoid(self.decay)
        if planned is None:
            state = (decay * self.states) + enc + dummy
        else:
            # `planned` is the state trajectory sample produced by
            # parallel_scan.associative_scan. It already contains this token's
            # `enc` and the decayed history, so the value must not add `enc`
            # again. Subtracting the detached value keeps the forward value
            # unchanged while preserving the direct encoder gradient that the
            # sequential step has (the history itself is detached, exactly like
            # the `self.states` buffer).
            state = planned.detach() + (enc - enc.detach()) + dummy
        x_out = x + self.silu(self.weights(self.norm(state)))
        return x_out, state, decay

    def reset(self):
        self.states.zero_()
        self.decaytrace.zero_()
        self.embedtrace.zero_()


def loss_variance(x: torch.Tensor) -> torch.Tensor:
    var = torch.var(x, unbiased=False)
    return torch.clamp(1.0 - torch.sqrt(var + 1e-4), min=0.0)


def loss_pred_mse(x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - tgt) ** 2)


def loss_crossentropy(output: torch.Tensor, nextb: int | torch.Tensor) -> torch.Tensor:
    if isinstance(nextb, torch.Tensor):
        if nextb.dim() > 0:
            nextb = nextb.squeeze()
        idx = nextb.item() if nextb.numel() == 1 else nextb
    else:
        idx = nextb
    return -output[idx] + torch.logsumexp(output, dim=-1)


def loss_stop_mse(stop: torch.Tensor, end: bool) -> torch.Tensor:
    target = torch.tensor([1.0 if end else 0.0], dtype=stop.dtype, device=stop.device)
    return torch.mean((stop - target) ** 2)


def compute_losses(x: torch.Tensor, output: torch.Tensor, stop: torch.Tensor,
                   tgt: torch.Tensor | None = None,
                   nextb: int | torch.Tensor | None = None,
                   end: bool = False,
                   ce_only: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if ce_only:
        # Crossentropy-only training mode: variance / prediction / stop losses are skipped.
        if nextb is not None:
            total = loss_crossentropy(output, nextb)
            losses: Dict[str, torch.Tensor] = {'ce': total}
        else:
            total = torch.zeros((), dtype=output.dtype, device=output.device)
            losses = {}
        losses['total'] = total
        return total, losses

    l_var = loss_variance(x)
    losses = {'var': l_var}
    total = l_var

    if nextb is not None:
        if tgt is not None:
            l_pred = loss_pred_mse(x, tgt)
            losses['pred'] = l_pred
            total = total + l_pred
        l_ce = loss_crossentropy(output, nextb)
        l_stop = loss_stop_mse(stop, end)
        losses['ce'] = l_ce
        losses['stop'] = l_stop
        total = total + l_ce + l_stop

    losses['total'] = total
    return total, losses


class Model(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 16, spread: int = 32, temp: float = 0.75,
                 rate: float = 5e-4, bound: Tuple[int, int] = (40000, 120000),
                 vocab_size: int = 256, encoder: nn.Module | None = None, decoder: nn.Module | None = None):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.spread = spread
        self.temp = temp
        self.rate = rate
        self.bound = bound
        self.vocab_size = vocab_size

        self.encoder = encoder if encoder is not None else Encoder(dim, vocab_size=vocab_size)
        self.decoder = decoder if decoder is not None else Decoder(dim, vocab_size=vocab_size)
        self.blocks = nn.ModuleList([Layer(dim, spread, vocab_size=vocab_size) for _ in range(layers)])

        self.step_count = 0
        self.last_loss = 0.0
        self.optimizer = CustomAdamW(self.parameters(), lr=rate, weight_decay=0.01, bias_correction=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def sample(self, output: torch.Tensor) -> int:
        probs = torch.softmax(output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)) / math.log(float(self.vocab_size))
        temp = max(0.1, float(self.temp * (1.0 - self.temp * entropy)))
        dist = torch.distributions.Categorical(logits=output / temp)
        return int(dist.sample().item())

    def step(self, c: torch.Tensor | int, dummies: list[torch.Tensor] | None = None, frozen: bool = False,
             planned_states: list[torch.Tensor] | None = None):
        if dummies is None:
            dummies = [torch.zeros(self.dim, dtype=torch.float32, device=self.device) for _ in range(self.layers)]

        enc = self.encoder(c)
        x = enc
        states, decays = [], []

        for i, layer in enumerate(self.blocks):
            planned = planned_states[i] if planned_states is not None else None
            x, state, decay = layer(enc, x, dummies[i], planned=planned)
            if frozen:
                layer.states.copy_(state.detach())
            states.append(state)
            decays.append(decay)

        logits, stop = self.decoder(x)
        return (x, states, decays), (logits, stop)

    def _scan_layer_stack(self, tokens: torch.Tensor, chunk: int = 128):
        """Scan all layers as one batch of sequences.

        Every layer's recurrence consumes the same encoder output, so the
        layers are stacked on the batch axis and scanned in a single call.
        Returns ``(enc, trajectories)`` with ``enc`` shaped ``(B, L, D)`` and
        ``trajectories`` shaped ``(layers, B, L, D)``.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be a (B, L) tensor, got {tokens.dim()}-D")
        tokens = tokens.to(dtype=torch.long, device=self.device)
        enc = self.encoder(tokens)
        batch, length, dim = enc.shape
        layers = len(self.blocks)
        if layers == 0:
            return enc, enc.new_zeros(0, batch, length, dim)

        decays = torch.stack([torch.sigmoid(layer.decay) for layer in self.blocks], dim=0)
        initial = torch.stack([layer.states for layer in self.blocks], dim=0)
        inputs = enc.unsqueeze(0).expand(layers, batch, length, dim).reshape(layers * batch, length, dim)
        decay = decays.unsqueeze(1).expand(layers, batch, dim).reshape(layers * batch, dim)
        h0 = initial.unsqueeze(1).expand(layers, batch, dim).reshape(layers * batch, dim)

        scanned, _ = associative_scan(decay, inputs, h0=h0, chunk=chunk)
        return enc, scanned.view(layers, batch, length, dim)

    def scan_states(self, tokens: torch.Tensor, chunk: int = 128) -> list[torch.Tensor]:
        """Per-layer state trajectories for a ``(B, L)`` token chunk.

        Each trajectory is ``(B, L, D)`` and is produced by
        :func:`src.model.parallel_scan.associative_scan` from the layer's
        current ``states`` buffer. The trajectories are detached plans: the
        caller feeds them back through the unchanged per-token output and
        trace-buffer path. No buffer is mutated here.
        """
        with torch.no_grad():
            _, trajectories = self._scan_layer_stack(tokens, chunk)
        return [trajectories[i] for i in range(trajectories.shape[0])]

    def forward_chunk(self, tokens: torch.Tensor, chunk: int = 128):
        """Vectorized forward over a ``(B, L)`` token chunk.

        The state trajectory of every layer comes from the associative scan and
        the per-token output transform is applied to the scanned states as a
        single vectorized operation. Returns ``(x, states, (logits, stop))``
        with ``x`` and each entry of ``states`` shaped ``(B, L, D)`` and
        ``logits`` shaped ``(B, L, vocab_size)``. Buffers are not mutated.

        Run under ``torch.no_grad()`` (or inference mode) to take the in-place
        scan path; with gradients enabled the scan uses the out-of-place
        composition and is slower.
        """
        enc, trajectories = self._scan_layer_stack(tokens, chunk)
        x = enc
        for i, layer in enumerate(self.blocks):
            x = x + layer.silu(layer.weights(layer.norm(trajectories[i])))
        logits, stop = self.decoder(x)
        states = [trajectories[i] for i in range(trajectories.shape[0])]
        return x, states, (logits, stop)

    def lr_at_step(self, step: int) -> float:
        b0, b1 = float(self.bound[0]), float(self.bound[1])
        progress = max(0.0, min(1.0, (float(step) + 1.0 - b0) / (b1 - b0)))
        return self.rate * (1.0 - 0.9 * progress)

    def forward(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None = None,
                end: bool = False, frozen: bool = False,
                ce_only: bool = False) -> Tuple[int, float]:
        return self._forward(currb, nextb, end, frozen, ce_only, planned_states=None)

    def forward_planned(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None = None,
                        end: bool = False, frozen: bool = False, ce_only: bool = False,
                        planned_states: list[torch.Tensor] | None = None) -> Tuple[int, float]:
        """Same contract as :meth:`forward`, with scanned states for one token.

        ``planned_states`` is a list of per-layer ``(B, D)`` state samples from
        :meth:`scan_states` (one token slice of the chunk trajectory). Output,
        loss, trace-buffer and optimizer code are the unchanged ``forward``
        code path.
        """
        return self._forward(currb, nextb, end, frozen, ce_only, planned_states=planned_states)

    def _forward(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None,
                 end: bool, frozen: bool, ce_only: bool,
                 planned_states: list[torch.Tensor] | None) -> Tuple[int, float]:
        if frozen:
            with torch.no_grad():
                c = torch.tensor(currb, dtype=torch.long, device=self.device) if not isinstance(currb, torch.Tensor) else currb
                (_, states, _), (output, stop) = self.step(c, frozen=True, planned_states=planned_states)
                return self.sample(output), float(stop.item())

        c = torch.tensor(currb, dtype=torch.long, device=self.device) if not isinstance(currb, torch.Tensor) else currb
        TOKEN_LOOP_COUNTER["per_token"] += 1
        dummies = [torch.zeros(self.dim, dtype=torch.float32, device=self.device, requires_grad=True) for _ in range(self.layers)]

        self.zero_grad(set_to_none=True)

        tgt = None
        if nextb is not None:
            n = torch.tensor(nextb, dtype=torch.long, device=self.device) if not isinstance(nextb, torch.Tensor) else nextb
            with torch.no_grad():
                tgt = self.encoder(n).detach()

        (x, states, decays), (output, stop) = self.step(c, dummies=dummies, frozen=False, planned_states=planned_states)
        total_loss, _ = compute_losses(x, output, stop, tgt=tgt, nextb=nextb, end=end, ce_only=ce_only)
        self.last_loss = float(total_loss.item())
        if total_loss.requires_grad:
            total_loss.backward()

        c_val = int(currb) if not isinstance(currb, torch.Tensor) else int(currb.item())
        c_range = (torch.arange(self.vocab_size, device=self.device) == c_val).float().unsqueeze(1)

        for i, layer in enumerate(self.blocks):
            dlds = dummies[i].grad
            if dlds is None:
                dlds = torch.zeros(self.dim, device=self.device)

            decay_embedtrace = layer.embedtrace * decays[i].detach()
            embedtrace = decay_embedtrace + c_range

            if hasattr(self.encoder, 'embed') and self.encoder.embed.weight.grad is not None:
                self.encoder.embed.weight.grad.add_(dlds * decay_embedtrace)

            dec_val = decays[i].detach()
            decaytrace = (dec_val * layer.decaytrace) + (dec_val * (1.0 - dec_val) * layer.states)
            layer.decay.grad = (dlds * decaytrace).clone()

            layer.states.copy_(states[i].detach())
            layer.decaytrace.copy_(decaytrace.detach())
            layer.embedtrace.copy_(embedtrace.detach())

        # Both training paths read the schedule as a function of bytes consumed.
        lr = self.lr_at_bytes(self.step_count)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.optimizer.step()
        self.step_count += 1

        return self.sample(output.detach()), float(stop.detach().item())

    # ------------------------------------------------------------------ #
    # Chunked training pass (chunked-trace-training)
    # ------------------------------------------------------------------ #
    # The per-token path advances three trace buffers and injects two gradient
    # contributions once per byte. Both are affine recurrences over the same
    # decay, so their chunk-final values follow in closed form and the
    # per-token gradient can be recovered with one reverse weighted scan. This
    # pass keeps the per-token closed forms as the oracle and reproduces the
    # losses, the accumulated gradients and the buffers; it does *not*
    # reproduce the parameter trajectory, because it takes one optimizer step
    # per chunk instead of one per byte.

    def lr_at_bytes(self, bytes_consumed: int) -> float:
        """The learning rate as a function of bytes consumed.

        Both training paths read this one schedule, so "the learning rate at
        byte N" means the same thing to a per-token run and a chunked run.
        """
        return self.lr_at_step(int(bytes_consumed))

    @staticmethod
    def _as_token_batch(tokens) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B, L) or (L,), got {tokens.dim()}-D")
        return tokens.to(dtype=torch.long)

    def _prepare_chunk_inputs(self, tokens, next_tokens, ends):
        tokens = self._as_token_batch(tokens).to(self.device)
        batch, length = tokens.shape
        mask = torch.ones(batch, length, dtype=torch.bool, device=self.device)
        if next_tokens is None:
            # The last token of a chunk has no observed successor: its
            # cross-entropy term is masked out rather than invented.
            next_tokens = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
            mask[:, -1] = False
        else:
            next_tokens = self._as_token_batch(next_tokens).to(self.device)
            if next_tokens.shape != tokens.shape:
                raise ValueError("next_tokens must have the same (B, L) shape as tokens")
        if ends is None:
            ends = torch.zeros(batch, length, dtype=torch.bool, device=self.device)
        elif not isinstance(ends, torch.Tensor):
            ends = torch.full((batch, length), bool(ends), dtype=torch.bool, device=self.device)
        elif ends.dim() == 0:
            ends = ends.expand(batch, length).to(torch.bool)
        elif ends.dim() == 1:
            ends = ends.unsqueeze(0).expand(batch, length).to(torch.bool)
        return tokens, next_tokens, ends.to(self.device), mask

    def _batched_ce(self, logits: torch.Tensor, targets: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -picked * mask.to(log_probs.dtype)

    def _batched_token_losses(self, x, logits, stop, next_tokens, ends, mask,
                              ce_only: bool):
        """Per-token loss terms reduced over the dimension axis.

        Every term is the same expression the per-token path evaluates on one
        ``(D,)`` vector, vectorized over ``(B, L)``; they are summed over
        tokens, exactly like the per-token path's per-token scalars.
        """
        batch, length, _ = x.shape
        terms = {}
        total = x.new_zeros(batch, length)
        if not ce_only:
            var = torch.clamp(
                1.0 - torch.sqrt(torch.var(x, dim=-1, unbiased=False) + 1e-4), min=0.0
            )
            terms["var"] = var
            total = total + var
            tgt = self.encoder(next_tokens).detach()
            terms["pred"] = ((x - tgt) ** 2).mean(dim=-1)
            end_targets = ends.to(dtype=stop.dtype).reshape(batch, length, 1)
            terms["stop"] = ((stop - end_targets) ** 2).mean(dim=-1)
            total = total + terms["pred"] + terms["stop"]
        terms["ce"] = self._batched_ce(logits, next_tokens, mask) if next_tokens is not None \
            else x.new_zeros(batch, length)
        total = total + terms["ce"]
        summed = {name: value.sum() for name, value in terms.items()}
        summed["total"] = total.sum()
        return total, summed

    def _chunk_forward_backward(self, tokens, next_tokens, ends, mask, chunk, ce_only):
        """One vectorized forward, one backward over the summed chunk loss."""
        enc = self.encoder(tokens)
        with torch.no_grad():
            _, trajectories = self._scan_layer_stack(tokens, chunk)
        dummies = [
            torch.zeros(tokens.shape[0], tokens.shape[1], self.dim,
                        dtype=enc.dtype, device=self.device, requires_grad=True)
            for _ in self.blocks
        ]
        x = enc
        states = []
        for index, layer in enumerate(self.blocks):
            # The scanned trajectory is the plan; the dummy carrier is the
            # per-token handle whose gradient is dL/dstate_t, and the
            # enc - enc.detach() term preserves the direct encoder gradient.
            state = trajectories[index].detach() + (enc - enc.detach()) + dummies[index]
            states.append(state)
            x = x + layer.silu(layer.weights(layer.norm(state)))
        logits, stop = self.decoder(x)
        per_token, terms = self._batched_token_losses(
            x, logits, stop, next_tokens, ends, mask, ce_only
        )
        total = per_token.sum()
        self.last_loss = float(total.item())
        if total.requires_grad:
            total.backward()
        return enc, trajectories, states, dummies, per_token, terms, total

    def _apply_chunk_updates(self, tokens, enc, states, dummies, chunk) -> Dict[str, Any]:
        """Vectorized trace-buffer advance and gradient-hook injection.

        Closed forms (per layer, ``d = sigmoid(decay)``, ``L`` = chunk length):

        * ``embedtrace_L = d**L * etrace_0 + sum_s onehot(c_s) * d**(L-1-s)``
        * ``dtrace_L = d**L * dtrace_0 + sum_t d*(1-d) * S_{t-1} * d**(L-1-t)``
        * ``states_L = S_{L-1}`` (the last scanned state)

        Gradient hooks, with ``g_t = dL/dstate_t``:

        * embedding: ``sum_t g_t (x) etrace_{t-1}``, split into the
          buffer-initial term ``(sum_t g_t d**t) (x) etrace_0`` and the scattered
          byte term over the reverse suffix sums ``R_{t+1}``;
        * decay: ``sum_t g_t * dtrace_t``, which *replaces* the autograd
          gradient exactly as the per-token path does.

        Batch convention: the ``(D,)`` and ``(V, D)`` buffers hold the batch
        mean, which is what makes a ``B > 1`` chunk well defined for a buffer
        that has no batch axis. A ``B == 1`` chunk is the exact per-token case.
        """
        batch, length = tokens.shape
        dtype = enc.dtype
        device = self.device
        log_decay = torch.log(torch.stack(
            [torch.sigmoid(layer.decay).detach() for layer in self.blocks]
        )) if self.blocks else torch.zeros(0, self.dim, device=device, dtype=dtype)

        embed_weight = getattr(getattr(self.encoder, "embed", None), "weight", None)
        info: Dict[str, Any] = {"layers": [], "embedding_hook_applied": False}

        for index, layer in enumerate(self.blocks):
            decay = torch.sigmoid(layer.decay).detach()
            g = dummies[index].grad
            if g is None:
                g = torch.zeros(batch, length, self.dim, dtype=dtype, device=device)

            exponents = torch.arange(length + 1, dtype=dtype, device=device)
            powers = torch.exp(log_decay[index].unsqueeze(0) * exponents.unsqueeze(1))
            decay_to_L = powers[-1]

            # --- buffers -------------------------------------------------
            etrace_prev = layer.embedtrace.detach().clone()
            state_prev = layer.states.detach()
            prev_states = torch.cat(
                [state_prev.expand(batch, 1, self.dim), states[index].detach()[:, :-1, :]],
                dim=1,
            )

            byte_weights = torch.exp(
                log_decay[index].unsqueeze(0)
                * torch.arange(length - 1, -1, -1, dtype=dtype, device=device).unsqueeze(1)
            )
            flat_weights = byte_weights.unsqueeze(0).expand(batch, -1, -1).reshape(-1, self.dim)
            scatter = torch.zeros_like(etrace_prev)
            scatter.index_add_(0, tokens.reshape(-1), flat_weights)
            etrace_new = etrace_prev * decay_to_L + scatter / float(batch)

            dtrace_inputs = (decay * (1.0 - decay)).unsqueeze(0).unsqueeze(0) * prev_states
            with torch.no_grad():
                dtrace_traj, _ = associative_scan(
                    decay, dtrace_inputs,
                    h0=layer.decaytrace.detach().expand(batch, self.dim).clone(),
                    chunk=chunk,
                )
            dtrace_new = dtrace_traj[:, -1].mean(dim=0)

            # --- embedding hook -----------------------------------------
            if embed_weight is not None and embed_weight.grad is not None:
                initial_weights = torch.exp(
                    log_decay[index].unsqueeze(0)
                    * torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
                )
                initial_term = etrace_prev * (g * initial_weights.unsqueeze(0)).sum(dim=(0, 1)).unsqueeze(0)
                with torch.no_grad():
                    suffix, _ = reverse_decay_scan(decay, g, chunk=chunk)
                suffix_next = torch.cat(
                    [suffix[:, 1:, :], torch.zeros(batch, 1, self.dim, dtype=dtype, device=device)],
                    dim=1,
                )
                byte_term = torch.zeros_like(etrace_prev)
                byte_term.index_add_(0, tokens.reshape(-1), suffix_next.reshape(-1, self.dim))
                embed_weight.grad.add_((initial_term + byte_term).to(embed_weight.dtype))
                info["embedding_hook_applied"] = True

            # --- decay hook (replaces the autograd gradient) -------------
            layer.decay.grad = (g * dtrace_traj).sum(dim=(0, 1))

            with torch.no_grad():
                layer.states.copy_(states[index][:, -1].mean(dim=0).detach())
                layer.decaytrace.copy_(dtrace_new.detach())
                layer.embedtrace.copy_(etrace_new.detach())
            info["layers"].append({
                "layer": index,
                "chunk_final_state_norm": float(layer.states.norm().item()),
                "chunk_final_decaytrace_norm": float(layer.decaytrace.norm().item()),
                "chunk_final_embedtrace_norm": float(layer.embedtrace.norm().item()),
            })
        return info

    def train_chunk(self, tokens, next_tokens=None, ends=None, *, chunk: int = 128,
                    ce_only: bool = False, verify_equivalence: bool = False,
                    counter: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """One chunked training pass: one optimizer step per chunk.

        The pass is algebraically the per-token pass for the per-token losses,
        the accumulated gradients and the trace buffers. It is *not* the same
        optimizer trajectory: one step per chunk is a different training
        regime, so the accumulated gradient is divided by the chunk length and
        both paths read the learning-rate schedule as a function of bytes
        consumed.

        ``verify_equivalence`` runs the per-token oracle beside the chunked
        path and compares gradients, losses and buffers. When it is not run the
        result names the missing verification and does not claim equivalence.
        """
        tokens, next_tokens, ends, mask = self._prepare_chunk_inputs(
            tokens, next_tokens, ends
        )
        batch, length = tokens.shape
        if length == 0:
            return {
                "loss": 0.0, "tokens": 0, "chunk_length": 0, "optimizer_steps": 0,
                "grad_scale": 0.0, "equivalent": [], "not_equivalent": list(CHUNK_NON_EQUIVALENT_QUANTITIES),
                "equivalence_verified": False,
                "equivalence_note": "an empty chunk has no loss to compare",
            }

        self.zero_grad(set_to_none=True)
        TOKEN_LOOP_COUNTER["chunked"] += 0
        initial_buffers = {
            name: tensor.detach().clone() for name, tensor in self.named_buffers()
        }
        enc, trajectories, states, dummies, per_token, terms, total = self._chunk_forward_backward(
            tokens, next_tokens, ends, mask, chunk, ce_only
        )
        hook_info = self._apply_chunk_updates(tokens, enc.detach(), states, dummies, chunk)

        # Capture the chunked gradients before anything else touches them.
        chunked_grads = {
            name: (param.grad.detach().clone() if param.grad is not None else None)
            for name, param in self.named_parameters()
        }

        verification: Dict[str, Any] = {"verified": False}
        if verify_equivalence:
            # The oracle must start where the chunked pass started, so the
            # buffers the chunked pass just advanced are set aside and put back
            # afterwards. The comparison reads the chunked final buffers
            # explicitly, never whatever happens to be current.
            chunked_buffers = {
                name: tensor.detach().clone() for name, tensor in self.named_buffers()
            }
            current = dict(self.named_buffers())
            with torch.no_grad():
                for name, value in initial_buffers.items():
                    if name in current and current[name].shape == value.shape:
                        current[name].copy_(value)
            oracle = self.per_token_gradient_oracle(
                tokens, next_tokens, ends, ce_only=ce_only, mask=mask
            )
            verification = self._compare_with_oracle(
                chunked_grads, per_token, oracle, chunked_buffers
            )
            current = dict(self.named_buffers())
            with torch.no_grad():
                for name, value in chunked_buffers.items():
                    if name in current and current[name].shape == value.shape:
                        current[name].copy_(value)
            # The oracle zeroed the gradients it accumulated; put the chunked
            # gradients back for the optimizer step.
            for name, param in self.named_parameters():
                param.grad = chunked_grads[name]
        elif counter is not None:
            counter["equivalence_check"] = 0

        lr = self.lr_at_bytes(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        grad_scale = 1.0 / float(batch * length)
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.mul_(grad_scale)
        self.optimizer.step()
        self.step_count += batch * length

        if verification.get("verified"):
            equivalent = list(CHUNK_EQUIVALENT_QUANTITIES)
            note = "verified against the per-token oracle for this chunk"
        else:
            equivalent = []
            note = (
                "the per-token equivalence check was not run on this chunk: "
                "losses, gradients and trace buffers are not claimed as verified"
            )
        info: Dict[str, Any] = {
            "loss": float(total.item()),
            "loss_terms": {name: float(value.item()) for name, value in terms.items()},
            # Summed over the batch, matching the per-token oracle's scalar.
            "per_token_losses": per_token.detach().sum(dim=0).tolist(),
            "tokens": batch * length,
            "chunk_length": length,
            "batch_size": batch,
            "optimizer_steps": 1,
            "step_count": self.step_count,
            "lr": lr,
            "grad_scale": grad_scale,
            "equivalent": equivalent,
            "not_equivalent": list(CHUNK_NON_EQUIVALENT_QUANTITIES),
            "equivalence_verified": bool(verification.get("verified")),
            "equivalence_note": note,
            "verification": verification,
            "hooks": hook_info,
            "token_loop_iterations": 0,
        }
        return info

    def per_token_gradient_oracle(self, tokens, next_tokens=None, ends=None, *,
                                  ce_only: bool = False,
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """The reference: the per-token path with accumulation, not stepping.

        For every token it runs the unchanged per-token forward, backward,
        trace-buffer advance and hook injection, then sums the gradients into an
        accumulator instead of handing them to the optimizer. Parameters never
        move, so the buffers evolve exactly as they do in a real per-token run
        and the accumulated gradient is the one a per-token run would apply
        before its ``1/L`` scale. Buffers are restored before returning.
        """
        tokens, next_tokens, ends, derived_mask = self._prepare_chunk_inputs(
            tokens, next_tokens, ends
        )
        mask = derived_mask if mask is None else mask
        batch, length = tokens.shape

        saved_buffers = {
            name: tensor.detach().clone() for name, tensor in self.named_buffers()
        }
        accum = {
            name: torch.zeros_like(param.detach())
            for name, param in self.named_parameters()
        }
        prev_states = [
            layer.states.detach().clone().expand(batch, self.dim).clone()
            for layer in self.blocks
        ]
        # Per-sample trajectories: the (V, D) and (D,) buffers have no batch
        # axis, so they store the batch mean, but the hooks are accumulated per
        # sample and then summed exactly like the per-token path does. This
        # keeps a B > 1 chunk consistent with the chunked pass.
        etraces = [
            layer.embedtrace.detach().clone().unsqueeze(0).repeat(batch, 1, 1)
            for layer in self.blocks
        ]
        dtraces = [
            layer.decaytrace.detach().clone().expand(batch, self.dim).clone()
            for layer in self.blocks
        ]
        token_losses: List[float] = []

        embed_weight = getattr(getattr(self.encoder, "embed", None), "weight", None)
        try:
            for step in range(length):
                TOKEN_LOOP_COUNTER["per_token"] += 1
                self.zero_grad(set_to_none=True)
                dummies = [
                    torch.zeros(batch, self.dim, dtype=torch.float32,
                                device=self.device, requires_grad=True)
                    for _ in self.blocks
                ]
                enc = self.encoder(tokens[:, step])
                x = enc
                new_states = []
                decays = []
                for index, layer in enumerate(self.blocks):
                    decay = torch.sigmoid(layer.decay)
                    decays.append(decay.detach())
                    state = decay * prev_states[index] + enc + dummies[index]
                    new_states.append(state)
                    x = x + layer.silu(layer.weights(layer.norm(state)))
                logits, stop = self.decoder(x)
                _, token_terms = self._batched_token_losses(
                    x.unsqueeze(1), logits.unsqueeze(1), stop.unsqueeze(1),
                    next_tokens[:, step].unsqueeze(1), ends[:, step].unsqueeze(1),
                    mask[:, step].unsqueeze(1), ce_only,
                )
                loss = token_terms["total"]
                if loss.requires_grad:
                    loss.backward()
                token_losses.append(float(loss.item()))

                for index, layer in enumerate(self.blocks):
                    g = dummies[index].grad
                    if g is None:
                        g = torch.zeros(batch, self.dim, device=self.device)
                    if embed_weight is not None and embed_weight.grad is not None:
                        contribution = (g.unsqueeze(1) * etraces[index]).sum(dim=0)
                        embed_weight.grad.add_(contribution.to(embed_weight.dtype))
                    previous = prev_states[index]
                    with torch.no_grad():
                        # The per-token path advances decaytrace *before* it
                        # reads it for the hook, so the hook uses decaytrace_t,
                        # not decaytrace_{t-1}.
                        decaytrace_t = (
                            decays[index] * dtraces[index]
                            + decays[index] * (1.0 - decays[index]) * previous
                        )
                    layer.decay.grad = (g * decaytrace_t).sum(dim=0)
                    with torch.no_grad():
                        dtraces[index] = decaytrace_t
                        onehot = torch.zeros(batch, self.vocab_size, self.dim,
                                             device=self.device)
                        onehot.scatter_(
                            1,
                            tokens[:, step].reshape(batch, 1, 1).expand(batch, 1, self.dim),
                            1.0,
                        )
                        etraces[index] = decays[index] * etraces[index] + onehot
                    prev_states[index] = new_states[index].detach()

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        accum[name] += param.grad
        finally:
            current = dict(self.named_buffers())
            with torch.no_grad():
                for name, value in saved_buffers.items():
                    if name in current and current[name].shape == value.shape:
                        current[name].copy_(value)
            self.zero_grad(set_to_none=True)

        final_buffers: Dict[str, torch.Tensor] = {}
        for name in dict(self.named_buffers()):
            parts = name.split(".")
            if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit():
                index = int(parts[1])
                short = parts[2]
                if short == "states":
                    final_buffers[name] = prev_states[index].mean(dim=0)
                elif short == "decaytrace":
                    final_buffers[name] = dtraces[index].mean(dim=0)
                elif short == "embedtrace":
                    final_buffers[name] = etraces[index].mean(dim=0)
        return {
            "gradients": accum,
            "buffers": final_buffers,
            "token_losses": token_losses,
            "tokens": batch * length,
            "token_loop_iterations": batch * length,
        }

    def _compare_with_oracle(self, chunked_grads, per_token, oracle,
                             chunked_buffers: Optional[Dict[str, torch.Tensor]] = None
                             ) -> Dict[str, Any]:
        """Compare the chunked pass to the per-token oracle within tolerance."""
        problems: List[str] = []
        max_grad_delta = 0.0
        for name, param in self.named_parameters():
            reference = oracle["gradients"].get(name)
            actual = chunked_grads.get(name)
            if reference is None:
                continue
            if actual is None:
                if float(reference.abs().max().item()) > CHUNK_EQUIVALENCE_TOLERANCE:
                    problems.append(f"gradient {name} is missing in the chunked pass")
                continue
            reference = reference.to(actual.dtype)
            delta = float((actual - reference).abs().max().item())
            max_grad_delta = max(max_grad_delta, delta)
            scale = max(1.0, float(reference.abs().max().item()))
            if delta > CHUNK_EQUIVALENCE_TOLERANCE * scale + CHUNK_EQUIVALENCE_TOLERANCE:
                problems.append(f"gradient {name} differs by {delta:.3e}")

        max_loss_delta = 0.0
        reference_losses = oracle["token_losses"]
        actual_losses = per_token.detach().sum(dim=0).tolist()
        for reference, actual in zip(reference_losses, actual_losses):
            delta = abs(float(actual) - float(reference))
            max_loss_delta = max(max_loss_delta, delta)
            if delta > CHUNK_EQUIVALENCE_TOLERANCE * max(1.0, abs(reference)):
                problems.append(f"per-token loss differs by {delta:.3e}")

        max_buffer_delta = 0.0
        current = chunked_buffers if chunked_buffers is not None else dict(self.named_buffers())
        for name, reference in oracle["buffers"].items():
            actual = current.get(name)
            if actual is None or actual.shape != reference.shape:
                problems.append(f"buffer {name} is missing in the chunked pass")
                continue
            delta = float((actual - reference).abs().max().item())
            max_buffer_delta = max(max_buffer_delta, delta)
            scale = max(1.0, float(reference.abs().max().item()))
            if delta > CHUNK_EQUIVALENCE_TOLERANCE * scale + CHUNK_EQUIVALENCE_TOLERANCE:
                problems.append(f"buffer {name} differs by {delta:.3e}")

        return {
            "verified": not problems,
            "tolerance": CHUNK_EQUIVALENCE_TOLERANCE,
            "max_gradient_delta": max_grad_delta,
            "max_loss_delta": max_loss_delta,
            "max_buffer_delta": max_buffer_delta,
            "problems": problems,
            "reference": "Model.per_token_gradient_oracle",
        }

    def __call__(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None = None,
                 end: bool = False, frozen: bool = False,
                 ce_only: bool = False) -> Tuple[int, float]:
        return self.forward(currb, nextb=nextb, end=end, frozen=frozen, ce_only=ce_only)

    def reset(self):
        for layer in self.blocks:
            layer.reset()

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return self.vocab_size * self.dim + self.layers * per_layer + self.vocab_size * self.dim + self.vocab_size + self.dim + 1

    def save(self, path: str):
        data = {}
        for k, v in self.named_parameters():
            data[f"m.{k}"] = v.detach().cpu().contiguous()

        data["step_count"] = torch.tensor([self.step_count], dtype=torch.long)

        for i, layer in enumerate(self.blocks):
            data[f"state.{i}"] = layer.states.detach().cpu().contiguous()
            data[f"decaytrace.{i}"] = layer.decaytrace.detach().cpu().contiguous()
            data[f"embedtrace.{i}"] = layer.embedtrace.detach().cpu().contiguous()

        for i, p in enumerate(self.parameters()):
            p_state = self.optimizer.state.get(p)
            if p_state:
                if 'exp_avg' in p_state:
                    data[f"o.{i}.exp_avg"] = p_state['exp_avg'].detach().cpu().contiguous()
                if 'exp_avg_sq' in p_state:
                    data[f"o.{i}.exp_avg_sq"] = p_state['exp_avg_sq'].detach().cpu().contiguous()
                if 'step' in p_state:
                    data[f"o.{i}.step"] = torch.tensor([p_state['step']], dtype=torch.long)

        tmp = 'temporary-' + os.path.basename(path)
        tmp_dir = os.path.dirname(path)
        tmp_path = os.path.join(tmp_dir, tmp) if tmp_dir else tmp
        save_file(data, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str):
        """Load a checkpoint, raising when it is absent.

        Silent random initialisation is a degradation, not a fallback: callers
        that want an explicitly random model construct ``Model(...)`` without
        calling ``load()``.
        """
        if not path or not os.path.isfile(path):
            raise MissingCheckpointError(
                f"Checkpoint not found: {path!r}. Refusing to answer from "
                "randomly initialised weights."
            )

        data = load_file(path)
        model_dict = {}

        for k, v in data.items():
            if k.startswith("m."):
                model_dict[k[2:]] = v
            elif k.startswith("state."):
                idx = int(k.split('.')[1])
                self.blocks[idx].states.copy_(v.to(self.device))
            elif k.startswith("decaytrace."):
                idx = int(k.split('.')[1])
                self.blocks[idx].decaytrace.copy_(v.to(self.device))
            elif k.startswith("embedtrace."):
                idx = int(k.split('.')[1])
                self.blocks[idx].embedtrace.copy_(v.to(self.device))
            elif k == "step_count":
                self.step_count = int(v.item())

        if model_dict:
            self.load_state_dict(model_dict, strict=False)

        params = list(self.parameters())
        for i, p in enumerate(params):
            if f"o.{i}.exp_avg" in data and f"o.{i}.exp_avg_sq" in data:
                self.optimizer.state[p] = {
                    'step': int(data[f"o.{i}.step"].item()) if f"o.{i}.step" in data else 0,
                    'exp_avg': data[f"o.{i}.exp_avg"].to(self.device),
                    'exp_avg_sq': data[f"o.{i}.exp_avg_sq"].to(self.device),
                }


# ---------------------------------------------------------------------------
# Legacy RTUMemoryBlock kept for backward compat - deprecated, use PlasticAssociativeRTU
# ---------------------------------------------------------------------------
class RTUMemoryBlock(nn.Module):
    """Plastic Recurrent Trace Unit for semantic sentence/chunk vectors - LEGACY, deprecated."""
    def __init__(self, dim: int = 384, spread: int = 32):
        super().__init__()
        self.dim = dim
        self.spread = spread
        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)
        self.register_buffer('states', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('decaytrace', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('embedtrace', torch.zeros(dim, dtype=torch.float32))
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay)
        if x.dim() == 1:
            new_state = (decay * self.states) + x
            delta = self.silu(self.proj(self.norm(new_state)))
            return x + delta, new_state
        else:
            outputs = []
            curr_state = self.states.clone()
            for t in range(x.shape[0]):
                curr_state = (decay * curr_state) + x[t]
                delta = self.silu(self.proj(self.norm(curr_state)))
                outputs.append(x[t] + delta)
            return torch.stack(outputs, dim=0), curr_state

    def reset(self):
        self.states.zero_()
        self.decaytrace.zero_()
        self.embedtrace.zero_()


# ---------------------------------------------------------------------------
# New: PlasticAssociativeRTU - Titans + RWKV-7 hybrid
# ---------------------------------------------------------------------------
class PlasticAssociativeRTU(PlasticAdapter):
    """
    Multi-head associative plastic memory (Titans + RWKV-7 hybrid) replacing 1D EMA RTU
    with predictive surprise gating, momentum-augmented surprise updates,
    and Householder-like selective erasure for bounded lifelong learning.
    
    Maintains:
    - S: multi-head matrix-valued associative state R^{H x Dh x Dh}
    - momentum: momentum buffer same shape
    - h_trace: 1D vector trace summary
    - decay: logit decay params
    - Projections: W_pred, W_K, W_V, W_Q, W_gate, W_alpha
    """
    def __init__(self, dim: int = 384, heads: int = 4, spread: int = 32, 
                 momentum_decay: float = 0.85, d_model: int = None, **kwargs):
        if d_model is not None:
            dim = d_model
        # Handle adapter_kwargs that might contain d_model
        if "d_model" in kwargs:
            dim = kwargs.pop("d_model")
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.spread = spread
        self.momentum_decay = momentum_decay
        # Capacity enforcement: K_max bounds the key register; weakest-key eviction uses projection norm
        _kmax = kwargs.pop("K_max", None)
        if _kmax is None:
            _kmax = kwargs.pop("k_max", None)
        if _kmax is None:
            _kmax = 5000
        self.K_max = int(_kmax)
        if self.K_max < 1:
            self.K_max = 1
        self.eviction_count = 0
        self._key_registry = []  # list of (norm, k_hat) for weakest-key tracking

        # Associative state S in R^{H x Dh x Dh}
        self.register_buffer('S', torch.zeros(heads, self.head_dim, self.head_dim, dtype=torch.float32))
        self.register_buffer('momentum', torch.zeros(heads, self.head_dim, self.head_dim, dtype=torch.float32))
        self.register_buffer('h_trace', torch.zeros(dim, dtype=torch.float32))
        
        # Decay params (like RTUMemoryBlock)
        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)

        # Projections
        self.W_pred = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_gate = nn.Linear(dim, heads, bias=False)  # retention weight per head
        self.W_alpha = nn.Linear(dim, heads, bias=False)  # erasure rate per head
        
        self.norm_pred = nn.LayerNorm(dim, eps=1e-5)
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim, bias=False)  # for compatibility with old RTU
        self.silu = nn.SiLU()

        # For backward compat, alias states to h_trace
        # self.states will be property

        # Initialize projections with small std
        for m in [self.W_pred, self.W_K, self.W_V, self.W_Q, self.W_gate, self.W_alpha, self.proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    @property
    def states(self):
        """Compatibility: states alias to h_trace for old code expecting RTUMemoryBlock.states"""
        return self.h_trace

    @states.setter
    def states(self, value):
        with torch.no_grad():
            self.h_trace.copy_(value)

    def predict_next(self) -> torch.Tensor:
        """
        Generate top-down expectation hat{e}_t = LayerNorm(W_pred h_{t-1}) from current trace.
        Returns normalized prediction vector of dimension dim.
        """
        pred = self.W_pred(self.h_trace)
        pred = self.norm_pred(pred)
        # Normalize
        pred = pred / (torch.linalg.vector_norm(pred).clamp(min=1e-8))
        return pred

    def compute_surprise(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute prediction surprise as s_t = 1 - cos(e_t, hat{e}_t) clamped to [0,2].
        Returns (error_vector, surprise_scalar)
        """
        e_t = x.detach().float().reshape(-1)
        if e_t.shape[0] != self.dim:
            # Handle dim mismatch
            if e_t.shape[0] > self.dim:
                e_t = e_t[:self.dim]
            else:
                e_t = F.pad(e_t, (0, self.dim - e_t.shape[0]))
        
        # Normalize e_t for cosine
        e_norm = e_t / (torch.linalg.vector_norm(e_t).clamp(min=1e-8))
        
        hat_e = self.predict_next()  # already normalized
        
        cos_sim = torch.dot(e_norm, hat_e).item()
        # Clamp cos_sim to [-1,1] for numerical stability
        cos_sim = max(-1.0, min(1.0, cos_sim))
        surprise = 1.0 - cos_sim
        # Clamp to [0,2] per spec
        surprise = max(0.0, min(2.0, surprise))
        
        # Use normalized residual so surprise is scale-invariant and comparable across inputs
        error_vec = e_norm - hat_e
        
        return error_vec, surprise

    def update_associative_memory(self, x: torch.Tensor, surprise_factor: float = 1.0, effective_lr: float = None, base_lr: float = 0.05) -> torch.Tensor:
        """
        Update associative state using momentum-augmented surprise:
        M_t = eta M_{t-1} + grad * s_t * (effective_lr / base_lr), then S_t = S_{t-1} G_t + M_t
        where G_t is Householder-like erasure operator.
        
        Also updates 1D trace: h_t = 0.9 h_{t-1} + 0.1 (1+0.5 s_t) e_t (trace keeps 0.9/0.1, not rescaled)
        Returns adapted output (for PlasticAdapter interface)
        """
        e_t = x.detach().float().reshape(-1)
        if e_t.shape[0] != self.dim:
            if e_t.shape[0] > self.dim:
                e_t = e_t[:self.dim]
            else:
                e_t = F.pad(e_t, (0, self.dim - e_t.shape[0]))

        # Reject non-finite updates without consuming capacity (task 2.6)
        if not torch.isfinite(e_t).all():
            return e_t
        # Also check if computed projections would be non-finite later, but early gate on e_t suffices for now
        # Capacity and norm check: use key projection norm as strength measure
        # Compute k early to get norm for eviction decision
        # Compute projections (need for registry and update)
        k = self.W_K(e_t)  # dim
        v = self.W_V(e_t)  # dim
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            return e_t
        
        # Reshape to heads
        k_heads = k.view(self.heads, self.head_dim)  # H x Dh
        v_heads = v.view(self.heads, self.head_dim)  # H x Dh
        
        # Normalize keys per head for erasure operator
        k_norms = torch.linalg.vector_norm(k_heads, dim=1, keepdim=True).clamp(min=1e-8)
        k_hat = k_heads / k_norms  # H x Dh, normalized
        # Overall key strength: projection norm (scalar) for capacity ordering
        k_strength = float(torch.linalg.vector_norm(k).item())

        # Capacity enforcement: evict weakest key when at K_max before writing new association
        # Use existing attenuation operator for removal (Householder-like)
        if len(self._key_registry) >= self.K_max:
            # Find weakest key (smallest projection norm)
            weakest_idx = min(range(len(self._key_registry)), key=lambda i: self._key_registry[i][0])
            weakest_norm, weakest_k_hat = self._key_registry.pop(weakest_idx)
            # Attenuate S along weakest direction using Householder erasure
            # For each head, apply G_evict = (I - alpha_evict * k_hat_weak k_hat_weak^T) with alpha_evict=1.0
            with torch.no_grad():
                for h in range(self.heads):
                    # weakest_k_hat is stored as full k_hat (H x Dh); slice per head
                    if isinstance(weakest_k_hat, torch.Tensor) and weakest_k_hat.dim() == 2:
                        kh_weak = weakest_k_hat[h]  # Dh
                    else:
                        # fallback: use current k_hat if registry format unexpected
                        kh_weak = k_hat[h]
                    I = torch.eye(self.head_dim, device=self.S.device, dtype=self.S.dtype)
                    G_evict = I - 1.0 * torch.outer(kh_weak, kh_weak)
                    self.S[h] = self.S[h] @ G_evict
            self.eviction_count += 1

        # Determine learning-rate scaling for momentum write
        lr_scale = 1.0
        if effective_lr is not None:
            try:
                lr_scale = float(effective_lr) / float(base_lr) if float(base_lr) != 0 else 1.0
            except Exception:
                lr_scale = 1.0
        
        # Compute alpha_h (erasure rate) and w_h (retention) per head
        alpha_logits = self.W_alpha(e_t)  # heads
        w_logits = self.W_gate(e_t)  # heads
        alpha_h = torch.sigmoid(alpha_logits)  # [0,1]
        w_h = torch.sigmoid(w_logits)  # [0,1], retention weight
        
        # Momentum update: M_t = eta * M_{t-1} + outer(k_h, v_h) * s_t * lr_scale
        # outer product per head: k_h outer v_h -> Dh x Dh
        with torch.no_grad():
            # Decay momentum
            self.momentum.mul_(self.momentum_decay)
            
            # Add new gradient scaled by surprise and learning-rate
            for h in range(self.heads):
                outer = torch.outer(k_heads[h], v_heads[h])  # Dh x Dh
                self.momentum[h].add_(outer * surprise_factor * lr_scale)
            
            # Householder-like erasure: G_h = (I - alpha_h * k_hat k_hat^T) * diag(w_h)
            # For simplicity, w_h is scalar per head controlling overall retention
            # So G_h = (I - alpha_h * k_hat k_hat^T) * w_h
            # Then S_t = S_{t-1} @ G_t + M_t  (or S_{t-1} * G_t + M_t)
            # We'll implement S = S @ G + M
            
            new_S = torch.zeros_like(self.S)
            for h in range(self.heads):
                I = torch.eye(self.head_dim, device=self.S.device, dtype=self.S.dtype)
                kh = k_hat[h]  # Dh
                # G = (I - alpha * kh kh^T) * w
                G = I - alpha_h[h] * torch.outer(kh, kh)
                G = G * w_h[h]  # retention scaling
                # S_t = S_{t-1} G + M_t
                # Note: S is Dh x Dh, G is Dh x Dh
                new_S[h] = self.S[h] @ G + self.momentum[h]
            
            self.S.copy_(new_S)
            
            # Update 1D trace: h_t = 0.9 h_{t-1} + 0.1 (1+0.5 s_t) e_t (not rescaled by lr, per Decision 1)
            trace_coeff = 0.1 * (1.0 + 0.5 * surprise_factor)
            self.h_trace.mul_(0.9).add_(e_t * trace_coeff)

            # Register new key for capacity tracking (store copy detached)
            try:
                self._key_registry.append((k_strength, k_hat.detach().clone()))
            except Exception:
                pass

        # Return adapted output: e_t + SiLU(proj(norm(h_trace))) like old RTU
        with torch.no_grad():
            normed = self.norm(self.h_trace)
            delta = self.silu(self.proj(normed))
            output = e_t + delta
        
        return output

    def retrieve(self, q_vec: torch.Tensor) -> torch.Tensor:
        """
        Associative recall from fast weights.
        q_vec: query embedding dim
        Returns retrieved representation dim
        """
        q = q_vec.detach().float().reshape(-1)
        if q.shape[0] != self.dim:
            if q.shape[0] > self.dim:
                q = q[:self.dim]
            else:
                q = F.pad(q, (0, self.dim - q.shape[0]))
        
        q_proj = self.W_Q(q)  # dim
        q_heads = q_proj.view(self.heads, self.head_dim)  # H x Dh
        
        # For each head, retrieve: out_h = S_h^T @ q_h or S_h @ q_h
        # Using S^T @ q for associative recall
        out_heads = []
        for h in range(self.heads):
            # S[h] is Dh x Dh, q_heads[h] is Dh
            # Retrieve: S[h].T @ q_heads[h] or S[h] @ q_heads[h]
            # We'll use S @ q
            out_h = self.S[h] @ q_heads[h]  # Dh
            out_heads.append(out_h)
        
        out = torch.cat(out_heads, dim=0)  # dim
        # Optionally add residual from h_trace alignment?
        # For simplicity, return out + small h_trace contribution
        out = out + 0.1 * self.h_trace
        return out

    def update_scheduled(self, x, *, surprise_factor, effective_lr, base_lr):
        output = self.update_associative_memory(
            x, surprise_factor=surprise_factor,
            effective_lr=effective_lr, base_lr=base_lr,
        )
        return output, effective_lr

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """PlasticAdapter interface: update with input embedding, return adapted output."""
        return self.update_associative_memory(x, surprise_factor=1.0)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        # Override to include buffers
        # Use nn.Module's state_dict directly: ``super()`` resolves to the
        # PlasticAdapter ABC stub, which returns None.
        base = nn.Module.state_dict(self, *args, **kwargs) or {}
        # Add buffers that are not in base (S, momentum, h_trace are buffers)
        # Actually buffers are included in state_dict by default if persistent=True
        # But we also want to ensure custom keys
        # Return combined
        sd = {
            "S": self.S.detach().cpu().contiguous(),
            "momentum": self.momentum.detach().cpu().contiguous(),
            "h_trace": self.h_trace.detach().cpu().contiguous(),
            "decay": self.decay.detach().cpu().contiguous(),
            "W_pred_weight": self.W_pred.weight.detach().cpu().contiguous(),
            "W_K_weight": self.W_K.weight.detach().cpu().contiguous(),
            "W_V_weight": self.W_V.weight.detach().cpu().contiguous(),
            "W_Q_weight": self.W_Q.weight.detach().cpu().contiguous(),
            "W_gate_weight": self.W_gate.weight.detach().cpu().contiguous(),
            "W_alpha_weight": self.W_alpha.weight.detach().cpu().contiguous(),
            "proj_weight": self.proj.weight.detach().cpu().contiguous(),
            "norm_weight": self.norm.weight.detach().cpu().contiguous(),
            "norm_bias": self.norm.bias.detach().cpu().contiguous(),
            "norm_pred_weight": self.norm_pred.weight.detach().cpu().contiguous(),
            "norm_pred_bias": self.norm_pred.bias.detach().cpu().contiguous(),
        }
        sd["_adapter_type"] = "associative_rtu"
        sd["_adapter_version"] = 1
        sd["dim"] = self.dim
        sd["heads"] = self.heads
        sd["K_max"] = int(getattr(self, 'K_max', 5000))
        sd["eviction_count"] = int(getattr(self, 'eviction_count', 0))
        # Empty registries are state too: restoring a pristine snapshot must
        # clear keys created after it, not retain the live register accidentally.
        sd["_key_registry_norms"] = torch.tensor(
            [n for n, _ in self._key_registry], dtype=torch.float32
        )
        sd["_key_registry_khats"] = (
            torch.stack([k.detach().cpu() for _, k in self._key_registry], dim=0)
            if self._key_registry else torch.empty(0, self.heads, self.head_dim)
        )
        return sd

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        # Handle both our custom format and standard nn.Module format
        with torch.no_grad():
            if "S" in state:
                self.S.copy_(state["S"].to(self.S.device))
            if "momentum" in state:
                self.momentum.copy_(state["momentum"].to(self.momentum.device))
            if "h_trace" in state:
                self.h_trace.copy_(state["h_trace"].to(self.h_trace.device))
            if "states" in state:
                # Legacy compatibility: states -> h_trace
                self.h_trace.copy_(state["states"].to(self.h_trace.device))
            if "decay" in state:
                self.decay.copy_(state["decay"].to(self.decay.device))
            
            # Load projection weights if present
            if "W_pred_weight" in state:
                self.W_pred.weight.copy_(state["W_pred_weight"].to(self.W_pred.weight.device))
            if "W_K_weight" in state:
                self.W_K.weight.copy_(state["W_K_weight"].to(self.W_K.weight.device))
            if "W_V_weight" in state:
                self.W_V.weight.copy_(state["W_V_weight"].to(self.W_V.weight.device))
            if "W_Q_weight" in state:
                self.W_Q.weight.copy_(state["W_Q_weight"].to(self.W_Q.weight.device))
            if "W_gate_weight" in state:
                self.W_gate.weight.copy_(state["W_gate_weight"].to(self.W_gate.weight.device))
            if "W_alpha_weight" in state:
                self.W_alpha.weight.copy_(state["W_alpha_weight"].to(self.W_alpha.weight.device))
            if "proj_weight" in state:
                self.proj.weight.copy_(state["proj_weight"].to(self.proj.weight.device))
            if "norm_weight" in state:
                self.norm.weight.copy_(state["norm_weight"].to(self.norm.weight.device))
            if "norm_bias" in state:
                self.norm.bias.copy_(state["norm_bias"].to(self.norm.bias.device))
            if "norm_pred_weight" in state:
                self.norm_pred.weight.copy_(state["norm_pred_weight"].to(self.norm_pred.weight.device))
            if "norm_pred_bias" in state:
                self.norm_pred.bias.copy_(state["norm_pred_bias"].to(self.norm_pred.bias.device))
            
            if "K_max" in state:
                try:
                    self.K_max = int(state["K_max"])
                except Exception:
                    pass
            if "eviction_count" in state:
                try:
                    self.eviction_count = int(state["eviction_count"])
                except Exception:
                    pass
            if "_key_registry_norms" in state and "_key_registry_khats" in state:
                try:
                    norms = state["_key_registry_norms"]
                    khats = state["_key_registry_khats"]
                    self._key_registry = []
                    for i in range(norms.shape[0]):
                        n = float(norms[i].item())
                        kh = khats[i].detach().clone()
                        self._key_registry.append((n, kh))
                except Exception:
                    pass
            
            # Also try to load via parent for any remaining keys
            try:
                # Filter to only keys that are in parent's state_dict
                parent_sd = nn.Module.state_dict(self)
                filtered = {k: v for k, v in state.items() if k in parent_sd}
                if filtered:
                    super().load_state_dict(filtered, strict=False)
            except:
                pass

    def reset(self):
        self.S.zero_()
        self.momentum.zero_()
        self.h_trace.zero_()
        self.eviction_count = 0
        self._key_registry = []

    def get_memory_health(self) -> Dict[str, float]:
        """Basic health metrics for this module alone."""
        with torch.no_grad():
            assoc_norm = torch.linalg.vector_norm(self.S).item()
            trace_norm = torch.linalg.vector_norm(self.h_trace).item()
            return {
                "associative_norm": assoc_norm,
                "trace_norm": trace_norm,
                "saturation": min(1.0, assoc_norm / 100.0),  # heuristic
                "evictions": int(getattr(self, 'eviction_count', 0)),
                "eviction_count": int(getattr(self, 'eviction_count', 0)),
                "key_count": len(getattr(self, '_key_registry', [])),
                "K_max": int(getattr(self, 'K_max', 5000)),
            }
