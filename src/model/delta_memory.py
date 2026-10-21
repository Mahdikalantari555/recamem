"""Gated Delta-Rule plastic memory and sparse high-dimensional coding.

This module implements the ``adaptation/delta-memory`` and
``adaptation/sparse-coding`` capabilities as additive alternatives to the
existing ``associative_rtu`` adapter:

* :class:`DeltaPlasticAdapter` writes only the residual between the observed
  value and the value already associated with the (normalised) key, gated per
  head by a learned sigmoid.  It exposes a Householder-based targeted
  :meth:`erase` and versioned persistence.
* :class:`SparseCoder` encodes 384-D anchor embeddings into a 2048-D code with
  a deterministic TopK-128 activation budget.
* :class:`SparseDeltaAdapter` / :class:`FlyModelAdapter` compose the coder with
  a Delta adapter on the sparse code while keeping the ``PlasticAdapter``
  contract.

Nothing here changes the behaviour of an existing ``associative_rtu`` profile:
new adapter names must be requested explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.plastic_adapter import PlasticAdapter

# Format version written into every state dictionary. Readers reject anything
# newer instead of silently reinterpreting an unknown layout.
DELTA_STATE_VERSION = 1
SPARSE_STATE_VERSION = 1

# 128 active entries out of 2048 is exactly 93.75% sparsity.
SPARSE_INPUT_DIM = 384
SPARSE_CODE_DIM = 2048
SPARSE_TOP_K = 128


class AdapterStateError(ValueError):
    """Serialized adapter state is incompatible with this adapter instance."""


def _householder(k: torch.Tensor) -> torch.Tensor:
    """Return an orthogonal ``H`` with ``H @ k == e0`` for unit vector ``k``."""
    dim = k.shape[0]
    e0 = torch.zeros_like(k)
    e0[0] = 1.0
    v = k - e0
    v_norm_sq = torch.dot(v, v)
    if float(v_norm_sq) < 1e-12:
        # ``k`` already points at e0 (or its negation): reflection is identity
        # up to the sign already encoded in the stored key.
        return torch.eye(dim, dtype=k.dtype, device=k.device)
    return torch.eye(dim, dtype=k.dtype, device=k.device) - 2.0 * torch.outer(v, v) / v_norm_sq


def _normalized(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v).clamp(min=eps)


class SparseCoder(nn.Module):
    """384 -> 2048 expansion with a deterministic TopK-128 code budget."""

    def __init__(
        self,
        input_dim: int = SPARSE_INPUT_DIM,
        code_dim: int = SPARSE_CODE_DIM,
        top_k: int = SPARSE_TOP_K,
        seed: int = 0,
    ):
        super().__init__()
        if input_dim <= 0 or code_dim <= 0:
            raise ValueError("input_dim and code_dim must be positive")
        if not 0 < top_k <= code_dim:
            raise ValueError(f"top_k must be in (0, {code_dim}], got {top_k}")
        self.input_dim = int(input_dim)
        self.code_dim = int(code_dim)
        self.top_k = int(top_k)
        generator = torch.Generator().manual_seed(int(seed))
        encoder = torch.empty(self.code_dim, self.input_dim)
        nn.init.normal_(encoder, mean=0.0, std=1.0 / math.sqrt(self.input_dim), generator=generator)
        self.encoder = nn.Linear(self.input_dim, self.code_dim, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(encoder)
        # Tied decoder keeps the parameter count and reconstruction meaningful.
        self.decoder = nn.Linear(self.code_dim, self.input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(encoder.T)

    # -- encoding -------------------------------------------------------
    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).detach().float().reshape(-1)
        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"SparseCoder expects a {self.input_dim}-D input, got {x.shape[0]}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("SparseCoder input must be finite")
        return x

    @staticmethod
    def topk_mask(activations: torch.Tensor, k: int) -> torch.Tensor:
        """Boolean mask of the ``k`` largest-magnitude entries.

        Ties are resolved by ascending feature index through a stable sort, so
        the code is reproducible for identical inputs on every run.
        """
        magnitudes = activations.detach().abs()
        order = torch.argsort(magnitudes, descending=True, stable=True)
        mask = torch.zeros_like(magnitudes, dtype=torch.bool)
        mask[order[:k]] = True
        return mask

    def encode(self, x: torch.Tensor, return_activations: bool = False):
        x = self._validate_input(x)
        activations = F.relu(self.encoder(x))
        if not bool((activations > 0).any()):
            # A degenerate all-zero projection would leave no active entries.
            # Keep the TopK contract by selecting the largest raw projections.
            activations = self.encoder(x)
        mask = self.topk_mask(activations, self.top_k)
        code = torch.where(mask, activations, torch.zeros_like(activations))
        if return_activations:
            return code, activations
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        code = torch.as_tensor(code).detach().float().reshape(-1)
        if code.shape[0] != self.code_dim:
            raise ValueError(
                f"SparseCoder expects a {self.code_dim}-D code, got {code.shape[0]}"
            )
        return self.decoder(code)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> Dict[str, float]:
        x = self._validate_input(x)
        recon = self.reconstruct(x)
        mse = float(torch.mean((recon - x) ** 2).item())
        cos = float(F.cosine_similarity(recon.unsqueeze(0), x.unsqueeze(0)).item())
        return {"mse": mse, "cosine": cos, "input_dim": self.input_dim,
                "code_dim": self.code_dim, "top_k": self.top_k}

    # -- diagnostics ----------------------------------------------------
    @property
    def sparsity(self) -> float:
        return 1.0 - (self.top_k / self.code_dim)

    def active_count(self, code: torch.Tensor) -> int:
        return int((code != 0).sum().item())

    def utilization(self, code: torch.Tensor) -> Dict[str, float]:
        code = code.reshape(-1)
        return {
            "active": self.active_count(code),
            "code_dim": self.code_dim,
            "sparsity": self.sparsity,
            "zero_ratio": float((code == 0).float().mean().item()),
            "l1": float(code.abs().sum().item()),
        }

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "coder_encoder_weight": self.encoder.weight.detach().cpu().contiguous(),
            "coder_decoder_weight": self.decoder.weight.detach().cpu().contiguous(),
            "coder_decoder_bias": self.decoder.bias.detach().cpu().contiguous(),
            "coder_input_dim": int(self.input_dim),
            "coder_code_dim": int(self.code_dim),
            "coder_top_k": int(self.top_k),
            "_coder_version": SPARSE_STATE_VERSION,
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        version = state.get("_coder_version", SPARSE_STATE_VERSION)
        if int(version) > SPARSE_STATE_VERSION:
            raise AdapterStateError(
                f"SparseCoder state version {version} is newer than supported "
                f"{SPARSE_STATE_VERSION}"
            )
        input_dim = int(state.get("coder_input_dim", self.input_dim))
        code_dim = int(state.get("coder_code_dim", self.code_dim))
        top_k = int(state.get("coder_top_k", self.top_k))
        if (input_dim, code_dim) != (self.input_dim, self.code_dim):
            raise AdapterStateError(
                "SparseCoder state shape "
                f"({input_dim}->{code_dim}) does not match configured "
                f"({self.input_dim}->{self.code_dim})"
            )
        with torch.no_grad():
            self.encoder.weight.copy_(state["coder_encoder_weight"].to(self.encoder.weight.device))
            self.decoder.weight.copy_(state["coder_decoder_weight"].to(self.decoder.weight.device))
            self.decoder.bias.copy_(state["coder_decoder_bias"].to(self.decoder.bias.device))
        self.top_k = top_k
        return self


class DeltaPlasticAdapter(PlasticAdapter):
    """Multi-head gated Delta-Rule associative memory.

    With ``S`` keyed by normalised keys, the read for a query ``q`` is
    ``S_h^T q_h`` and the update is

        ``S_h <- (1 - beta_h) S_h + beta_h * outer(k_h, v_h - S_h^T k_h)``

    which corrects an existing association instead of accumulating an
    uncorrected outer product.
    """

    _ADAPTER_VERSION = DELTA_STATE_VERSION

    def __init__(
        self,
        dim: int = 384,
        heads: int = 4,
        spread: int = 32,
        K_max: int = 5000,
        beta_bias: float = 0.0,
        normalize_keys: bool = True,
        adapter_type: str = "delta_rtu",
        **kwargs,
    ):
        if "d_model" in kwargs:
            dim = kwargs.pop("d_model")
        kwargs.pop("k_max", None)
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0 or dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.spread = int(spread)
        self.K_max = max(1, int(K_max))
        self.normalize_keys = bool(normalize_keys)
        self._adapter_type = adapter_type
        self._adapter_version = self._ADAPTER_VERSION
        self.eviction_count = 0
        self._key_registry = []  # list[(strength: float, k_hat: Tensor)]

        self.register_buffer(
            "S", torch.zeros(self.heads, self.head_dim, self.head_dim, dtype=torch.float32)
        )
        self.register_buffer("h_trace", torch.zeros(self.dim, dtype=torch.float32))

        halflives = torch.exp(torch.linspace(0.0, math.log(float(max(2, spread))), self.dim))
        retention = torch.exp(-math.log(2.0) / halflives)
        self.decay = nn.Parameter(torch.log(retention) - torch.log1p(-retention))

        self.W_K = nn.Linear(self.dim, self.dim, bias=False)
        self.W_V = nn.Linear(self.dim, self.dim, bias=False)
        self.W_pred = nn.Linear(self.dim, self.dim, bias=False)
        self.W_beta = nn.Linear(self.dim, self.heads, bias=True)
        self.norm = nn.LayerNorm(self.dim, eps=1e-5)
        self.norm_pred = nn.LayerNorm(self.dim, eps=1e-5)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)

        for module in (self.W_K, self.W_V, self.W_pred, self.proj):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.dim))
        # A deterministic, small default gate: writes are correctable without
        # immediately saturating the state.
        with torch.no_grad():
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(float(beta_bias))

    # -- PlasticAdapter contract ---------------------------------------
    @property
    def states(self) -> torch.Tensor:
        return self.h_trace

    @states.setter
    def states(self, value: torch.Tensor):
        with torch.no_grad():
            self.h_trace.copy_(value)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).detach().float().reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"DeltaPlasticAdapter expects a {self.dim}-D input, got {x.shape[0]}")
        return x

    def _heads_of(self, projection: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return projection(x).view(self.heads, self.head_dim)

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head bounded write/forget gate ``beta = sigmoid(W_beta x)``."""
        x = self._prepare(x)
        return torch.sigmoid(self.W_beta(x))

    def beta(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`gates` used by diagnostics and tests."""
        return self.gates(x)

    def _read(self, k_hat: torch.Tensor) -> torch.Tensor:
        """Value currently associated with normalised per-head keys."""
        return torch.stack([self.S[h].T @ k_hat[h] for h in range(self.heads)])

    def _unit_keys(self, k: torch.Tensor) -> torch.Tensor:
        if not self.normalize_keys:
            return k
        return torch.stack([_normalized(k[h]) for h in range(self.heads)])

    def _write(self, x: torch.Tensor, beta: Optional[torch.Tensor] = None,
               surprise_factor: float = 1.0) -> torch.Tensor:
        x = self._prepare(x)
        if not torch.isfinite(x).all():
            return x
        k = self._heads_of(self.W_K, x)
        v = self._heads_of(self.W_V, x)
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            return x
        k_hat = self._unit_keys(k)
        if beta is None:
            beta = self.gates(x)
        self._apply_delta(k_hat, v, beta, surprise_factor)
        with torch.no_grad():
            self.h_trace.mul_(0.9).add_(
                x * (0.1 * (1.0 + 0.5 * float(surprise_factor)))
            )
        return self._adapt(x)

    def _apply_delta(self, k_hat: torch.Tensor, v: torch.Tensor,
                     beta: torch.Tensor, surprise_factor: float = 0.0) -> None:
        """One gated residual write for pre-projected keys and values."""
        beta = torch.as_tensor(beta).detach().reshape(-1).clamp(0.0, 1.0)
        if beta.shape[0] != self.heads:
            raise ValueError(
                f"beta must have one gate per head ({self.heads}), got {beta.shape[0]}"
            )
        with torch.no_grad():
            predicted = self._read(k_hat)
            self._enforce_capacity(k_hat)
            for h in range(self.heads):
                b = float(beta[h].item())
                if b <= 1e-12:
                    continue
                # Gated Delta Rule in its rank-one form:
                #   S <- S - beta * k (k^T S) + beta * k v^T
                #     = S + beta * outer(k, v - S^T k)
                # The retention term is selective: only the component along the
                # written key is decayed, so repeating an unchanged write is a
                # no-op instead of erasing the association.
                self.S[h].sub_(b * torch.outer(k_hat[h], predicted[h]))
                self.S[h].add_(b * torch.outer(k_hat[h], v[h]))
            self._remember_key(k_hat)
        del surprise_factor  # the trace is the caller's responsibility

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            delta = F.silu(self.proj(self.norm(self.h_trace)))
        return x + delta

    def write_association(self, key: torch.Tensor, value: torch.Tensor,
                          beta: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Write ``key -> value`` through the gated Delta rule.

        ``beta`` overrides the learned gate (one value per head, or a scalar)
        so no-op and saturated writes can be tested deterministically.
        """
        key = self._prepare(key)
        value = self._prepare(value)
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("key and value must be finite")
        k_hat = self._unit_keys(self._heads_of(self.W_K, key))
        v = self._heads_of(self.W_V, value)
        if beta is None:
            beta = self.gates(key)
        gates = torch.as_tensor(beta).detach().reshape(-1).clamp(0.0, 1.0)
        if gates.numel() == 1:
            gates = gates.repeat(self.heads)
        self._apply_delta(k_hat, v, gates)
        return {
            "status": "written",
            "beta": [float(b) for b in gates.tolist()],
            "state_norm": float(torch.linalg.vector_norm(self.S).item()),
        }

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """Memory read for ``key`` without the trace residual."""
        key = self._prepare(key)
        k_hat = self._unit_keys(self._heads_of(self.W_K, key))
        return self._read(k_hat).reshape(-1)

    def set_beta_constant(self, value: float) -> None:
        """Pin every head's gate to ``value`` (diagnostic/test hook)."""
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {value}")
        logit = math.log(1e-12) if value == 0.0 else (
            math.log(1e12) if value == 1.0 else math.log(value / (1.0 - value))
        )
        with torch.no_grad():
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(logit)

    def _remember_key(self, k_hat: torch.Tensor) -> None:
        strength = float(torch.linalg.vector_norm(k_hat).item())
        self._key_registry.append((strength, k_hat.detach().clone()))

    def _enforce_capacity(self, k_hat: torch.Tensor) -> None:
        if len(self._key_registry) < self.K_max:
            return
        weakest = min(range(len(self._key_registry)), key=lambda i: self._key_registry[i][0])
        _norm, weak_key = self._key_registry.pop(weakest)
        # Selective attenuation of the evicted direction, not a global decay.
        for h in range(self.heads):
            key = _normalized(weak_key[h]) if self.normalize_keys else weak_key[h]
            self._attenuate_head(h, key, gamma=1.0)
        self.eviction_count += 1

    def _attenuate_head(self, head: int, k_unit: torch.Tensor, gamma: float) -> None:
        """Scale memory components along ``k_unit`` by ``1 - gamma``."""
        if gamma <= 0.0:
            return
        H = _householder(k_unit)
        decay = torch.ones(self.head_dim, dtype=self.S.dtype, device=self.S.device)
        decay[0] = 1.0 - min(1.0, float(gamma))
        D = torch.diag(decay)
        self.S[head] = H.T @ D @ H @ self.S[head]

    def update_associative_memory(
        self,
        x: torch.Tensor,
        surprise_factor: float = 1.0,
        effective_lr: float = None,
        base_lr: float = 0.05,
    ) -> torch.Tensor:
        beta = self.gates(x)
        if effective_lr is not None and base_lr:
            # The requested schedule scales the write gate; the applied gate is
            # still bounded by [0, 1].
            scale = max(0.0, float(effective_lr) / float(base_lr))
            beta = beta * scale
        return self._write(x, beta=beta, surprise_factor=surprise_factor)

    def update_scheduled(self, x, *, surprise_factor, effective_lr, base_lr):
        output = self.update_associative_memory(
            x, surprise_factor=surprise_factor, effective_lr=effective_lr, base_lr=base_lr
        )
        return output, float(effective_lr)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        return self._write(x)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        return self.read(q) + 0.1 * self.h_trace

    def predict_next(self) -> torch.Tensor:
        pred = self.norm_pred(self.W_pred(self.h_trace))
        return _normalized(pred)

    def compute_surprise(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        x = self._prepare(x)
        e_norm = _normalized(x)
        cos = torch.dot(e_norm, self.predict_next().detach()).item()
        cos = max(-1.0, min(1.0, cos))
        surprise = max(0.0, min(2.0, 1.0 - cos))
        return e_norm - self.predict_next().detach(), surprise

    # -- diagnostics ----------------------------------------------------
    def recall_score(self, key: torch.Tensor) -> float:
        """Mean associative read norm across heads for a normalised key."""
        key = self._prepare(key)
        k_hat = self._unit_keys(self._heads_of(self.W_K, key))
        return float(self._read(k_hat).norm(dim=1).mean().item())

    def gate_report(self, x: torch.Tensor) -> Dict[str, Any]:
        beta = self.gates(x)
        return {
            "per_head": [float(b) for b in beta.tolist()],
            "min": float(beta.min().item()),
            "max": float(beta.max().item()),
            "heads": self.heads,
        }

    def erase(self, key: torch.Tensor, strength: float = 1.0) -> Dict[str, Any]:
        """Attenuate the selected key direction via a Householder reflection.

        Only components along the selected (normalised) key direction are
        scaled by ``1 - strength``; unrelated directions are preserved.
        """
        strength = float(strength)
        if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError(f"erase strength must be in [0, 1], got {strength}")
        key = torch.as_tensor(key).detach().float().reshape(-1)
        if key.shape[0] != self.dim:
            raise ValueError(f"erase key must have {self.dim} dimensions, got {key.shape[0]}")
        if not torch.isfinite(key).all():
            raise ValueError("erase key must be finite")

        k_hat = self._unit_keys(self._heads_of(self.W_K, key))
        before = float(self._read(k_hat).norm(dim=1).mean().item())

        # Snapshot independent remembered directions and their reads so the
        # report can quantify collateral change instead of asserting zero.
        controls = []
        remaining = []
        removed = 0
        for norm, stored in self._key_registry:
            per_head_cos = torch.stack(
                [F.cosine_similarity(stored[h], k_hat[h], dim=0) for h in range(self.heads)]
            )
            if float(per_head_cos.mean().item()) > 0.99:
                removed += 1
                continue
            remaining.append((norm, stored))
            controls.append((stored.clone(), self._read(stored).clone()))

        with torch.no_grad():
            for h in range(self.heads):
                if torch.linalg.vector_norm(k_hat[h]) < 1e-8:
                    continue
                self._attenuate_head(h, _normalized(k_hat[h]), gamma=strength)

        after = float(self._read(k_hat).norm(dim=1).mean().item())

        max_relative_change = 0.0
        min_cosine = 1.0
        for stored, read_before in controls:
            read_after = self._read(stored)
            before_norm = float(read_before.norm().item())
            after_norm = float(read_after.norm().item())
            if before_norm > 1e-12:
                max_relative_change = max(
                    max_relative_change, abs(after_norm - before_norm) / before_norm
                )
            cosine = float(
                F.cosine_similarity(read_before.reshape(1, -1), read_after.reshape(1, -1)).item()
            )
            min_cosine = min(min_cosine, cosine)
        self._key_registry = remaining
        return {
            "status": "erased",
            "strength": strength,
            "selected_before": before,
            "selected_after": after,
            "selected_ratio": (after / before) if before > 1e-12 else 0.0,
            "collateral_change": max_relative_change,
            "collateral_min_cosine": min_cosine if controls else 1.0,
            "control_keys": len(controls),
            "keys_removed": removed,
            "heads": self.heads,
        }

    # -- persistence ----------------------------------------------------
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        sd: Dict[str, Any] = {
            "S": self.S.detach().cpu().contiguous(),
            "h_trace": self.h_trace.detach().cpu().contiguous(),
            "decay": self.decay.detach().cpu().contiguous(),
            "W_K_weight": self.W_K.weight.detach().cpu().contiguous(),
            "W_V_weight": self.W_V.weight.detach().cpu().contiguous(),
            "W_pred_weight": self.W_pred.weight.detach().cpu().contiguous(),
            "W_beta_weight": self.W_beta.weight.detach().cpu().contiguous(),
            "W_beta_bias": self.W_beta.bias.detach().cpu().contiguous(),
            "proj_weight": self.proj.weight.detach().cpu().contiguous(),
            "norm_weight": self.norm.weight.detach().cpu().contiguous(),
            "norm_bias": self.norm.bias.detach().cpu().contiguous(),
            "norm_pred_weight": self.norm_pred.weight.detach().cpu().contiguous(),
            "norm_pred_bias": self.norm_pred.bias.detach().cpu().contiguous(),
            "_key_registry_norms": torch.tensor(
                [n for n, _ in self._key_registry], dtype=torch.float32
            ),
            "_key_registry_khats": (
                torch.stack([k.detach().cpu() for _, k in self._key_registry], dim=0)
                if self._key_registry
                else torch.empty(0, self.heads, self.head_dim)
            ),
        }
        sd["_adapter_type"] = self._adapter_type
        sd["_adapter_version"] = self._ADAPTER_VERSION
        sd["dim"] = self.dim
        sd["heads"] = self.heads
        sd["K_max"] = self.K_max
        sd["eviction_count"] = self.eviction_count
        sd["normalize_keys"] = self.normalize_keys
        return sd

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        version = int(state.get("_adapter_version", 1))
        if version > self._ADAPTER_VERSION:
            raise AdapterStateError(
                f"Delta adapter state version {version} is newer than supported "
                f"{self._ADAPTER_VERSION}"
            )
        dim = int(state.get("dim", self.dim))
        heads = int(state.get("heads", self.heads))
        if dim != self.dim or heads != self.heads:
            raise AdapterStateError(
                f"Delta adapter state geometry (dim={dim}, heads={heads}) does not "
                f"match configured (dim={self.dim}, heads={self.heads})"
            )
        if "S" in state:
            tensor = state["S"]
            expected = (self.heads, self.head_dim, self.head_dim)
            if tuple(tensor.shape) != expected:
                raise AdapterStateError(
                    f"associative state shape {tuple(tensor.shape)} != expected {expected}"
                )
        with torch.no_grad():
            for name, target in (
                ("S", self.S),
                ("h_trace", self.h_trace),
                ("decay", self.decay),
                ("W_K_weight", self.W_K.weight),
                ("W_V_weight", self.W_V.weight),
                ("W_pred_weight", self.W_pred.weight),
                ("W_beta_weight", self.W_beta.weight),
                ("W_beta_bias", self.W_beta.bias),
                ("proj_weight", self.proj.weight),
                ("norm_weight", self.norm.weight),
                ("norm_bias", self.norm.bias),
                ("norm_pred_weight", self.norm_pred.weight),
                ("norm_pred_bias", self.norm_pred.bias),
            ):
                if name not in state:
                    continue
                value = state[name]
                if tuple(value.shape) != tuple(target.shape):
                    raise AdapterStateError(
                        f"{name} shape {tuple(value.shape)} != expected {tuple(target.shape)}"
                    )
                target.copy_(value.to(target.device))
            self.K_max = max(1, int(state.get("K_max", self.K_max)))
            self.eviction_count = int(state.get("eviction_count", 0))
            if "_key_registry_norms" in state and "_key_registry_khats" in state:
                norms = state["_key_registry_norms"].reshape(-1)
                khats = state["_key_registry_khats"]
                self._key_registry = [
                    (float(norms[i].item()), khats[i].detach().clone())
                    for i in range(int(norms.shape[0]))
                ]
        return self

    def reset(self) -> None:
        self.S.zero_()
        self.h_trace.zero_()
        self.eviction_count = 0
        self._key_registry = []

    def get_memory_health(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "associative_norm": float(torch.linalg.vector_norm(self.S).item()),
                "trace_norm": float(torch.linalg.vector_norm(self.h_trace).item()),
                "saturation": min(1.0, float(torch.linalg.vector_norm(self.S).item()) / 100.0),
                "evictions": int(self.eviction_count),
                "key_count": len(self._key_registry),
                "K_max": int(self.K_max),
                "adapter": self._adapter_type,
            }

    def consolidate(self) -> Dict[str, Any]:
        """Small orthogonality regularisation on the associative state."""
        before = float(torch.linalg.vector_norm(self.S).item())
        with torch.no_grad():
            for h in range(self.heads):
                Sh = self.S[h]
                grad = 2.0 * Sh @ (Sh.T @ Sh - torch.eye(
                    self.head_dim, dtype=Sh.dtype, device=Sh.device
                ))
                self.S[h].sub_(1e-4 * grad)
        return {
            "status": "consolidated",
            "updated": True,
            "norm_before": before,
            "norm_after": float(torch.linalg.vector_norm(self.S).item()),
        }


class SparseDeltaAdapter(PlasticAdapter):
    """Sparse TopK code followed by a gated Delta memory in code space."""

    _adapter_version = SPARSE_STATE_VERSION

    def __init__(
        self,
        dim: int = SPARSE_INPUT_DIM,
        code_dim: int = SPARSE_CODE_DIM,
        top_k: int = SPARSE_TOP_K,
        heads: int = 8,
        K_max: int = 5000,
        adapter_type: str = "sparse_delta",
        seed: int = 0,
        **kwargs,
    ):
        if "d_model" in kwargs:
            dim = kwargs.pop("d_model")
        kwargs.pop("k_max", None)
        super().__init__()
        if code_dim % heads != 0:
            raise ValueError(f"code_dim {code_dim} must be divisible by heads {heads}")
        self.dim = int(dim)
        self.code_dim = int(code_dim)
        self.top_k = int(top_k)
        self.heads = int(heads)
        self.K_max = max(1, int(K_max))
        self._adapter_type = adapter_type
        self._adapter_version = self._adapter_version
        self.coder = SparseCoder(self.dim, self.code_dim, self.top_k, seed=seed)
        self.delta = DeltaPlasticAdapter(
            dim=self.code_dim,
            heads=self.heads,
            K_max=self.K_max,
            adapter_type=adapter_type,
        )
        self.register_buffer("h_trace", torch.zeros(self.dim, dtype=torch.float32))

    # -- PlasticAdapter contract ---------------------------------------
    @property
    def S(self) -> torch.Tensor:  # pragma: no cover - compatibility accessor
        return self.delta.S

    def update(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        code = self.coder.encode(x)
        adapted = self.delta.update(code)
        with torch.no_grad():
            self.h_trace.mul_(0.9).add_(0.1 * x)
        return self.coder.decode(adapted)

    def update_associative_memory(self, x, surprise_factor: float = 1.0,
                                  effective_lr: float = None, base_lr: float = 0.05):
        x = self._prepare(x)
        code = self.coder.encode(x)
        adapted = self.delta.update_associative_memory(
            code, surprise_factor=surprise_factor,
            effective_lr=effective_lr, base_lr=base_lr,
        )
        with torch.no_grad():
            self.h_trace.mul_(0.9).add_(0.1 * x)
        return self.coder.decode(adapted)

    def update_scheduled(self, x, *, surprise_factor, effective_lr, base_lr):
        out = self.update_associative_memory(
            x, surprise_factor=surprise_factor,
            effective_lr=effective_lr, base_lr=base_lr,
        )
        return out, float(effective_lr)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        q = self._prepare(q)
        code = self.coder.encode(q)
        read = self.delta.retrieve(code)
        return self.coder.decode(read) + 0.1 * self.h_trace

    def predict_next(self) -> torch.Tensor:
        return _normalized(self.delta.predict_next())

    def compute_surprise(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        x = self._prepare(x)
        return self.delta.compute_surprise(self.coder.encode(x))

    def erase(self, key: torch.Tensor, strength: float = 1.0) -> Dict[str, Any]:
        key = self._prepare(key)
        report = self.delta.erase(self.coder.encode(key), strength=strength)
        report["adapter"] = self._adapter_type
        return report

    def set_beta_constant(self, value: float) -> None:
        self.delta.set_beta_constant(value)

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        return self.delta.gates(self.coder.encode(self._prepare(x)))

    def gate_report(self, x: torch.Tensor) -> Dict[str, Any]:
        report = self.delta.gate_report(self.coder.encode(self._prepare(x)))
        report["adapter"] = self._adapter_type
        return report

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """Memory read in the 384-D embedding space."""
        code = self.coder.encode(self._prepare(key))
        return self.coder.decode(self.delta.read(code))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.coder.encode(x)

    def differentiable_output(self, x: torch.Tensor) -> torch.Tensor:
        code = self.coder.encode(self._prepare(x))
        return self.coder.decode(self.delta.differentiable_output(code))

    def code_report(self, x: torch.Tensor) -> Dict[str, Any]:
        code = self.coder.encode(self._prepare(x))
        report = self.coder.utilization(code)
        report["adapter"] = self._adapter_type
        return report

    # -- persistence ----------------------------------------------------
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        sd = {f"delta_{key}": value for key, value in self.delta.state_dict().items()}
        sd.update(self.coder.state_dict())
        sd["h_trace"] = self.h_trace.detach().cpu().contiguous()
        sd["_adapter_type"] = self._adapter_type
        sd["_adapter_version"] = SPARSE_STATE_VERSION
        sd["dim"] = self.dim
        sd["code_dim"] = self.code_dim
        sd["top_k"] = self.top_k
        sd["heads"] = self.heads
        sd["K_max"] = self.K_max
        sd["eviction_count"] = self.delta.eviction_count
        return sd

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        version = int(state.get("_adapter_version", 1))
        if version > SPARSE_STATE_VERSION:
            raise AdapterStateError(
                f"sparse adapter state version {version} is newer than supported "
                f"{SPARSE_STATE_VERSION}"
            )
        dim = int(state.get("dim", self.dim))
        code_dim = int(state.get("code_dim", self.code_dim))
        top_k = int(state.get("top_k", self.top_k))
        heads = int(state.get("heads", self.heads))
        if (dim, code_dim, heads) != (self.dim, self.code_dim, self.heads):
            raise AdapterStateError(
                "sparse adapter geometry "
                f"(dim={dim}, code_dim={code_dim}, heads={heads}) does not match "
                f"configured (dim={self.dim}, code_dim={self.code_dim}, heads={self.heads})"
            )
        if top_k != self.top_k:
            raise AdapterStateError(
                f"sparse code budget top_k={top_k} does not match configured "
                f"top_k={self.top_k}; migrate the profile explicitly"
            )
        self.coder.load_state_dict(state)
        self.top_k = top_k
        delta_state = {
            key[len("delta_"):]: value
            for key, value in state.items()
            if key.startswith("delta_")
        }
        self.delta.load_state_dict(delta_state)
        if "h_trace" in state:
            with torch.no_grad():
                self.h_trace.copy_(state["h_trace"].to(self.h_trace.device))
        return self

    def reset(self) -> None:
        self.delta.reset()
        self.h_trace.zero_()

    def get_memory_health(self) -> Dict[str, float]:
        health = self.delta.get_memory_health()
        health["code_dim"] = float(self.code_dim)
        health["top_k"] = float(self.top_k)
        health["adapter"] = self._adapter_type
        return health

    # -- helpers --------------------------------------------------------
    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).detach().float().reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"{type(self).__name__} expects a {self.dim}-D input, got {x.shape[0]}")
        return x


class FlyModelAdapter(SparseDeltaAdapter):
    """FlyModel-style configured sparse/delta profile.

    Same public contract and default 384 -> 2048 / TopK-128 pipeline as
    :class:`SparseDeltaAdapter`; the distinct registry name lets profiles pin
    the FlyModel configuration explicitly.
    """

    def __init__(self, dim: int = SPARSE_INPUT_DIM, code_dim: int = SPARSE_CODE_DIM,
                 top_k: int = SPARSE_TOP_K, heads: int = 8, **kwargs):
        super().__init__(
            dim=dim, code_dim=code_dim, top_k=top_k, heads=heads,
            adapter_type="fly_model", **kwargs,
        )


def _register_delta_adapters() -> None:
    """Register the new adapter names without touching existing entries."""
    from src.model.plastic_adapter import register_adapter

    register_adapter("delta_rtu", DeltaPlasticAdapter, strict=False)
    register_adapter("sparse_delta", SparseDeltaAdapter, strict=False)
    register_adapter("fly_model", FlyModelAdapter, strict=False)


__all__ = [
    "AdapterStateError",
    "DeltaPlasticAdapter",
    "FlyModelAdapter",
    "SparseCoder",
    "SparseDeltaAdapter",
    "DELTA_STATE_VERSION",
    "SPARSE_STATE_VERSION",
]
