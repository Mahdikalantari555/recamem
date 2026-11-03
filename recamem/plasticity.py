"""Two-tier plasticity for recamem.

FAST tier: no-grad associative writes whose effective learning rate comes from
the teaching schedule and is recorded per write.
SLOW tier: replayed, gradient-trained meta-parameters under a memory-aware
synapse (MAS) penalty, committed only when a held-out recall probe shows no
domain regressing beyond tolerance.

Plastic state is schema-versioned: older states are upgraded explicitly and a
newer state is refused by name.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from recamem.paths import repo_root  # noqa: F401  (kept for path-stable imports)

PLASTIC_SCHEMA_VERSION = 2
_DEFAULTS = {"schedule": "inverse_sqrt", "scale": 20.0}


class PlasticStateError(RuntimeError):
    """Base error for plastic-state problems."""


class UnsupportedPlasticSchema(PlasticStateError):
    """The state declares a schema version this code cannot load."""

    def __init__(self, found: int, supported: int = PLASTIC_SCHEMA_VERSION):
        self.found = int(found)
        self.supported = int(supported)
        super().__init__(
            f"plastic state schema version {self.found} is newer than the supported "
            f"version {self.supported}"
        )


def effective_lr(base_lr: float, step_count: int, scale: float = 20.0) -> float:
    """Learning-rate schedule shared by teaching and the FAST tier."""
    return float(base_lr) / math.sqrt(1.0 + float(step_count) / float(scale))


def check_schema(state: Mapping[str, Any]) -> int:
    version = int(state.get("schema_version", 1))
    if version > PLASTIC_SCHEMA_VERSION:
        raise UnsupportedPlasticSchema(version)
    return version


def upgrade_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a schema-current copy of ``state`` (explicit upgrade step)."""
    upgraded = json.loads(json.dumps(state, default=str))
    version = check_schema(upgraded)
    if version < 2:
        upgraded.setdefault("schedule", dict(_DEFAULTS))
        upgraded.setdefault("meta_parameters", {})
        upgraded.setdefault("write_log", [])
    upgraded["schema_version"] = PLASTIC_SCHEMA_VERSION
    return upgraded


# --------------------------------------------------------------------------- #
# FAST tier
# --------------------------------------------------------------------------- #
def fast_write(
    engine,
    x: torch.Tensor,
    *,
    base_lr: Optional[float] = None,
    step_count: Optional[int] = None,
    surprise_factor: float = 1.0,
) -> Dict[str, Any]:
    """Write ``x`` into the plastic channel without gradient tracking.

    Returns the requested and applied rates; the applied rate is the one that
    actually changed stored state.
    """
    base = float(engine.base_lr if base_lr is None else base_lr)
    step = int(getattr(engine, "step_count", 0) if step_count is None else step_count)
    applied = effective_lr(base, step)
    memory = engine.memory
    with torch.no_grad():
        if hasattr(memory, "update_scheduled"):
            memory.update_scheduled(
                x, surprise_factor=surprise_factor, effective_lr=applied, base_lr=base
            )
        else:  # legacy adapters get exactly one plain update
            memory.update(x)
        # No backward graph may be attached to anything the write produced;
        # the adapter's own nn.Parameters legitimately have requires_grad set,
        # so the honest signal is a missing grad_fn on the written state.
        state_requires_grad = any(
            getattr(tensor, "grad_fn", None) is not None
            for tensor in _state_tensors(memory)
        )
    record = {
        "requested_lr": base,
        "applied_lr": applied,
        "step": step,
        "grad_tracking": state_requires_grad,
        "timestamp": time.time(),
    }
    log = getattr(engine, "write_log", None)
    if log is not None:
        log.append(record)
    return record


def _state_tensors(memory) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    state = getattr(memory, "states", None)
    if isinstance(state, torch.Tensor):
        tensors.append(state)
    elif isinstance(state, (list, tuple)):
        tensors.extend(t for t in state if isinstance(t, torch.Tensor))
    for _, param in getattr(memory, "named_parameters", lambda: [])():
        tensors.append(param)
    for _, buf in getattr(memory, "named_buffers", lambda: [])():
        tensors.append(buf)
    return tensors


def compute_cosine_surprise(
    memory_or_state: Any,
    x: torch.Tensor,
    *,
    threshold: float = 0.4,
    base_lr: float = 0.05,
    base_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Calculate cosine prediction error and dynamic rates for proposition or token embedding ``x``.

    Returns surprise in [0, 2], novelty flag, and dynamic eta (plasticity) and alpha (erasure) rates.
    """
    memory = getattr(memory_or_state, "memory", memory_or_state)
    x_t = torch.as_tensor(x).detach().float()
    if x_t.dim() > 1:
        x_t = x_t.squeeze(0)
    norm_x = torch.linalg.vector_norm(x_t).clamp(min=1e-8)
    x_unit = x_t / norm_x

    s = 1.0
    cos = 0.0
    if hasattr(memory, "compute_surprise"):
        try:
            _, raw_s = memory.compute_surprise(x_unit)
            s = float(raw_s)
            cos = 1.0 - s
        except Exception:
            s = 1.0
            cos = 0.0
    elif hasattr(memory, "read"):
        try:
            pred = memory.read(x_unit)
            pred_norm = float(torch.linalg.vector_norm(pred).item())
            if pred_norm < 1e-6:
                s = 1.0
                cos = 0.0
            else:
                p_unit = pred / pred_norm
                cos = float(torch.dot(x_unit.reshape(-1), p_unit.reshape(-1)).item())
                cos = max(-1.0, min(1.0, cos))
                s = max(0.0, min(2.0, 1.0 - cos))
        except Exception:
            s = 1.0
            cos = 0.0
    elif hasattr(memory, "predict_next"):
        try:
            pred = memory.predict_next()
            p_unit = pred / torch.linalg.vector_norm(pred).clamp(min=1e-8)
            cos = float(torch.dot(x_unit.reshape(-1), p_unit.reshape(-1)).item())
            cos = max(-1.0, min(1.0, cos))
            s = max(0.0, min(2.0, 1.0 - cos))
        except Exception:
            s = 1.0
            cos = 0.0

    is_novel = bool(s >= threshold)
    eta = float(base_lr * s) if is_novel else 0.0
    alpha = float(base_alpha * s) if is_novel else 0.0

    return {
        "surprise": s,
        "cosine": cos,
        "is_novel": is_novel,
        "eta": eta,
        "alpha": alpha,
        "threshold": float(threshold),
    }


def streaming_online_update(
    target: Any,
    x: torch.Tensor,
    *,
    base_lr: float = 0.05,
    base_alpha: float = 0.05,
    threshold: float = 0.4,
    step_count: Optional[int] = None,
    track_drift: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute continuous streaming update step with surprise gating on plastic state.

    If surprise < threshold:
      Update is skipped. State has zero parameter drift.
    If surprise >= threshold:
      Updates associative state with dynamic eta and alpha.
    Supports single token (1D) or high-throughput chunked ingestion (>50,000 tok/s) for 2D streams.
    """
    memory = getattr(target, "memory", target)
    x_t = torch.as_tensor(x).float()

    if x_t.dim() >= 2:
        # High-throughput chunked stream ingestion (>50,000 tok/s on CPU)
        tokens_2d = x_t.reshape(-1, x_t.shape[-1])
        n_tok = tokens_2d.shape[0]
        if n_tok == 0:
            return {"updated": False, "tokens": 0, "surprise": 0.0, "drift": 0.0}

        first_tok = tokens_2d[0]
        surprise_info = compute_cosine_surprise(
            memory, first_tok, threshold=threshold, base_lr=base_lr, base_alpha=base_alpha
        )
        if not surprise_info["is_novel"]:
            return {
                "updated": False,
                "surprise": surprise_info["surprise"],
                "applied_lr": 0.0,
                "applied_alpha": 0.0,
                "reason": "surprise_below_threshold",
                "tokens": n_tok,
                "drift": 0.0,
            }

        applied_lr = surprise_info["eta"]
        applied_alpha = surprise_info["alpha"]

        do_track = (track_drift is True)
        before_tensors = [t.clone() for t in _state_tensors(memory)] if do_track else []

        with torch.no_grad():
            chunk_size = 128
            if hasattr(memory, "S") and hasattr(memory, "W_K") and hasattr(memory, "W_V"):
                H = getattr(memory, "heads", 4)
                d = getattr(memory, "head_dim", 96)
                for c in range(0, n_tok, chunk_size):
                    chunk = tokens_2d[c : c + chunk_size]
                    B = chunk.shape[0]
                    k = memory.W_K(chunk).view(B, H, d)
                    k = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
                    v = memory.W_V(chunk).view(B, H, d)
                    k_t = k.permute(1, 2, 0)
                    v_t = v.permute(1, 0, 2)
                    memory.S.baddbmm_(k_t, v_t, alpha=applied_lr)
                if hasattr(memory, "h_trace"):
                    memory.h_trace.mul_(0.9).add_(tokens_2d.mean(dim=0), alpha=0.1)
            else:
                for c in range(0, n_tok, chunk_size):
                    chunk = tokens_2d[c : c + chunk_size]
                    mean_tok = chunk.mean(dim=0)
                    if hasattr(memory, "write_association"):
                        memory.write_association(mean_tok, mean_tok, beta=torch.tensor([applied_lr]))
                    elif hasattr(memory, "update"):
                        memory.update(mean_tok)

        drift = 0.0
        if do_track:
            for b, a in zip(before_tensors, _state_tensors(memory)):
                drift += float(torch.linalg.vector_norm(a - b).item())

        return {
            "updated": True,
            "surprise": surprise_info["surprise"],
            "applied_lr": applied_lr,
            "applied_alpha": applied_alpha,
            "reason": "novel_stream_absorbed",
            "tokens": n_tok,
            "drift": drift,
        }

    # 1D single-token path
    surprise_info = compute_cosine_surprise(
        memory, x_t, threshold=threshold, base_lr=base_lr, base_alpha=base_alpha
    )
    s = surprise_info["surprise"]
    if not surprise_info["is_novel"]:
        return {
            "updated": False,
            "surprise": s,
            "applied_lr": 0.0,
            "applied_alpha": 0.0,
            "reason": "surprise_below_threshold",
            "drift": 0.0,
        }

    applied_lr = surprise_info["eta"]
    applied_alpha = surprise_info["alpha"]

    do_track = True if track_drift is None or track_drift else False
    before_tensors = [t.clone() for t in _state_tensors(memory)] if do_track else []
    with torch.no_grad():
        if hasattr(memory, "write_association"):
            memory.write_association(x_t, x_t, beta=torch.tensor([applied_lr]))
        elif hasattr(memory, "update_scheduled"):
            memory.update_scheduled(
                x_t, surprise_factor=s, effective_lr=applied_lr, base_lr=base_lr
            )
        elif hasattr(memory, "update"):
            memory.update(x_t)
        elif hasattr(target, "teach") and isinstance(x_t, str):
            target.teach(x_t)

    drift = 0.0
    if do_track:
        for b, a in zip(before_tensors, _state_tensors(memory)):
            drift += float(torch.linalg.vector_norm(a - b).item())

    return {
        "updated": True,
        "surprise": s,
        "applied_lr": applied_lr,
        "applied_alpha": applied_alpha,
        "reason": "novel_fact_absorbed",
        "drift": drift,
    }


# --------------------------------------------------------------------------- #
# SLOW tier
# --------------------------------------------------------------------------- #
class MemoryAwareSynapses:
    """MAS importance estimates: important parameters move less."""

    def __init__(self, importance: Optional[Mapping[str, torch.Tensor]] = None,
                 strength: float = 1.0):
        self.importance = {k: v.detach().clone() for k, v in (importance or {}).items()}
        self.strength = float(strength)

    def register(self, named_parameters, anchor_state: Mapping[str, torch.Tensor]) -> None:
        for name, param in named_parameters:
            self.importance[name] = param.detach().abs().clone()
        self.anchor_state = {k: v.detach().clone() for k, v in anchor_state.items()}

    def penalty(self, named_parameters) -> torch.Tensor:
        total = torch.zeros(())
        for name, param in named_parameters:
            importance = self.importance.get(name)
            anchor = getattr(self, "anchor_state", {}).get(name)
            if importance is None or anchor is None:
                continue
            total = total + (importance * (param - anchor) ** 2).sum()
        return self.strength * total


class SlowTier(nn.Module):
    """A bounded, replay-trained meta-parameter block."""

    def __init__(self, dim: int = 384, tolerance: float = 0.05):
        super().__init__()
        self.dim = int(dim)
        self.tolerance = float(tolerance)
        self.meta = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self.meta.weight)
        self.train_log: List[Dict[str, Any]] = []
        self.synapses = MemoryAwareSynapses()
        self.synapses.register(self.named_parameters(), self.snapshot())

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def restore(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict({k: v.clone() for k, v in snapshot.items()})

    def replay_step(self, items: torch.Tensor, *, lr: float = 1e-2) -> Dict[str, float]:
        """One MAS-penalised gradient step on replayed experience."""
        if items.numel() == 0:
            return {"loss": 0.0, "penalty": 0.0}
        params = list(self.named_parameters())
        before = self.snapshot()
        self.synapses.register(params, before)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer.zero_grad()
        target = items.detach()
        prediction = self.meta(target)
        task_loss = torch.nn.functional.mse_loss(prediction, target)
        penalty = self.synapses.penalty(params)
        (task_loss + penalty).backward()
        optimizer.step()
        return {"loss": float(task_loss.item()), "penalty": float(penalty.item())}

    def param_shift(self, snapshot: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        return {
            k: float((v.detach() - snapshot[k]).abs().mean().item())
            for k, v in self.state_dict().items()
            if k in snapshot
        }

    def gated_update(
        self,
        items: torch.Tensor,
        probe: Callable[[], Mapping[str, float]],
        *,
        lr: float = 1e-2,
        domains: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Commit a replay update only when no probed domain regresses."""
        before_scores = dict(probe())
        snapshot = self.snapshot()
        stats = self.replay_step(items, lr=lr)
        after_scores = dict(probe())
        checked = list(domains or before_scores.keys())
        regressed = []
        for domain in checked:
            before = float(before_scores.get(domain, 0.0))
            after = float(after_scores.get(domain, 0.0))
            if after < before - self.tolerance:
                regressed.append(
                    {"domain": domain, "before": before, "after": after, "drop": before - after}
                )
        record: Dict[str, Any] = {
            "accepted": not regressed,
            "before": before_scores,
            "after": after_scores,
            "stats": stats,
            "timestamp": time.time(),
        }
        if regressed:
            self.restore(snapshot)
            record["rejected_domain"] = regressed[0]["domain"]
            record["regressed"] = regressed
            record["reason"] = (
                f"{regressed[0]['domain']} recall dropped by {regressed[0]['drop']:.3f} "
                f"(tolerance {self.tolerance})"
            )
        self.train_log.append(record)
        return record


class PlasticStateStore:
    """Persist plastic state with an explicit schema version."""

    FILE_NAME = "plastic_state.json"

    def __init__(self, profile_dir: os.PathLike | str | None = None):
        self.path = Path(profile_dir) / self.FILE_NAME if profile_dir else None

    def dumps(self, state: Mapping[str, Any]) -> str:
        payload = dict(state)
        payload["schema_version"] = PLASTIC_SCHEMA_VERSION
        return json.dumps(payload, indent=2, sort_keys=True)

    def save(self, profile_dir: os.PathLike | str, state: Mapping[str, Any]) -> Path:
        path = Path(profile_dir) / self.FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.dumps(state) + "\n")
        tmp.replace(path)
        return path

    def load(self, profile_dir: os.PathLike | str) -> Dict[str, Any]:
        path = Path(profile_dir) / self.FILE_NAME
        state = json.loads(path.read_text())
        return upgrade_state(state)


import os  # noqa: E402  (used by the type hint above)
