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
