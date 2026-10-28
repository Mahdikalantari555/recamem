"""Autonomous sleep consolidation, MAS importance pruning and RDB schedule.

Implements the ``memory/sleep-consolidation``, ``memory/memory-aware-synapses``
and ``memory/decreasing-backprop`` capabilities.

Design constraints encoded here:

* :meth:`SleepConsolidation.autonomous_retrieval` never accepts an expected
  pattern list.  Replay candidates come from the engine's own knowledge
  vectors and a decaying self-inhibition/refractory trace.
* Every replay loop is bounded by steps, wall-clock time, a cancellation
  event, a duplicate cap and a confidence floor.
* MAS uses one backward pass per sampled batch of the squared-L2 output
  objective and only touches parameters in an explicit allowlist.
* Pruning keeps a reversible mask and the original tensor values.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from recamem.plastic_adapter import apply_plastic_update

MAS_STATE_VERSION = 1
SLEEP_CONFIG_VERSION = 1

DEFAULT_SLEEP_CONFIG: Dict[str, Any] = {
    "version": SLEEP_CONFIG_VERSION,
    "enabled": False,
    "idle_seconds": 30.0,
    "interval_seconds": 5.0,
    "max_replay_steps": 128,
    "max_seconds": 10.0,
    "min_similarity": 0.0,
    "duplicate_cap": 2,
    "inhibition_strength": 0.6,
    "inhibition_decay": 0.5,
    "fact_prune_ratio": 0.1,
    "mas_enabled": False,
    "parameter_prune_ratio": 0.1,
    "schedule": {"base_lr": 0.05, "decay_factor": 0.95, "index": 0},
}


class ScheduleValidationError(ValueError):
    """A DecreasingBackprop configuration was invalid."""


class DecreasingBackprop:
    """Persisted learning-rate schedule ``lr_i = base_lr * decay_factor ** i``."""

    def __init__(self, base_lr: float = 0.05, decay_factor: float = 0.95,
                 index: int = 0, experimental_relu_down: bool = False):
        self.base_lr = self._validate_rate(base_lr)
        self.decay_factor = self._validate_decay(decay_factor)
        self.index = self._validate_index(index)
        # ReLUDown is not verified against the reference implementation, so it
        # stays off unless explicitly requested and is never applied to a
        # default model.
        self.experimental_relu_down = bool(experimental_relu_down)

    @staticmethod
    def _validate_rate(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ScheduleValidationError(f"base_lr must be finite and positive, got {value}")
        return value

    @staticmethod
    def _validate_decay(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ScheduleValidationError(
                f"decay_factor must be in (0, 1], got {value}"
            )
        return value

    @staticmethod
    def _validate_index(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScheduleValidationError(f"index must be a non-negative integer, got {value!r}")
        if value < 0:
            raise ScheduleValidationError(f"index must be non-negative, got {value}")
        return value

    def lr_at(self, index: Optional[int] = None) -> float:
        i = self.index if index is None else self._validate_index(index)
        return float(self.base_lr * (self.decay_factor ** i))

    @property
    def current_lr(self) -> float:
        return self.lr_at()

    def advance(self) -> int:
        """Advance exactly once after a committed task/update boundary."""
        self.index += 1
        return self.index

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "decay_factor": self.decay_factor,
            "i": self.index,
            "lr_i": self.lr_at(),
            "experimental_relu_down": self.experimental_relu_down,
        }

    def apply_to_optimizer(self, optimizer) -> Dict[str, Any]:
        lr = self.lr_at()
        for group in optimizer.param_groups:
            group["lr"] = lr
        return {"applied_lr": lr, "index": self.index}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "base_lr": self.base_lr,
            "decay_factor": self.decay_factor,
            "index": self.index,
            "experimental_relu_down": self.experimental_relu_down,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "DecreasingBackprop":
        version = int(state.get("version", 1))
        if version > 1:
            raise ScheduleValidationError(f"Unsupported schedule version {version}")
        self.base_lr = self._validate_rate(state.get("base_lr", self.base_lr))
        self.decay_factor = self._validate_decay(state.get("decay_factor", self.decay_factor))
        self.index = self._validate_index(int(state.get("index", 0)))
        self.experimental_relu_down = bool(state.get("experimental_relu_down", False))
        return self


def relu_down(x: torch.Tensor, slope: float = 0.5) -> torch.Tensor:
    """Experimental ReLUDown transform (opt-in; not used by default models)."""
    slope = float(slope)
    if not 0.0 < slope <= 1.0:
        raise ValueError("slope must be in (0, 1]")
    return torch.where(x > 0, x, slope * x)


@dataclass
class MASReport:
    status: str
    ratio: float
    selected: int
    total: int
    threshold: float
    rounding: str = "floor"
    quality_before: Optional[float] = None
    quality_after: Optional[float] = None
    quality_delta: Optional[float] = None
    warning: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "ratio": self.ratio,
            "selected": self.selected,
            "total": self.total,
            "threshold": self.threshold,
            "rounding": self.rounding,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "quality_delta": self.quality_delta,
            "warning": self.warning,
        }


class MemoryAwareSynapses:
    """Output-sensitivity importance with reversible low-importance pruning."""

    def __init__(self, adapter, ratio: float = 0.1,
                 eligible: Optional[Sequence[str]] = None,
                 quality_fn: Optional[Callable[[], Optional[float]]] = None):
        if adapter is None:
            raise ValueError("MemoryAwareSynapses requires an adapter")
        ratio = float(ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1), got {ratio}")
        self.adapter = adapter
        self.ratio = ratio
        self.eligible = list(eligible) if eligible is not None else None
        self.quality_fn = quality_fn
        self.importance: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        self.originals: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, Any] = {}
        self.last_report: Optional[MASReport] = None

    # -- eligibility ----------------------------------------------------
    def eligible_parameters(self) -> Dict[str, torch.nn.Parameter]:
        params = {}
        for name, param in self.adapter.named_parameters():
            if not param.requires_grad and name not in self._explicit_names():
                continue
            if "anchor" in name.lower():
                continue
            if self.eligible is not None and name not in self.eligible:
                continue
            params[name] = param
        return params

    def _explicit_names(self) -> List[str]:
        return list(self.eligible or [])

    def non_eligible_buffers(self) -> Dict[str, torch.Tensor]:
        return dict(self.adapter.named_buffers())

    # -- importance -----------------------------------------------------
    def estimate_importance(self, inputs: Iterable[torch.Tensor],
                            batch_size: int = 8) -> Dict[str, Any]:
        """Accumulate ``mean(abs(grad))`` of ``||f(x)||^2`` (one backward/batch)."""
        batches = self._as_batches(inputs, batch_size)
        params = self.eligible_parameters()
        if not params:
            raise ValueError("No eligible parameters for MAS importance")
        self.importance = {name: torch.zeros_like(param.detach())
                           for name, param in params.items()}
        seen = 0
        backward_passes = 0
        for batch in batches:
            if batch.numel() == 0:
                continue
            for param in params.values():
                param.grad = None
            with torch.enable_grad():
                output = self.adapter.differentiable_output(batch)
                objective = torch.sum(output ** 2)
                if not torch.isfinite(objective):
                    continue
                objective.backward()
            backward_passes += 1
            for name, param in params.items():
                grad = param.grad
                if grad is None:
                    grad = torch.zeros_like(param)
                grad = torch.nan_to_num(grad.detach().abs(), nan=0.0, posinf=0.0, neginf=0.0)
                self.importance[name] = (
                    self.importance[name] * seen + grad
                ) / (seen + 1)
            seen += 1
            for param in params.values():
                param.grad = None
        self.stats = {
            "batches": seen,
            "backward_passes": backward_passes,
            "eligible_parameters": sorted(params),
            "eligible_elements": int(sum(p.numel() for p in params.values())),
        }
        return dict(self.stats)

    def _as_batches(self, inputs: Iterable[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            tensor = inputs
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return [tensor[i:i + batch_size] for i in range(0, tensor.shape[0], batch_size)]
        items = [torch.as_tensor(x).detach().float().reshape(-1) for x in inputs]
        if not items:
            return []
        return [torch.stack(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]

    # -- pruning --------------------------------------------------------
    def _quality(self) -> Optional[float]:
        if self.quality_fn is None:
            return None
        try:
            value = self.quality_fn()
            return None if value is None else float(value)
        except Exception:
            return None

    def prune(self, ratio: Optional[float] = None,
              apply: bool = True) -> Dict[str, Any]:
        if not self.importance:
            raise RuntimeError("estimate_importance() must run before prune()")
        ratio = self.ratio if ratio is None else float(ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1), got {ratio}")
        params = self.eligible_parameters()
        missing = set(params) - set(self.importance)
        if missing:
            raise RuntimeError(f"importance missing for eligible parameters: {sorted(missing)}")

        quality_before = self._quality()
        # Deterministic ordering: parameter names sorted, then flat index.
        # A stable sort over ``(score, concatenation order)`` resolves ties by
        # name and index rather than by dictionary iteration order.
        names = sorted(params)
        sizes = [params[name].numel() for name in names]
        all_scores = torch.cat([
            self.importance[name].detach().reshape(-1) for name in names
        ]) if names else torch.empty(0)
        total = int(all_scores.numel())
        selected_count = int(math.floor(total * ratio))
        order = torch.argsort(all_scores, descending=False, stable=True)
        selected_flat = order[:selected_count]
        threshold = (
            float(all_scores[selected_flat[-1]].item()) if selected_count > 0 else 0.0
        )

        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        selected_mask = torch.zeros(total, dtype=torch.bool)
        selected_mask[selected_flat] = True

        new_masks: Dict[str, torch.Tensor] = {}
        for index, name in enumerate(names):
            param = params[name]
            mask = ~selected_mask[offsets[index]:offsets[index + 1]].reshape(param.shape)
            new_masks[name] = mask
            if name not in self.originals:
                # Keep the pre-pruning values so restore is lossless.
                self.originals[name] = param.detach().clone()
        self.masks = new_masks

        if apply:
            with torch.no_grad():
                for name, mask in self.masks.items():
                    params[name].mul_(mask.to(params[name].dtype))

        quality_after = self._quality()
        delta = (
            None if quality_before is None or quality_after is None
            else quality_after - quality_before
        )
        warning = None
        if delta is not None and delta < 0:
            warning = f"retrieval quality regressed by {abs(delta):.4f} after pruning"
        report = MASReport(
            status="pruned" if selected_count else "no_eligible_elements",
            ratio=ratio,
            selected=selected_count,
            total=total,
            threshold=threshold,
            quality_before=quality_before,
            quality_after=quality_after,
            quality_delta=delta,
            warning=warning,
        )
        self.last_report = report
        self.ratio = ratio
        return report.as_dict()

    def restore(self) -> Dict[str, Any]:
        """Recover the exact pre-pruning parameter values."""
        if not self.originals:
            return {"status": "nothing_to_restore", "restored": 0}
        params = self.eligible_parameters()
        restored = 0
        with torch.no_grad():
            for name, original in self.originals.items():
                if name in params:
                    params[name].copy_(original.to(params[name].device))
                    restored += 1
        self.originals = {}
        self.masks = {}
        return {"status": "restored", "restored": restored}

    # -- persistence ----------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, score in self.importance.items():
            key = name.replace(".", "__")
            state[f"score__{key}"] = score.detach().cpu().contiguous()
            if name in self.masks:
                state[f"mask__{key}"] = self.masks[name].to(torch.uint8).cpu().contiguous()
            if name in self.originals:
                state[f"original__{key}"] = self.originals[name].detach().cpu().contiguous()
        return state

    def metadata(self) -> Dict[str, Any]:
        return {
            "version": MAS_STATE_VERSION,
            "ratio": self.ratio,
            "selected": self.last_report.selected if self.last_report else 0,
            "total": self.last_report.total if self.last_report else 0,
            "parameters": sorted(self.importance),
        }

    def save_state(self, path: str, metadata_path: Optional[str] = None) -> None:
        from safetensors.torch import save_file
        save_file(self.state_dict(), str(path))
        if metadata_path:
            with open(metadata_path, "w") as handle:
                json.dump(self.metadata(), handle, indent=2)

    def load_state(self, path: str, metadata_path: Optional[str] = None,
                   apply: bool = True) -> Dict[str, Any]:
        from safetensors.torch import load_file
        state = load_file(str(path))
        params = self.eligible_parameters()
        self.importance, self.masks, self.originals = {}, {}, {}
        for key, tensor in state.items():
            prefix, _, name = key.partition("__")
            name = name.replace("__", ".")
            if name not in params:
                continue
            expected = params[name].shape
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(
                    f"MAS state for {name} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected)}"
                )
            if prefix == "score":
                self.importance[name] = tensor.float()
            elif prefix == "mask":
                self.masks[name] = tensor.bool()
            elif prefix == "original":
                self.originals[name] = tensor.float()
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as handle:
                meta = json.load(handle)
            if int(meta.get("version", 1)) > MAS_STATE_VERSION:
                raise ValueError(f"Unsupported MAS state version {meta.get('version')}")
            self.ratio = float(meta.get("ratio", self.ratio))
        if apply and self.masks:
            with torch.no_grad():
                for name, mask in self.masks.items():
                    if name in params:
                        params[name].mul_(mask.to(params[name].dtype))
        return {
            "status": "loaded",
            "parameters": len(self.importance),
            "masks": len(self.masks),
            "originals": len(self.originals),
        }


class SleepConsolidation:
    """Bounded autonomous replay driven by internal state and refractory inhibition."""

    def __init__(self, droid, max_replay_steps: int = 128,
                 max_seconds: float = 10.0, min_similarity: float = 0.0,
                 duplicate_cap: int = 2, inhibition_strength: float = 0.6,
                 inhibition_decay: float = 0.5, duplicate_penalty: float = 0.5,
                 lr: float = 1e-3, base_lr: float = 0.05):
        if max_replay_steps < 0:
            raise ValueError("max_replay_steps must be non-negative")
        if max_seconds < 0:
            raise ValueError("max_seconds must be non-negative")
        if duplicate_cap < 1:
            raise ValueError("duplicate_cap must be at least 1")
        if not 0.0 <= inhibition_strength <= 1.0:
            raise ValueError("inhibition_strength must be in [0, 1]")
        if not 0.0 <= inhibition_decay <= 1.0:
            raise ValueError("inhibition_decay must be in [0, 1]")
        self.droid = droid
        self.max_replay_steps = int(max_replay_steps)
        self.max_seconds = float(max_seconds)
        self.min_similarity = float(min_similarity)
        self.duplicate_cap = int(duplicate_cap)
        self.inhibition_strength = float(inhibition_strength)
        self.inhibition_decay = float(inhibition_decay)
        if duplicate_penalty < 0:
            raise ValueError("duplicate_penalty must be non-negative")
        self.duplicate_penalty = float(duplicate_penalty)
        self.lr = float(lr)
        self.base_lr = float(base_lr)

    def _active_memory(self):
        store = getattr(self.droid, "knowledge", None)
        if store is None:
            return [], torch.zeros(0), torch.zeros(0)
        vecs, ids = store._load_dense()
        if len(ids) == 0:
            return [], vecs, vecs
        rows = store.conn.execute(
            "SELECT id, novelty FROM facts WHERE superseded = 0"
        ).fetchall()
        active = {row[0]: (row[1] if row[1] is not None else 0.0) for row in rows}
        keep = [i for i, fact_id in enumerate(ids) if fact_id in active]
        if not keep:
            return [], vecs[:0], vecs[:0]
        index = torch.tensor(keep, dtype=torch.long)
        ids_active = [ids[i] for i in keep]
        vecs_active = vecs.index_select(0, index).contiguous()
        novelty = torch.tensor([active[i] for i in ids_active], dtype=torch.float32)
        return ids_active, vecs_active, novelty

    def _seed_query(self, vecs: torch.Tensor) -> torch.Tensor:
        memory = getattr(self.droid, "memory", None)
        if memory is not None and hasattr(memory, "predict_next"):
            try:
                with torch.no_grad():
                    seed = memory.predict_next().detach().float().reshape(-1)
                if seed.shape[0] == vecs.shape[1] and float(seed.norm()) > 1e-8:
                    return seed / seed.norm()
            except Exception:
                pass
        return torch.nn.functional.normalize(vecs.mean(dim=0), dim=0)

    def autonomous_retrieval(self, cancel_event: Optional[threading.Event] = None,
                             **overrides) -> Dict[str, Any]:
        """Run a bounded replay cycle; deliberately takes no expected-pattern list."""
        started = time.perf_counter()
        max_steps = int(overrides.get("max_replay_steps", self.max_replay_steps))
        max_seconds = float(overrides.get("max_seconds", self.max_seconds))
        duplicate_cap = int(overrides.get("duplicate_cap", self.duplicate_cap))
        min_similarity = float(overrides.get("min_similarity", self.min_similarity))
        lr = float(overrides.get("lr", self.lr))
        base_lr = float(overrides.get("base_lr", self.base_lr))

        ids, vecs, novelty = self._active_memory()
        report: Dict[str, Any] = {
            "status": "no_patterns",
            "eligible_patterns": len(ids),
            "steps": 0,
            "unique_recovered": 0,
            "duplicates": 0,
            "coverage": 0.0,
            "retrieved_ids": [],
            "duplicate_ids": [],
            "replay_writes": 0,
            "stop_reason": "empty_memory",
            "duration_ms": 0.0,
        }
        if len(ids) == 0:
            report["duration_ms"] = (time.perf_counter() - started) * 1000
            return report

        query = self._seed_query(vecs)
        inhibition = torch.zeros(len(ids))
        counts = torch.zeros(len(ids), dtype=torch.int32)
        retrieved: List[int] = []
        duplicates: List[int] = []
        replay_writes = 0
        stop_reason = "budget_exhausted"
        steps = 0
        cancelled = False

        while steps < max_steps and (time.perf_counter() - started) < max_seconds:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                stop_reason = "cancelled"
                break
            if not torch.isfinite(query).all():
                stop_reason = "non_finite_query"
                break
            sims = vecs @ query
            # Refractory inhibition plus a per-repeat penalty keeps duplicates
            # from crowding out unseen patterns while still allowing a
            # configured number of revisits.
            scores = (sims + 0.05 * novelty - inhibition
                      - self.duplicate_penalty * counts.float())
            blocked = counts >= duplicate_cap
            if bool(blocked.all()):
                stop_reason = "coverage_complete"
                break
            scores = torch.where(blocked, torch.full_like(scores, -float("inf")), scores)
            best = int(torch.argmax(scores).item())
            # The floor applies to the actual similarity, not to the inhibited
            # score: inhibition is a ranking device, not evidence that memory
            # no longer contains the pattern.
            best_score = float(sims[best].item())
            if steps > 0 and best_score < min_similarity:
                stop_reason = "below_similarity_floor"
                break

            fact_id = int(ids[best])
            if counts[best] > 0:
                duplicates.append(fact_id)
            else:
                retrieved.append(fact_id)
            counts[best] += 1
            steps += 1

            # Replay the internal pattern through the plastic adapter.
            try:
                apply_plastic_update(
                    self.droid.memory, vecs[best],
                    surprise_factor=max(0.0, 1.0 - best_score),
                    effective_lr=lr, base_lr=max(base_lr, 1e-9),
                )
                replay_writes += 1
            except Exception as exc:  # replay failure must not abort the cycle
                report.setdefault("warnings", []).append(
                    f"replay failed for pattern {fact_id}: {exc}"
                )

            # Self-inhibition: a retrieved pattern and its correlated
            # neighbours become harder to select next, so another attractor can
            # surface.  Inhibition decays every step.
            correlation = torch.clamp(vecs @ vecs[best], min=0.0)
            inhibition = self.inhibition_decay * inhibition + self.inhibition_strength * correlation
            query = torch.nn.functional.normalize(
                0.7 * query + 0.3 * vecs[best], dim=0
            )

        unique = len(set(retrieved))
        report.update({
            "status": "cancelled" if cancelled else "completed",
            "steps": steps,
            "unique_recovered": unique,
            "duplicates": len(duplicates),
            "coverage": unique / max(1, len(ids)),
            "retrieved_ids": retrieved,
            "duplicate_ids": duplicates,
            "replay_writes": replay_writes,
            "stop_reason": stop_reason,
            "duration_ms": (time.perf_counter() - started) * 1000,
        })
        return report


class AutoSleepWorker:
    """One cancellable daemon worker per droid; repeated starts are idempotent."""

    def __init__(self, droid, config: Optional[Dict[str, Any]] = None):
        merged = dict(DEFAULT_SLEEP_CONFIG)
        if config:
            merged.update(config)
        schedule_cfg = merged.get("schedule") or {}
        merged["schedule"] = {**DEFAULT_SLEEP_CONFIG["schedule"], **schedule_cfg}
        self.droid = droid
        self.config = merged
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.cycles = 0
        self.last_report: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    @property
    def interval(self) -> float:
        return max(0.01, float(self.config.get("interval_seconds", 5.0)))

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"auto-sleep-{getattr(self.droid, 'name', 'droid')}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        joined = not thread.is_alive()
        if joined:
            self._thread = None
        return joined

    def _run(self) -> None:
        idle_seconds = float(self.config.get("idle_seconds", 30.0))
        while not self._stop.wait(self.interval):
            try:
                last_activity = float(getattr(self.droid, "last_activity", 0.0))
                if time.time() - last_activity < idle_seconds:
                    continue
                report = self.droid.sleep(cancel_event=self._stop)
                self.last_report = report
                self.cycles += 1
            except Exception as exc:  # keep the worker alive, surface the error
                self.last_error = str(exc)

    def status(self) -> Dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "cycles": self.cycles,
            "interval_seconds": self.interval,
            "idle_seconds": float(self.config.get("idle_seconds", 30.0)),
            "last_error": self.last_error,
            "last_report_status": (self.last_report or {}).get("status"),
        }


__all__ = [
    "AutoSleepWorker",
    "DEFAULT_SLEEP_CONFIG",
    "DecreasingBackprop",
    "MemoryAwareSynapses",
    "ScheduleValidationError",
    "SleepConsolidation",
    "relu_down",
]
