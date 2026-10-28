
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Dict, List, Type, Any, Optional
import torch
import importlib.metadata

class PlasticAdapter(torch.nn.Module, ABC):
    @abstractmethod
    def update(self, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def predict_next(self) -> torch.Tensor:
        pass
    @abstractmethod
    def compute_surprise(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        pass
    @abstractmethod
    def state_dict(self) -> dict:
        pass
    @abstractmethod
    def load_state_dict(self, state: dict):
        pass
    def update_associative_memory(self, x: torch.Tensor, surprise_factor: float = 1.0) -> torch.Tensor:
        # Compatibility bridge, never retry an update that may already have mutated state.
        return self.update(x)

    def consolidate(self) -> dict:
        """Optional optimization is not supported by the default implementation."""
        return {"status": "unsupported", "updated": False}

    def get_health(self) -> dict:
        return {"status": "unsupported", "metrics": {}}

    def create_snapshot(self) -> dict:
        return clone_state(self.state_dict())

    def restore_snapshot(self, snapshot: dict):
        # Some third-party loaders retain references to their inputs.
        return self.load_state_dict(clone_state(snapshot))

    def differentiable_output(self, x: torch.Tensor) -> torch.Tensor:
        """State-free differentiable forward used by MAS importance.

        Exercises every same-width Linear projection without mutating any
        buffer or associative state, so output sensitivity can be measured
        with one backward pass and no labels.  Adapters with a different
        geometry override this method.
        """
        x = torch.as_tensor(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        width = x.shape[-1]
        out = x
        for module in self.modules():
            if (isinstance(module, torch.nn.Linear)
                    and module.in_features == width
                    and module.out_features == width):
                out = out + module(out)
        if hasattr(self, "norm") and isinstance(self.norm, torch.nn.Module):
            out = self.norm(out)
        return out


def clone_state(state):
    """Detach/clone tensors recursively; deepcopy all other mutable metadata."""
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {key: clone_state(value) for key, value in state.items()}
    if isinstance(state, list):
        return [clone_state(value) for value in state]
    if isinstance(state, tuple):
        return tuple(clone_state(value) for value in state)
    return deepcopy(state)


class PlasticUpdateError(RuntimeError):
    """A plastic update failed. Its state may be partially changed; never retry implicitly."""


@dataclass(frozen=True)
class PlasticUpdateResult:
    output: torch.Tensor
    requested_lr: float
    applied_lr: Optional[float]
    scheduling_supported: bool


def apply_plastic_update(adapter, x, *, surprise_factor, effective_lr, base_lr):
    """Call exactly one declared capability, without probing signatures by execution.

    Scheduled adapters implement update_scheduled(...) and return (output, applied_lr).
    Legacy adapters receive only their historical arguments. Exceptions propagate as
    PlasticUpdateError with the original cause, including failures after mutation.
    """
    if not math.isfinite(effective_lr) or effective_lr < 0:
        raise ValueError("effective_lr must be finite and non-negative")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    scheduled = getattr(adapter, "update_scheduled", None)
    try:
        if callable(scheduled):
            output, applied_lr = scheduled(
                x, surprise_factor=surprise_factor,
                effective_lr=effective_lr, base_lr=base_lr,
            )
            applied_lr = float(applied_lr)
            if not math.isfinite(applied_lr) or applied_lr < 0:
                raise ValueError("adapter reported an invalid applied learning rate")
            return PlasticUpdateResult(output, effective_lr, applied_lr, True)
        associative = getattr(adapter, "update_associative_memory", None)
        update = getattr(adapter, "update", None)
        if callable(associative):
            output = associative(x, surprise_factor=surprise_factor)
        elif callable(update):
            output = update(x)
        else:
            output, new_state = adapter(x)
            if hasattr(adapter, "states"):
                with torch.no_grad():
                    adapter.states.copy_(new_state)
        return PlasticUpdateResult(output, effective_lr, None, False)
    except Exception as exc:
        raise PlasticUpdateError(
            f"{type(adapter).__name__} plastic update failed; state may be partially changed: {exc}"
        ) from exc


_ADAPTER_REGISTRY: Dict[str, Type[PlasticAdapter]] = {}

def register_adapter(name: str, cls: Type[PlasticAdapter], strict: bool = True):
    if strict and not issubclass(cls, PlasticAdapter):
        required = ["update", "retrieve", "predict_next", "compute_surprise", "state_dict", "load_state_dict"]
        has_all = all(hasattr(cls, m) for m in required)
        if not has_all:
            raise TypeError(f"Adapter class must inherit from PlasticAdapter, got {cls}")
    _ADAPTER_REGISTRY[name] = cls

def list_adapters() -> List[str]:
    return list(_ADAPTER_REGISTRY.keys())

def identify_adapter(adapter) -> Optional[str]:
    """Registered name for ``adapter``, without inferring from class-name text.

    Preference order: the adapter's own ``_adapter_type`` if it is a registered
    name, then the registry entry whose class is exactly ``type(adapter)``.
    Returns ``None`` when the adapter was never registered.
    """
    if adapter is None:
        return None
    declared = getattr(adapter, "_adapter_type", None)
    if isinstance(declared, str) and declared in _ADAPTER_REGISTRY:
        if _ADAPTER_REGISTRY[declared] is type(adapter):
            return declared
    for name, cls in _ADAPTER_REGISTRY.items():
        if cls is type(adapter):
            return name
    return None

def get_adapter_class(name: str) -> Type[PlasticAdapter]:
    if name not in _ADAPTER_REGISTRY:
        _discover_entry_points()
        _register_builtins()
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Adapter '{name}' not found. Available: {list_adapters()}")
    return _ADAPTER_REGISTRY[name]

def get_adapter(name: str, **kwargs) -> PlasticAdapter:
    if name not in _ADAPTER_REGISTRY:
        _discover_entry_points()
        _register_builtins()
        if name not in _ADAPTER_REGISTRY:
            raise KeyError(f"Adapter '{name}' not found. Available: {list_adapters()}")
    cls = _ADAPTER_REGISTRY[name]
    if "d_model" in kwargs and "dim" not in kwargs:
        kwargs["dim"] = kwargs.pop("d_model")
    return cls(**kwargs)

def _discover_entry_points():
    try:
        try:
            eps = importlib.metadata.entry_points(group="tmt.plastic_adapters")
        except TypeError:
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get("tmt.plastic_adapters", [])
        for ep in eps:
            try:
                cls = ep.load()
                if issubclass(cls, PlasticAdapter):
                    register_adapter(ep.name, cls)
            except Exception:
                continue
    except Exception:
        pass

_discover_entry_points()

def _register_builtins():
    try:
        from src.model.rtu import PlasticAssociativeRTU
        register_adapter("associative_rtu", PlasticAssociativeRTU, strict=False)
    except Exception:
        try:
            from .rtu import PlasticAssociativeRTU
            register_adapter("associative_rtu", PlasticAssociativeRTU, strict=False)
        except Exception:  # noqa: BLE001
            pass
    try:
        from src.model.rtu import RTUMemoryBlock
        class LegacyRTUAdapter(PlasticAdapter):
            def __init__(self, dim=384, **kwargs):
                super().__init__()
                if "d_model" in kwargs:
                    dim = kwargs.pop("d_model")
                self.rtu = RTUMemoryBlock(dim=dim)
                self.dim = dim
                self.register_buffer('h_trace', torch.zeros(dim))
            def update(self, x):
                out, new_state = self.rtu(x)
                self.h_trace.copy_(new_state)
                return out
            def retrieve(self, q):
                return self.h_trace
            def predict_next(self):
                return self.h_trace / (torch.linalg.vector_norm(self.h_trace).clamp(min=1e-8))
            def compute_surprise(self, x):
                pred = self.predict_next()
                e_norm = x / (torch.linalg.vector_norm(x).clamp(min=1e-8))
                cos_sim = torch.dot(e_norm, pred).item()
                surprise = max(0.0, min(2.0, 1.0 - cos_sim))
                return e_norm - pred, surprise
            def state_dict(self, *args, **kwargs):
                return {
                    "states": self.rtu.states.detach().cpu(),
                    "decay": self.rtu.decay.detach().cpu(),
                    "proj_weight": self.rtu.proj.weight.detach().cpu(),
                    "h_trace": self.h_trace.detach().cpu(),
                    "_adapter_type": "legacy_rtu",
                    "_adapter_version": 1,
                }
            def load_state_dict(self, state, strict=True):
                with torch.no_grad():
                    if "states" in state:
                        self.rtu.states.copy_(state["states"].to(self.rtu.states.device))
                        self.h_trace.copy_(state["states"].to(self.h_trace.device))
                    if "h_trace" in state:
                        self.h_trace.copy_(state["h_trace"].to(self.h_trace.device))
                    if "decay" in state:
                        self.rtu.decay.copy_(state["decay"].to(self.rtu.decay.device))
                    if "proj_weight" in state:
                        self.rtu.proj.weight.copy_(state["proj_weight"].to(self.rtu.proj.weight.device))
        register_adapter("legacy_rtu", LegacyRTUAdapter, strict=False)
        register_adapter("rtu", LegacyRTUAdapter, strict=False)
    except Exception:
        pass
    # Delta / sparse adapters are additive: they never replace an existing
    # name, and a failure to import them must not break the builtins.
    try:
        from recamem.delta_memory import _register_delta_adapters
        _register_delta_adapters()
    except Exception:
        try:
            from .delta_memory import _register_delta_adapters
            _register_delta_adapters()
        except Exception:
            pass

_register_builtins()
