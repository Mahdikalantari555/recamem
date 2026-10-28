"""Backend capability and injection-result contracts.

The point of this module is honesty about what actually ran. A backend
advertises which memory-injection operations it can perform; the orchestrator
reports the *requested* and *executed* strategy plus a reason whenever the two
differ. Nothing here fabricates tokens, projections or "injected" labels.

Contracts
---------
``BackendCapabilities``
    Immutable description of a generation backend: prompt rendering, optional
    hidden-state injection, optional KV priming, its context window and the
    output it reserves.
``fit_prompt``
    Budgets a complete prompt (facts + separators + query) against the backend
    tokenizer, reserving output space. Facts are trimmed whole; the query is
    never truncated, and an oversized query raises ``QueryBudgetError``.
``HiddenInjection``
    Scoped forward hook on a locally supplied hookable PyTorch decoder. It is
    applied during generation and removed on exit, including on exception.
``KvPrefill``
    Scoped cache prefill through the binding's own API, with reset on cleanup.
``resolve_strategy``
    Capability-aware strategy selection returning requested/executed/reason.

See ``docs/backend-contracts.md`` for the support matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PROMPT = "prompt"
HIDDEN = "hidden"
KV_CACHE = "kv_cache"
AUTO = "auto"
STRATEGIES = (PROMPT, HIDDEN, KV_CACHE)


class UnsupportedCapabilityError(RuntimeError):
    """Raised when a strategy is requested from a backend that cannot perform it."""

    def __init__(self, strategy: str, backend: str = "backend", reason: str = ""):
        self.strategy = strategy
        self.backend = backend
        self.reason = reason or f"{backend} does not advertise {strategy}"
        super().__init__(f"unsupported capability {strategy!r}: {self.reason}")


class QueryBudgetError(ValueError):
    """Raised when the user query alone does not fit the context budget."""

    def __init__(self, query_tokens: int, budget: int):
        self.query_tokens = query_tokens
        self.budget = budget
        super().__init__(
            f"query needs {query_tokens} tokens but only {budget} are available; "
            "the query is never truncated"
        )


class TraceValidationError(ValueError):
    """Raised for a trace that cannot be injected (type, shape, non-finite)."""


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
def _declared(obj: Any, name: str) -> bool:
    """True when the object declares ``name`` (so a False answer is final)."""
    return getattr(obj, name, None) is not None


def _bool_call(obj: Any, name: str) -> bool:
    value = getattr(obj, name, None)
    if value is None:
        return False
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def _int_call(obj: Any, name: str, default: Optional[int] = None) -> Optional[int]:
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return int(value() if callable(value) else value)
    except Exception:
        return default


@dataclass(frozen=True)
class BackendCapabilities:
    """What a generation backend can actually do."""

    name: str = "backend"
    prompt: bool = True
    hidden: bool = False
    kv_prime: bool = False
    context_window: int = 4096
    output_reserve: int = 512
    tokenizer: str = "bytes"

    def supports(self, strategy: str) -> bool:
        if strategy == PROMPT:
            return self.prompt
        if strategy == HIDDEN:
            return self.hidden
        if strategy == KV_CACHE:
            return self.kv_prime
        return False

    @property
    def prompt_only(self) -> bool:
        return self.prompt and not self.hidden and not self.kv_prime

    @property
    def budget(self) -> int:
        return max(0, self.context_window - self.output_reserve)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "hidden": self.hidden,
            "kv_prime": self.kv_prime,
            "context_window": self.context_window,
            "output_reserve": self.output_reserve,
            "tokenizer": self.tokenizer,
            "budget": self.budget,
        }


def capabilities_of(backend: Any) -> BackendCapabilities:
    """Infer capabilities from the backend's advertised attributes.

    A backend that advertises nothing is prompt-only, which is the safe
    default: prompt rendering never mutates model state.
    """
    if backend is None:
        return BackendCapabilities(name="none", hidden=False, kv_prime=False)
    if isinstance(backend, BackendCapabilities):
        return backend

    name = type(backend).__name__
    explicit = getattr(backend, "capabilities", None)
    if isinstance(explicit, BackendCapabilities):
        return explicit

    # An explicit declaration is authoritative, including an explicit False.
    if _declared(backend, "supports_hidden_injection"):
        hidden = _bool_call(backend, "supports_hidden_injection")
    elif _declared(backend, "supports_hidden"):
        hidden = bool(getattr(backend, "supports_hidden"))
    else:
        hidden = False

    if _declared(backend, "supports_kv_priming"):
        kv_prime = _bool_call(backend, "supports_kv_priming")
    else:
        # A binding that exposes both prefill and reset can prime its cache.
        model = getattr(backend, "model", None)
        kv_prime = callable(getattr(model, "eval", None)) and callable(
            getattr(model, "reset", None)
        )

    window = _int_call(backend, "context_window")
    if window is None:
        window = _int_call(backend, "n_ctx", 4096)
    reserve = _int_call(backend, "output_reserve", 512)
    return BackendCapabilities(
        name=name,
        prompt=True,
        hidden=bool(hidden),
        kv_prime=bool(kv_prime),
        context_window=int(window or 4096),
        output_reserve=int(reserve or 0),
        tokenizer=str(getattr(backend, "tokenizer_kind", "bytes")),
    )


def count_tokens(backend: Any, text: str) -> int:
    """Measure ``text`` with the backend's own tokenizer, else UTF-8 bytes."""
    if backend is not None:
        counter = getattr(backend, "count_tokens", None)
        if callable(counter):
            try:
                return int(counter(text))
            except Exception:
                pass
        tokenizer = getattr(backend, "tokenize", None)
        if callable(tokenizer):
            try:
                return len(tokenizer(text))
            except Exception:
                pass
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Prompt budgeting
# --------------------------------------------------------------------------- #
CONTEXT_HEADER = "Context:"
FACT_PREFIX = "- "
QUERY_PREFIX = "\n\nQuery: "


def render_prompt(query: str, fact_texts: Sequence[str]) -> str:
    if not fact_texts:
        return query
    body = "\n".join(f"{FACT_PREFIX}{text}" for text in fact_texts)
    return f"{CONTEXT_HEADER}\n{body}{QUERY_PREFIX}{query}"


@dataclass(frozen=True)
class PromptPlan:
    """A rendered prompt plus the budget accounting behind it."""

    prompt: str
    query: str
    fact_texts: Tuple[str, ...]
    tokens_used: int
    query_tokens: int
    budget: int
    context_window: int
    reserved_output: int
    trimmed_facts: int
    tokenizer: str

    @property
    def fact_count(self) -> int:
        return len(self.fact_texts)

    @property
    def prompt_ends_with_query(self) -> bool:
        return self.prompt.endswith(self.query) and (
            self.prompt == self.query or self.prompt.endswith(QUERY_PREFIX + self.query)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_count": self.fact_count,
            "tokens_used": self.tokens_used,
            "query_tokens": self.query_tokens,
            "budget": self.budget,
            "context_window": self.context_window,
            "reserved_output": self.reserved_output,
            "trimmed_facts": self.trimmed_facts,
            "tokenizer": self.tokenizer,
            "prompt_ends_with_query": self.prompt_ends_with_query,
        }


def _fact_text(fact: Any) -> str:
    if isinstance(fact, str):
        return fact.strip()
    if isinstance(fact, dict):
        return str(fact.get("text", "")).strip()
    return ""


def fit_prompt(
    query: str,
    facts: Iterable[Any],
    *,
    backend: Any = None,
    capabilities: Optional[BackendCapabilities] = None,
    max_context_tokens: Optional[int] = None,
    output_reserve: Optional[int] = None,
) -> PromptPlan:
    """Fit as many whole facts as the budget allows, keeping the query intact."""
    caps = capabilities or capabilities_of(backend)
    context_window = caps.context_window
    reserve = caps.output_reserve if output_reserve is None else int(output_reserve)
    budget = caps.budget if max_context_tokens is None else int(max_context_tokens)
    if max_context_tokens is not None and output_reserve is None:
        reserve = 0
    budget = max(0, int(budget))

    query_tokens = count_tokens(backend, QUERY_PREFIX.strip("\n") + " " + query)
    if query_tokens > budget:
        raise QueryBudgetError(query_tokens, budget)

    texts: List[str] = [text for text in (_fact_text(f) for f in facts) if text]
    accepted: List[str] = []
    for text in texts:
        candidate = render_prompt(query, accepted + [text])
        if count_tokens(backend, candidate) <= budget:
            accepted.append(text)
        else:
            # Deterministic prefix trimming: stop at the first fact that does
            # not fit rather than skipping ahead and reordering the evidence.
            break
    prompt = render_prompt(query, accepted)
    return PromptPlan(
        prompt=prompt,
        query=query,
        fact_texts=tuple(accepted),
        tokens_used=count_tokens(backend, prompt),
        query_tokens=query_tokens,
        budget=budget,
        context_window=context_window,
        reserved_output=reserve,
        trimmed_facts=len(texts) - len(accepted),
        tokenizer=caps.tokenizer if backend is None else "backend",
    )


# --------------------------------------------------------------------------- #
# Strategy selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategySelection:
    requested: str
    executed: str
    reason: str
    fact_count: int
    capabilities: Dict[str, Any] = field(default_factory=dict)

    @property
    def fell_back(self) -> bool:
        return self.executed != self.requested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.executed,
            "requested": self.requested,
            "executed": self.executed,
            "reason": self.reason,
            "fallback": self.fell_back,
            "fact_count": self.fact_count,
            "capabilities": self.capabilities,
        }


def auto_preference(fact_count: int) -> str:
    if fact_count < 5:
        return PROMPT
    if fact_count <= 20:
        return HIDDEN
    return KV_CACHE


def resolve_strategy(
    fact_count: int,
    query_type: str = "specific_fact",
    profile_config: Optional[Dict[str, Any]] = None,
    capabilities: Optional[BackendCapabilities] = None,
    *,
    requested: Optional[str] = None,
    strict: bool = False,
) -> StrategySelection:
    """Resolve the strategy that will actually execute for this backend."""
    caps = capabilities or BackendCapabilities(name="none", hidden=False)
    pinned = None
    if profile_config:
        candidate = profile_config.get("injection_strategy")
        if candidate in STRATEGIES:
            pinned = candidate
    if requested in STRATEGIES:
        target = requested
    elif pinned is not None:
        target = pinned
    elif requested == AUTO or pinned is None:
        target = AUTO
    else:
        target = AUTO

    if target == AUTO:
        preferred = auto_preference(int(fact_count))
        if caps.supports(preferred):
            return StrategySelection(AUTO, preferred, "auto", int(fact_count), caps.to_dict())
        return StrategySelection(
            AUTO, PROMPT, f"unsupported_capability:{preferred}", int(fact_count),
            caps.to_dict(),
        )

    if caps.supports(target):
        return StrategySelection(target, target, "explicit", int(fact_count), caps.to_dict())
    if strict:
        raise UnsupportedCapabilityError(target, caps.name)
    if caps.prompt_only:
        return StrategySelection(
            target, PROMPT, f"prompt_only_backend:{caps.name}", int(fact_count),
            caps.to_dict(),
        )
    return StrategySelection(
        target, PROMPT, f"unsupported_capability:{target}", int(fact_count), caps.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Hidden-state injection
# --------------------------------------------------------------------------- #
def _validate_trace(trace: Any) -> torch.Tensor:
    if not isinstance(trace, torch.Tensor):
        raise TraceValidationError(
            f"hidden trace must be a torch.Tensor, got {type(trace).__name__}"
        )
    tensor = trace.detach().float()
    if tensor.dim() == 0:
        raise TraceValidationError("hidden trace must have at least one dimension")
    if tensor.dim() > 1:
        tensor = tensor.reshape(-1) if tensor.numel() > 0 else tensor.reshape(0)
        if tensor.dim() != 1:
            raise TraceValidationError(
                f"hidden trace must be 1-D, got shape {tuple(trace.shape)}"
            )
    if tensor.numel() == 0:
        raise TraceValidationError("hidden trace is empty")
    if not bool(torch.isfinite(tensor).all()):
        raise TraceValidationError("hidden trace contains non-finite values")
    return tensor


def project_trace(trace: torch.Tensor, hidden_size: int, *, seed: int = 0xC0FFEE) -> torch.Tensor:
    """Deterministic projection onto ``hidden_size`` using a local generator.

    The generator is local: global torch RNG state is untouched, so an injection
    can never change sampling for the rest of the process.
    """
    tensor = trace.detach().float()
    source = tensor.numel()
    if source == hidden_size:
        return tensor.reshape(hidden_size).clone()
    generator = torch.Generator(device="cpu").manual_seed(seed ^ (source * 131 + hidden_size))
    matrix = torch.randn(source, hidden_size, generator=generator) / math.sqrt(max(1, source))
    return (tensor.reshape(source) @ matrix).clone()


class HiddenInjection:
    """Scoped residual-stream injection for a hookable PyTorch decoder.

    ``with HiddenInjection(backend, trace, layer_idx=12):`` installs one forward
    hook that adds the projected trace to that layer's output. The hook is
    removed on exit, including when generation raises, so no later request or
    profile inherits the trace.
    """

    def __init__(
        self,
        backend: Any,
        trace: torch.Tensor,
        *,
        layer_idx: Optional[int] = None,
        scale: float = 1.0,
        seed: int = 0xC0FFEE,
    ):
        self.backend = backend
        self.capabilities = capabilities_of(backend)
        if not self.capabilities.hidden:
            raise UnsupportedCapabilityError(HIDDEN, self.capabilities.name)
        self.trace = _validate_trace(trace)
        self.layers = self._resolve_layers(backend)
        self.hidden_size = self._resolve_hidden_size(backend)
        if layer_idx is None:
            declared = getattr(backend, "injection_layer", None)
            try:
                layer_idx = int(declared()) if callable(declared) else (
                    int(declared) if declared is not None else len(self.layers) - 1
                )
            except Exception:
                layer_idx = len(self.layers) - 1
        self.layer_idx = int(layer_idx)
        if not 0 <= self.layer_idx < len(self.layers):
            raise ValueError(
                f"layer_idx {self.layer_idx} is out of range for {len(self.layers)} layers"
            )
        self.scale = float(scale)
        self.projected = project_trace(self.trace, self.hidden_size, seed=seed) * self.scale
        self.fired = 0
        self._handles: List[Any] = []
        self._closed = False

    @staticmethod
    def _resolve_layers(backend: Any) -> Sequence[torch.nn.Module]:
        getter = getattr(backend, "hidden_layers", None)
        layers = getter() if callable(getter) else None
        if not layers:
            raise UnsupportedCapabilityError(
                HIDDEN, type(backend).__name__, "backend exposes no hidden_layers()"
            )
        return list(layers)

    @staticmethod
    def _resolve_hidden_size(backend: Any) -> int:
        getter = getattr(backend, "hidden_size", None)
        if callable(getter):
            return int(getter())
        value = getattr(backend, "hidden_size", None)
        if isinstance(value, int):
            return value
        model = getattr(backend, "model", None)
        n_embd = getattr(model, "n_embd", None)
        if callable(n_embd):
            return int(n_embd())
        if isinstance(n_embd, int):
            return n_embd
        raise UnsupportedCapabilityError(
            HIDDEN, type(backend).__name__, "backend exposes no hidden size"
        )

    # -- lifecycle ------------------------------------------------------ #
    def _hook(self, module, inputs, output):
        if self._closed:
            return output
        delta = self.projected.to(dtype=torch.float32)
        self.fired += 1

        def add(tensor: torch.Tensor) -> torch.Tensor:
            contribution = delta.to(device=tensor.device, dtype=tensor.dtype)
            if tensor.dim() == 0:
                return tensor + contribution[0]
            while contribution.dim() > tensor.dim():
                contribution = contribution.sum(dim=0)
            return tensor + contribution

        if isinstance(output, tuple):
            if not output or not torch.is_tensor(output[0]):
                return output
            return (add(output[0]), *output[1:])
        if torch.is_tensor(output):
            return add(output)
        return output

    def apply(self) -> "HiddenInjection":
        if self._closed:
            raise RuntimeError("injection has already been closed")
        if self._handles:
            return self
        handle = self.layers[self.layer_idx].register_forward_hook(self._hook)
        self._handles.append(handle)
        return self

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> "HiddenInjection":
        return self.apply()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.remove()
        return False

    @property
    def applied(self) -> bool:
        return self.fired > 0

    @property
    def cleanup_scope(self) -> str:
        return "forward-hook" if not self._closed else "removed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": HIDDEN,
            "requested": HIDDEN,
            "executed": HIDDEN if self.applied else HIDDEN,
            "applied": self.applied,
            "layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
            "trace_norm": float(torch.linalg.vector_norm(self.trace).item()),
            "projected_norm": float(torch.linalg.vector_norm(self.projected).item()),
            "router_calls": self.fired,
            "cleanup": self.cleanup_scope,
            "capabilities": self.capabilities.to_dict(),
        }


def inject_hidden(backend: Any, trace: torch.Tensor, *, layer_idx: Optional[int] = None,
                  scale: float = 1.0) -> HiddenInjection:
    """Build (but do not apply) a scoped hidden injection."""
    return HiddenInjection(backend, trace, layer_idx=layer_idx, scale=scale)


# --------------------------------------------------------------------------- #
# KV cache prefill
# --------------------------------------------------------------------------- #
class KvPrefill:
    """Scoped KV-cache prefill through the backend binding.

    The backend must implement ``prime_kv_cache(texts)`` returning the number of
    primed tokens (or a mapping with ``tokens``) and ``reset_kv_cache()``.
    """

    def __init__(self, backend: Any, fact_texts: Sequence[str]):
        self.backend = backend
        self.capabilities = capabilities_of(backend)
        if not self.capabilities.kv_prime:
            raise UnsupportedCapabilityError(KV_CACHE, self.capabilities.name)
        self.fact_texts = [text for text in fact_texts if text]
        self.tokens = 0
        self.binding: Dict[str, Any] = {}
        self._primed = False
        self._cleaned = False
        self._ever_applied = False

    def prime(self) -> "KvPrefill":
        if self._primed:
            return self
        prime = getattr(self.backend, "prime_kv_cache", None)
        if not callable(prime):
            raise UnsupportedCapabilityError(
                KV_CACHE, self.capabilities.name, "backend has no prime_kv_cache()"
            )
        result = prime(list(self.fact_texts))
        if isinstance(result, dict):
            tokens = int(result.get("tokens", 0))
            self.binding = dict(result)
        else:
            tokens = int(result)
            self.binding = {"tokens": tokens}
        if tokens <= 0:
            raise UnsupportedCapabilityError(
                KV_CACHE, self.capabilities.name,
                "backend reported no primed tokens; refusing to claim prefill",
            )
        self.tokens = tokens
        self._primed = True
        self._ever_applied = True
        return self

    def cleanup(self) -> None:
        if self._cleaned:
            return
        reset = getattr(self.backend, "reset_kv_cache", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        self._primed = False
        self._cleaned = True

    def __enter__(self) -> "KvPrefill":
        return self.prime()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False

    @property
    def applied(self) -> bool:
        """True when a prefill actually happened, even after cleanup."""
        return self._ever_applied and self.tokens > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": KV_CACHE,
            "requested": KV_CACHE,
            "executed": KV_CACHE if self.applied else KV_CACHE,
            "applied": self.applied,
            "fact_count": len(self.fact_texts),
            "tokens": self.tokens,
            "primed_tokens": self.tokens,
            "binding": self.binding,
            "cleanup": "released" if self._cleaned else "held",
            "capabilities": self.capabilities.to_dict(),
        }


def inject_kv_cache(backend: Any, facts: Iterable[Any]) -> KvPrefill:
    """Build (but do not prime) a scoped KV prefill from fact records."""
    return KvPrefill(backend, [_fact_text(fact) for fact in facts])


__all__ = [
    "AUTO",
    "BackendCapabilities",
    "CONTEXT_HEADER",
    "HIDDEN",
    "HiddenInjection",
    "KV_CACHE",
    "KvPrefill",
    "PROMPT",
    "PromptPlan",
    "QueryBudgetError",
    "STRATEGIES",
    "StrategySelection",
    "TraceValidationError",
    "UnsupportedCapabilityError",
    "auto_preference",
    "capabilities_of",
    "count_tokens",
    "fit_prompt",
    "inject_hidden",
    "inject_kv_cache",
    "project_trace",
    "render_prompt",
    "resolve_strategy",
]
