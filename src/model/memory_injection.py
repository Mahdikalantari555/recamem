"""Memory-injection helpers: prompt budgeting, strategy selection, scoped ops.

Every function here reports what actually happened. A projection that was
prepared but never applied is not injection, token IDs that were never fed to a
cache are not priming, and a fallback names the capability it lacked.

The heavy lifting lives in :mod:`src.model.backend_contracts`; this module keeps
the historical entry points used by the engine, CLI and tests.
"""

from typing import Any, Dict, List, Optional

import torch

from src.model.backend_contracts import (
    AUTO,
    HIDDEN,
    KV_CACHE,
    PROMPT,
    STRATEGIES,
    BackendCapabilities,
    HiddenInjection,
    KvPrefill,
    PromptPlan,
    QueryBudgetError,
    StrategySelection,
    TraceValidationError,
    UnsupportedCapabilityError,
    auto_preference,
    capabilities_of,
    count_tokens,
    fit_prompt,
    inject_kv_cache as _build_kv_prefill,
    inject_hidden as _build_hidden,
    resolve_strategy as _resolve_strategy,
)

__all__ = [
    "HIDDEN",
    "KV_CACHE",
    "PROMPT",
    "BackendCapabilities",
    "HiddenInjection",
    "KvPrefill",
    "PromptPlan",
    "QueryBudgetError",
    "StrategySelection",
    "TraceValidationError",
    "UnsupportedCapabilityError",
    "inject_hidden",
    "inject_kv_cache",
    "inject_prompt",
    "plan_prompt",
    "resolve_strategy",
    "select_strategy",
]


def plan_prompt(
    query: str,
    facts: List[Dict[str, Any]],
    *,
    backend: Any = None,
    capabilities: Optional[BackendCapabilities] = None,
    max_context_tokens: Optional[int] = None,
) -> PromptPlan:
    """Budget a complete prompt against the backend tokenizer."""
    return fit_prompt(
        query,
        facts,
        backend=backend,
        capabilities=capabilities,
        max_context_tokens=max_context_tokens,
    )


def inject_prompt(
    query: str,
    facts: List[Dict[str, Any]],
    max_context_tokens: int = 3000,
    backend: Any = None,
) -> str:
    """Render ``Context: ... Query: <query>`` within the prompt budget.

    The query is preserved verbatim and stays last; facts that do not fit are
    dropped whole. An oversized query raises :class:`QueryBudgetError` instead
    of being truncated.
    """
    if not facts:
        return query
    plan = fit_prompt(
        query,
        facts,
        backend=backend,
        max_context_tokens=max_context_tokens,
        output_reserve=0,
    )
    return plan.prompt


def inject_hidden(
    backend: Any,
    query: str,
    h_trace: torch.Tensor,
    layer_idx: int = 12,
    *,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Prepare a scoped hidden injection for ``backend``.

    Returns the injection result; it is *not* applied until the caller enters
    the returned :class:`HiddenInjection` context (the engine does this around
    generation). A backend without hidden support raises
    :class:`UnsupportedCapabilityError` unless ``allow_fallback`` is set, in
    which case the result reports ``executed="prompt"`` and the reason.
    """
    caps = capabilities_of(backend)
    try:
        injection = _build_hidden(backend, h_trace, layer_idx=layer_idx)
    except (UnsupportedCapabilityError, TraceValidationError) as exc:
        if isinstance(exc, TraceValidationError) or not allow_fallback:
            raise
        return {
            "strategy": HIDDEN,
            "requested": HIDDEN,
            "executed": PROMPT,
            "applied": False,
            "reason": str(exc),
            "fallback": PROMPT,
            "query": query,
            "capabilities": caps.to_dict(),
        }
    info = injection.to_dict()
    info.update({"query": query, "requested": HIDDEN, "executed": HIDDEN})
    return info


def inject_kv_cache(
    backend: Any,
    facts: List[Dict[str, Any]],
    query: str,
    *,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Prepare a scoped KV prefill; the caller primes it around generation."""
    caps = capabilities_of(backend)
    try:
        prefill = _build_kv_prefill(backend, facts)
    except UnsupportedCapabilityError as exc:
        if not allow_fallback:
            raise
        return {
            "strategy": KV_CACHE,
            "requested": KV_CACHE,
            "executed": PROMPT,
            "applied": False,
            "reason": str(exc),
            "fallback": PROMPT,
            "fact_count": len(facts),
            "query": query,
            "capabilities": caps.to_dict(),
        }
    info = prefill.to_dict()
    info.update({"query": query, "requested": KV_CACHE, "executed": KV_CACHE})
    return info


def resolve_strategy(
    fact_count: int,
    query_type: str = "specific_fact",
    profile_config: Optional[Dict[str, Any]] = None,
    capabilities: Optional[BackendCapabilities] = None,
    *,
    requested: Optional[str] = None,
    strict: bool = False,
) -> StrategySelection:
    return _resolve_strategy(
        fact_count,
        query_type,
        profile_config,
        capabilities,
        requested=requested,
        strict=strict,
    )


def select_strategy(
    fact_count: int,
    query_type: str,
    profile_config: Optional[Dict] = None,
    capabilities: Optional[BackendCapabilities] = None,
) -> str:
    """Return the preferred strategy, or the executed one when capabilities are known.

    Without a backend there is nothing to check, so this returns the request
    that *would* be made (pinned setting first, then the documented thresholds).
    When capabilities are supplied the answer reflects what can actually run.
    """
    pinned = None
    if profile_config:
        candidate = profile_config.get("injection_strategy")
        if candidate in STRATEGIES:
            pinned = candidate
    if capabilities is None:
        return pinned if pinned is not None else auto_preference(int(fact_count))
    return _resolve_strategy(
        fact_count, query_type, profile_config, capabilities, requested=None
    ).executed


def strategy_report(
    fact_count: int,
    query_type: str = "specific_fact",
    profile_config: Optional[Dict[str, Any]] = None,
    capabilities: Optional[BackendCapabilities] = None,
    *,
    requested: Optional[str] = None,
) -> Dict[str, Any]:
    """Diagnostics view of strategy resolution (requested/executed/reason)."""
    return _resolve_strategy(
        fact_count, query_type, profile_config, capabilities, requested=requested
    ).to_dict()
