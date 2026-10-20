"""Contract tests for src/model/memory_injection.py.

Invariants under test (see openspec/specs/model/memory-injection):
- the prompt strategy renders a ``Context:`` block and places ``Query:`` last;
- the strategy selector honours a pinned profile config and otherwise picks by
  fact count (<5 prompt, 5-20 hidden, >20 kv_cache);
- the byte-backend fallback in ``inject_kv_cache`` is ``[1, 2, 3]`` and never a
  silent success for the KV path;
- the hidden-state path either projects to the backend hidden size or reports an
  error with a prompt fallback - it never silently returns a wrong shape.
"""

import pytest
import torch

from src.model.memory_injection import (
    inject_hidden,
    inject_kv_cache,
    inject_prompt,
    select_strategy,
)


# --------------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------------- #
def test_inject_prompt_without_facts_returns_query():
    assert inject_prompt("What is radar?", []) == "What is radar?"
    assert inject_prompt("What is radar?", [{"text": "   "}]) == "What is radar?"


def test_inject_prompt_layout_puts_query_last():
    facts = [{"text": "Radar uses radio waves."}, {"text": "Lidar uses lasers."}]
    prompt = inject_prompt("Compare radar and lidar", facts)

    lines = prompt.splitlines()
    assert lines[0] == "Context:"
    assert "- Radar uses radio waves." in lines
    assert "- Lidar uses lasers." in lines
    assert prompt.index("Query:") > prompt.index("Context:")
    assert prompt.endswith("Query: Compare radar and lidar")
    assert prompt.count("Query:") == 1
    assert prompt.count("Context:") == 1


def test_inject_prompt_respects_token_budget():
    facts = [{"text": "x" * 400} for _ in range(20)]
    prompt = inject_prompt("q", facts, max_context_tokens=50)
    # The budget is exceeded by the first fact only, so no facts fit.
    assert prompt == "q"

    prompt = inject_prompt("q", facts, max_context_tokens=1000)
    assert prompt.startswith("Context:")
    assert prompt.count("- ") < len(facts)


def test_inject_prompt_skips_blank_facts_only():
    facts = [{"text": ""}, {"text": "Only survivor."}]
    prompt = inject_prompt("q", facts)
    assert "- Only survivor." in prompt
    assert prompt.count("- ") == 1


# --------------------------------------------------------------------------- #
# Strategy selection
# --------------------------------------------------------------------------- #
def test_select_strategy_auto_thresholds():
    assert select_strategy(0, "definitional") == "prompt"
    assert select_strategy(4, "specific_fact") == "prompt"
    assert select_strategy(5, "relational") == "hidden"
    assert select_strategy(20, "temporal") == "hidden"
    assert select_strategy(21, "ambiguous") == "kv_cache"


def test_select_strategy_pinned_config_wins():
    for pinned in ("prompt", "hidden", "kv_cache"):
        assert select_strategy(30, "definitional", {"injection_strategy": pinned}) == pinned


def test_select_strategy_auto_and_invalid_pins_fall_through():
    assert select_strategy(30, "definitional", {"injection_strategy": "auto"}) == "kv_cache"
    assert select_strategy(1, "definitional", {"injection_strategy": "bogus"}) == "prompt"
    assert select_strategy(6, "definitional", {"other_key": "prompt"}) == "hidden"


# --------------------------------------------------------------------------- #
# KV cache priming
# --------------------------------------------------------------------------- #
class _ByteBackend:
    """Backend without a tokenizer: exercises the documented mock fallback."""


class _TokenizerBackend:
    def tokenize(self, text: str):
        return [len(text), len(text) + 1]


def test_inject_kv_cache_uses_byte_fallback_without_tokenizer():
    result = inject_kv_cache(_ByteBackend(), [{"text": "a fact"}], "a query")
    assert result["strategy"] == "kv_cache"
    assert result["fact_tokens"] == [1, 2, 3]
    assert result["query_tokens"] == [4, 5, 6]
    assert result["fact_count"] == 1
    assert "a fact" in result["prompt"]
    assert result["prompt"].endswith("a query")


def test_inject_kv_cache_uses_backend_tokenizer():
    result = inject_kv_cache(_TokenizerBackend(), [{"text": "abcdef"}], "q")
    assert result["fact_tokens"] == [6, 7]
    assert result["query_tokens"] == [1, 2]
    assert result["fact_count"] == 1


def test_inject_kv_cache_failure_is_reported_with_prompt_fallback():
    class _Boom:
        def tokenize(self, text):
            raise RuntimeError("tokenizer offline")

    result = inject_kv_cache(_Boom(), [{"text": "fallback fact"}], "q")
    assert result["strategy"] == "kv_cache"
    assert "tokenizer offline" in result["error"]
    assert result["fallback"] == "prompt"
    assert "fallback fact" in result["prompt"]


# --------------------------------------------------------------------------- #
# Hidden-state injection
# --------------------------------------------------------------------------- #
class _ModelWithEmbDim:
    @staticmethod
    def n_embd():
        return 32


class _HiddenBackend:
    model = _ModelWithEmbDim()


def test_inject_hidden_projects_to_backend_dim():
    trace = torch.randn(384, generator=torch.Generator().manual_seed(0))
    result = inject_hidden(_HiddenBackend(), "q", trace, layer_idx=7)
    assert result["strategy"] == "hidden"
    assert result["layer_idx"] == 7
    assert result["projected_trace"].shape == (32,)
    assert result["query"] == "q"
    assert "layer 7" in result["injection"]


def test_inject_hidden_defaults_to_llama_hidden_size():
    trace = torch.randn(384, generator=torch.Generator().manual_seed(0))
    result = inject_hidden(object(), "q", trace)
    assert result["layer_idx"] == 12
    assert result["projected_trace"].shape == (4096,)


def test_inject_hidden_pads_or_truncates_nonstandard_trace():
    backend = _HiddenBackend()
    for trace in (torch.randn(100), torch.randn(500)):
        result = inject_hidden(backend, "q", trace)
        assert result["projected_trace"].shape == (32,)


def test_inject_hidden_reports_error_with_prompt_fallback():
    class _BrokenBackend:
        class model:
            @staticmethod
            def n_embd():
                raise RuntimeError("hidden size unavailable")

    trace = torch.randn(384, generator=torch.Generator().manual_seed(1))
    result = inject_hidden(_BrokenBackend(), "q", trace)
    assert result["strategy"] == "hidden"
    assert "hidden size unavailable" in result["error"]
    assert result["fallback"] == "prompt"
    assert result["query"] == "q"
    # The fallback is a prompt-stuffed memory note, not an empty string.
    assert result["prompt"].startswith("Query: q") or "Memory trace norm" in result["prompt"]
    assert "Memory trace norm" in result["prompt"]
