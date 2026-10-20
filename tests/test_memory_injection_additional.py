"""Contract tests for `src/model/memory_injection.py` (surface-test-coverage 3.4).

Covered: the prompt layout with the query last, the byte-backend `tokenize()`
fallback, a pinned config returning the configured strategy, and explicit
handling of the hidden/KV stub paths.
"""

import torch
from types import SimpleNamespace

import pytest

from src.model.memory_injection import inject_hidden, inject_kv_cache, inject_prompt, select_strategy


def test_inject_prompt_returns_query_when_no_facts():
    assert inject_prompt("what is the query?", []) == "what is the query?"


def test_inject_prompt_renders_context_then_query_last():
    facts = [{"text": "Fact one about the topic."}, {"text": "Fact two about the topic."}]
    prompt = inject_prompt("what is the query?", facts)

    assert "Context:" in prompt
    assert "Query: what is the query?" in prompt
    context_part, _, _ = prompt.partition("\n\nQuery:")
    assert "Fact one" in context_part
    assert "Fact two" in context_part


def test_inject_prompt_respects_max_context_tokens():
    long_fact = {"text": "x" * 5000}
    prompt = inject_prompt("q", [long_fact, long_fact, long_fact], max_context_tokens=100)
    # No single fact fits, so the context block collapses and only the query
    # is returned: output stays bounded and still ends with the query.
    assert prompt == "q"


def test_inject_prompt_keeps_facts_that_fit():
    facts = [{"text": "short fact."}, {"text": "x" * 5000}, {"text": "another short fact."}]
    prompt = inject_prompt("q", facts, max_context_tokens=100)
    assert "short fact." in prompt
    assert "Query: q" in prompt
    assert "x" * 5000 not in prompt


def _raising_n_embd():
    def n_embd():
        raise AttributeError("no n_embd")

    return n_embd


def test_inject_hidden_projects_trace_and_reports_layer():
    h_trace = torch.randn(384)
    backend = SimpleNamespace(model=SimpleNamespace(n_embd=lambda: 32))
    info = inject_hidden(backend, "query text", h_trace, layer_idx=6)

    assert info["strategy"] == "hidden"
    assert info["layer_idx"] == 6
    assert info["query"] == "query text"
    assert info["projected_trace"].shape == (32,)
    assert "Hidden injection at layer 6" in info["injection"]


def test_inject_hidden_handles_backend_without_model():
    h_trace = torch.randn(20)
    info = inject_hidden(None, "query text", h_trace)
    assert info["strategy"] == "hidden"
    assert info["projected_trace"].shape == (4096,)


def test_inject_hidden_falls_back_to_prompt_on_error():
    h_trace = torch.randn(384)
    backend = SimpleNamespace(model=SimpleNamespace(n_embd=_raising_n_embd()))
    info = inject_hidden(backend, "query text", h_trace)
    assert info["fallback"] == "prompt"
    assert "Context:" in info["prompt"]
    assert "Query: query text" in info["prompt"]


def test_inject_kv_cache_uses_backend_tokenize():
    calls = []

    class FakeBackend:
        def tokenize(self, text):
            calls.append(text)
            return [len(text) % 1000]

    facts = [{"text": "fact alpha"}, {"text": "fact beta"}]
    info = inject_kv_cache(FakeBackend(), facts, "query text")

    assert info["strategy"] == "kv_cache"
    assert info["fact_count"] == 2
    assert info["fact_tokens"] == [len("fact alpha\nfact beta") % 1000]
    assert info["query_tokens"] == [len("query text") % 1000]
    assert calls == ["fact alpha\nfact beta", "query text"]


def test_inject_kv_cache_falls_back_when_backend_lacks_tokenize():
    info = inject_kv_cache(object(), [{"text": "a fact"}], "a query")
    assert info["strategy"] == "kv_cache"
    assert info["fact_tokens"] == [1, 2, 3]
    assert info["query_tokens"] == [4, 5, 6]
    assert "prompt" in info


def test_select_strategy_respects_pinned_config():
    assert select_strategy(1, "x", profile_config={"injection_strategy": "hidden"}) == "hidden"
    assert select_strategy(100, "x", profile_config={"injection_strategy": "kv_cache"}) == "kv_cache"
    # Invalid config values fall through to auto-selection.
    assert select_strategy(1, "x", profile_config={"injection_strategy": "nope"}) == "prompt"


def test_select_strategy_auto_selection():
    assert select_strategy(0, "x") == "prompt"
    assert select_strategy(4, "x") == "prompt"
    assert select_strategy(5, "x") == "hidden"
    assert select_strategy(20, "x") == "hidden"
    assert select_strategy(21, "x") == "kv_cache"
    assert select_strategy(100, "x") == "kv_cache"
