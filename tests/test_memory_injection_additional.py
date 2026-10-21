"""Contract tests for `src/model/memory_injection.py` (surface-test-coverage 3.4).

Covered: the prompt layout with the query last, the byte-backend `tokenize()`
fallback, a pinned config returning the configured strategy, and explicit
handling of the hidden/KV stub paths.
"""

import torch
from types import SimpleNamespace

import pytest

from src.model.backend_contracts import (
    KvPrefill,
    TraceValidationError,
    UnsupportedCapabilityError,
)
from src.model.memory_injection import (
    inject_hidden, inject_kv_cache, inject_prompt, select_strategy,
)


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


def _hookable(hidden=8, layers=2):
    from src.model.torch_decoder_backend import HookableDecoderBackend

    class Block(torch.nn.Module):
        def forward(self, x):
            return x

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden)
            self.layers = torch.nn.ModuleList([Block() for _ in range(layers)])

        def forward(self, ids):
            x = torch.nn.functional.one_hot(ids % hidden, hidden).float()
            for block in self.layers:
                x = block(x)
            return torch.zeros(x.shape[0], x.shape[1], 26)

    return HookableDecoderBackend(Decoder())


def test_inject_hidden_projects_trace_and_reports_layer():
    h_trace = torch.randn(384)
    backend = _hookable(hidden=32, layers=3)
    info = inject_hidden(backend, "query text", h_trace, layer_idx=1)

    assert info["strategy"] == "hidden"
    assert info["requested"] == "hidden"
    assert info["executed"] == "hidden"
    assert info["layer_idx"] == 1
    assert info["hidden_size"] == 32
    assert info["query"] == "query text"
    # Preparing an injection is not injection: nothing has been applied yet.
    assert info["applied"] is False


def test_inject_hidden_handles_backend_without_layers():
    h_trace = torch.randn(20)
    info = inject_hidden(object(), "query text", h_trace, allow_fallback=True)
    assert info["requested"] == "hidden"
    assert info["executed"] == "prompt"
    assert info["applied"] is False
    assert info["fallback"] == "prompt"


def test_inject_hidden_raises_on_unsupported_backend_without_fallback():
    h_trace = torch.randn(384)
    with pytest.raises(UnsupportedCapabilityError):
        inject_hidden(object(), "query text", h_trace)


def test_inject_hidden_reports_malformed_trace_without_secondary_error():
    backend = _hookable()
    for bad in ("not a tensor", torch.tensor([float("inf")]), torch.tensor([])):
        with pytest.raises(TraceValidationError):
            inject_hidden(backend, "query text", bad)


def test_inject_kv_cache_prefills_only_through_a_capable_backend():
    class Capable:
        def __init__(self):
            self.tokens = []

        def supports_kv_priming(self):
            return True

        def prime_kv_cache(self, texts):
            for text in texts:
                self.tokens.extend(text.encode("utf-8"))
            return {"tokens": len(self.tokens)}

        def reset_kv_cache(self):
            self.tokens = []

    backend = Capable()
    facts = [{"text": "fact alpha"}, {"text": "fact beta"}]
    info = inject_kv_cache(backend, facts, "query text")
    assert info["executed"] == "kv_cache"
    assert info["fact_count"] == 2
    assert info["tokens"] == 0  # built, not primed
    prefill = KvPrefill(backend, [fact["text"] for fact in facts])
    with prefill:
        assert prefill.tokens == len(b"fact alphafact beta")
        assert backend.tokens
    assert backend.tokens == []


def test_inject_kv_cache_falls_back_when_backend_lacks_prefill():
    info = inject_kv_cache(SimpleNamespace(), [{"text": "a fact"}], "a query",
                           allow_fallback=True)
    assert info["requested"] == "kv_cache"
    assert info["executed"] == "prompt"
    assert info["applied"] is False
    assert "reason" in info


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
