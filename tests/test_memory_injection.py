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

from src.model.backend_contracts import (
    HiddenInjection,
    TraceValidationError,
    UnsupportedCapabilityError,
)
from src.model.memory_injection import (
    inject_hidden,
    inject_kv_cache,
    inject_prompt,
    select_strategy,
)
from src.model.backend_contracts import KvPrefill


def prefill_session(backend, facts):
    """Small helper: a KvPrefill built from fact dicts."""
    return KvPrefill(backend, [fact["text"] for fact in facts])


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
# KV cache priming: only a capable binding may claim prefill
# --------------------------------------------------------------------------- #
class _ByteBackend:
    """Prompt-only backend: no tokenizer, no prefill, no hidden hook."""

    def __init__(self):
        self.cache = []

    def supports_hidden_injection(self):
        return False

    def supports_kv_priming(self):
        return False


class _CapableKvBackend:
    """Binding double exposing prefill + reset (the llama.cpp contract)."""

    def __init__(self):
        self.events = []
        self.cache = []

    def tokenize(self, text):
        return list(text.encode("utf-8"))

    def supports_kv_priming(self):
        return True

    def prime_kv_cache(self, texts):
        tokens = [token for text in texts for token in self.tokenize(text)]
        self.events.append(("prefill", len(tokens)))
        self.cache.extend(tokens)
        return {"tokens": len(tokens), "binding": "double"}

    def reset_kv_cache(self):
        self.events.append(("reset", len(self.cache)))
        self.cache.clear()


def test_inject_kv_cache_prefills_through_the_binding_then_resets():
    backend = _CapableKvBackend()
    prefill = inject_kv_cache(backend, [{"text": "a fact"}], "a query", allow_fallback=False)
    assert prefill["executed"] == "kv_cache"
    with prefill_session(backend, [{"text": "a fact"}]) as session:
        assert session.applied
        assert session.tokens == len("a fact")
        assert backend.cache == list(b"a fact")
        backend.events.append(("generate", 0))
    # Cleanup runs on exit, after generation.
    assert backend.cache == []
    assert [event[0] for event in backend.events] == ["prefill", "generate", "reset"]


def test_inject_kv_cache_unsupported_backend_raises_without_fake_tokens():
    with pytest.raises(UnsupportedCapabilityError) as exc:
        inject_kv_cache(_ByteBackend(), [{"text": "a fact"}], "a query")
    assert "kv_cache" in str(exc.value)


def test_inject_kv_cache_can_fall_back_with_an_explicit_reason():
    info = inject_kv_cache(_ByteBackend(), [{"text": "a fact"}], "a query",
                           allow_fallback=True)
    assert info["requested"] == "kv_cache"
    assert info["executed"] == "prompt"
    assert info["applied"] is False
    assert "unsupported" in info["reason"]
    assert "fact_tokens" not in info  # never fabricate token ids


def test_inject_kv_cache_rejects_a_binding_that_primes_nothing():
    class _Empty(_CapableKvBackend):
        def prime_kv_cache(self, texts):
            return {"tokens": 0}

    with pytest.raises(UnsupportedCapabilityError):
        KvPrefill(_Empty(), ["a fact"]).prime()


# --------------------------------------------------------------------------- #
# Hidden-state injection: a scoped hook, not a projection
# --------------------------------------------------------------------------- #
def _decoder(hidden=8, layers=2):
    from src.model.torch_decoder_backend import HookableDecoderBackend

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(hidden, hidden)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x + torch.tanh(self.linear(x))

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": hidden})()
            self.layers = torch.nn.ModuleList([Block() for _ in range(layers)])
            self.head = torch.nn.Linear(hidden, 26)

        def forward(self, ids):
            x = torch.nn.functional.one_hot(ids % hidden, hidden).float()
            for block in self.layers:
                x = block(x)
            return self.head(x)

    return HookableDecoderBackend(Decoder())


def test_inject_hidden_projects_to_backend_dim_and_applies_at_the_layer():
    backend = _decoder(hidden=8, layers=2)
    trace = torch.randn(384, generator=torch.Generator().manual_seed(0))
    injection = inject_hidden(backend, "q", trace, layer_idx=1)
    assert injection["layer_idx"] == 1
    assert injection["hidden_size"] == 8
    session = HiddenInjection(backend, trace, layer_idx=1)
    assert session.projected.shape == (8,)
    assert session.applied is False  # preparing is not injecting

    with session:
        backend.generate("hello", max_tokens=2)
    assert session.applied is True
    assert session.fired == 1
    assert session.cleanup_scope == "removed"


def test_inject_hidden_changes_logits_only_while_hooked():
    backend = _decoder(hidden=8, layers=2)
    trace = torch.randn(384, generator=torch.Generator().manual_seed(1))
    baseline = backend.generate("same prompt", max_tokens=2)
    baseline_logits = backend.last_logits.clone()
    with HiddenInjection(backend, trace, layer_idx=1, scale=4.0):
        injected = backend.generate("same prompt", max_tokens=2)
        injected_logits = backend.last_logits.clone()
    after = backend.generate("same prompt", max_tokens=2)
    assert not torch.allclose(baseline_logits, injected_logits)
    assert after == baseline
    assert torch.allclose(backend.last_logits, baseline_logits)


def test_inject_hidden_hook_is_removed_on_exception():
    backend = _decoder()
    layer = backend.hidden_layers()[1]
    trace = torch.randn(64, generator=torch.Generator().manual_seed(2))
    session = HiddenInjection(backend, trace, layer_idx=1)
    with pytest.raises(RuntimeError):
        with session:
            raise RuntimeError("generation failed")
    assert session._handles == []
    assert layer._forward_hooks == {}
    # A later call is unaffected.
    backend.generate("x", max_tokens=1)
    assert session.fired == 0


def test_inject_hidden_defaults_to_the_last_layer():
    backend = _decoder(hidden=8, layers=3)
    session = HiddenInjection(backend, torch.randn(16), layer_idx=None)
    assert session.layer_idx == 2


def test_inject_hidden_rejects_invalid_traces_without_touching_state_or_rng():
    backend = _decoder()
    rng_before = torch.get_rng_state().clone()
    for bad in ([[1.0, 2.0]], torch.tensor([1.0, float("nan")]), torch.tensor([]),
                torch.zeros(0)):
        with pytest.raises(TraceValidationError):
            HiddenInjection(backend, bad)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert backend.hidden_layers()[0].calls == 0


def test_inject_hidden_requires_a_capable_backend():
    with pytest.raises(UnsupportedCapabilityError):
        HiddenInjection(_ByteBackend(), torch.randn(8))
    # An explicit hidden request on an unsupported backend is a capability
    # error, not a silent fallback.
    with pytest.raises(UnsupportedCapabilityError):
        inject_hidden(_ByteBackend(), "q", torch.randn(8))


def test_inject_hidden_fallback_is_explicit_and_never_applies():
    info = inject_hidden(_ByteBackend(), "q", torch.randn(8), allow_fallback=True)
    assert info["requested"] == "hidden"
    assert info["executed"] == "prompt"
    assert info["applied"] is False
    assert "supports_hidden_injection" in info["reason"] or "capability" in info["reason"]
