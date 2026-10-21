"""Contract tests for backend capabilities, prompt budgeting and scoped ops.

Covers post-merge-completion tasks 3.1, 3.2, 3.5 and 3.6:
- capabilities are declared, not guessed, and an explicit ``False`` is final;
- prompt budgets count the whole prompt (facts, separators, query) with the
  backend tokenizer or UTF-8 bytes, reserve output space, keep the query last
  and never truncate it;
- strategy resolution reports requested/executed/reason;
- invalid traces fail without mutating state or global RNG.
"""

import json

import pytest
import torch

from src.model.backend_contracts import (
    AUTO,
    HIDDEN,
    KV_CACHE,
    PROMPT,
    BackendCapabilities,
    HiddenInjection,
    KvPrefill,
    QueryBudgetError,
    TraceValidationError,
    UnsupportedCapabilityError,
    capabilities_of,
    count_tokens,
    fit_prompt,
    project_trace,
    render_prompt,
    resolve_strategy,
)


class _PromptOnly:
    def supports_hidden_injection(self):
        return False

    def supports_kv_priming(self):
        return False


class _Capable:
    def __init__(self, window=128, reserve=32):
        self._window = window
        self._reserve = reserve

    def supports_hidden_injection(self):
        return True

    def supports_kv_priming(self):
        return True

    def context_window(self):
        return self._window

    def output_reserve(self):
        return self._reserve

    def count_tokens(self, text):
        return len(text.encode("utf-8"))

    # -- hidden --------------------------------------------------------- #
    def hidden_layers(self):
        return self.layers

    def hidden_size(self):
        return 4

    # -- kv ------------------------------------------------------------- #
    def prime_kv_cache(self, texts):
        self.primed = sum(len(text) for text in texts)
        return {"tokens": self.primed}

    def reset_kv_cache(self):
        self.primed = 0


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
def test_advertised_capabilities_are_authoritative():
    caps = capabilities_of(_PromptOnly())
    assert caps.prompt is True
    assert caps.hidden is False
    assert caps.kv_prime is False
    assert caps.prompt_only is True


def test_an_explicit_false_is_not_overridden_by_duck_typing():
    class Module(torch.nn.Module):
        pass

    class Backend:
        model = Module()  # nn.Module has .eval(), but this backend says no

        def supports_kv_priming(self):
            return False

    assert capabilities_of(Backend()).kv_prime is False


def test_undeclared_backend_with_prefill_and_reset_can_prime():
    class Binding:
        def eval(self, tokens):
            return tokens

        def reset(self):
            return None

    class Backend:
        model = Binding()

    assert capabilities_of(Backend()).kv_prime is True


def test_budget_reserves_output_space():
    caps = capabilities_of(_Capable(window=128, reserve=32))
    assert caps.budget == 96
    assert caps.to_dict()["budget"] == 96


def test_none_backend_is_prompt_only():
    caps = capabilities_of(None)
    assert caps.hidden is False and caps.kv_prime is False
    assert caps.prompt is True


# --------------------------------------------------------------------------- #
# Prompt budgeting
# --------------------------------------------------------------------------- #
def test_prompt_layout_keeps_the_query_last():
    plan = fit_prompt("what is radar?", [{"text": "Radar uses radio."}])
    assert plan.prompt == "Context:\n- Radar uses radio.\n\nQuery: what is radar?"
    assert plan.prompt_ends_with_query
    assert plan.fact_count == 1


def test_budget_counts_separators_and_reserved_output():
    backend = _Capable(window=48, reserve=8)  # 40 tokens for the whole prompt
    query = "q"
    facts = [{"text": "a" * 10}, {"text": "b" * 10}, {"text": "c" * 10}]
    plan = fit_prompt(query, facts, backend=backend)
    assert plan.budget == 40
    assert plan.tokens_used <= plan.budget
    assert plan.fact_count < len(facts)
    assert plan.trimmed_facts == len(facts) - plan.fact_count
    assert plan.prompt_ends_with_query


def test_multibyte_text_is_counted_in_bytes_not_characters():
    # "é" is 2 UTF-8 bytes; a character/4 estimate would accept far more.
    backend = _Capable(window=40, reserve=0)
    plan = fit_prompt("q", [{"text": "é" * 20}], backend=backend)
    assert count_tokens(backend, "é" * 20) == 40
    assert plan.fact_count == 0  # the 20-char fact needs 40 bytes plus scaffold
    assert plan.prompt == "q"


def test_exact_limit_is_inclusive():
    fact = {"text": "abcd"}
    plan_free = fit_prompt("q", [fact])
    exact = count_tokens(None, plan_free.prompt)
    plan = fit_prompt("q", [fact], max_context_tokens=exact)
    assert plan.fact_count == 1
    plan_tight = fit_prompt("q", [fact], max_context_tokens=exact - 1)
    assert plan_tight.fact_count == 0


def test_oversized_query_raises_instead_of_truncating():
    with pytest.raises(QueryBudgetError) as exc:
        fit_prompt("q" * 100, [{"text": "fact"}], max_context_tokens=10)
    assert exc.value.query_tokens > exc.value.budget
    assert "never truncated" in str(exc.value)


def test_no_fitting_facts_returns_the_query_unchanged():
    plan = fit_prompt("query", [{"text": "x" * 500}], max_context_tokens=20)
    assert plan.prompt == "query"
    assert plan.fact_count == 0
    assert plan.trimmed_facts == 1


def test_trimming_is_a_deterministic_prefix():
    facts = [{"text": f"fact{index}"} for index in range(6)]
    first = fit_prompt("q", facts, max_context_tokens=60)
    second = fit_prompt("q", facts, max_context_tokens=60)
    assert first.fact_texts == second.fact_texts
    assert first.fact_texts == tuple(f"fact{i}" for i in range(len(first.fact_texts)))


def test_blank_facts_are_skipped_not_counted():
    plan = fit_prompt("q", [{"text": "  "}, {"text": "real"}])
    assert plan.fact_texts == ("real",)
    assert plan.trimmed_facts == 0


def test_render_prompt_without_facts_is_the_query():
    assert render_prompt("q", []) == "q"


# --------------------------------------------------------------------------- #
# Strategy resolution
# --------------------------------------------------------------------------- #
def test_auto_follows_documented_thresholds_when_capable():
    caps = capabilities_of(_Capable())
    assert resolve_strategy(3, capabilities=caps).executed == PROMPT
    assert resolve_strategy(15, capabilities=caps).executed == HIDDEN
    assert resolve_strategy(21, capabilities=caps).executed == KV_CACHE


def test_auto_falls_back_with_a_reason_on_a_prompt_only_backend():
    caps = capabilities_of(_PromptOnly())
    selection = resolve_strategy(15, capabilities=caps)
    assert selection.requested == AUTO
    assert selection.executed == PROMPT
    assert selection.reason == "unsupported_capability:hidden"
    assert selection.fell_back is True


def test_pinned_strategy_wins_when_supported():
    caps = capabilities_of(_Capable())
    selection = resolve_strategy(3, profile_config={"injection_strategy": "hidden"},
                                 capabilities=caps)
    assert selection.requested == HIDDEN
    assert selection.executed == HIDDEN
    assert selection.reason == "explicit"


def test_byte_rtu_profile_stays_prompt_for_every_fact_count():
    caps = capabilities_of(_PromptOnly())
    for count in (3, 15, 21, 40):
        selection = resolve_strategy(count, capabilities=caps)
        assert selection.executed == PROMPT


def test_explicit_unsupported_request_raises_when_strict():
    hidden_only = BackendCapabilities(name="hidden-only", hidden=True, kv_prime=False)
    with pytest.raises(UnsupportedCapabilityError):
        resolve_strategy(5, capabilities=hidden_only, requested=KV_CACHE, strict=True)
    # The non-strict path documents the fallback instead of raising.
    selection = resolve_strategy(5, capabilities=hidden_only, requested=KV_CACHE)
    assert selection.executed == PROMPT
    assert selection.reason == "unsupported_capability:kv_cache"


def test_explicit_unsupported_request_can_report_a_documented_fallback():
    caps = capabilities_of(_Capable())
    selection = resolve_strategy(5, capabilities=caps, requested=HIDDEN, strict=False)
    assert selection.executed == HIDDEN  # capable backend honours it


# --------------------------------------------------------------------------- #
# Trace validation and projection
# --------------------------------------------------------------------------- #
def test_nonfinite_and_empty_traces_are_rejected():
    for bad in (torch.tensor([float("nan")]), torch.tensor([])):
        with pytest.raises(TraceValidationError):
            HiddenInjection(_CapableWithLayers(), bad)


class _CapableWithLayers(_Capable):
    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Identity(), torch.nn.Identity()]

    @property
    def capabilities(self):
        return BackendCapabilities(name="capable-layers", hidden=True, kv_prime=True)


def test_project_trace_is_deterministic_and_shape_shifting():
    trace = torch.randn(384, generator=torch.Generator().manual_seed(3))
    a = project_trace(trace, 16)
    b = project_trace(trace, 16)
    assert a.shape == (16,)
    assert torch.allclose(a, b)
    assert not torch.allclose(project_trace(trace, 16)[:8], trace[:8])


def test_projection_does_not_touch_global_rng_state():
    trace = torch.randn(64)
    before = torch.get_rng_state().clone()
    project_trace(trace, 32)
    fit_prompt("q", [{"text": "x"}])
    assert torch.equal(torch.get_rng_state(), before)


def test_kv_prefill_reports_tokens_and_releases_the_cache():
    backend = _CapableWithLayers()
    prefill = KvPrefill(backend, ["fact one", "fact two"])
    with prefill:
        assert prefill.applied is True
        assert prefill.tokens == len("fact one" + "fact two")
        assert backend.primed == prefill.tokens
    assert backend.primed == 0
    assert prefill.to_dict()["cleanup"] == "released"


def test_injection_results_are_json_serialisable():
    backend = _CapableWithLayers()
    injection = HiddenInjection(backend, torch.randn(8), layer_idx=0)
    payload = json.dumps(injection.to_dict())
    assert '"strategy": "hidden"' in payload
    assert injection.applied is False
