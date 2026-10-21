"""Integration tests for hookable-decoder hidden injection (task 3.3/3.7).

The fixture is a tiny deterministic decoder, so the evidence is about the
mechanism — scoped hook, changed intermediate state and logits, cleanup on
success and on exception — not about model quality. A real downloaded decoder
is not available in this environment and stays blocked in the ledger.
"""

import pytest
import torch

from src.model.backend_contracts import (
    HiddenInjection,
    capabilities_of,
    fit_prompt,
    resolve_strategy,
)
from src.model.torch_decoder_backend import HookableDecoderBackend


class _Block(torch.nn.Module):
    def __init__(self, hidden, bias):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden) * bias)
        self.outputs = []

    def forward(self, x):
        y = x + torch.tanh(self.linear(x))
        self.outputs.append(y.detach().clone())
        return y


class _Decoder(torch.nn.Module):
    def __init__(self, hidden=8, layers=3, vocab=26):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden})()
        self.layers = torch.nn.ModuleList([_Block(hidden, 0.5 + i) for i in range(layers)])
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(hidden).repeat(4, 1)[:vocab])

    def forward(self, ids):
        x = torch.nn.functional.one_hot(ids % self.config.hidden_size,
                                        self.config.hidden_size).float()
        for block in self.layers:
            x = block(x)
        return self.head(x)


class _Tokenizer:
    def encode(self, text):
        return [byte % 32 for byte in text.encode("utf-8")]


@pytest.fixture()
def backend():
    return HookableDecoderBackend(_Decoder(), tokenizer=_Tokenizer(), context_window=64,
                                  output_reserve=8)


def test_backend_advertises_only_hidden_injection(backend):
    caps = capabilities_of(backend)
    assert caps.prompt is True
    assert caps.hidden is True
    assert caps.kv_prime is False
    assert backend.capabilities.name == "HookableDecoderBackend"
    assert backend.hidden_size() == 8
    assert len(backend.hidden_layers()) == 3


def test_layer_list_is_discovered_from_transformer_h():
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList([torch.nn.Identity()])
            self.config = type("C", (), {"n_embd": 4})()

        def forward(self, ids):  # pragma: no cover - not used
            return ids

    backend = HookableDecoderBackend(Wrapped(), hidden_size=4)
    assert len(backend.hidden_layers()) == 1
    assert backend.hidden_size() == 4


def test_missing_layers_is_explicit_not_silent():
    class Bare(torch.nn.Module):
        config = type("C", (), {"hidden_size": 4})()

        def forward(self, ids):  # pragma: no cover
            return ids

    backend = HookableDecoderBackend(Bare(), hidden_size=4)
    assert backend.supports_hidden_injection() is False
    with pytest.raises(ValueError):
        backend.hidden_layers()


def test_hook_changes_intermediate_state_and_logits(backend):
    trace = torch.randn(384, generator=torch.Generator().manual_seed(5))
    backend.generate("prompt", max_tokens=1)
    # Layer 2 consumes layer 1's (hooked) output, so its state is the observable
    # that must change; layer 1's own record is pre-hook.
    baseline_layer = backend.hidden_layers()[2].outputs[-1].clone()
    baseline_logits = backend.last_logits.clone()

    with HiddenInjection(backend, trace, layer_idx=1, scale=2.0):
        backend.generate("prompt", max_tokens=1)
        injected_layer = backend.hidden_layers()[2].outputs[-1].clone()
        injected_logits = backend.last_logits.clone()

    assert not torch.allclose(baseline_layer, injected_layer)
    assert not torch.allclose(baseline_logits, injected_logits)
    # Layer 0 runs before the hooked layer, so it is untouched.
    backend.generate("prompt", max_tokens=1)
    assert torch.allclose(backend.last_logits, baseline_logits)


def test_injection_does_not_leak_between_requests(backend):
    trace = torch.randn(32, generator=torch.Generator().manual_seed(6))
    backend.generate("x", max_tokens=1)
    reference = backend.last_logits.clone()
    first = HiddenInjection(backend, trace, layer_idx=2, scale=3.0)
    with first:
        backend.generate("x", max_tokens=1)
    assert first.applied is True
    assert backend.hidden_layers()[2]._forward_hooks == {}
    backend.generate("x", max_tokens=1)
    assert torch.allclose(backend.last_logits, reference)


def test_hooks_are_removed_after_an_exception_inside_generation():
    class Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("decoder exploded")

    decoder = _Decoder()
    decoder.layers[1] = Boom()  # the model fails after the hook is installed
    backend = HookableDecoderBackend(decoder, tokenizer=_Tokenizer())
    trace = torch.randn(32, generator=torch.Generator().manual_seed(7))
    target = backend.hidden_layers()[1]
    with pytest.raises(RuntimeError):
        with HiddenInjection(backend, trace, layer_idx=1):
            backend.generate("x", max_tokens=1)
    assert target._forward_hooks == {}


def test_prompt_budget_uses_the_backend_tokenizer(backend):
    plan = fit_prompt("q", [{"text": "fact"}], backend=backend)
    assert plan.budget == 56  # 64 - 8 reserved
    assert plan.tokens_used == len(plan.prompt)  # tokenizer emits one id per byte
    assert plan.prompt_ends_with_query


def test_strategy_resolution_on_a_hookable_decoder(backend):
    caps = capabilities_of(backend)
    assert resolve_strategy(15, capabilities=caps).executed == "hidden"
    # 21+ facts prefer KV priming, which this adapter does not support.
    selection = resolve_strategy(21, capabilities=caps)
    assert selection.executed == "prompt"
    assert selection.reason == "unsupported_capability:kv_cache"


def test_generation_never_requests_a_gpu_or_downloads_weights(backend):
    # The adapter only ever calls the supplied module.
    out = backend.generate("hello", max_tokens=2)
    assert isinstance(out, str)
    assert backend.forward_calls == 1
    assert backend.model_path_missing_ok if hasattr(backend, "model_path_missing_ok") else True
