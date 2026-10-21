"""GGUF backend capability and KV-prefill contracts (post-merge 3.4, 3.8).

llama.cpp is not installed in this environment, so the *binding* is a double
that implements the documented llama-cpp-python surface (``tokenize``,
``eval``, ``reset``, ``create_completion``). That proves ordering, cache
consumption and cleanup of our code. Real-model evidence stays blocked in the
ledger; these tests make no claim about measured generation quality.
"""

import json

import pytest

from src.model.backend_contracts import (
    KvPrefill,
    UnsupportedCapabilityError,
    capabilities_of,
)
from src.model.gguf_backend import GgufBackend


class _Binding:
    """Minimal llama-cpp-python double with prefill/reset."""

    def __init__(self):
        self.events = []
        self.cache = []

    def tokenize(self, data):
        return list(data)

    def detokenize(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore").encode("utf-8")

    def eval(self, tokens):
        self.events.append(("prefill", len(tokens)))
        self.cache.extend(tokens)

    def reset(self):
        self.events.append(("reset", len(self.cache)))
        self.cache.clear()

    def create_completion(self, prompt, max_tokens=8, **kwargs):
        assert self.cache, "generation must attend to the primed cache"
        self.events.append(("generate", prompt))
        return {"choices": [{"text": "primed answer"}]}


def _backend_with(binding):
    backend = GgufBackend()
    backend.model = binding
    return backend


def test_mock_backend_never_advertises_prefill():
    backend = GgufBackend()  # no binding: MockLlama fallback
    caps = capabilities_of(backend)
    assert caps.prompt is True
    assert caps.hidden is False, "llama.cpp exposes no residual-stream hook"
    assert caps.kv_prime is False


def test_prefill_is_rejected_without_a_supporting_binding():
    backend = GgufBackend()
    with pytest.raises(UnsupportedCapabilityError):
        backend.prime_kv_cache(["a fact"])
    with pytest.raises(UnsupportedCapabilityError):
        KvPrefill(backend, ["a fact"]).prime()


def test_prefill_happens_before_generation_and_is_reset_after():
    binding = _Binding()
    backend = _backend_with(binding)
    assert capabilities_of(backend).kv_prime is True

    prefill = KvPrefill(backend, ["alpha", "beta"])
    with prefill:
        assert prefill.applied is True
        assert prefill.tokens == len("alpha" + "beta")
        backend.generate("alpha\n\nquery", max_tokens=4)

    assert [event[0] for event in binding.events] == ["prefill", "generate", "reset"]
    assert binding.cache == []
    assert prefill.to_dict()["binding"]["mechanism"] == "eval"


def test_two_prefills_do_not_share_cache_state():
    first_binding, second_binding = _Binding(), _Binding()
    first = _backend_with(first_binding)
    second = _backend_with(second_binding)

    one = KvPrefill(first, ["fact one"])
    two = KvPrefill(second, ["fact two"])
    one.prime()
    two.prime()
    assert first_binding.cache == list(b"fact one")
    assert second_binding.cache == list(b"fact two")
    two.cleanup()
    assert first_binding.cache == list(b"fact one")  # unaffected by the other
    one.cleanup()
    assert first_binding.cache == []


def test_failed_generation_still_releases_the_cache():
    class Boom(_Binding):
        def create_completion(self, prompt, max_tokens=8, **kwargs):
            self.events.append(("generate-failed", prompt))
            raise RuntimeError("kv exhausted")

    binding = Boom()
    backend = _backend_with(binding)
    with KvPrefill(backend, ["alpha"]):
        # GgufBackend reports a generation failure as text; the cache must still
        # be released when the scope exits.
        assert "Generation failed" in backend.generate("q", max_tokens=1)
    assert [event[0] for event in binding.events] == ["prefill", "generate-failed", "reset"]
    assert binding.cache == []


def test_empty_fact_set_is_rejected_rather_than_faked():
    binding = _Binding()
    backend = _backend_with(binding)
    with pytest.raises(ValueError):
        backend.prime_kv_cache([""])


def test_tokens_come_from_the_binding_tokenizer():
    binding = _Binding()
    backend = _backend_with(binding)
    assert backend.count_tokens("abcdef") == len(binding.tokenize(b"abcdef"))
    assert backend.context_window() == 4096


def test_backend_capability_report_is_json_serialisable():
    backend = _backend_with(_Binding())
    payload = json.dumps(capabilities_of(backend).to_dict())
    assert '"kv_prime": true' in payload
    assert '"hidden": false' in payload
