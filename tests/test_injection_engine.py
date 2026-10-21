"""Engine-level injection wiring (post-merge-completion tasks 3.1, 3.4, 3.6).

The engine must report what executed, raise for explicit unsupported modes,
honour a profile-pinned strategy after reload, and clean up scoped hooks and
primed caches even when generation fails.
"""

import pytest
import torch

from src.model.backend_contracts import (
    UnsupportedCapabilityError,
    capabilities_of,
    resolve_strategy,
)
from src.model.droid import DroidEngine
from src.model.memory_injection import select_strategy
from src.model.torch_decoder_backend import HookableDecoderBackend


class _Block(torch.nn.Module):
    def forward(self, x):
        return x + torch.tanh(x)


class _Decoder(torch.nn.Module):
    def __init__(self, hidden=8, vocab=26):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden})()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(2)])
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, ids):
        x = torch.nn.functional.one_hot(ids % 8, 8).float()
        for block in self.layers:
            x = block(x)
        return self.head(x)


class _KvBackend:
    """Greedy double: records prefill/generate/reset ordering."""

    def __init__(self):
        self.events = []
        self.cache = []

    def supports_hidden_injection(self):
        return False

    def supports_kv_priming(self):
        return True

    def count_tokens(self, text):
        return len(text.encode("utf-8"))

    def prime_kv_cache(self, texts):
        tokens = [ord(c) for text in texts for c in text]
        self.cache.extend(tokens)
        self.events.append("prefill")
        return {"tokens": len(tokens)}

    def reset_kv_cache(self):
        self.cache.clear()
        self.events.append("reset")

    def generate(self, prompt, max_tokens=8, **kwargs):
        assert self.cache, "generation must start with a primed cache"
        self.events.append("generate")
        return "answer"


def _droid(tmp_path, backend, **kwargs):
    return DroidEngine(name="inject", db_path=str(tmp_path / "k.db"),
                       gguf_backend=backend, **kwargs)


def _facts(count):
    return [{"id": index, "text": f"fact number {index}", "similarity": 0.9}
            for index in range(count)]


# --------------------------------------------------------------------------- #
# Strategy selection through the engine
# --------------------------------------------------------------------------- #
def test_auto_selection_uses_3_15_21_fact_thresholds(tmp_path):
    backend = HookableDecoderBackend(_Decoder())
    droid = _droid(tmp_path, backend)
    three = droid.generate_with_memory("query", strategy="auto", facts=_facts(3))
    fifteen = droid.generate_with_memory("query", strategy="auto", facts=_facts(15))
    twenty_one = droid.generate_with_memory("query", strategy="auto", facts=_facts(21))
    assert three["strategy"] == "prompt"
    assert fifteen["strategy"] == "hidden"
    # KV priming is not advertised by this adapter: prompt with a reason.
    assert twenty_one["strategy"] == "prompt"
    assert twenty_one["strategy_reason"] == "unsupported_capability:kv_cache"


def test_byte_rtu_backend_is_prompt_only_for_every_fact_count(tmp_path):
    from src.model.byte_rtu_backend import ByteRtuBackend

    droid = _droid(tmp_path, ByteRtuBackend())
    for count in (3, 15, 21):
        result = droid.generate_with_memory("query", strategy="auto", facts=_facts(count))
        assert result["strategy"] == "prompt"
        assert result["capabilities"]["hidden"] is False
        assert result["capabilities"]["kv_prime"] is False


def test_explicit_unsupported_mode_raises(tmp_path):
    from src.model.byte_rtu_backend import ByteRtuBackend

    droid = _droid(tmp_path, ByteRtuBackend())
    with pytest.raises(UnsupportedCapabilityError):
        droid.generate_with_memory("query", strategy="hidden", facts=_facts(5))


def test_hidden_mode_applies_and_cleans_up(tmp_path):
    backend = HookableDecoderBackend(_Decoder())
    droid = _droid(tmp_path, backend)
    result = droid.generate_with_memory("query", strategy="hidden", facts=_facts(6))
    assert result["strategy"] == "hidden"
    assert result["injection_info"]["applied"] is True
    assert result["injection_info"]["executed"] == "hidden"
    assert backend.hidden_layers()[1]._forward_hooks == {}
    assert result["injection_info"]["cleanup"] == "removed"


def test_kv_mode_prefills_before_generation_and_resets_after(tmp_path):
    backend = _KvBackend()
    droid = _droid(tmp_path, backend)
    result = droid.generate_with_memory("query", strategy="kv_cache", facts=_facts(3))
    assert result["strategy"] == "kv_cache"
    assert result["injection_info"]["applied"] is True
    assert backend.events == ["prefill", "generate", "reset"]
    assert backend.cache == []


def test_kv_cache_is_released_when_generation_raises(tmp_path):
    class Boom(_KvBackend):
        def generate(self, prompt, max_tokens=8, **kwargs):
            self.events.append("generate")
            raise RuntimeError("decoder offline")

    backend = Boom()
    droid = _droid(tmp_path, backend)
    result = droid.generate_with_memory("query", strategy="kv_cache", facts=_facts(2))
    assert "generation_error" in result["injection_info"]
    assert backend.events == ["prefill", "generate", "reset"]
    assert backend.cache == []


def test_pinned_strategy_survives_reload_and_stays_prompt(tmp_path):
    from src.model.byte_rtu_backend import ByteRtuBackend

    droid = _droid(tmp_path, ByteRtuBackend())
    droid.gguf_config["injection_strategy"] = "prompt"
    profile = tmp_path / "profile"
    droid.save_profile(str(profile))

    loaded = DroidEngine(name="inject")
    loaded.load_profile(str(profile))
    assert loaded.gguf_config.get("injection_strategy") == "prompt"
    caps = capabilities_of(loaded.gguf_backend)
    for count in (3, 15, 21):
        assert select_strategy(count, "specific_fact", loaded.gguf_config, caps) == "prompt"


def test_prompt_budget_error_surfaces_instead_of_truncating(tmp_path):
    from src.model.backend_contracts import QueryBudgetError

    backend = HookableDecoderBackend(_Decoder(), context_window=32, output_reserve=8)
    droid = _droid(tmp_path, backend)
    with pytest.raises(QueryBudgetError):
        droid.generate_with_memory("q" * 100, strategy="prompt", facts=_facts(2))


def test_reported_capabilities_match_the_backend(tmp_path):
    from src.model.byte_rtu_backend import ByteRtuBackend

    droid = _droid(tmp_path, ByteRtuBackend())
    result = droid.generate_with_memory("query", strategy="prompt", facts=_facts(2))
    assert result["capabilities"]["name"] == "ByteRtuBackend"
    assert result["requested_strategy"] == "prompt"
    assert result["strategy_reason"] == "explicit"


def test_resolution_is_json_friendly_and_never_claims_hidden_on_prompt_only():
    from src.model.byte_rtu_backend import ByteRtuBackend

    caps = capabilities_of(ByteRtuBackend())
    selection = resolve_strategy(15, capabilities=caps)
    payload = selection.to_dict()
    assert payload["executed"] == "prompt"
    assert payload["fallback"] is True
    assert payload["reason"] == "unsupported_capability:hidden"
