"""Contract tests for `src/model/gguf_backend.py` (surface-test-coverage 3.7).

All tests are CPU-only and offline: `llama_cpp` is stubbed via `sys.modules`,
the download path is exercised without network, and the backend imports
without `llama_cpp` installed.
"""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from src.model.gguf_backend import GgufBackend, GgufLoadError


def test_constructor_stores_arguments_and_refuses_a_missing_path():
    with pytest.raises(GgufLoadError):
        GgufBackend(
            model_path="/tmp/does-not-exist.gguf",
            repo_id="repo",
            filename="model.gguf",
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=True,
        )
    backend = GgufBackend(n_ctx=2048, n_gpu_layers=0, verbose=True)
    assert backend.repo_id is None
    assert backend.filename is None
    assert backend.n_ctx == 2048
    assert backend.n_gpu_layers == 0
    assert backend.verbose is True
    assert backend.model is None


def test_generate_without_a_model_raises_named_error():
    backend = GgufBackend()
    with pytest.raises(GgufLoadError):
        backend.generate("hello world", max_tokens=16, temperature=0.7)


def test_tokenize_and_detokenize_without_a_model():
    backend = GgufBackend()
    ids = backend.tokenize("a b c")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    text = backend.detokenize([65, 66, 67])
    assert isinstance(text, str)


def test_generate_call_shape_with_stubbed_llama(tmp_path):
    """Fake `llama_cpp` module: `create_completion` is invoked with the
    generation kwargs and `generate` returns its text."""
    calls = {}

    class FakeLlama:
        def __init__(self, model_path, **kwargs):
            calls["init"] = (model_path, kwargs)

        def create_completion(self, prompt, max_tokens, **kwargs):
            calls["completion"] = (prompt, max_tokens, kwargs)
            return {"choices": [{"text": f"synthetic:{prompt}"}]}

    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.Llama = FakeLlama

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake-gguf")

    with mock.patch.dict(sys.modules, {"llama_cpp": fake_llama}):
        backend = GgufBackend(model_path=str(model_file), n_ctx=1024, n_gpu_layers=1)
        assert calls["init"][0] == str(model_file)
        assert calls["init"][1]["n_ctx"] == 1024
        assert calls["init"][1]["n_gpu_layers"] == 1

        out = backend.generate("prompt text", max_tokens=64, temperature=0.5, top_p=0.9)
        assert out == "synthetic:prompt text"
        prompt, max_tokens, kwargs = calls["completion"]
        assert prompt == "prompt text"
        assert max_tokens == 64
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9


def test_download_model_missing_huggingface_hub_raises_clear_error():
    backend = GgufBackend()
    with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(ImportError, match="huggingface_hub not installed"):
            backend.download_model("org/repo", "model.gguf")


def test_backend_imports_and_constructs_without_llama_cpp():
    # `llama_cpp` is not installed in this environment; importing the module
    # and constructing the backend must not crash, and no mock is invented.
    assert "llama_cpp" not in sys.modules
    backend = GgufBackend()
    with pytest.raises(GgufLoadError):
        backend.generate("anything")
