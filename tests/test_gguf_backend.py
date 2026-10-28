"""Contract tests for src/model/gguf_backend.py.

``llama-cpp-python`` is an optional dependency, so these tests never import it:
every model interaction goes through a stub module or a stub object injected at
the module boundary. What is pinned here:

- importing the module does not require ``llama_cpp``;
- ``load_model`` forwards ``n_ctx``/``n_gpu_layers``/``verbose``/extra params to
  ``Llama`` and degrades to the in-process mock when the package is missing;
- ``generate`` builds the documented ``create_completion`` call shape, supports
  streaming callbacks, and returns an explicit error string on failure;
- ``download_model`` reports a clear error when ``huggingface_hub`` is absent and
  propagates other failures.

The honest limit: stubbing ``llama_cpp`` proves the call shape, not inference.
"""

import importlib
import importlib.util
import sys
import types

import pytest

from src.model import gguf_backend as gguf_module
from src.model.gguf_backend import GgufBackend, GgufLoadError, MockLlama
pytestmark = pytest.mark.phase0



# --------------------------------------------------------------------------- #
# Optional dependency handling
# --------------------------------------------------------------------------- #
def test_module_import_does_not_require_llama_cpp(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    spec = importlib.util.spec_from_file_location("gguf_backend_fresh", gguf_module.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert hasattr(fresh, "GgufBackend")
    assert hasattr(fresh, "MockLlama")
    # The fresh module's mock is independent of the package-level import path.
    backend = fresh.GgufBackend(mock=True)
    assert backend.is_mock is True
    assert "Mock LLM output" in backend.generate("hello")


def test_missing_llama_cpp_raises_instead_of_mocking(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"gguf")

    with pytest.raises(GgufLoadError) as excinfo:
        GgufBackend(model_path=str(model_file))
    assert "llama-cpp-python is not installed" in str(excinfo.value)

    # The explicit mock stays available and is labelled.
    backend = GgufBackend(model_path=str(model_file), mock=True)
    assert isinstance(backend.model, MockLlama)
    assert backend.is_mock is True
    assert "Mock LLM output" in backend.generate("hello", max_tokens=8)


# --------------------------------------------------------------------------- #
# Construction and loading
# --------------------------------------------------------------------------- #
def test_constructor_defaults():
    backend = GgufBackend()
    assert backend.model is None
    assert backend.n_ctx == 4096
    assert backend.n_gpu_layers == -1
    assert backend.model_path is None


def test_constructor_keeps_extra_params():
    backend = GgufBackend(n_ctx=128, n_gpu_layers=0, verbose=True, seed=7, logits_all=True)
    assert backend.n_ctx == 128
    assert backend.n_gpu_layers == 0
    assert backend.verbose is True
    assert backend.params == {"seed": 7, "logits_all": True}


def test_load_model_forwards_arguments_to_llama(monkeypatch, tmp_path):
    calls = {}

    class StubLlama:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.kwargs = kwargs

    stub = types.ModuleType("llama_cpp")
    stub.Llama = StubLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"gguf")
    backend = GgufBackend(model_path=str(model_file), n_ctx=256, n_gpu_layers=2, seed=11)

    assert isinstance(backend.model, StubLlama)
    assert calls["model_path"] == str(model_file)
    assert calls["n_ctx"] == 256
    assert calls["n_gpu_layers"] == 2
    assert calls["verbose"] is False
    assert calls["seed"] == 11


def test_constructor_rejects_missing_model_path():
    with pytest.raises(GgufLoadError) as excinfo:
        GgufBackend(model_path="/definitely/not/here.gguf")
    assert "/definitely/not/here.gguf" in str(excinfo.value)


def test_constructor_auto_downloads_when_repo_given(monkeypatch, tmp_path):
    downloaded = tmp_path / "downloaded.gguf"
    downloaded.write_bytes(b"gguf")

    def fake_download(self, repo_id, filename, local_dir="models"):
        assert repo_id == "org/model"
        assert filename == "model.gguf"
        return str(downloaded)

    monkeypatch.setattr(GgufBackend, "download_model", fake_download)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    with pytest.raises(GgufLoadError):
        GgufBackend(repo_id="org/model", filename="model.gguf")


def test_constructor_survives_download_failure(monkeypatch):
    def boom(self, repo_id, filename, local_dir="models"):
        raise RuntimeError("offline")

    monkeypatch.setattr(GgufBackend, "download_model", boom)
    with pytest.raises(GgufLoadError) as excinfo:
        GgufBackend(repo_id="org/model", filename="model.gguf", verbose=True)
    assert "offline" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class _RecordingModel:
    def __init__(self, text="generated text"):
        self.text = text
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"text": self.text}]}


def test_generate_without_model_raises_named_error():
    backend = GgufBackend()
    with pytest.raises(GgufLoadError):
        backend.generate("what is radar?")


def test_generate_builds_completion_call():
    backend = GgufBackend()
    backend.model = _RecordingModel("an answer")
    out = backend.generate("q", max_tokens=64, temperature=0.2, top_p=0.5, top_k=7, repeat_penalty=1.3)

    assert out == "an answer"
    call = backend.model.calls[0]
    assert call == {
        "prompt": "q",
        "max_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": 7,
        "repeat_penalty": 1.3,
    }


def test_generate_streams_through_callback():
    class StreamingModel:
        def __init__(self):
            self.calls = []

        def create_completion(self, **kwargs):
            self.calls.append(kwargs)
            for chunk in ("one ", "two ", "three"):
                yield {"choices": [{"text": chunk}]}

    backend = GgufBackend()
    backend.model = StreamingModel()
    seen = []
    out = backend.generate("q", stream=True, callback=seen.append)

    assert out == "one two three"
    assert seen == ["one ", "two ", "three"]
    assert backend.model.calls[0]["stream"] is True


def test_generate_stream_without_callback_uses_blocking_path():
    backend = GgufBackend()
    backend.model = _RecordingModel("blocking")
    assert backend.generate("q", stream=True, callback=None) == "blocking"


def test_generate_reports_failure():
    class BrokenModel:
        def create_completion(self, **kwargs):
            raise RuntimeError("kv cache exhausted")

    backend = GgufBackend()
    backend.model = BrokenModel()
    out = backend.generate("q")
    assert out.startswith("[Generation failed: ")
    assert "kv cache exhausted" in out


# --------------------------------------------------------------------------- #
# Tokenizer helpers and KV cache
# --------------------------------------------------------------------------- #
def test_tokenize_and_detokenize_without_model():
    backend = GgufBackend()
    tokens = backend.tokenize("two words")
    assert tokens == [hash("two") % 10000, hash("words") % 10000]
    assert backend.detokenize([1, 2, 3]) == "[Mock detokenize 3 tokens]"


def test_tokenize_uses_model_bytes_api():
    class TokenModel:
        def __init__(self):
            self.received = None

        def tokenize(self, data):
            self.received = data
            return [1, 2, 3]

        def detokenize(self, tokens):
            return b"decoded"

    backend = GgufBackend()
    backend.model = TokenModel()
    assert backend.tokenize("hello") == [1, 2, 3]
    assert backend.model.received == b"hello"
    assert backend.detokenize([1, 2, 3]) == "decoded"


def test_tokenize_falls_back_to_ordinals_on_model_error():
    class FailingTokenModel:
        def tokenize(self, data):
            raise RuntimeError("no tokenizer")

    backend = GgufBackend()
    backend.model = FailingTokenModel()
    assert backend.tokenize("ab") == [ord("a"), ord("b")]


def test_kv_cache_accessors():
    backend = GgufBackend()
    assert backend.get_kv_cache() is None
    backend.set_kv_cache("ignored")  # no model -> no-op

    backend.model = types.SimpleNamespace(kv_cache=None)
    assert backend.get_kv_cache() is None
    backend.set_kv_cache({"tokens": 3})
    assert backend.model.kv_cache == {"tokens": 3}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def test_download_requires_huggingface_hub(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    backend = GgufBackend()
    with pytest.raises(ImportError, match="huggingface_hub not installed"):
        backend.download_model("org/model", "model.gguf", str(tmp_path / "models"))


def test_download_success_creates_local_dir(monkeypatch, tmp_path):
    calls = {}

    def hf_hub_download(repo_id, filename, local_dir):
        calls.update(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return f"{local_dir}/{filename}"

    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    local_dir = tmp_path / "models"
    backend = GgufBackend()
    path = backend.download_model("org/model", "model.gguf", str(local_dir))

    assert path == f"{local_dir}/model.gguf"
    assert local_dir.is_dir()
    assert calls == {"repo_id": "org/model", "filename": "model.gguf", "local_dir": str(local_dir)}


def test_download_propagates_non_wildcard_failure(monkeypatch, tmp_path):
    def hf_hub_download(repo_id, filename, local_dir):
        raise RuntimeError("401 unauthorized")

    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    with pytest.raises(RuntimeError, match="401 unauthorized"):
        GgufBackend().download_model("org/private", "model.gguf", str(tmp_path / "models"))


def test_download_resolves_wildcard_filename(monkeypatch, tmp_path):
    attempts = []

    def hf_hub_download(repo_id, filename, local_dir):
        attempts.append(filename)
        if "*" in filename:
            raise RuntimeError("no such file")
        return f"{local_dir}/{filename}"

    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = hf_hub_download
    stub.list_repo_files = lambda repo_id: ["README.md", "model-Q4_K_M.gguf"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    path = GgufBackend().download_model("org/model", "*.gguf", str(tmp_path / "models"))
    assert path.endswith("model-Q4_K_M.gguf")
    assert attempts == ["*.gguf", "model-Q4_K_M.gguf"]
