import time
import torch
import pytest
from src.model.onnx_anchor import AnchorUnavailableError, OnnxMiniLM
pytestmark = pytest.mark.phase0


MOCK = dict(mode="mock")

def test_onnx_anchor_shapes():
    anchor = OnnxMiniLM(**MOCK)
    emb_single = anchor.embed("Remote sensing is Earth observation from satellites.")
    assert isinstance(emb_single, torch.Tensor)
    assert emb_single.shape == (384,)
    
    # Check L2 unit norm
    norm = torch.norm(emb_single, p=2).item()
    assert abs(norm - 1.0) < 1e-4

    emb_batch = anchor.embed(["First sentence", "Second sentence"])
    assert isinstance(emb_batch, torch.Tensor)
    assert emb_batch.shape == (2, 384)

def test_onnx_anchor_latency():
    anchor = OnnxMiniLM(**MOCK)
    # Warmup
    anchor.embed("Warmup query")
    
    t0 = time.perf_counter()
    anchor.embed("What is remote sensing?")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"Inference took {elapsed_ms:.2f}ms"

def test_onnx_anchor_semantics():
    anchor = OnnxMiniLM(**MOCK)
    e_rs1 = anchor.embed("Remote sensing acquires information about Earth from satellites without physical contact.")
    e_rs2 = anchor.embed("Earth observation using satellite imagery.")
    e_pizza = anchor.embed("Making pizza with mozzarella cheese and tomato sauce.")

    sim_rs = torch.dot(e_rs1, e_rs2).item()
    sim_unrelated = torch.dot(e_rs1, e_pizza).item()

    assert sim_rs > sim_unrelated, f"Expected {sim_rs} > {sim_unrelated}"
    assert sim_rs > 0.5, f"Expected strong semantic correlation, got {sim_rs}"


def test_real_anchor_fails_fast_without_weights():
    """No silent mock fallback: the real anchor names the missing path."""
    with pytest.raises(AnchorUnavailableError) as excinfo:
        OnnxMiniLM(mode="real", cache_dir="/tmp/recamem-no-anchor-weights")
    assert "Model not found" in str(excinfo.value)


def test_mock_is_only_selected_explicitly():
    assert OnnxMiniLM(mode="mock").is_mock is True
    with pytest.raises(ValueError):
        OnnxMiniLM(mode="sometimes")
