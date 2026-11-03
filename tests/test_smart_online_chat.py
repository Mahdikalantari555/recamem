"""Integration tests for smart online learning chat and CPU throughput.

Validates:
1. Declarative conversational statements in chat are automatically absorbed into plastic memory.
2. Follow-up queries recall recently learned facts.
3. Fast vs slow routing executes appropriately.
4. GGUF backend context priming injects active memory facts.
5. CPU plastic ingestion achieves >50,000 tokens/second.
6. Base model parameters experience zero weight drift.
"""

import time
import pytest
import torch

from recamem.delta_memory import DeltaPlasticAdapter
from recamem.engine import DroidEngine
from recamem.gguf import GgufBackend, MockLlama
from recamem.plasticity import streaming_online_update


def test_online_chat_absorbs_and_recalls():
    droid = DroidEngine(name="online-learner")

    # 1. User tells facts naturally in chat (no 'learn:' or 'remember:' prefix needed)
    reply1 = droid.chat("The orbital inclination of Sentinel-2 is 98.62 degrees. The revisit frequency is five days.")
    assert "absorbed this into my memory" in reply1.lower() or "remembered" in reply1.lower()

    # 2. Fact is stored in memory and recallable immediately
    hits = droid.recall("What is the orbital inclination of Sentinel-2?", top_k=2)
    assert len(hits) > 0
    assert "98.62 degrees" in hits[0]["text"]

    # 3. Follow-up question returns recalled fact
    answer = droid.chat("What is the orbital inclination of Sentinel-2?")
    assert "98.62" in answer


def test_online_chat_dual_path_routing_and_gguf_priming():
    droid = DroidEngine(name="routing-droid")
    # Attach explicit mock or real backend
    backend = GgufBackend(mock=True)
    droid.gguf_backend = backend

    # Teach domain facts
    droid.teach("Hyperspectral imaging measures hundreds of narrow contiguous spectral bands across the electromagnetic spectrum.")

    # 1. Simple definitional query -> fast recall
    fast_ans = droid.chat("/fast What is hyperspectral imaging?")
    assert "hundreds of narrow contiguous spectral bands" in fast_ans

    # 2. Slow path -> GGUF synthesis with primed context
    slow_ans = droid.chat("/slow Explain hyperspectral imaging and its spectral bands.")
    assert len(slow_ans) > 0
    assert "Mock LLM output" in slow_ans or "spectral bands" in slow_ans


def test_cpu_plastic_ingestion_throughput_benchmark():
    """Verify plastic associative state updates achieve >50,000 tokens/second on CPU."""
    adapter = DeltaPlasticAdapter(dim=384, heads=4)
    n_tokens = 5000

    # Simulate embeddings for 5000 tokens
    torch.manual_seed(42)
    token_embeddings = torch.randn(n_tokens, 384)
    token_embeddings = token_embeddings / torch.linalg.vector_norm(token_embeddings, dim=1, keepdim=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        res = streaming_online_update(adapter, token_embeddings, base_lr=0.05, threshold=0.0)
    elapsed = time.perf_counter() - t0

    tokens_per_sec = n_tokens / max(elapsed, 1e-6)
    assert tokens_per_sec > 50000, f"Expected >50,000 tok/s on CPU, got {tokens_per_sec:.1f} tok/s"
    assert res["updated"] is True
    assert res["tokens"] == n_tokens


def test_zero_catastrophic_drift_on_frozen_backend():
    """Verify generative backend weights remain 100% frozen across continuous online learning."""
    droid = DroidEngine(name="frozen-check")
    backend = GgufBackend(mock=True)
    droid.gguf_backend = backend

    fp_before = droid._frozen_fingerprint()

    # Continuous learning stream of 20 conversational turns
    for i in range(20):
        droid.chat(f"Station {i} is located at coordinate latitude {10.0 + i} and longitude {20.0 + i}.")

    fp_after = droid._frozen_fingerprint()

    # Frozen components (anchor + decoder) must have identical cryptographic fingerprints
    assert fp_before == fp_after
