"""End-to-end integration test with real GGUF model and DroidEngine."""
from __future__ import annotations

from pathlib import Path
import pytest

from recamem.engine import DroidEngine
from recamem.gguf import GgufBackend

GGUF_PATH = "/mnt/d/AI/.lmstudio/models/OpenBMB/MiniCPM-V-4_7-1B-Q8_0/MiniCPM-V-4_7-1B-Q8_0.gguf"


@pytest.mark.skipif(not Path(GGUF_PATH).is_file(), reason="Local GGUF model file not found")
def test_droid_engine_with_real_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAMEM_ANCHOR", "mock")
    backend = GgufBackend(model_path=GGUF_PATH, n_ctx=512, verbose=False)
    engine = DroidEngine(
        name="real-gguf-droid",
        db_path=str(tmp_path / "test.db"),
        gguf_backend=backend,
        gguf_config={"injection_strategy": "prompt", "max_context_tokens": 256},
    )

    # Teach fact
    learn_res = engine.teach("The primary sensor payload is named Hyperion-X.")
    assert learn_res["status"] == "learned"
    assert learn_res["propositions"] >= 1

    # Slow path generation query
    response = engine.chat("/slow What is the primary sensor payload?")
    assert len(response) > 0
    assert engine._last_answer_path in ("generative", "fallback", "extractive", "recall")
