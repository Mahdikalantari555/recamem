"""Real inference verification using the user-provided MiniCPM-V-4_7-1B GGUF model."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from recamem.gguf import GgufBackend

GGUF_PATH = "/mnt/d/AI/.lmstudio/models/OpenBMB/MiniCPM-V-4_7-1B-Q8_0/MiniCPM-V-4_7-1B-Q8_0.gguf"


@pytest.mark.skipif(not Path(GGUF_PATH).is_file(), reason="Local GGUF model file not found")
def test_real_gguf_backend_inference():
    """Verify real inference, tokens generation, and prompt handling with local GGUF."""
    backend = GgufBackend(model_path=GGUF_PATH, n_ctx=512, verbose=False)
    assert backend.is_mock is False
    assert backend.model is not None

    prompt = "The capital of France is"
    out = backend.generate(prompt, max_tokens=10, temperature=0.1)
    assert isinstance(out, str)
    assert len(out.strip()) > 0
    assert "Paris" in out
