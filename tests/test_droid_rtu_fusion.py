import itertools
import os

import pytest
import torch

from src.model.byte_rtu_backend import ByteRtuBackend
from src.model.droid import DroidEngine, NO_INFORMATION_MESSAGE
from src.model.rtu import Model

REMOTE_SENSING_TEXT = (
    "Remote sensing is the acquisition of information about an object or "
    "phenomenon without making physical contact with the object, in contrast "
    "to in situ or on-site observation."
)


def _raw_ui_loop(model, prompt, out_cap=300, stop_threshold=0.35):
    """Verbatim replica of the standalone raw-UI loop in app.py:364-382."""
    data = (prompt + "\n").encode("utf-8")
    out_bytes = []
    for i, (c, n) in enumerate(itertools.pairwise(data)):
        model(c, nextb=n, end=(i == len(data) - 2), frozen=True)
    b = data[-1]
    for _ in range(out_cap):
        b, stop = model(b, nextb=None, end=False, frozen=True)
        out_bytes.append(b)
        if stop > stop_threshold:
            break
    return bytes(out_bytes)


def _small_backend(checkpoint="", **kwargs):
    params = dict(dim=64, layers=2, spread=16, checkpoint=checkpoint)
    params.update(kwargs)
    return ByteRtuBackend(**params)


def test_byte_rtu_backend_matches_raw_ui_loop(tmp_path):
    prompt = (
        "Context:\n"
        "- Remote sensing is the acquisition of information about an object.\n\n"
        "Query: What is remote sensing?"
    )
    ckpt = str(tmp_path / "rtu.safetensors")
    torch.manual_seed(11)
    ui_model = Model(dim=64, layers=2, spread=16)
    ui_model.save(ckpt)

    backend = _small_backend(checkpoint=ckpt, max_output_bytes=300, stop_threshold=0.35)

    torch.manual_seed(1)
    ui_text = _raw_ui_loop(ui_model, prompt).decode("utf-8", errors="replace")

    torch.manual_seed(1)
    backend_text = backend.generate(prompt)

    assert backend_text == ui_text


def test_no_free_generation_without_facts():
    engine = DroidEngine(
        name="fused-empty", byte_rtu_backend=_small_backend()
    )
    reply = engine.chat("/slow What is remote sensing?")
    assert reply == NO_INFORMATION_MESSAGE


def test_byte_rtu_backend_bounds_and_unicode(tmp_path):
    ckpt = str(tmp_path / "rtu.safetensors")
    torch.manual_seed(7)
    m = Model(dim=64, layers=2, spread=16)
    m.save(ckpt)
    backend = _small_backend(checkpoint=ckpt)

    assert backend.generate("") == ""
    assert backend.generate("   ") == ""
    assert isinstance(
        backend.generate("remnote sensing \u63a2\u6d4b \U0001f30d"), str
    )

    # State carries context: same sampling seed, different prompt bytes produce
    # different output (priming changes what the component generates).
    torch.manual_seed(99)
    out_a = backend.generate("Context:\n- Alpha fact.\n\nQuery: alpha?")
    torch.manual_seed(99)
    out_b = backend.generate("Context:\n- Beta fact.\n\nQuery: beta?")

    assert out_a != out_b


def test_fused_chat_is_grounded(monkeypatch):
    checkpoint = os.environ.get("RTU_CHECKPOINT", "")
    backend = ByteRtuBackend(checkpoint=checkpoint) if checkpoint else _small_backend()
    engine = DroidEngine(name="fused-grounded", byte_rtu_backend=backend)
    engine.teach(REMOTE_SENSING_TEXT)

    # Plumbing assertion (always runs): taught facts must reach the priming bytes.
    captured = {}
    original = engine.gguf_backend.generate

    def spy(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return original(prompt, *args, **kwargs)

    monkeypatch.setattr(engine.gguf_backend, "generate", spy)
    engine.chat("/slow What is remote sensing?")
    assert "acquisition" in captured["prompt"].lower()

    if not checkpoint:
        pytest.skip(
            "RTU_CHECKPOINT is not set and no trained byte-RTU checkpoint "
            "exists in the repo, so taught-substring grounding cannot be "
            "asserted (random-init quality ceiling, see spec Assumptions)"
        )

    grounded = DroidEngine(
        name="fused-grounded-trained",
        byte_rtu_backend=ByteRtuBackend(checkpoint=checkpoint),
    )
    grounded.teach(REMOTE_SENSING_TEXT)
    reply = grounded.chat("/slow What is remote sensing?")
    assert "acquisition" in reply.lower()


def test_rtu_weights_frozen_during_inference():
    backend = _small_backend()
    engine = DroidEngine(name="fused-frozen", byte_rtu_backend=backend)
    engine.teach(REMOTE_SENSING_TEXT)

    model = backend.model

    def snapshot():
        snap = {k: v.detach().clone() for k, v in model.named_parameters()}
        for i, layer in enumerate(model.blocks):
            snap[f"state.{i}"] = layer.states.detach().clone()
            snap[f"decaytrace.{i}"] = layer.decaytrace.detach().clone()
            snap[f"embedtrace.{i}"] = layer.embedtrace.detach().clone()
        snap["step_count"] = torch.tensor([model.step_count], dtype=torch.long)
        return snap

    before = snapshot()
    for q in [
        "What is remote sensing?",
        "Tell me about physical contact.",
        "Define acquisition of information.",
    ]:
        engine.chat(f"/slow {q}")
    after = snapshot()

    assert set(before) == set(after)
    for key in before:
        assert torch.equal(before[key], after[key]), f"frozen rule violated: {key}"
