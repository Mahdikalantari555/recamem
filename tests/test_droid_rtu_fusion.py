import itertools
import json
import os
import shutil
import tempfile

import pytest
import torch

from src.model.byte_rtu_backend import ByteRtuBackend
from src.model.droid import DroidEngine, NO_INFORMATION_MESSAGE
from src.model.droid_manager import DroidManager
from src.model.memory_injection import select_strategy
from src.model.rtu import InvalidProfileError, MissingCheckpointError, Model

REMOTE_SENSING_TEXT = (
    "Remote sensing is the acquisition of information about an object or "
    "phenomenon without making physical contact with the object, in contrast "
    "to in situ or on-site observation."
)


def _raw_ui_loop(model, data, out_cap=300, stop_threshold=0.35):
    """Verbatim replica of the standalone raw-UI loop in app.py (chat tab).

    ``data`` is the already-encoded byte stream. The UI loop appends its own
    ``"\\n"`` for the interactive terminal; the backend never does, so callers
    pass ``prompt + "\\n"`` when comparing the two.
    """
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


def _longest_common_substring(a, b):
    """Length of the longest common substring (grounding strength)."""
    from difflib import SequenceMatcher

    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size


def _new_checkpoint(tmp_path, seed=7):
    ckpt = str(tmp_path / "rtu.safetensors")
    torch.manual_seed(seed)
    Model(dim=64, layers=2, spread=16).save(ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Backend conformance (1.7 - 1.10)
# ---------------------------------------------------------------------------


def test_byte_rtu_backend_matches_raw_ui_loop(tmp_path):
    prompt = (
        "Context:\n"
        "- Remote sensing is the acquisition of information about an object.\n\n"
        "Query: What is remote sensing?"
    )
    ckpt = _new_checkpoint(tmp_path, seed=11)
    torch.manual_seed(11)
    ui_model = Model(dim=64, layers=2, spread=16)
    ui_model.load(ckpt)

    backend = _small_backend(checkpoint=ckpt, max_output_bytes=300, stop_threshold=0.35)

    # The backend never appends "\n"; the UI loop's stream carries its own.
    torch.manual_seed(1)
    ui_bytes = _raw_ui_loop(ui_model, prompt.encode("utf-8"))
    torch.manual_seed(1)
    backend_text = backend.generate(prompt, max_tokens=300)
    assert backend_text == ui_bytes.decode("utf-8", errors="ignore")

    # Same parity when the caller supplies the UI loop's trailing newline.
    torch.manual_seed(5)
    ui_bytes_nl = _raw_ui_loop(ui_model, (prompt + "\n").encode("utf-8"))
    torch.manual_seed(5)
    backend_text_nl = backend.generate(prompt + "\n", max_tokens=300)
    assert backend_text_nl == ui_bytes_nl.decode("utf-8", errors="ignore")


def test_reset_clears_prompt_and_state(tmp_path):
    backend = _small_backend(checkpoint=_new_checkpoint(tmp_path))
    backend.generate("Query: reset me", max_tokens=4)

    assert backend._prompt_bytes == b""
    for layer in backend.model.blocks:
        assert torch.all(layer.states == 0.0)
        assert torch.all(layer.decaytrace == 0.0)
        assert torch.all(layer.embedtrace == 0.0)

    # A manual prime is cleared by the public reset().
    backend._prompt_bytes = b"primed"
    backend.model.blocks[0].states.add_(1.0)
    backend.reset()
    assert backend._prompt_bytes == b""
    assert torch.all(backend.model.blocks[0].states == 0.0)


def test_prompt_has_no_trailing_newline(tmp_path):
    backend = _small_backend(checkpoint=_new_checkpoint(tmp_path))
    prompt = "Context:\n- A fact.\n\nQuery: A?"
    backend.generate(prompt, max_tokens=1)

    assert backend._last_prompt_bytes == prompt.encode("utf-8")
    assert not backend._last_prompt_bytes.endswith(b"\n")
    assert backend.encode_prompt(prompt) == prompt.encode("utf-8")
    # The query stays last in the primed bytes.
    assert backend._last_prompt_bytes.endswith(b"Query: A?")


def test_output_bounded_by_min_of_tokens_and_bytes(tmp_path):
    # stop_threshold > 1 makes the stop head unable to end the loop early.
    backend = _small_backend(
        checkpoint=_new_checkpoint(tmp_path), max_output_bytes=10, stop_threshold=2.0
    )
    assert len(backend.generate("abc", max_tokens=4).encode("utf-8")) <= 4
    assert len(backend.generate("abc", max_tokens=100).encode("utf-8")) <= 10
    assert backend.generate("abc", max_tokens=0) == ""
    assert backend.generate("abc", max_tokens=-3) == ""


# ---------------------------------------------------------------------------
# Fused chat behaviour
# ---------------------------------------------------------------------------


def test_no_free_generation_without_facts():
    engine = DroidEngine(name="fused-empty", byte_rtu_backend=_small_backend())
    reply = engine.chat("/slow What is remote sensing?")
    assert reply == NO_INFORMATION_MESSAGE
    assert engine.answer_path == "recall"


def test_byte_rtu_backend_bounds_and_unicode(tmp_path):
    backend = _small_backend(checkpoint=_new_checkpoint(tmp_path))

    assert backend.generate("") == ""
    assert backend.generate("   ") == ""
    assert isinstance(backend.generate("remnote sensing \u63a2\u6d4b \U0001f30d"), str)

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

    # Plumbing assertion (always runs): taught facts reach the priming bytes.
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
    # Grounding contract (task 1.29): at least one taught fact substring
    # appears in the answer. A small-corpus checkpoint is not byte-perfect,
    # so the assertion is on a shared substring, not on exact words.
    overlap = _longest_common_substring(reply.lower(), REMOTE_SENSING_TEXT.lower())
    assert overlap >= 8, f"answer is not grounded in taught facts: {reply!r}"
    assert grounded.answer_path == "fused"


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


def test_dual_backend_construction_raises():
    with pytest.raises(ValueError):
        DroidEngine(
            name="both",
            gguf_backend=object(),
            byte_rtu_backend=_small_backend(),
        )


def test_answer_path_values(monkeypatch):
    # recall: extractive answer.
    engine = DroidEngine(name="paths")
    engine.teach(REMOTE_SENSING_TEXT)
    engine.chat("/fast What is remote sensing?")
    assert engine.answer_path == "recall"
    assert engine.legacy_answer_path == "extractive"

    # llm: a non-byte backend that generates text.
    class StubLlm:
        def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
            return "generated answer"

    llm_engine = DroidEngine(name="paths-llm", gguf_backend=StubLlm())
    llm_engine.teach(REMOTE_SENSING_TEXT)
    llm_engine.chat("/slow What is remote sensing?")
    assert llm_engine.answer_path == "llm"

    # fused: a byte backend that generates text.
    backend = _small_backend()
    monkeypatch.setattr(backend, "generate", lambda *a, **k: "fused answer")
    fused_engine = DroidEngine(name="paths-fused", byte_rtu_backend=backend)
    fused_engine.teach(REMOTE_SENSING_TEXT)
    reply = fused_engine.chat("/slow What is remote sensing?")
    assert reply == "fused answer"
    assert fused_engine.answer_path == "fused"
    assert fused_engine.legacy_answer_path == "generative"


# ---------------------------------------------------------------------------
# Recall ranking (1.11 - 1.12)
# ---------------------------------------------------------------------------


def test_associative_bonus_reranks_and_cold_start_is_neutral():
    engine = DroidEngine(name="rank")
    engine.test_time_training = False

    vectors = {
        "alpha beta one": torch.tensor([1.0, 0.0, 0.0]),
        "alpha beta two": torch.tensor([0.0, 1.0, 0.0]),
    }
    query = "alpha beta"

    def embed(text):
        if text in vectors:
            return vectors[text].clone()
        return torch.tensor([1.0, 1.0, 0.0]) / (2 ** 0.5)

    monkeypatch_embed = embed
    engine.anchor.embed = monkeypatch_embed

    facts = [
        {
            "id": 1,
            "text": "alpha beta one",
            "timestamp": 0.0,
            "novelty": 0.0,
            "dense_similarity": 0.60,
        },
        {
            "id": 2,
            "text": "alpha beta two",
            "timestamp": 0.0,
            "novelty": 0.0,
            "dense_similarity": 0.60,
        },
    ]
    engine.knowledge.total_count = lambda: 2
    engine.knowledge.recall = lambda *a, **k: [dict(f) for f in facts]

    # Cold start: S is empty, so the bonus is zero and the order is unchanged.
    engine.memory.S.zero_()
    cold = engine.recall(query, top_k=2, threshold=0.0)
    assert [h["id"] for h in cold] == [1, 2]
    assert all(h["associative_bonus"] == 0.0 for h in cold)

    # Warm associative state aligned with fact 2 re-orders the same candidates.
    engine.memory.S.fill_(0.01)
    engine.memory.retrieve = lambda q: vectors["alpha beta two"].clone()
    warm = engine.recall(query, top_k=2, threshold=0.0)
    assert [h["id"] for h in warm] == [2, 1]
    assert warm[0]["associative_bonus"] > 0.0
    assert warm[1]["associative_bonus"] == 0.0


def test_recall_score_composition():
    from src.model.droid import (
        DENSE_GATE,
        RECALL_SCORE_WEIGHTS,
        _confidence_score,
        _lexical_overlap,
        _recency_score,
    )

    assert RECALL_SCORE_WEIGHTS["associative"] == 0.15
    assert RECALL_SCORE_WEIGHTS["recency"] == 0.02
    assert RECALL_SCORE_WEIGHTS["confidence"] == 0.05
    assert DENSE_GATE == 0.48
    assert _lexical_overlap("remote sensing", "remote sensing is acquisition") == 1.0
    assert _lexical_overlap("remote sensing", "lidar measures distance") == 0.0
    assert _recency_score(0.0) == 0.0
    assert _confidence_score({"novelty": 5.0}) == 1.0
    assert _confidence_score({}) == 0.0


# ---------------------------------------------------------------------------
# Profile round trip, manifest, errors, isolation (1.1 - 1.6, 1.13 - 1.17)
# ---------------------------------------------------------------------------


@pytest.fixture
def manager_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _fused_profile(manager, name="fused"):
    engine = manager.create_droid(name, byte_rtu=True, rtu_dim=64, rtu_layers=2, rtu_spread=16)
    engine.teach(REMOTE_SENSING_TEXT)
    manager.save_droid(name)
    return engine


def test_fused_profile_roundtrip(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    engine = _fused_profile(manager)
    assert isinstance(engine.gguf_backend, ByteRtuBackend)

    reloaded = DroidManager(base_dir=manager_dir).get_droid("fused")
    assert reloaded.engine == "byte_rtu"
    assert reloaded.byte_rtu_checkpoint == "rtu.safetensors"
    assert reloaded.fusion_mode == "prompt_stuffing"
    assert isinstance(reloaded.gguf_backend, ByteRtuBackend)
    assert reloaded.knowledge.active_count() == engine.knowledge.active_count()
    assert reloaded.step_count == engine.step_count

    before = engine.memory.state_dict()
    after = reloaded.memory.state_dict()
    assert set(before) == set(after)
    for key in before:
        if isinstance(before[key], torch.Tensor):
            assert torch.equal(before[key], after[key]), key
    assert torch.equal(engine.gguf_backend.model.blocks[0].states,
                       reloaded.gguf_backend.model.blocks[0].states)


def test_missing_checkpoint_raises(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    _fused_profile(manager)
    os.remove(os.path.join(manager_dir, "fused", "rtu.safetensors"))

    with pytest.raises(MissingCheckpointError) as excinfo:
        DroidManager(base_dir=manager_dir).get_droid("fused")
    assert "rtu.safetensors" in str(excinfo.value)


def test_model_load_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(MissingCheckpointError):
        Model(dim=16, layers=1).load(str(tmp_path / "absent.safetensors"))
    with pytest.raises(MissingCheckpointError):
        Model(dim=16, layers=1).load("")
    # Explicit random init is constructing a model, not loading one.
    model = Model(dim=16, layers=1)
    assert model.step_count == 0


def test_invalid_profile_manifest(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    _fused_profile(manager)
    profile = os.path.join(manager_dir, "fused")

    # Missing manifest.
    os.remove(os.path.join(profile, "config.json"))
    with pytest.raises(InvalidProfileError):
        DroidEngine(name="x").load_profile(profile)

    # Corrupt JSON.
    with open(os.path.join(profile, "config.json"), "w") as f:
        f.write("{not json")
    with pytest.raises(InvalidProfileError):
        DroidEngine(name="x").load_profile(profile)

    # Valid JSON, wrong type.
    with open(os.path.join(profile, "config.json"), "w") as f:
        json.dump([1, 2, 3], f)
    with pytest.raises(InvalidProfileError):
        DroidEngine(name="x").load_profile(profile)


def test_checkpoint_path_escape_rejected(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    _fused_profile(manager)
    profile = os.path.join(manager_dir, "fused")
    config_path = os.path.join(profile, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["byte_rtu_checkpoint"] = "../evil.safetensors"
    with open(config_path, "w") as f:
        json.dump(config, f)

    with pytest.raises(InvalidProfileError):
        DroidEngine(name="x").load_profile(profile)

    config["byte_rtu_checkpoint"] = "/tmp/evil.safetensors"
    with open(config_path, "w") as f:
        json.dump(config, f)
    with pytest.raises(InvalidProfileError):
        DroidEngine(name="x").load_profile(profile)


def test_manifest_fields_and_save_order(manager_dir, monkeypatch):
    manager = DroidManager(base_dir=manager_dir)
    engine = manager.create_droid("ordered", byte_rtu=True, rtu_dim=64, rtu_layers=2, rtu_spread=16)
    engine.teach(REMOTE_SENSING_TEXT)

    profile = os.path.join(manager_dir, "ordered")
    seen = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        # The manifest write is the last step; every artifact it names must
        # already exist when os.replace promotes config.json.tmp.
        seen["knowledge"] = os.path.isfile(os.path.join(profile, "knowledge.db"))
        seen["memory"] = os.path.isfile(os.path.join(profile, "memory.safetensors"))
        seen["checkpoint"] = os.path.isfile(os.path.join(profile, "rtu.safetensors"))
        seen["src"] = os.path.basename(src)
        seen["dst"] = os.path.basename(dst)
        return real_replace(src, dst)

    monkeypatch.setattr("src.model.droid.os.replace", spy_replace)
    engine.save_profile(profile)

    assert seen == {
        "knowledge": True,
        "memory": True,
        "checkpoint": True,
        "src": "config.json.tmp",
        "dst": "config.json",
    }
    with open(os.path.join(profile, "config.json")) as f:
        config = json.load(f)
    assert config["engine"] == "byte_rtu"
    assert config["fusion_mode"] == "prompt_stuffing"
    assert config["byte_rtu_checkpoint"] == "rtu.safetensors"
    assert config["rtu_checkpoint"] == "rtu.safetensors"  # legacy alias
    assert config["rtu_dim"] == 64
    assert config["rtu_layers"] == 2
    assert config["rtu_spread"] == 16
    assert config["injection_strategy"] == "prompt"
    assert not os.path.exists(os.path.join(profile, "config.json.tmp"))


def test_fused_profile_pins_prompt_strategy(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    _fused_profile(manager)
    reloaded = DroidManager(base_dir=manager_dir).get_droid("fused")

    assert reloaded.gguf_config.get("injection_strategy") == "prompt"
    for count in (3, 15, 40):
        assert select_strategy(count, "specific_fact", reloaded.gguf_config) == "prompt"


def test_droid_profile_has_no_checkpoint(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    engine = manager.create_droid("plain")
    engine.teach(REMOTE_SENSING_TEXT)
    manager.save_droid("plain")

    profile = os.path.join(manager_dir, "plain")
    assert not os.path.isfile(os.path.join(profile, "rtu.safetensors"))
    with open(os.path.join(profile, "config.json")) as f:
        config = json.load(f)
    assert config["engine"] == "droid"
    assert config["fusion_mode"] == "off"
    assert config["byte_rtu_checkpoint"] is None

    reloaded = DroidManager(base_dir=manager_dir).get_droid("plain")
    assert reloaded.engine == "droid"
    assert reloaded.byte_rtu_checkpoint is None


def test_profiles_are_isolated(manager_dir):
    manager = DroidManager(base_dir=manager_dir)
    a = manager.create_droid("iso-a", byte_rtu=True, rtu_dim=64, rtu_layers=2, rtu_spread=16)
    b = manager.create_droid("iso-b", byte_rtu=True, rtu_dim=64, rtu_layers=2, rtu_spread=16)
    a.teach("Remote sensing is the acquisition of information.")
    b.teach("Cardiology is the study of disorders of the heart.")
    manager.save_droid("iso-a")
    manager.save_droid("iso-b")

    b_before = {k: v.clone() for k, v in b.memory.state_dict().items()
                if isinstance(v, torch.Tensor)}
    b_facts = b.knowledge.active_count()

    # Write to A and re-load B from disk in a fresh manager.
    a.teach("Lidar uses laser pulses to measure distances.")
    manager.save_droid("iso-a")
    b_reloaded = DroidManager(base_dir=manager_dir).get_droid("iso-b")

    b_after = b_reloaded.memory.state_dict()
    assert b_reloaded.knowledge.active_count() == b_facts
    for key, value in b_before.items():
        assert torch.equal(value, b_after[key]), key

    assert os.path.abspath(os.path.join(manager_dir, "iso-a", "rtu.safetensors")) != os.path.abspath(
        os.path.join(manager_dir, "iso-b", "rtu.safetensors")
    )
    assert os.path.abspath(os.path.join(manager_dir, "iso-a", "knowledge.db")) != os.path.abspath(
        os.path.join(manager_dir, "iso-b", "knowledge.db")
    )
