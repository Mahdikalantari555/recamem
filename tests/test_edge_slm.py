"""Edge SLM catalog: provenance, hard budgets and deterministic recommendations."""
import pytest

from src.model.edge_slm import EdgeSLMRegistry, REGISTRY_VERSION


EXPECTED_KEYS = {
    "llama-3.2-1b", "llama-3.2-3b",
    "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b",
    "smollm2-135m", "smollm2-360m", "smollm2-1.7b", "smollm2-3b",
    "gemma-3-270m", "gemma-3-1b", "gemma-3-4b",
    "phi-4-mini",
}


@pytest.fixture
def registry():
    return EdgeSLMRegistry()


def test_catalog_has_the_thirteen_requested_models(registry):
    assert len(registry) == 13
    assert {entry.key for entry in registry.all()} == EXPECTED_KEYS
    assert set(registry.families()) == {"llama-3.2", "qwen2.5", "smollm2",
                                        "gemma-3", "phi-4"}
    metadata = registry.metadata()
    assert metadata["version"] == REGISTRY_VERSION
    assert metadata["verified"] == "2026-10-21"
    assert metadata["count"] == 13


def test_source_reported_reference_values(registry):
    llama = registry.get("llama-3.2-1b").to_dict()
    assert llama["ram_gb"] == pytest.approx(1.5)
    assert llama["ram_source"] == "source-reported"
    assert llama["tok_s"] == [30.0, 50.0]
    assert llama["tok_s_source"] == "source-reported"
    assert llama["provenance"]["local_measurement"] is False

    assert "reasoning" in registry.get("qwen2.5-1.5b").task_tags
    assert any("11T" in note for note in registry.get("smollm2-1.7b").notes)

    gemma = registry.get("gemma-3-1b").to_dict()
    assert gemma["ram_gb"] == pytest.approx(0.8)
    assert gemma["quality"]["gsm8k_pct"] == 62.8
    assert gemma["quality"]["source_label"] == "source-reported"


def test_unknown_throughput_is_not_fabricated(registry):
    for key in ("llama-3.2-3b", "qwen2.5-3b", "phi-4-mini"):
        entry = registry.get(key)
        assert entry.tok_s is None
        assert entry.tok_s_source is None
        assert entry.to_dict()["tok_s_source"] == "unknown"


def test_every_entry_carries_quantization_provenance_and_gguf_identifiers(registry):
    for entry in registry.all():
        payload = entry.to_dict()
        assert payload["quantizations"]
        assert payload["default_quantization"]
        assert payload["model_id"]
        assert payload["filename"].lower().endswith(".gguf")
        assert payload["provenance"]["verified"]
        assert payload["provenance"]["type"] in ("source-reported", "estimate")


def test_recommendation_respects_hard_ram_budget(registry):
    result = registry.recommend(ram_gb=0.4)
    assert result["status"] == "ok"
    assert result["recommendations"]
    assert all(item["ram_gb"] <= 0.4 for item in result["recommendations"])
    assert any(item["key"] == "smollm2-135m" for item in result["recommendations"])
    assert any(item["reason"] == "ram_exceeds_budget" for item in result["excluded"])


def test_recommendation_throughput_unknown_is_not_a_pass(registry):
    result = registry.recommend(min_tok_s=30)
    keys = {item["key"] for item in result["recommendations"]}
    assert "llama-3.2-1b" in keys  # source-reported 30-50 tok/s
    assert "gemma-3-1b" not in keys  # throughput unknown
    reasons = {item["key"]: item["reason"] for item in result["excluded"]}
    assert reasons["gemma-3-1b"] == "throughput_unknown_for_hard_minimum"
    assert reasons["llama-3.2-3b"] == "throughput_unknown_for_hard_minimum"


def test_ram_and_throughput_can_produce_no_match(registry):
    result = registry.recommend(ram_gb=0.05, min_tok_s=100)
    assert result["status"] == "no_match"
    assert result["recommendations"] == []
    assert result["excluded"]


def test_reasoning_task_ranks_with_benchmark_evidence(registry):
    result = registry.recommend(ram_gb=2.0, task="reasoning")
    assert result["status"] == "ok"
    keys = [item["key"] for item in result["recommendations"]]
    assert "qwen2.5-1.5b" in keys
    assert "gemma-3-1b" in keys
    top = result["recommendations"][0]
    assert any("tagged for reasoning" in line for line in top["rationale"])
    gemma = next(item for item in result["recommendations"] if item["key"] == "gemma-3-1b")
    assert any("GSM8K" in line for line in gemma["rationale"])
    assert "source-reported" in " ".join(gemma["rationale"])


def test_general_chat_and_summarization_rankings(registry):
    chat = registry.recommend(ram_gb=4.0, task="general_chat")
    assert chat["recommendations"]
    assert all(any("general_chat" in tag for tag in item["task_tags"])
               for item in chat["recommendations"])
    summary = registry.recommend(ram_gb=4.0, task="summarization")
    assert summary["recommendations"]


def test_recommendation_is_deterministic(registry):
    first = registry.recommend(ram_gb=2.0, task="reasoning")
    second = EdgeSLMRegistry().recommend(ram_gb=2.0, task="reasoning")
    assert [item["key"] for item in first["recommendations"]] == [
        item["key"] for item in second["recommendations"]
    ]


def test_invalid_filters_are_rejected(registry):
    with pytest.raises(ValueError):
        registry.recommend(ram_gb=-1)
    with pytest.raises(ValueError):
        registry.recommend(min_tok_s=-5)
    with pytest.raises(ValueError):
        registry.recommend(task="telepathy")


def test_gguf_config_maps_without_loading(registry):
    config = registry.gguf_config("llama-3.2-1b")
    assert config["gguf_repo_id"].startswith("bartowski/")
    assert config["gguf_filename"].endswith(".gguf")
    assert config["gguf_quantization"] == "Q4_K_M"
    assert config["load_requested"] is False
    # No backend attribute exists: browsing cannot trigger a download.
    assert not hasattr(registry, "gguf_backend")


def test_unknown_model_key_raises_key_error(registry):
    with pytest.raises(KeyError):
        registry.get("not-a-model")
