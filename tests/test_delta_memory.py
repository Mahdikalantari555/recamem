"""Delta / sparse plastic memory: gated residual writes, erase, persistence.

Covers ``adaptation/delta-memory`` and ``adaptation/sparse-coding`` (tasks
1.1-1.5, 2.1-2.4, 3.1-3.3).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.model.delta_memory import (
    DELTA_STATE_VERSION,
    SPARSE_CODE_DIM,
    SPARSE_INPUT_DIM,
    SPARSE_TOP_K,
    AdapterStateError,
    DeltaPlasticAdapter,
    FlyModelAdapter,
    SparseCoder,
    SparseDeltaAdapter,
)
from src.model.droid import DroidEngine
from src.model.plastic_adapter import get_adapter, list_adapters
from src.model.rtu import InvalidProfileError, PlasticAssociativeRTU


@pytest.fixture
def adapter():
    torch.manual_seed(0)
    return DeltaPlasticAdapter(dim=64, heads=4)


# ---------------------------------------------------------------------------
# Registry (task 2.3)
# ---------------------------------------------------------------------------
def test_delta_and_sparse_names_are_registered_and_constructible():
    names = list_adapters()
    for name in ("associative_rtu", "delta_rtu", "sparse_delta", "fly_model"):
        assert name in names
    assert isinstance(get_adapter("delta_rtu", dim=64, heads=4), DeltaPlasticAdapter)
    sparse = get_adapter("sparse_delta", dim=384)
    assert isinstance(sparse, SparseDeltaAdapter)
    assert sparse.code_dim == SPARSE_CODE_DIM
    assert sparse.top_k == SPARSE_TOP_K
    fly = get_adapter("fly_model", dim=384)
    assert isinstance(fly, FlyModelAdapter)
    assert fly._adapter_type == "fly_model"


def test_unknown_adapter_name_still_raises_key_error():
    with pytest.raises(KeyError) as exc:
        get_adapter("not_a_memory")
    assert "not_a_memory" in str(exc.value)


def test_associative_adapter_remains_the_unchanged_default():
    assert isinstance(get_adapter("associative_rtu", dim=32), PlasticAssociativeRTU)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        DeltaPlasticAdapter(dim=10, heads=4)
    with pytest.raises(ValueError):
        DeltaPlasticAdapter(dim=0, heads=1)
    with pytest.raises(ValueError):
        SparseCoder(384, 2048, top_k=0)


# ---------------------------------------------------------------------------
# Gated delta rule (tasks 1.1-1.2)
# ---------------------------------------------------------------------------
def test_zero_gate_leaves_state_unchanged(adapter):
    adapter.set_beta_constant(0.0)
    before = adapter.S.clone()
    key, value = torch.randn(64), torch.randn(64)
    report = adapter.write_association(key, value)
    assert all(b == pytest.approx(0.0) for b in report["beta"])
    assert torch.allclose(adapter.S, before, atol=1e-12)
    assert torch.allclose(adapter.read(key), torch.zeros(64), atol=1e-8)


def test_full_gate_stores_the_value_and_reads_it_back(adapter):
    adapter.set_beta_constant(1.0)
    key, value = torch.randn(64), torch.randn(64)
    adapter.write_association(key, value)
    expected = adapter.W_V(value)
    assert F.cosine_similarity(adapter.read(key), expected, dim=0) > 0.999


def test_repeated_writes_converge_instead_of_growing(adapter):
    adapter.set_beta_constant(1.0)
    key, value = torch.randn(64), torch.randn(64)
    for _ in range(20):
        adapter.write_association(key, value)
    norm_after_repeats = float(adapter.S.norm())
    expected = adapter.W_V(value)
    assert F.cosine_similarity(adapter.read(key), expected, dim=0) > 0.999
    # A correct residual write does not accumulate unbounded outer products.
    assert norm_after_repeats < 10 * float(expected.detach().norm())


def test_changed_value_overwrites_existing_association(adapter):
    adapter.set_beta_constant(1.0)
    key = torch.randn(64)
    v1, v2 = torch.randn(64), torch.randn(64)
    adapter.write_association(key, v1)
    first = adapter.read(key)
    adapter.write_association(key, v2)
    second = adapter.read(key)
    target = adapter.W_V(v2)
    assert F.cosine_similarity(first, adapter.W_V(v1), dim=0) > 0.99
    assert F.cosine_similarity(second, target, dim=0) > 0.99
    assert F.cosine_similarity(second, adapter.W_V(v1), dim=0) < 0.5


def test_partial_gate_moves_read_toward_the_new_value(adapter):
    adapter.set_beta_constant(1.0)
    key = torch.randn(64)
    v1, v2 = torch.randn(64), torch.randn(64)
    adapter.write_association(key, v1)
    before = adapter.read(key)
    adapter.write_association(key, v2, beta=torch.tensor([0.5] * adapter.heads))
    after = adapter.read(key)
    target = adapter.W_V(v2)
    assert torch.dot(after, target) > torch.dot(before, target)
    assert after.norm() <= before.norm() + target.norm()


def test_gate_is_bounded_and_per_head(adapter):
    x = torch.randn(64)
    beta = adapter.gates(x)
    assert beta.shape == (adapter.heads,)
    assert torch.all(beta >= 0) and torch.all(beta <= 1)
    report = adapter.gate_report(x)
    assert report["heads"] == adapter.heads
    assert report["min"] >= 0 and report["max"] <= 1


# ---------------------------------------------------------------------------
# Householder selective erase (task 1.3)
# ---------------------------------------------------------------------------
def test_erase_removes_selected_key_and_preserves_orthogonal_control(adapter):
    adapter.set_beta_constant(1.0)
    torch.manual_seed(3)
    selected = torch.randn(64)
    # Build a control embedding whose key projection is orthogonal to the
    # selected key: the erase must be selective, not a global decay.
    W = adapter.W_K.weight.detach()
    selected_keys = (W @ selected).view(adapter.heads, adapter.head_dim)
    torch.manual_seed(11)
    target = torch.randn(adapter.heads, adapter.head_dim)
    # Remove the component of each head's control key along the selected key.
    for h in range(adapter.heads):
        target[h] -= (
            torch.dot(target[h], selected_keys[h])
            / torch.dot(selected_keys[h], selected_keys[h])
        ) * selected_keys[h]
    control = torch.linalg.solve(W, target.reshape(-1))
    adapter.write_association(selected, torch.randn(64))
    adapter.write_association(control, torch.randn(64))
    control_before = adapter.read(control)
    report = adapter.erase(selected, strength=1.0)
    assert report["status"] == "erased"
    assert report["selected_after"] < 0.01 * report["selected_before"]
    assert report["keys_removed"] >= 1
    control_after = adapter.read(control)
    cosine = F.cosine_similarity(control_before, control_after, dim=0).item()
    assert cosine > 0.99
    assert report["collateral_min_cosine"] > 0.99
    assert report["collateral_change"] < 0.02


def test_erase_is_bounded(adapter):
    adapter.set_beta_constant(1.0)
    key = torch.randn(64)
    adapter.write_association(key, torch.randn(64))
    before = adapter.read(key)
    half = adapter.erase(key, strength=0.5)
    assert 0.3 * half["selected_before"] < half["selected_after"] < 0.7 * half["selected_before"]
    assert half["selected_ratio"] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_invalid_erase_strength_rejected_without_mutation(adapter, strength):
    key = torch.randn(64)
    adapter.write_association(key, torch.randn(64))
    before = adapter.S.clone()
    with pytest.raises(ValueError):
        adapter.erase(key, strength=strength)
    assert torch.allclose(adapter.S, before)


def test_nonfinite_erase_key_rejected_without_mutation(adapter):
    key = torch.randn(64)
    adapter.write_association(key, torch.randn(64))
    before = adapter.S.clone()
    bad = key.clone()
    bad[0] = float("inf")
    with pytest.raises(ValueError):
        adapter.erase(bad, strength=0.5)
    assert torch.allclose(adapter.S, before)


# ---------------------------------------------------------------------------
# Versioned persistence (tasks 1.4-1.5)
# ---------------------------------------------------------------------------
def test_state_round_trip_matches_retrieval_and_update(adapter):
    adapter.set_beta_constant(0.7)
    keys = [torch.randn(64) for _ in range(3)]
    for key in keys:
        adapter.write_association(key, torch.randn(64))
    state = adapter.state_dict()
    assert state["_adapter_version"] == DELTA_STATE_VERSION
    clone = DeltaPlasticAdapter(dim=64, heads=4)
    clone.load_state_dict(state)
    for key in keys:
        assert torch.allclose(adapter.read(key), clone.read(key), atol=1e-6)
    x = torch.randn(64)
    assert torch.allclose(adapter.update(x), clone.update(x), atol=1e-6)


def test_incompatible_state_shape_is_rejected(adapter):
    state = adapter.state_dict()
    state["S"] = torch.zeros(2, 16, 16)
    with pytest.raises(AdapterStateError):
        DeltaPlasticAdapter(dim=64, heads=4).load_state_dict(state)

    other = DeltaPlasticAdapter(dim=32, heads=4)
    with pytest.raises(AdapterStateError):
        other.load_state_dict(adapter.state_dict())


def test_newer_state_version_is_rejected(adapter):
    state = adapter.state_dict()
    state["_adapter_version"] = DELTA_STATE_VERSION + 1
    with pytest.raises(AdapterStateError):
        DeltaPlasticAdapter(dim=64, heads=4).load_state_dict(state)


def test_capacity_eviction_is_persisted(adapter):
    small = DeltaPlasticAdapter(dim=64, heads=4, K_max=2)
    small.set_beta_constant(1.0)
    for _ in range(4):
        small.write_association(torch.randn(64), torch.randn(64))
    assert small.eviction_count == 2
    state = small.state_dict()
    clone = DeltaPlasticAdapter(dim=64, heads=4, K_max=2)
    clone.load_state_dict(state)
    assert clone.eviction_count == 2
    assert len(clone._key_registry) == len(small._key_registry)


# ---------------------------------------------------------------------------
# Sparse coding (tasks 2.1, 2.4)
# ---------------------------------------------------------------------------
def test_sparse_encoder_dimensions_and_fixed_sparsity():
    coder = SparseCoder()
    x = torch.randn(SPARSE_INPUT_DIM)
    code = coder.encode(x)
    assert code.shape == (SPARSE_CODE_DIM,)
    assert coder.active_count(code) == SPARSE_TOP_K
    assert code.nonzero().shape[0] == SPARSE_TOP_K
    utilization = coder.utilization(code)
    assert utilization["sparsity"] == pytest.approx(0.9375)
    assert utilization["zero_ratio"] == pytest.approx(0.9375)


def test_sparse_topk_ties_are_stable():
    activations = torch.zeros(2048)
    activations[:300] = 1.0  # 300 identical magnitudes -> index order decides
    first = SparseCoder.topk_mask(activations, 128)
    second = SparseCoder.topk_mask(activations.clone(), 128)
    assert torch.equal(first, second)
    assert first.nonzero().reshape(-1).tolist() == list(range(128))


def test_sparse_reconstruction_returns_input_space():
    coder = SparseCoder()
    x = torch.randn(SPARSE_INPUT_DIM)
    recon = coder.reconstruct(x)
    assert recon.shape == (SPARSE_INPUT_DIM,)
    metrics = coder.reconstruction_error(x)
    assert math.isfinite(metrics["mse"]) and metrics["mse"] >= 0
    assert math.isfinite(metrics["cosine"])


@pytest.mark.parametrize("bad", [torch.randn(383), torch.randn(385)])
def test_sparse_coder_rejects_wrong_input_shape(bad):
    with pytest.raises(ValueError):
        SparseCoder().encode(bad)


def test_sparse_coder_rejects_wrong_code_shape():
    with pytest.raises(ValueError):
        SparseCoder().decode(torch.zeros(1024))


def test_sparse_delta_adapter_contract_and_erasure():
    torch.manual_seed(1)
    adapter = SparseDeltaAdapter(dim=384, heads=8)
    x = torch.randn(384)
    adapter.update(x)
    assert adapter.retrieve(x).shape == (384,)
    assert adapter.compute_surprise(x)[1] >= 0
    report = adapter.code_report(x)
    assert report["active"] == SPARSE_TOP_K
    erase = adapter.erase(x, strength=1.0)
    assert erase["adapter"] == "sparse_delta"

    state = adapter.state_dict()
    clone = SparseDeltaAdapter(dim=384, heads=8)
    clone.load_state_dict(state)
    assert torch.allclose(adapter.retrieve(x), clone.retrieve(x), atol=1e-6)
    assert clone.top_k == adapter.top_k


def test_sparse_state_geometry_mismatch_is_rejected():
    adapter = SparseDeltaAdapter(dim=384)
    with pytest.raises(AdapterStateError):
        SparseDeltaAdapter(dim=384, top_k=64).load_state_dict(adapter.state_dict())


# ---------------------------------------------------------------------------
# Droid integration (tasks 3.1-3.3)
# ---------------------------------------------------------------------------
def _droid(tmp_path, adapter_name):
    return DroidEngine(
        name="delta-droid", plastic_adapter=adapter_name,
        db_path=str(tmp_path / "k.db"), dim=384,
    )


@pytest.mark.parametrize("name", ["delta_rtu", "sparse_delta", "fly_model"])
def test_profile_round_trip_keeps_selected_adapter(tmp_path, name):
    droid = _droid(tmp_path, name)
    droid.teach("The sensor captures multispectral imagery every morning.")
    profile = tmp_path / "profile"
    droid.save_profile(str(profile))

    import json
    manifest = json.loads((profile / "config.json").read_text())
    assert manifest["plastic_adapter"] == name
    assert manifest["adapter_version"] == 1

    loaded = DroidEngine(name="delta-droid")
    loaded.load_profile(str(profile))
    assert loaded.adapter_name == name
    assert type(loaded.memory) is type(droid.memory)
    assert loaded.knowledge.active_count() == droid.knowledge.active_count()


def test_legacy_profile_still_loads_associative_adapter(tmp_path):
    droid = DroidEngine(name="legacy", plastic_adapter="associative_rtu",
                        db_path=str(tmp_path / "k.db"))
    profile = tmp_path / "legacy-profile"
    droid.save_profile(str(profile))
    loaded = DroidEngine(name="legacy")
    loaded.load_profile(str(profile))
    assert isinstance(loaded.memory, PlasticAssociativeRTU)
    assert loaded.adapter_name == "associative_rtu"


def test_explicit_unknown_adapter_fails_without_default_substitution(tmp_path):
    with pytest.raises(InvalidProfileError) as exc:
        DroidEngine(name="bad", plastic_adapter="does_not_exist")
    assert "does_not_exist" in str(exc.value)

    profile = tmp_path / "bad-profile"
    droid = DroidEngine(name="bad", plastic_adapter="associative_rtu",
                        db_path=str(tmp_path / "k.db"))
    droid.save_profile(str(profile))
    import json
    manifest = json.loads((profile / "config.json").read_text())
    manifest["plastic_adapter"] = "does_not_exist"
    (profile / "config.json").write_text(json.dumps(manifest))
    fresh = DroidEngine(name="bad")
    with pytest.raises(InvalidProfileError):
        fresh.load_profile(str(profile))
    # The previously active engine state is untouched.
    assert isinstance(fresh.memory, PlasticAssociativeRTU)


def test_legacy_manifest_without_identity_records_migration(tmp_path):
    import json
    droid = DroidEngine(name="migrate", plastic_adapter="associative_rtu",
                        db_path=str(tmp_path / "k.db"))
    profile = tmp_path / "migrate-profile"
    droid.save_profile(str(profile))
    manifest = json.loads((profile / "config.json").read_text())
    manifest.pop("plastic_adapter")
    manifest.pop("adapter_identity_source", None)
    (profile / "config.json").write_text(json.dumps(manifest))
    loaded = DroidEngine(name="migrate")
    loaded.load_profile(str(profile))
    assert loaded.adapter_name == "associative_rtu"
    assert any(m["migration"] == "adapter_identity_default" for m in loaded.migrations)


def test_incompatible_profile_dimension_fails_without_corruption(tmp_path):
    droid = _droid(tmp_path, "delta_rtu")
    droid.teach("Lidar measures distance with laser pulses.")
    profile = tmp_path / "delta-profile"
    droid.save_profile(str(profile))

    fresh = DroidEngine(name="delta-droid")
    fresh.dim = 128  # force an incompatible adapter geometry
    with pytest.raises(InvalidProfileError):
        fresh.load_profile(str(profile))


def test_teach_and_recall_route_through_delta_adapter(tmp_path):
    droid = _droid(tmp_path, "delta_rtu")
    result = droid.teach("Berlin is the capital of Germany.")
    assert result["status"] == "learned"
    assert result["scheduling_supported"] is True
    hits = droid.recall("What is the capital of Germany?")
    assert hits
    assert any("Berlin" in hit["text"] for hit in hits)


def test_custom_adapter_registry_name_survives_create_save_load_resave(tmp_path):
    """post-merge 2.5: persist the registry id, never the class name."""
    import json
    from src.model import plastic_adapter as registry_module
    from src.model.rtu import PlasticAssociativeRTU

    class InheritedAssociation(PlasticAssociativeRTU):
        """Class name differs from the registry name on purpose."""

    registry_name = "renamed-association"
    registry_module.register_adapter(registry_name, InheritedAssociation)
    try:
        droid = DroidEngine(name="custom", plastic_adapter=registry_name,
                            db_path=str(tmp_path / "k.db"))
        assert droid.adapter_name == registry_name
        droid.teach("The renamed adapter stores this fact.")

        first = tmp_path / "first"
        droid.save_profile(str(first))
        manifest = json.loads((first / "config.json").read_text())
        assert manifest["plastic_adapter"] == registry_name

        loaded = DroidEngine(name="custom")
        loaded.load_profile(str(first))
        assert loaded.adapter_name == registry_name
        assert type(loaded.memory).__name__ == "InheritedAssociation"

        # Re-saving keeps the registry identity, not the inferred class name.
        second = tmp_path / "second"
        loaded.save_profile(str(second))
        resaved = json.loads((second / "config.json").read_text())
        assert resaved["plastic_adapter"] == registry_name
    finally:
        registry_module._ADAPTER_REGISTRY.pop(registry_name, None)
