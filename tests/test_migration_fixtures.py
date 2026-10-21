"""Migration fixtures without model weights (post-merge task 1.4).

Three profile shapes are built deterministically in a temporary directory:

* **legacy** — no adapter identity, no metadata sidecar, `states`-style tensors
  and a JSON knowledge store;
* **default** — the current `associative_rtu` profile written by `save_profile`;
* **custom** — a registered adapter whose class name differs from its registry
  identifier, with constructor kwargs and a request object.

Their checksums and normalized configurations are recorded in
`tests/fixtures/profile_checksums.json`, so a change in the written artifact
layout has to be acknowledged explicitly rather than silently accepted.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.model.droid import DroidEngine
from src.model.plastic_adapter import PlasticAdapter, register_adapter
from src.model.rtu import InvalidProfileError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = REPO_ROOT / "tests" / "fixtures" / "profile_checksums.json"
DIM = 384


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CustomTraceAdapter(PlasticAdapter):
    """Registry id 'custom_trace' but a class name that shares no substring."""

    _adapter_type = "custom_trace"
    _adapter_version = 3

    def __init__(self, dim: int = DIM, gain: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.gain = float(gain)
        self.register_buffer("trace", torch.zeros(dim))
        self.register_buffer("bias", torch.zeros(dim))

    def update(self, x):
        with torch.no_grad():
            self.trace.mul_(0.9).add_(x * self.gain)

    def retrieve(self, x):
        return self.trace + self.bias

    def predict_next(self):
        return self.trace + self.bias

    def compute_surprise(self, x):
        return x - self.trace, float((x - self.trace).norm())

    def state_dict(self):
        return {
            "trace": self.trace.detach().clone(),
            "bias": self.bias.detach().clone(),
            "dim": self.dim,
            "gain": self.gain,
            "_adapter_type": self._adapter_type,
            "_adapter_version": self._adapter_version,
        }

    def load_state_dict(self, state):
        with torch.no_grad():
            self.trace.copy_(state["trace"])
            self.bias.copy_(state["bias"])
        self.dim = int(state.get("dim", self.dim))
        self.gain = float(state.get("gain", self.gain))


def _build_legacy(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(11)
    trace = torch.randn(DIM, generator=generator)
    save_file(
        {
            "states": trace,
            "decay": torch.ones(1),
            "proj_weight": torch.eye(DIM),
        },
        str(directory / "memory.safetensors"),
    )
    facts = [
        {"text": "The mixer is on site B", "source": "legacy", "timestamp": 1.0,
         "step": 0, "novelty": 0.5},
        {"text": "The crane is inspected weekly", "source": "legacy",
         "timestamp": 2.0, "step": 1, "novelty": 0.4},
    ]
    (directory / "knowledge.json").write_text(json.dumps(facts, indent=2))
    manifest = {
        "name": "legacy",
        "dim": DIM,
        "step_count": 2,
        "base_lr": 0.05,
        "total_facts": len(facts),
        "surprise_threshold": 0.4,
        "K_max": 5000,
        "test_time_training": True,
        "engine": "droid",
        "fusion_mode": "off",
        "rtu_dim": DIM,
        "rtu_layers": 8,
        "rtu_spread": 32,
        "byte_rtu_checkpoint": None,
    }
    (directory / "config.json").write_text(json.dumps(manifest, indent=2))


def _build_default(directory: Path):
    engine = DroidEngine(name="default", db_path=str(directory.parent / "default.db"))
    engine.teach("The mixer is on site B")
    engine.save_profile(str(directory))
    engine.knowledge.close()


def _build_custom(directory: Path):
    register_adapter("custom_trace", CustomTraceAdapter, strict=False)
    engine = DroidEngine(
        name="custom",
        db_path=str(directory.parent / "custom.db"),
        plastic_adapter="custom_trace",
        adapter_kwargs={"dim": DIM, "gain": 2.0},
    )
    engine.teach("The crane is inspected weekly")
    engine.save_profile(str(directory))
    engine.knowledge.close()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    _build_legacy(root / "legacy")
    _build_default(root / "default")
    _build_custom(root / "custom")
    return root


VOLATILE_CONFIG_KEYS = {"updated_at", "platform", "python"}


def _normalised_config(directory: Path) -> dict:
    raw = json.loads((directory / "config.json").read_text())
    return {k: v for k, v in raw.items() if k not in VOLATILE_CONFIG_KEYS}


def describe(directory: Path) -> dict:
    """Checksum + layout contract for one fixture profile.

    Volatile values (wall-clock timestamps, SQLite page bytes) are excluded;
    what is asserted is the stable artifact layout, tensor schema and manifest
    configuration that migrations consume.
    """
    from safetensors.torch import load_file

    config = _normalised_config(directory)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    tensors = sorted(load_file(str(directory / "memory.safetensors")))
    description = {
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "config_keys": sorted(config),
        "plastic_adapter": config.get("plastic_adapter"),
        "adapter_kwargs": config.get("adapter_kwargs"),
        "tensor_keys": tensors,
        "tensor_bytes": (directory / "memory.safetensors").stat().st_size,
        "has_adapter_meta": (directory / "adapter_meta.json").exists(),
        "has_knowledge_db": (directory / "knowledge.db").exists(),
    }
    db = directory / "knowledge.db"
    if db.exists():
        import sqlite3

        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            description["db_tables"] = sorted(
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            )
            description["fact_rows"] = conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        finally:
            conn.close()
    else:
        description["legacy_facts"] = len(
            json.loads((directory / "knowledge.json").read_text())
        )
    return description


def test_fixture_checksums_and_configurations_are_recorded(built):
    observed = {shape: describe(built / shape)
                for shape in ("legacy", "default", "custom")}
    import os

    if os.environ.get("UPDATE_FIXTURE_MANIFEST") == "1":
        EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    expected = json.loads(EXPECTED.read_text())
    assert observed == expected, (
        "Fixture layout changed. Review it and regenerate with "
        "UPDATE_FIXTURE_MANIFEST=1 pytest tests/test_migration_fixtures.py"
    )

    legacy_config = json.loads((built / "legacy" / "config.json").read_text())
    assert "plastic_adapter" not in legacy_config
    assert not (built / "legacy" / "adapter_meta.json").exists()
    assert not (built / "legacy" / "knowledge.db").exists()

    custom_config = json.loads((built / "custom" / "config.json").read_text())
    assert custom_config["plastic_adapter"] == "custom_trace"
    assert custom_config["adapter_kwargs"]["gain"] == 2.0
    assert expected["custom"]["adapter_kwargs"]["gain"] == 2.0


def test_legacy_fixture_loads_with_a_migration_record(built):
    engine = DroidEngine(name="legacy")
    engine.load_profile(str(built / "legacy"))
    assert engine.adapter_name == "associative_rtu"
    assert any(m["migration"] == "adapter_identity_default" for m in engine.migrations)
    assert engine.knowledge.active_count() == 2
    engine.knowledge.close()


def test_default_fixture_round_trips(built):
    engine = DroidEngine(name="default")
    engine.load_profile(str(built / "default"))
    assert engine.adapter_name == "associative_rtu"
    assert engine.knowledge.active_count() == 1
    engine.knowledge.close()


def test_custom_fixture_round_trips_with_identity_and_kwargs(built, tmp_path):
    register_adapter("custom_trace", CustomTraceAdapter, strict=False)
    engine = DroidEngine(name="custom")
    engine.load_profile(str(built / "custom"))
    assert engine.adapter_name == "custom_trace"
    assert isinstance(engine.memory, CustomTraceAdapter)
    assert engine.memory.gain == pytest.approx(2.0)
    engine.teach("The pump is on site D")
    resaved = tmp_path / "resaved"
    engine.save_profile(str(resaved))
    manifest = json.loads((resaved / "config.json").read_text())
    assert manifest["plastic_adapter"] == "custom_trace"
    assert manifest["adapter_kwargs"]["gain"] == pytest.approx(2.0)
    engine.knowledge.close()


def test_unknown_adapter_fixture_is_rejected(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    _build_legacy(directory)
    manifest = json.loads((directory / "config.json").read_text())
    manifest["plastic_adapter"] = "definitely_not_registered"
    (directory / "config.json").write_text(json.dumps(manifest))
    engine = DroidEngine(name="broken")
    with pytest.raises(InvalidProfileError, match="unsupported plastic_adapter"):
        engine.load_profile(str(directory))
    engine.knowledge.close()


def test_old_database_schema_is_migrated_in_place(tmp_path):
    """A pre-lifelong DB (no decay/lifecycle columns) opens and is upgraded."""
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT,
            timestamp REAL,
            step INTEGER,
            novelty REAL,
            superseded INTEGER DEFAULT 0,
            valid_until REAL,
            superseded_by INTEGER
        );
        CREATE TABLE facts_vecs (
            id INTEGER PRIMARY KEY REFERENCES facts(id),
            vec BLOB
        );
        CREATE VIRTUAL TABLE facts_fts USING fts5(text, content='facts', content_rowid='id');
        """
    )
    conn.execute(
        "INSERT INTO facts(text, source, timestamp, step, novelty) VALUES (?,?,?,?,?)",
        ("The mixer is on site B", "legacy", 1.0, 0, 0.5),
    )
    conn.commit()
    conn.close()

    from src.model.knowledge_store import KnowledgeStore

    store = KnowledgeStore(str(path), dim=8)
    columns = {row[1] for row in store.conn.execute("PRAGMA table_info(facts)")}
    assert {"decay_beta", "access_count", "last_retrieved", "version_id",
            "feedback", "needs_review"} <= columns
    tables = {row[0] for row in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"retrieval_feedback", "feedback_log", "semantic_facts"} <= tables
    assert store.get_audit_trail()[0]["text"] == "The mixer is on site B"
    assert store.decay_exponent(1) == pytest.approx(store.DECAY_BETA_INITIAL)
    store.close()
