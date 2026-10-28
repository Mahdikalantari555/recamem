"""Core install works without optional packages (post-merge task 3.8).

The check runs in a subprocess with an import blocker installed *before* the
model imports, so it cannot be satisfied by packages the test session already
loaded.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED = [
    "llama_cpp",
    "hnswlib",
    "onnxruntime",
    "onnx",
    "transformers",
    "huggingface_hub",
    "streamlit",
    "pandas",
    "sentence_transformers",
]

PROBE = """
import importlib.abc, importlib.machinery, sys, json

BLOCKED = %r


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"blocked optional dependency: {root}")
        return None


sys.meta_path.insert(0, Blocker())

from src.model.droid import DroidEngine
from src.model.plastic_adapter import list_adapters
from src.model.backend_contracts import capabilities_of
from src.model.memory_injection import inject_prompt
from src.model.hybrid_retrieval import build_dense_index
from src.model.edge_slm import EdgeSLMRegistry

anchor_report = {}
from src.model.onnx_anchor import OnnxMiniLM
anchor = OnnxMiniLM(mode="mock")
anchor_report = {"is_mock": bool(getattr(anchor, "is_mock", False))}

index = build_dense_index("auto", dim=8)
print(json.dumps({
    "adapters": list_adapters(),
    "anchor": anchor_report,
    "index_mode": index.mode,
    "catalog": len(EdgeSLMRegistry().all()),
    "prompt": inject_prompt("q", [{"text": "fact"}]),
    "capabilities": capabilities_of(None).to_dict(),
}))
""" % (BLOCKED,)


def test_core_imports_and_degrades_without_optional_packages():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert "associative_rtu" in payload["adapters"]
    assert payload["anchor"]["is_mock"] is True, (
        "without onnxruntime the anchor must report mock embeddings"
    )
    assert payload["index_mode"] == "exact-fallback"
    assert payload["catalog"] == 13
    assert payload["prompt"] == "Context:\n- fact\n\nQuery: q"


def test_requirement_files_separate_core_from_optional():
    core = (ROOT / "requirements-core.txt").read_text()
    optional = (ROOT / "requirements-optional.txt").read_text()
    full = (ROOT / "requirements.txt").read_text()

    for required in ("torch", "numpy", "safetensors", "pytest"):
        assert required in core
    for heavy in ("llama-cpp-python", "onnxruntime", "hnswlib", "streamlit"):
        assert heavy in optional
        assert heavy not in core
    assert "-r requirements-core.txt" in full
    assert "-r requirements-optional.txt" in full


def test_documentation_names_every_fallback():
    doc = (ROOT / "docs/optional-dependencies.md").read_text()
    for token in ("is_mock", "exact-fallback", "fallback-lexical", "MockLlama"):
        assert token in doc
