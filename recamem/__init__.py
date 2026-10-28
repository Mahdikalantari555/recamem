"""RecaMem — a local-first plastic memory engine.

The package is the primary home of the engine code. The historical
``src.model`` modules remain as deprecation shims for one release.

Public surface is imported lazily so that ``import recamem`` stays cheap and
does not pull in torch until a caller actually needs an engine.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

#: Contract version written into plastic state and profile manifests.
PLASTIC_SCHEMA_VERSION = 2

_LAZY: dict[str, tuple[str, str]] = {
    "DroidEngine": ("recamem.engine", "DroidEngine"),
    "MemoryEngine": ("recamem.engine", "DroidEngine"),
    "DroidManager": ("recamem.manager", "DroidManager"),
    "AgentManager": ("recamem.manager", "DroidManager"),
    "KnowledgeStore": ("recamem.episodic", "KnowledgeStore"),
    "DecisionHead": ("recamem.decision_head", "DecisionHead"),
    "DroidPackage": ("recamem.package", "DroidPackage"),
    "AgentPackage": ("recamem.package", "DroidPackage"),
    "GgufBackend": ("recamem.gguf", "GgufBackend"),
    "ByteRtuBackend": ("recamem.byte_rtu", "ByteRtuBackend"),
    "MiniLMAnchor": ("recamem.anchor", "OnnxMiniLM"),
    "OnnxMiniLM": ("recamem.anchor", "OnnxMiniLM"),
    "AnchorUnavailableError": ("recamem.anchor", "AnchorUnavailableError"),
    "Model": ("src.model.rtu", "Model"),
    "RTUMemoryBlock": ("src.model.rtu", "RTUMemoryBlock"),
    "MissingCheckpointError": ("src.model.rtu", "MissingCheckpointError"),
    "InvalidProfileError": ("src.model.rtu", "InvalidProfileError"),
    "extract_propositions": ("recamem.segmenter", "extract_propositions"),
    "extract_propositions_with_fallback": (
        "recamem.segmenter",
        "extract_propositions_with_fallback",
    ),
    "build_recamem_core": ("src.model.recamem_core", "build_recamem_core"),
}

__all__ = sorted(_LAZY) + [
    "PLASTIC_SCHEMA_VERSION",
    "agent_home",
    "agents_dir",
    "get_adapter",
    "list_adapters",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    target = _LAZY.get(name)
    if target is None:
        if name in {"agent_home", "agents_dir"}:
            from recamem import identity

            return getattr(identity, name)
        if name in {"get_adapter", "list_adapters", "PlasticAdapter"}:
            from recamem import plastic_adapter

            return getattr(plastic_adapter, name)
        raise AttributeError(f"module 'recamem' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:  # pragma: no cover - convenience only
    return sorted(set(globals()) | set(_LAZY))
