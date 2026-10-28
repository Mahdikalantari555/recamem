"""Generation-backend capability registry.

One place answers "what can this backend actually do?", so the engine, CLI,
MCP server and UI cannot claim a capability the backend does not have. Mock
backends are labelled here and only ever constructed when a caller asks for one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from recamem.byte_rtu import ByteRtuBackend
from recamem.gguf import GgufBackend, GgufLoadError


def backend_capabilities(backend: Any) -> Dict[str, Any]:
    """Return a capability matrix entry for ``backend``.

    Unknown backends get the conservative answer: prompt injection only, no
    hidden states, no KV priming, not mock.
    """
    if backend is None:
        return {
            "backend": "none",
            "is_mock": False,
            "loaded": False,
            "supports_prompt_injection": False,
            "supports_hidden_injection": False,
            "supports_kv_priming": False,
        }
    if hasattr(backend, "capability_report"):
        report = dict(backend.capability_report())
    else:
        report = {
            "backend": type(backend).__name__,
            "is_mock": bool(getattr(backend, "is_mock", False)),
            "loaded": getattr(backend, "model", None) is not None,
            "supports_hidden_injection": bool(
                getattr(backend, "supports_hidden_injection", lambda: False)()
            ),
            "supports_kv_priming": bool(
                getattr(backend, "supports_kv_priming", lambda: False)()
            ),
        }
    report.setdefault("supports_prompt_injection", True)
    return report


def build_backend(kind: str, *, mock: bool = False, **kwargs: Any):
    """Construct a backend by name, refusing silent mock substitution."""
    kind = (kind or "none").lower()
    if kind in {"none", "off"}:
        return None
    if kind == "gguf":
        return GgufBackend(mock=mock, **kwargs)
    if kind in {"byte_rtu", "byte-rtu", "rtu"}:
        return ByteRtuBackend(**kwargs)
    if kind == "mock":
        return GgufBackend(mock=True, **kwargs)
    raise ValueError(f"Unknown backend {kind!r}; expected gguf, byte_rtu, mock or none")


__all__ = [
    "GgufBackend",
    "GgufLoadError",
    "ByteRtuBackend",
    "backend_capabilities",
    "build_backend",
]
