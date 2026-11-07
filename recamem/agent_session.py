"""Shared session entry point for existing RecaMem runner operations.

The first migration step intentionally preserves the legacy runner contract;
it does not assert that all API, MCP or memory paths are unified yet.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AgentSession:
    """Execute a single request against an explicitly supplied engine."""

    def __init__(self, engine: Any, *, profile: Optional[str] = None) -> None:
        self.engine = engine
        self.profile = profile

    def handle(self, prompt: str, *, mode: str = "chat", top_k: int = 3) -> Dict[str, Any]:
        # Deferred import avoids a circular import with the compatibility runner.
        from recamem.runner import _run_once_impl

        return _run_once_impl(prompt, self.engine, mode=mode, profile=self.profile, top_k=top_k)
