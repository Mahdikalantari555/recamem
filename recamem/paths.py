"""Filesystem roots for recamem.

Repository-relative paths used to be computed as ``Path(__file__).parents[N]``.
After the package move the depth differs per module, so the lookup walks up to
the directory that holds ``pyproject.toml`` (or ``configs/``) instead of
hard-coding a parent index.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("pyproject.toml", "openspec")
_root: Path | None = None


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing this package."""
    global _root
    if start is None and _root is not None:
        return _root
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if any((candidate / marker).exists() for marker in _MARKERS):
            if start is None:
                _root = candidate
            return candidate
    return here.parent


def package_root() -> Path:
    return Path(__file__).resolve().parent


def configs_dir() -> Path:
    return repo_root() / "configs"


def evals_dir() -> Path:
    return repo_root() / "evals"


def cache_dir() -> Path:
    """Download-once cache, outside the repository."""
    explicit = os.environ.get("RECAMEM_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "recamem"
