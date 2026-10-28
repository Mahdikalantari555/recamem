"""web_teacher — compatibility shim (recamem identity migration).

The implementation lives in :mod:`recamem.web_teacher`. This module is an alias for it:
it replaces itself in ``sys.modules`` so existing imports, ``monkeypatch.setattr``
calls and pickle paths keep resolving to the real implementation. It is removed
one release after the rename.
"""
import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module

_warnings.warn(
    "src.model.web_teacher is deprecated; import recamem.web_teacher instead",
    DeprecationWarning,
    stacklevel=2,
)
_target = _import_module("recamem.web_teacher")
_sys.modules[__name__] = _target
