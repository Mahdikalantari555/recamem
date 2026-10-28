"""torch_decoder_backend — compatibility shim (recamem identity migration).

The implementation lives in :mod:`recamem.torch_decoder_backend`. This module is an alias for it:
it replaces itself in ``sys.modules`` so existing imports, ``monkeypatch.setattr``
calls and pickle paths keep resolving to the real implementation. It is removed
one release after the rename.
"""
import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module

_warnings.warn(
    "src.model.torch_decoder_backend is deprecated; import recamem.torch_decoder_backend instead",
    DeprecationWarning,
    stacklevel=2,
)
_target = _import_module("recamem.torch_decoder_backend")
_sys.modules[__name__] = _target
