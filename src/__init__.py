"""Transitional shim package (removed one release after the recamem rename).

``src.model.*`` re-export the real implementations from :mod:`recamem`; the
neural substrate modules that have not moved yet still live here. Keeping the
shims importable is what lets existing scripts and tests run unchanged
(``docs/migration.md``).
"""
