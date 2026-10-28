"""Benchmark surface for recamem.

``recamem.bench.metrics`` and ``recamem.bench.continual`` carry the metrics
module that used to live in ``src/model/benchmarks.py`` and
``src/model/continual_bench.py``. ``recamem.bench.runners`` adds the LoCoMo,
LongMemEval-S and DMR runners with download-once caching and labelled offline
fixtures.
"""

from recamem.bench import continual, metrics  # noqa: F401

__all__ = ["continual", "metrics", "runners"]


def __getattr__(name):
    if name == "runners":
        import importlib

        return importlib.import_module("recamem.bench.runners")
    raise AttributeError(f"module 'recamem.bench' has no attribute {name!r}")
