"""The asv harness runs, so a perf regression can actually be measured.

asv itself is never invoked in CI, so nothing caught the benchmark API
drifting away from the library. This runs every ``setup`` and one ``time_*``
method per class at import cost only.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

BENCH_MODULES = sorted(
    m.name for m in pkgutil.iter_modules(["benchmarks"]) if m.name.startswith("bench_")
)


def _classes(module):
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and not obj.__name__.startswith("_")
    ]


def test_every_benchmark_module_is_discovered():
    assert len(BENCH_MODULES) == 7, BENCH_MODULES


@pytest.mark.slow
@pytest.mark.parametrize("name", BENCH_MODULES)
def test_benchmark_class_runs(name):
    module = importlib.import_module(f"benchmarks.{name}")
    classes = _classes(module)
    assert classes, f"{name} defines no benchmark class"
    for cls in classes:
        bench = cls()
        if hasattr(bench, "setup"):
            bench.setup()
        timed = [n for n in dir(bench) if n.startswith("time_")]
        assert timed, f"{cls.__name__} has no time_* method"
        getattr(bench, timed[0])()
