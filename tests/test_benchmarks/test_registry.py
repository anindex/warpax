"""Smoke test for benchmarks registry."""

from warpax.benchmarks.registry import create_default_registry


def test_default_registry_lists_core_metrics():
    reg = create_default_registry()
    names = set(reg.list_metrics())
    assert "Alcubierre" in names
    assert "Minkowski" in names
    assert "Schwarzschild" in names
