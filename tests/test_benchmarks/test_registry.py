"""Tests for the benchmarks registry."""

import pytest

from warpax.benchmarks.registry import create_default_registry


def test_default_registry_lists_core_metrics():
    reg = create_default_registry()
    names = set(reg.list_metrics())
    assert "Alcubierre" in names
    assert "Minkowski" in names
    assert "Schwarzschild" in names
    assert len(reg) == 3


def test_registry_ground_truth_payloads():
    reg = create_default_registry()

    metric, gt = reg.get("Minkowski")
    assert metric.name() == "Minkowski"
    assert gt["kretschmann"] == 0.0
    assert gt["stress_energy_zero"] is True

    _, gt_alc = reg.get("Alcubierre")
    assert gt_alc["energy_conditions"] == {
        "WEC": False,
        "NEC": False,
        "DEC": False,
        "SEC": False,
    }
    assert gt_alc["stress_energy_zero"] is False

    _, gt_schw = reg.get("Schwarzschild")
    assert gt_schw["stress_energy_zero"] is True


def test_registry_contains_and_unknown_name():
    reg = create_default_registry()
    assert "Minkowski" in reg
    assert "Nope" not in reg
    with pytest.raises(KeyError):
        reg.get("Nope")
