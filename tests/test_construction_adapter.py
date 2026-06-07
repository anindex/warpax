"""Tests for the cross-construction audit adapter."""
from __future__ import annotations

import jax.numpy as jnp

from warpax.analysis.construction_adapter import (
    MIN_WALL_CELLS,
    ConstructionSpec,
    construction_registry,
    is_resolved,
    wall_cells,
)
from warpax.geometry.metric import MetricSpecification


class TestRegistry:
    def test_registry_has_all_constructions(self):
        reg = construction_registry()
        for name in ("Alcubierre", "Rodal", "Fuchs", "WarpShell",
                     "S-shell", "T-shell"):
            assert name in reg
            assert isinstance(reg[name], ConstructionSpec)

    def test_each_spec_builds_a_metric(self):
        reg = construction_registry()
        for name, spec in reg.items():
            m = spec.metric()
            assert isinstance(m, MetricSpecification), name
            g = m(jnp.array([0.0, spec.bounds[0][1] * 0.4, 0.1, 0.0]))
            assert bool(jnp.all(jnp.isfinite(g))), name

    def test_tshell_uses_v0_and_is_static(self):
        reg = construction_registry()
        ts = reg["T-shell"]
        assert ts.speed_param == "v_0"
        assert ts.is_comoving is False


class TestResolutionGate:
    def test_all_constructions_resolve_at_default_n(self):
        reg = construction_registry()
        for name, spec in reg.items():
            resolved, cells = is_resolved(spec)
            assert resolved, f"{name} unresolved at default N ({cells} cells)"
            assert cells >= MIN_WALL_CELLS

    def test_coarse_grid_flags_unresolved(self):
        # A very coarse grid must fail the gate for a compact wall.
        reg = construction_registry()
        spec = reg["Alcubierre"]
        cells = wall_cells(spec, n=4)
        assert cells < MIN_WALL_CELLS
