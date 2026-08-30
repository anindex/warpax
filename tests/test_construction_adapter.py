"""Tests for the cross-construction adapter."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

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
        expected = ("Alcubierre", "Rodal", "Fuchs", "WarpShell", "Garattini", "S-shell", "T-shell")
        for name in expected:
            assert name in reg
            assert isinstance(reg[name], ConstructionSpec)
        assert len(reg) == len(expected)

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


def test_garattini_wall_resolves_when_the_reduction_is_bubble_centred():
    """The A3 withholding was a grid artefact, not a property of the construction.

    Garattini-Zatrimaylov places its bubble at ``r_0 = v_s / H`` by the matching
    condition, not at the coordinate origin. An origin-centred axisymmetric reduction
    clusters its radial nodes on a sphere the wall merely *crosses*, so the wall spans
    1.5 cells at the coarsest ladder level and the panel withheld it as under-resolved.
    The reduction is exact about either point, the configuration is axisymmetric
    about the propagation axis, which passes through both, and taken about the bubble
    the same ladder level spans 4.5.
    """
    import numpy as np

    from warpax.analysis.construction_adapter import (
        construction_registry,
        matched_registry,
    )
    from warpax.grids import axisymmetric_grid, wall_cells_on_axis

    for registry in (construction_registry, matched_registry):
        spec = registry()["Garattini"]
        metric = spec.metric()
        center = spec.center_of(metric)
        assert center == pytest.approx(metric.v_s / metric.H), "centre is r_0 = v_s / H"
        assert center != 0.0, "the whole point is that the bubble is off-origin"

        grid = axisymmetric_grid(
            spec.r_max, 32, 32, wall_radius=spec.wall_radius, a=spec.cluster_a, center=center
        )
        axis = center + np.concatenate([-grid.r[::-1], grid.r])
        assert wall_cells_on_axis(metric, axis).cells >= 4.0

        # And the origin-centred axis is what it used to be measured on.
        origin_axis = np.concatenate([-grid.r[::-1], grid.r])
        assert wall_cells_on_axis(metric, origin_axis).cells < 4.0


def test_axisymmetric_center_shifts_only_the_axis():
    """The offset must move the sampled points and nothing else."""
    import numpy as np

    from warpax.grids import axisymmetric_grid

    a = axisymmetric_grid(3.0, 12, 8, wall_radius=1.0)
    b = axisymmetric_grid(3.0, 12, 8, wall_radius=1.0, center=2.5)
    np.testing.assert_allclose(np.asarray(b.coords)[:, 1], np.asarray(a.coords)[:, 1] + 2.5)
    for col in (0, 2, 3):
        np.testing.assert_array_equal(np.asarray(b.coords)[:, col], np.asarray(a.coords)[:, col])
    np.testing.assert_array_equal(b.weights, a.weights)
