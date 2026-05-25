"""Visualization smoke tests: Manim scenes and frame-data helpers."""

from __future__ import annotations
import importlib
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest



matplotlib.use("Agg")

import matplotlib.pyplot as plt

from warpax.geometry.types import GridSpec

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


def test_comparison_plots_smoke():
    from warpax.visualization.comparison_plots import plot_comparison_panel

    shape = (4, 4, 4)
    euler = np.ones(shape)
    robust = euler * 0.9
    missed = np.zeros(shape, dtype=bool)
    fig = plot_comparison_panel(
        euler, robust, missed,
        grid_bounds=[(-1.0, 1.0)] * 3,
        grid_shape=shape,
        title="smoke",
    )
    assert fig is not None


def test_convergence_plots_smoke():
    from pathlib import Path

    from warpax.visualization.convergence_plots import plot_convergence

    json_path = Path("results/convergence_data.json")
    if not json_path.exists():
        pytest.skip("convergence_data.json not present")
    fig = plot_convergence(str(json_path))
    assert fig is not None


def test_kinematic_plots_smoke():
    from warpax.visualization.kinematic_plots import plot_kinematic_scalars

    field = np.zeros((4, 4, 4))
    fig = plot_kinematic_scalars(
        field, field, field,
        grid_bounds=[(-1.0, 1.0)] * 3,
        grid_shape=(4, 4, 4),
    )
    assert fig is not None


def test_geodesic_plots_smoke():
    from warpax.visualization.geodesic_plots import plot_tidal_evolution

    tau = np.linspace(0, 1, 10)
    eig = np.random.default_rng(0).normal(size=(10, 3)) * 1e-3
    fig = plot_tidal_evolution(eig, tau)
    assert fig is not None


def test_direction_fields_smoke():
    from warpax.visualization.direction_fields import plot_worst_observer_field

    grid = GridSpec(bounds=[(-1.0, 1.0)] * 3, shape=(4, 4, 4))
    params = np.zeros((*grid.shape, 3))
    ax = plot_worst_observer_field(params, grid, slice_index=2)
    assert ax is not None


@pytest.mark.smoke
def test_dark_diverge_colormaps_register() -> None:
    """All four ``dark_diverge_*`` colormaps register without error."""
    pytest.importorskip("manim")

    import matplotlib as mpl

    for mod_name in (
        "warpax.visualization.manim._split_screen",
        "warpax.visualization.manim._expansion_shear",
        "warpax.visualization.manim._heatmap_contour",
        "warpax.visualization.manim._boost_arrows",
    ):
        importlib.import_module(mod_name)

    for cmap_name in (
        "dark_diverge_ss",
        "dark_diverge_theta",
        "dark_diverge_hc",
        "dark_diverge",
    ):
        cmap = mpl.colormaps[cmap_name]
        # warm endpoint must be reachable (not a transparent placeholder)
        rgba_hot = cmap(1.0)
        assert rgba_hot[3] > 0.0
        assert max(rgba_hot[:3]) > 0.5


@pytest.mark.smoke
def test_eulerian_energy_density_matches_alcubierre_analytic() -> None:
    """``eulerian_energy_density_grid`` recovers the analytical Alcubierre rho.

    For Alcubierre alpha = 1, beta^x = -v_s f, so

        rho_Eul = T_{ab} n^a n^b = -(v_s^2 / (32 pi)) (df/dr)^2 (y^2+z^2)/r^2

    (Alcubierre 1994 Eq. 19 / arXiv:gr-qc/0009013).
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.benchmarks.alcubierre import eulerian_energy_density
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common import eulerian_energy_density_grid

    grid_spec = GridSpec(
        bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
        shape=(16, 16, 16),
    )
    metric = AlcubierreMetric(v_s=0.5)

    result = evaluate_curvature_grid(metric, grid_spec, compute_invariants=False)
    rho_numeric = eulerian_energy_density_grid(
        result.stress_energy, result.metric_inv
    )

    X, Y, Z = grid_spec.meshgrid
    rho_analytic = eulerian_energy_density(
        jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z), v_s=0.5
    )
    rho_analytic = np.asarray(rho_analytic)

    # Mask the wall (where df/dr is concentrated and the analytical
    # formula is well-defined)
    mask = np.abs(rho_analytic) > 1e-6
    if not mask.any():
        pytest.skip("Wall region empty in the chosen grid")

    rel_err = np.abs(
        rho_numeric[mask] - rho_analytic[mask]
    ) / np.abs(rho_analytic[mask])
    assert rel_err.max() < 1e-2, (
        f"max rel err {rel_err.max():.3e} between Eulerian density "
        "from T_{ab} n^a n^b and Alcubierre analytical form"
    )


@pytest.mark.smoke
def test_eulerian_density_differs_from_T00_with_shift() -> None:
    """For Alcubierre with non-zero v_s, rho_Eul != T_{00}.

    Sanity check that we are *not* secretly plotting the same field
    under two different labels.
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common import eulerian_energy_density_grid

    grid_spec = GridSpec(
        bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
        shape=(16, 16, 16),
    )
    metric = AlcubierreMetric(v_s=0.5)
    result = evaluate_curvature_grid(metric, grid_spec, compute_invariants=False)

    rho_eul = eulerian_energy_density_grid(
        result.stress_energy, result.metric_inv
    )
    T_00 = np.asarray(result.stress_energy[..., 0, 0])

    diff = rho_eul - T_00
    assert np.max(np.abs(diff)) > 1e-8, (
        "Eulerian density and T_{00} are bit-identical: shift correction "
        "is missing or not applied"
    )


@pytest.mark.smoke
def test_eulerian_density_equals_T00_for_minkowski() -> None:
    """For Minkowski (alpha=1, beta=0), rho_Eul == T_{00}."""
    from warpax.benchmarks import MinkowskiMetric
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common import eulerian_energy_density_grid

    grid_spec = GridSpec(
        bounds=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        shape=(8, 8, 8),
    )
    result = evaluate_curvature_grid(
        MinkowskiMetric(), grid_spec, compute_invariants=False
    )

    rho_eul = eulerian_energy_density_grid(
        result.stress_energy, result.metric_inv
    )
    T_00 = np.asarray(result.stress_energy[..., 0, 0])
    np.testing.assert_allclose(rho_eul, T_00, atol=1e-12)


@pytest.mark.smoke
def test_freeze_curvature_exposes_both_fields() -> None:
    """``freeze_curvature`` exposes ``energy_density`` and ``T_00_covariant``."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common import freeze_curvature

    grid_spec = GridSpec(
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        shape=(8, 8, 8),
    )
    metric = AlcubierreMetric(v_s=0.5)
    result = evaluate_curvature_grid(metric, grid_spec, compute_invariants=True)

    frame = freeze_curvature(result, grid_spec, metric_name="Alcubierre", v_s=0.5)
    assert "energy_density" in frame.scalar_fields
    assert "T_00_covariant" in frame.scalar_fields
    assert "energy_density" in frame.colormaps
    assert "energy_density" in frame.clim
