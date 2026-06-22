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
        euler,
        robust,
        missed,
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
        field,
        field,
        field,
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
        "warpax.visualization.manim._eulerian_kinematics",
        "warpax.visualization.manim._nec_margin",
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
    rho_numeric = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)

    X, Y, Z = grid_spec.meshgrid
    rho_analytic = eulerian_energy_density(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z), v_s=0.5)
    rho_analytic = np.asarray(rho_analytic)

    # Mask the wall (where df/dr is concentrated and the analytical
    # formula is well-defined)
    mask = np.abs(rho_analytic) > 1e-6
    if not mask.any():
        pytest.skip("Wall region empty in the chosen grid")

    rel_err = np.abs(rho_numeric[mask] - rho_analytic[mask]) / np.abs(rho_analytic[mask])
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

    rho_eul = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
    T_00 = np.asarray(result.stress_energy[..., 0, 0])

    diff = rho_eul - T_00
    assert np.max(np.abs(diff)) > 1e-8, (
        "Eulerian density and T_{00} are bit-identical: shift correction is missing or not applied"
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
    result = evaluate_curvature_grid(MinkowskiMetric(), grid_spec, compute_invariants=False)

    rho_eul = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
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


@pytest.mark.smoke
def test_observer_robust_nec_nonpositive_dense_sampling() -> None:
    """Dense null sampling makes the observer-robust NEC <= 0 everywhere.

    The Alcubierre NEC is violated for v_s > 0 and the worst-case over the
    null sphere is <= 0 at every point. Sampling only the few axis-aligned
    null rays of the rapidity observers leaves spurious positive "satisfied"
    pixels; the dense default must not (regression guard for B1).
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.sweep import make_rapidity_observers
    from warpax.geometry import GridSpec
    from warpax.visualization.common import build_ec_frame_sequence

    grid_spec = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(16, 16, 16))
    metric = AlcubierreMetric(v_s=0.9)

    dense = build_ec_frame_sequence(metric, grid_spec, v_s_values=[0.9], progress=False)[0]
    nec_dense = np.asarray(dense.scalar_fields["nec_margin_sweep"])
    assert nec_dense.max() <= 1e-6, "dense NEC has spurious satisfied pixels"
    assert nec_dense.min() < -0.1, "dense NEC missed the wall violation"

    coarse = build_ec_frame_sequence(
        metric,
        grid_spec,
        v_s_values=[0.9],
        nec_observer_params=make_rapidity_observers(),
        progress=False,
    )[0]
    nec_coarse = np.asarray(coarse.scalar_fields["nec_margin_sweep"])
    # The coarse axis-only sampling demonstrably over-reports satisfaction.
    assert nec_coarse.max() > nec_dense.max() + 1e-3


@pytest.mark.smoke
def test_ec_frames_expose_real_shape_function() -> None:
    """EC frames carry the true f(r_s) field, not a circular fallback (B3)."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.metrics._common import alcubierre_shape
    from warpax.visualization.common import build_ec_frame_sequence

    grid_spec = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(17, 17, 17))
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    frame = build_ec_frame_sequence(metric, grid_spec, v_s_values=[0.5], progress=False)[0]

    assert "shape_function" in frame.scalar_fields
    f = np.asarray(frame.scalar_fields["shape_function"])

    X, Y, Z = grid_spec.meshgrid
    r_s = np.sqrt(np.asarray(X) ** 2 + np.asarray(Y) ** 2 + np.asarray(Z) ** 2)
    f_analytic = np.asarray(alcubierre_shape(jnp.asarray(r_s), 1.0, 8.0))
    # The field is the genuine f(r_s), so it matches the analytic form -- a
    # circular fallback (radius x_extent/4 = 1.5) would not.
    np.testing.assert_allclose(f, f_analytic, atol=1e-5)
    assert f.max() > 0.99 and f.min() < 0.01


@pytest.mark.smoke
def test_image_contour_orientation_x_horizontal() -> None:
    """Contours place physical x on the horizontal axis, y on the vertical.

    Regression guard for B4 (image/contour were transposed: motion axis x
    rendered vertical and inverted). Uses a non-square grid to expose swaps.
    """
    from warpax.visualization.manim._image_utils import extract_zero_contour

    nx, ny = 9, 15
    x = np.linspace(-2, 2, nx)
    data_x = np.repeat(x[:, None], ny, axis=1)  # f = x -> zero line at x = 0
    vx = np.concatenate(extract_zero_contour(data_x, (-2, 2), (-3, 3), level=0.0), axis=0)
    assert np.allclose(vx[:, 0], 0.0, atol=0.1)
    assert vx[:, 1].max() > 1.5 and vx[:, 1].min() < -1.5

    y = np.linspace(-3, 3, ny)
    data_y = np.repeat(y[None, :], nx, axis=0)  # f = y -> zero line at y = 0
    vy = np.concatenate(extract_zero_contour(data_y, (-2, 2), (-3, 3), level=0.0), axis=0)
    assert np.allclose(vy[:, 1], 0.0, atol=0.1)
    assert vy[:, 0].max() > 1.0 and vy[:, 0].min() < -1.0


@pytest.mark.smoke
def test_extract_contours_multi_level() -> None:
    """``extract_contours`` returns one path-group per level, nested and NaN-safe.

    Backs the finer graded contour overlays (NECMargin2D / EulerianKinematics2D /
    EulerianVsWorstCaseNEC): several iso-levels traced in one call, deeper levels
    enclosing larger rings, and a stray NaN neither crashing nor wiping a contour.
    """
    from warpax.visualization.manim._image_utils import (
        extract_contours,
        extract_zero_contour,
    )

    n = 41
    xs = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    data = -(X**2 + Y**2)  # radial bowl, most negative at the centre
    gmin = float(data.min())
    # Fractions kept < 0.5 so every level is a full ring inside [-3, 3].
    levels = sorted(gmin * f for f in (0.1, 0.2, 0.3))

    res = extract_contours(data, (-3, 3), (-3, 3), levels, scene_width=6.0)
    assert len(res) == len(levels)
    got = [lvl for lvl, _ in res]
    np.testing.assert_allclose(got, sorted(levels), atol=1e-6)
    for _lvl, paths in res:
        assert paths, "expected a closed contour at this level"

    def _max_radius(paths: list[np.ndarray]) -> float:
        pts = np.concatenate(paths, axis=0)
        return float(np.max(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))

    # f decreases outward, so the most-negative (deepest) level is the outer ring.
    assert _max_radius(res[0][1]) > _max_radius(res[-1][1])

    # A stray NaN must not crash or erase the contour (order=1 bilinear fill).
    d2 = data.copy()
    d2[0, 0] = np.nan
    res2 = extract_contours(d2, (-3, 3), (-3, 3), [gmin * 0.2])
    assert res2 and res2[0][1]

    # Back-compat single-level wrapper still yields a flat list of (N, 2) segs.
    zc = extract_zero_contour(data, (-3, 3), (-3, 3), level=gmin * 0.2)
    assert isinstance(zc, list) and zc and zc[0].shape[1] == 2


@pytest.mark.smoke
def test_eulerian_kinematics_shape_function_is_real() -> None:
    """EulerianKinematics2D's bubble wall uses the real f(r_s), not a circle.

    Regression guard: EulerianKinematics2D builds frames from kinematic scalars
    and must attach the analytic shape function so the f = 0.5 overlay is the true
    bubble wall, not the heuristic-circle fallback in ``extract_bubble_contour``.
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.metrics._common import alcubierre_shape
    from warpax.visualization.common._physics import _shape_function_grid

    grid_spec = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(17, 17, 17))
    metric = AlcubierreMetric(v_s=0.6, R=1.0, sigma=8.0)

    f = _shape_function_grid(metric, grid_spec, 0.0)
    assert f is not None, "Alcubierre must expose a shape function for the wall"
    f = np.asarray(f)

    X, Y, Z = grid_spec.meshgrid
    r_s = np.sqrt(np.asarray(X) ** 2 + np.asarray(Y) ** 2 + np.asarray(Z) ** 2)
    f_analytic = np.asarray(alcubierre_shape(jnp.asarray(r_s), 1.0, 8.0))
    np.testing.assert_allclose(f, f_analytic, atol=1e-5)
    # Spans the f = 0.5 wall so the contour is actually extractable.
    assert f.min() < 0.5 < f.max()


@pytest.mark.smoke
def test_oneside_neg_clim_for_nonpositive_field() -> None:
    """``_oneside_neg_clim`` returns ``(vmin, 0)`` for a strictly-non-positive field."""
    from warpax.visualization.common._conversion import _oneside_neg_clim

    arr = np.array([-3.0, -1.0, 0.0, -0.5, np.nan])
    vmin, vmax = _oneside_neg_clim(arr)
    assert vmax == 0.0
    assert vmin == -3.0


@pytest.mark.smoke
def test_eulerian_wec_fields_bounded_and_cap_free() -> None:
    """Bounded WEC fields are well-posed; zeta_th is degenerate for Alcubierre.

    Replaces the misleading rapidity-capped ``min_u T_ab u^a u^b`` (which diverges
    to -inf with rapidity) with the invariant Type-I rest-frame WEC margin and the
    closed-form threshold rapidity. For Alcubierre the Type-I margin is defined
    only on the (sparse) Type-I points and zeta_th is degenerate.
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import eulerian_wec_fields

    grid_spec = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(20, 20, 20))
    res = evaluate_curvature_grid(AlcubierreMetric(v_s=0.5), grid_spec, compute_invariants=True)
    wf = eulerian_wec_fields(res.stress_energy, res.metric, res.metric_inv)

    wec = wf["wec_margin_eulerian"]
    finite = wec[np.isfinite(wec)]
    # Where the Type-I rest-frame margin is defined, it is <= 0 (within fp slack).
    assert finite.size and np.all(finite <= 1e-6)

    zt = wf["zeta_th"]
    # For Alcubierre the threshold rapidity is degenerate: 0 (rest frame already
    # violates) or +inf (WEC holds for all boosts) -- never finite positive.
    finite_pos = np.isfinite(zt) & (zt > 0)
    assert not finite_pos.any()
    assert wf["boost_dir"].shape == (*grid_spec.shape, 3)


@pytest.mark.smoke
def test_kretschmann_invariant_is_signed_and_finite() -> None:
    """K = R_abcd R^abcd is finite, structured, and sign-indefinite (Lorentzian).

    Regression guard for the Kretschmann scene: the invariant is NOT one-sided --
    for the Alcubierre wall it dips strongly negative -- so the scene must use a
    diverging scale, not a >= 0 ramp.
    """
    pytest.importorskip("manim")
    importlib.import_module("warpax.visualization.manim._kretschmann")  # imports cleanly

    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.visualization.common._physics import build_frame_sequence

    grid_spec = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(24, 24, 24))
    frames = build_frame_sequence(
        AlcubierreMetric(v_s=0.5),
        grid_spec,
        v_s_values=[0.5],
        compute_invariants=True,
        progress=False,
    )
    k = np.asarray(frames[0].scalar_fields["kretschmann"])
    assert np.isfinite(k).all()
    # Sign-indefinite in Lorentzian signature: both signs present on the wall.
    assert float(np.max(k)) > 0.0 and float(np.min(k)) < 0.0
