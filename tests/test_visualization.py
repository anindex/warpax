"""Visualization smoke tests: Manim scenes and frame-data helpers."""

from __future__ import annotations

import importlib
import json
import math

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from warpax.geometry.types import GridSpec


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
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["Eulerian", "Robust", "Missed (0 pts)"]
    assert len(fig.axes) == 5


@pytest.mark.parametrize("values", [[1.0, 1.0, 1.0], [1.0, 1.4, 1.1]])
def test_descriptive_convergence_plots(tmp_path, values):
    from warpax.visualization.convergence_plots import plot_convergence, plot_convergence_table

    mean = math.fsum(values) / len(values)
    path = tmp_path / "grid_samples.json"
    path.write_text(
        json.dumps(
            {
                "summary_method": "observed_grid_spread",
                "metric": "fixture",
                "resolutions": [25, 49, 97],
                "min_margin_nec": {
                    "values": values,
                    "mean": mean,
                    "max_abs_deviation_from_mean": max(abs(value - mean) for value in values),
                },
            }
        )
    )
    fig = plot_convergence(str(path))
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [25, 49, 97])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), values)
    assert len(ax.lines) == 1
    assert ax.get_yscale() == "linear"

    table_fig = plot_convergence_table(str(path))
    labels = [
        cell.get_text().get_text() for cell in table_fig.axes[0].tables[0].get_celld().values()
    ]
    assert "Max departure from mean" in labels
    assert "Extrapolated" not in labels


def test_tidal_plot_marks_peak_without_identifying_a_wall():
    from warpax.visualization.geodesic_plots import plot_tidal_evolution

    tau = np.array([0.0, 1.0, 2.0])
    eig = np.array([[1.0, -1.0, 0.0], [3.0, -2.0, 0.0], [1.0, 0.0, 0.0]])
    ax = plot_tidal_evolution(eig, tau).axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), tau)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), eig[:, 0])
    assert ax.lines[3].get_label() == "Peak tidal magnitude"
    assert ax.lines[3].get_xdata() == [1.0, 1.0]


def test_density_normalization_and_wec_threshold():
    """A non-unit lapse rescales T_00 and a non-spacelike slice has no Eulerian
    normal; negative rest density with NEC satisfied, positive density with NEC
    violated, and ordinary matter require thresholds zero, asinh(1), and
    infinity."""
    from warpax.visualization.common._conversion import (
        eulerian_energy_density_grid,
        eulerian_wec_fields,
    )

    g_inv = jnp.stack([jnp.diag(jnp.array([-0.25, 1.0, 1.0, 1.0])), jnp.eye(4)])
    density = eulerian_energy_density_grid(jnp.broadcast_to(jnp.eye(4), (2, 4, 4)), g_inv)
    assert density[0] == pytest.approx(0.25)
    assert np.isnan(density[1])

    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    tensors = jnp.stack(
        [
            jnp.diag(jnp.array(row))
            for row in [(-1.0, 2.0, 3.0, 4.0), (1.0, -2.0, 3.0, 4.0), (1.0, 2.0, 3.0, 4.0)]
        ]
    )
    fields = eulerian_wec_fields(tensors, jnp.broadcast_to(eta, (3, 4, 4)))
    np.testing.assert_allclose(fields["zeta_th"], [0.0, np.arcsinh(1.0), np.inf])


def test_direction_fields_smoke():
    from warpax.visualization.direction_fields import plot_worst_observer_field

    grid = GridSpec(bounds=[(-1.0, 1.0)] * 3, shape=(4, 4, 4))
    from matplotlib.quiver import Quiver

    params = np.zeros((*grid.shape, 3))
    params[..., 0] = 1.0
    params[..., 1] = np.pi / 2
    ax = plot_worst_observer_field(params, grid, slice_index=2)
    quivers = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(quivers) == 1
    assert quivers[0].U.shape == quivers[0].V.shape


@pytest.mark.parametrize(
    "vmin, vmax", [(-1.0, 0.0), (-1.0, -0.5), (0.2, 1.0), (0.0, 1.0), (-1.0, 1.0)]
)
def test_color_stops_are_monotone(vmin, vmax):
    pytest.importorskip("manim")
    from warpax.visualization.manim._image_utils import NEC_DEPTH_COLORS, color_stops

    positions = [pos for _, pos in color_stops(NEC_DEPTH_COLORS, vmin, vmax)]
    assert positions[0] == vmin
    assert positions[-1] == vmax
    assert np.all(np.diff(positions) > 0)


def test_save_or_return_saves_closes_and_returns(tmp_path):
    from warpax.visualization._style import save_or_return
    from warpax.visualization.shift_vorticity_plots import plot_shift_vorticity

    fig, _ = plt.subplots()
    assert save_or_return(fig, None) is fig
    assert fig.number in plt.get_fignums()
    path = tmp_path / "fig.pdf"
    assert save_or_return(fig, str(path)) is fig
    assert path.stat().st_size > 0
    assert fig.number not in plt.get_fignums()

    before = plt.get_fignums()
    fingerprint = {"Rodal": {"expansion": 0.5, "shear": 0.5, "vorticity": 0.0}}
    out = plot_shift_vorticity(
        fingerprint, {"Rodal": [(0.5, 0.0, 0.0)]}, ["Rodal"], str(tmp_path / "sv.pdf")
    )
    assert plt.get_fignums() == before
    assert len(out.axes) == 2


def test_symlog_height_keeps_ordering_and_opens_the_dynamic_range() -> None:
    """The embedding height map must reorder nothing and flatten nothing.

    A parameter sweep whose field grows by 10^3 is the case this exists for: a
    linear height leaves the weakest frame at 0.1% of the strongest, which
    renders as a flat plane. A flat field must not divide by zero
    (auto_linthresh(0.0) == 1.0); the map is strictly monotone and sign
    preserving, so no value reorders and the NEC verdict survives; and 10^3 of
    field range must not compress to 10^-3 of height.
    """
    pytest.importorskip("manim")

    from warpax.visualization.manim._surface import auto_linthresh, symlog_height

    lt = auto_linthresh(0.1474)
    assert lt == pytest.approx(1.474e-4)
    assert auto_linthresh(0.0) == 1.0

    v = np.linspace(-0.15, 0.15, 201)
    h = symlog_height(v, lt)
    assert np.all(np.diff(h) > 0)
    assert np.all(np.sign(h) == np.sign(v))
    assert h[np.argmin(np.abs(v))] == pytest.approx(0.0, abs=1e-12)

    weak, strong = 9.3e-4, 1.19e-1
    linear_ratio = weak / strong
    symlog_ratio = symlog_height(np.array([weak]), lt)[0] / symlog_height(np.array([strong]), lt)[0]
    assert linear_ratio < 0.01
    assert symlog_ratio > 0.25


def test_headline_sweep_stays_centered_and_reaches_flat_spacetime():
    pytest.importorskip("manim")
    from warpax.analysis.kinematic_scalars import compute_kinematic_scalars_grid
    from warpax.benchmarks import AlcubierreMetric
    from warpax.benchmarks.alcubierre import eulerian_energy_density
    from warpax.visualization.manim._wall_velocity_sweep import WallAndVelocitySweep

    grid = GridSpec(bounds=[(-2, 2), (-2, 2), (0, 0)], shape=(33, 33, 1))
    frames = WallAndVelocitySweep._build_rampdown_frames(grid, n_frames=3)
    assert [frame.v_s for frame in frames] == [0.5, 0.25, 0.0]
    for frame in frames:
        assert frame.t == 0.0
        rho = frame.scalar_fields["energy_density"]
        theta = frame.scalar_fields["theta"]
        expected = eulerian_energy_density(frame.x, frame.y, frame.z, v_s=frame.v_s, sigma=16)
        np.testing.assert_allclose(rho, expected, atol=1e-12)
        np.testing.assert_allclose(rho, rho[::-1], atol=1e-12)
        np.testing.assert_allclose(theta, -theta[::-1], atol=1e-12)
        if frame.v_s > 0:
            assert theta[8, 16, 0] > 0
            assert theta[24, 16, 0] < 0
    expected_theta, _, _ = compute_kinematic_scalars_grid(AlcubierreMetric(v_s=0.5, sigma=16), grid)
    np.testing.assert_allclose(frames[0].scalar_fields["theta"], expected_theta, atol=1e-12)
    for name, values in frames[-1].scalar_fields.items():
        if name != "shape_function":
            np.testing.assert_array_equal(values, np.zeros_like(values))


def test_headline_grid_resolves_the_thinnest_energy_wall():
    pytest.importorskip("manim")
    from warpax.benchmarks.alcubierre import eulerian_energy_density
    from warpax.visualization.manim._image_utils import bilinear_sampler
    from warpax.visualization.manim._wall_velocity_sweep import WallAndVelocitySweep

    axis = np.linspace(-2, 2, WallAndVelocitySweep.GRID_N)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    rho = np.asarray(eulerian_energy_density(x, y, np.zeros_like(x), sigma=16))
    sample = bilinear_sampler(axis, axis, rho)
    angles = np.linspace(0, 2 * np.pi, 401)
    exact = np.asarray(
        eulerian_energy_density(np.cos(angles), np.sin(angles), np.zeros_like(angles), sigma=16)
    )
    interpolated = np.array([sample(np.cos(a), np.sin(a)) for a in angles])
    assert np.max(np.abs(interpolated - exact)) / np.max(np.abs(exact)) < 0.025


def test_dark_diverge_colormap_registers() -> None:
    """The shared ``dark_diverge`` colormap registers once and reaches its warm end."""
    pytest.importorskip("manim")

    import matplotlib as mpl

    importlib.import_module("warpax.visualization.manim._image_utils")
    cmap = mpl.colormaps["dark_diverge"]
    rgba_hot = cmap(1.0)
    assert rgba_hot[3] > 0.0
    assert max(rgba_hot[:3]) > 0.5
    assert "dark_diverge_theta" not in mpl.colormaps


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

    mask = np.abs(rho_analytic) > 1e-6
    if not mask.any():
        pytest.skip("Wall region empty in the chosen grid")

    rel_err = np.abs(rho_numeric[mask] - rho_analytic[mask]) / np.abs(rho_analytic[mask])
    assert rel_err.max() < 1e-2, (
        f"max rel err {rel_err.max():.3e} between Eulerian density "
        "from T_{ab} n^a n^b and Alcubierre analytical form"
    )


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
    for name in ("kretschmann", "ricci_squared", "weyl_squared"):
        assert frame.colormaps[name] == "RdBu_r"
        lo, hi = frame.clim[name]
        assert lo == -hi


def test_observer_robust_nec_nonpositive_dense_sampling() -> None:
    """Dense null sampling makes the observer-robust NEC <= 0 everywhere.

    The Alcubierre NEC is violated for v_s > 0 and the worst-case over the
    null sphere is <= 0 at every point. Sampling only the few axis-aligned
    null rays of the rapidity observers leaves spurious positive "satisfied"
    pixels; the dense default must not.
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
    assert nec_coarse.max() > nec_dense.max() + 1e-3


def test_nec_comparison_adds_directions_in_the_same_frame():
    pytest.importorskip("manim")
    from warpax.visualization.manim._split_screen import (
        _build_eulerian_frames,
        _build_robust_frames,
    )

    kwargs = {
        "v_s_values": [0.5],
        "grid_shape": (33, 33, 1),
        "bounds": [(-2, 2), (-2, 2), (0, 0)],
    }
    axes = _build_eulerian_frames(**kwargs)[0]
    sampled = _build_robust_frames(**kwargs)[0]
    np.testing.assert_allclose(
        axes.scalar_fields["energy_density"], sampled.scalar_fields["energy_density"], atol=1e-12
    )
    six = axes.scalar_fields["nec_margin"]
    dense = sampled.scalar_fields["nec_margin_sweep"]
    assert np.all(dense <= six + 1e-12)
    assert np.max(six - dense) > 1e-3


def test_ec_frames_expose_real_shape_function() -> None:
    """EC frames carry the true f(r_s) field, which matches the analytic form;
    a circular fallback (radius x_extent/4 = 1.5) would not."""
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
    np.testing.assert_allclose(f, f_analytic, atol=1e-5)
    assert f.max() > 0.99 and f.min() < 0.01


def test_image_contour_orientation_x_horizontal() -> None:
    """Contours place physical x on the horizontal axis, y on the vertical.

    Regression guard for B4 (image/contour were transposed: motion axis x
    rendered vertical and inverted). Uses a non-square grid to expose swaps.
    """
    from warpax.visualization.manim._image_utils import extract_zero_contour

    nx, ny = 9, 15
    x = np.linspace(-2, 2, nx)
    data_x = np.repeat(x[:, None], ny, axis=1)
    vx = np.concatenate(extract_zero_contour(data_x, (-2, 2), (-3, 3), level=0.0), axis=0)
    assert np.allclose(vx[:, 0], 0.0, atol=0.1)
    assert vx[:, 1].max() > 1.5 and vx[:, 1].min() < -1.5

    y = np.linspace(-3, 3, ny)
    data_y = np.repeat(y[None, :], nx, axis=0)
    vy = np.concatenate(extract_zero_contour(data_y, (-2, 2), (-3, 3), level=0.0), axis=0)
    assert np.allclose(vy[:, 1], 0.0, atol=0.1)
    assert vy[:, 0].max() > 1.0 and vy[:, 0].min() < -1.0


def test_extract_contours_multi_level() -> None:
    """``extract_contours`` returns one path-group per level, nested and NaN-safe.

    On a radial bowl -(X^2 + Y^2) the level fractions are kept < 0.5 so every
    level is a full ring inside [-3, 3]; f decreases outward, so the deepest
    level is the outer ring; a stray NaN must not crash or erase the contour
    (order=1 bilinear fill); the single-level wrapper still yields a flat list
    of (N, 2) segments.

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
    data = -(X**2 + Y**2)
    gmin = float(data.min())
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

    assert _max_radius(res[0][1]) > _max_radius(res[-1][1])

    d2 = data.copy()
    d2[0, 0] = np.nan
    res2 = extract_contours(d2, (-3, 3), (-3, 3), [gmin * 0.2])
    assert res2 and res2[0][1]

    zc = extract_zero_contour(data, (-3, 3), (-3, 3), level=gmin * 0.2)
    assert isinstance(zc, list) and zc and zc[0].shape[1] == 2


def test_eulerian_kinematics_shape_function_is_real() -> None:
    """EulerianKinematics2D's bubble wall uses the real f(r_s), not a circle.

    The field spans the f = 0.5 wall so the contour is actually extractable.

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
    assert f.min() < 0.5 < f.max()


def test_oneside_neg_clim_for_nonpositive_field() -> None:
    """``_oneside_neg_clim`` returns ``(vmin, 0)`` for a strictly-non-positive field."""
    from warpax.visualization.common._conversion import _oneside_neg_clim

    arr = np.array([-3.0, -1.0, 0.0, -0.5, np.nan])
    vmin, vmax = _oneside_neg_clim(arr)
    assert vmax == 0.0
    assert vmin == -3.0


def test_eulerian_wec_fields_bounded_and_cap_free() -> None:
    """Bounded WEC fields are well-posed; zeta_th is degenerate for Alcubierre.

    Where the Type-I rest-frame margin is defined it is <= 0 (within fp slack).
    For Alcubierre the threshold rapidity is 0 (rest frame already violates)
    or +inf (WEC holds for all boosts), never finite positive.

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
    assert finite.size and np.all(finite <= 1e-6)

    zt = wf["zeta_th"]
    finite_pos = np.isfinite(zt) & (zt > 0)
    assert not finite_pos.any()
    assert wf["boost_dir"].shape == (*grid_spec.shape, 3)


def test_kretschmann_invariant_is_signed_and_finite() -> None:
    """K = R_abcd R^abcd is finite, structured, and sign-indefinite (Lorentzian).

    Both signs must be present on the wall.

    Regression guard for the Kretschmann scene: the invariant is NOT one-sided,
    for the Alcubierre wall it dips strongly negative, so the scene must use a
    diverging scale, not a >= 0 ramp.
    """
    pytest.importorskip("manim")
    importlib.import_module("warpax.visualization.manim._kretschmann")

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
    assert float(np.max(k)) > 0.0 and float(np.min(k)) < 0.0


def test_slice_axes_and_style_table_helpers():
    from warpax.visualization._style import slice_axes, style_table

    x_ax, y_ax, labels = slice_axes((4, 5, 6), ((-1.0, 1.0), (0.0, 2.0), (3.0, 4.0)), 1)
    assert labels == ["x", "z"]
    assert np.array_equal(x_ax, np.linspace(-1.0, 1.0, 4))
    assert np.array_equal(y_ax, np.linspace(3.0, 4.0, 6))

    fig, ax = plt.subplots()
    table = ax.table(cellText=[["a", "b"], ["c", "d"]], colLabels=["p", "q"], loc="center")
    style_table(table, 2, 7, [["white", "#FEE08B"], ["white", "white"]])
    assert table[0, 0].get_facecolor()[:3] == matplotlib.colors.to_rgb("#4472C4")
    assert table[1, 1].get_facecolor()[:3] == matplotlib.colors.to_rgb("#FEE08B")
    assert table[2, 0].get_facecolor()[:3] == matplotlib.colors.to_rgb("white")
    plt.close(fig)
