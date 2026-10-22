"""The grid anchors its densest sampling at both axial wall crossings.

The shared ladder checks symmetric refinement and the four-cell criterion.
One graded axis per ladder level costs tens of seconds, so the module fixture
is built once; the xdist_group keeps the cases on one worker or each rebuilds
it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.grids import wall_cells_on_axis, wall_clustered

pytestmark = pytest.mark.xdist_group("wall_clustered")

BOUNDS = ((-3.0, 3.0),) * 3
LADDER = (80, 100, 120)
WALL_R = 1.0
CLUSTER_A = 2.0


@pytest.fixture(scope="module")
def ladder():
    """``(metric, {n: axis}, {n: wall_cells_on_axis(axis)})`` over the whole ladder."""
    metric = AlcubierreMetric(v_s=0.5, R=WALL_R, sigma=8.0)
    axes = {
        n: np.asarray(wall_clustered(metric, BOUNDS, (n, n, n), a=CLUSTER_A).axes[0], dtype=float)
        for n in LADDER
    }
    return metric, axes, {n: wall_cells_on_axis(metric, xs) for n, xs in axes.items()}


def test_axis_is_a_graded_map_anchored_on_the_wall(ladder):
    """Exact endpoints, strictly increasing, densest node pair straddling |x| = R
    within one grid interval."""
    _, axes, _ = ladder
    for n, xs in axes.items():
        assert xs[0] == pytest.approx(BOUNDS[0][0], abs=1e-12), n
        assert xs[-1] == pytest.approx(BOUNDS[0][1], abs=1e-12), n
        assert np.all(np.diff(xs) > 0), f"N={n} coordinate map is not strictly increasing"

        mid = 0.5 * (xs[:-1] + xs[1:])
        densest = abs(mid[np.argmin(np.diff(xs))])
        assert densest == pytest.approx(WALL_R, abs=max(np.min(np.diff(xs)), 0.05)), (
            f"N={n} densest sampling at |x|={densest:.4f}, wall at {WALL_R}"
        )


def test_both_wall_crossings_clear_four_cells(ladder):
    """A sphere is crossed twice per axis; the two must match and both must resolve."""
    _, _, cells = ladder
    for n, res in cells.items():
        assert res.n_crossings == 2, f"N={n} expected 2 axial crossings, got {res.n_crossings}"
        first, second = (c[3] for c in res.per_crossing)
        assert first == pytest.approx(second, rel=1e-6), f"N={n} asymmetric: {first}, {second}"
        assert res.cells >= 4.0, f"N={n} resolves the wall with only {res.cells:.2f} cells"


def test_refinement_is_monotone_and_beats_a_uniform_grid(ladder):
    """Wall resolution improves at each level and exceeds the uniform-grid value
    (measured gain ~1.63x at N=80; asserted > 1.4x)."""
    metric, axes, cells = ladder
    got = [cells[n].cells for n in LADDER]
    assert got == sorted(got), f"refinement is not monotone: {got}"
    assert got == pytest.approx([5.905, 7.578, 8.874], rel=1e-3), got

    n = LADDER[0]
    uniform = wall_cells_on_axis(metric, np.linspace(*BOUNDS[0], n)).cells
    assert got[0] > 1.4 * uniform, f"graded {got[0]:.2f} vs uniform {uniform:.2f}"


def test_volume_weights_come_from_the_stored_axes():
    """The spec stores only axes; cell volumes are recomputed from them on demand."""
    from warpax.grids._volume_weights import compute_volume_weights

    metric = AlcubierreMetric(v_s=0.5, R=WALL_R, sigma=8.0)
    spec = wall_clustered(metric, BOUNDS, (8, 6, 5), wall_radius=WALL_R, a=CLUSTER_A)
    assert not hasattr(spec, "volume_weights")
    assert all(isinstance(ax, tuple) for ax in spec.coord_arrays)
    assert all(type(v) is float for ax in spec.coord_arrays for v in ax)
    vw = spec.volume_weights_array
    assert vw.shape == (8, 6, 5)
    assert np.array_equal(np.asarray(vw), np.asarray(compute_volume_weights(*spec.axes)))


def test_axisymmetric_radial_weights_are_the_shared_trapezoid_widths():
    """The axisymmetric grid and the Cartesian volume weights share one cell-width rule."""
    from warpax.grids import axisymmetric_grid
    from warpax.grids._volume_weights import cell_widths

    grid = axisymmetric_grid(3.0, 12, 6, wall_radius=1.0)
    r = grid.r
    expected = np.empty_like(r)
    expected[1:-1] = 0.5 * (r[2:] - r[:-2])
    expected[0] = 0.5 * (r[1] - r[0])
    expected[-1] = 0.5 * (r[-1] - r[-2])
    assert np.array_equal(np.asarray(cell_widths(jnp.asarray(r))), expected)
    _, w_mu = np.polynomial.legendre.leggauss(6)
    weights = (2.0 * np.pi * np.outer(r**2 * expected, w_mu)).ravel()
    np.testing.assert_allclose(np.asarray(grid.weights), weights, rtol=1e-14, atol=0)
