"""The wall-clustered grid must actually refine the wall, on both sides.

These pin the two defects fixed in the second revision:

1. the coordinate map was not *anchored* -- it clustered, but the densest
   sampling drifted off the wall (x = 1.266 for a wall at r = 1);
2. it clustered only toward ``+wall_radius``, so the ``-x`` crossing of the same
   spherical wall was measurably coarser and fell below the four-cell criterion
   at the coarsest ladder level.
"""
from __future__ import annotations

import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.grids import wall_cells_on_axis, wall_clustered

BOUNDS = ((-3.0, 3.0),) * 3
LADDER = (80, 100, 120)
WALL_R = 1.0
CLUSTER_A = 2.0


@pytest.fixture(scope="module")
def metric():
    return AlcubierreMetric(v_s=0.5, R=WALL_R, sigma=8.0)


def _axis(metric, n):
    return np.asarray(
        wall_clustered(metric, BOUNDS, (n, n, n), a=CLUSTER_A).axes[0], dtype=float
    )


@pytest.mark.parametrize("n", LADDER)
def test_endpoints_are_exact(metric, n):
    xs = _axis(metric, n)
    assert xs[0] == pytest.approx(BOUNDS[0][0], abs=1e-12)
    assert xs[-1] == pytest.approx(BOUNDS[0][1], abs=1e-12)
    assert np.all(np.diff(xs) > 0), "coordinate map must be strictly increasing"


@pytest.mark.parametrize("n", LADDER)
def test_densest_sampling_sits_on_the_wall(metric, n):
    """The minimum-spacing node pair must straddle |x| = R, not drift off it."""
    xs = _axis(metric, n)
    mid = 0.5 * (xs[:-1] + xs[1:])
    densest = abs(mid[np.argmin(np.diff(xs))])
    # One grid interval of tolerance; the pre-fix map missed by 27% of R.
    assert densest == pytest.approx(WALL_R, abs=max(np.min(np.diff(xs)), 0.05)), (
        f"densest sampling at |x|={densest:.4f}, wall at {WALL_R}"
    )


@pytest.mark.parametrize("n", LADDER)
def test_both_wall_crossings_equally_resolved(metric, n):
    """A sphere is crossed twice per axis; the two must not differ."""
    res = wall_cells_on_axis(metric, _axis(metric, n))
    assert res.n_crossings == 2, f"expected 2 axial crossings, got {res.n_crossings}"
    cells = [c[3] for c in res.per_crossing]
    assert cells[0] == pytest.approx(cells[1], rel=1e-6), (
        f"asymmetric wall resolution: {cells}"
    )


@pytest.mark.parametrize("n", LADDER)
def test_ladder_clears_four_cells_worst_case(metric, n):
    """WALL_CELL_FLOOR must hold at the *worst* crossing, not the best."""
    res = wall_cells_on_axis(metric, _axis(metric, n))
    assert res.cells >= 4.0, f"N={n} resolves the wall with only {res.cells:.2f} cells"


def test_resolution_increases_monotonically_with_n(metric):
    cells = [wall_cells_on_axis(metric, _axis(metric, n)).cells for n in LADDER]
    assert cells == sorted(cells), f"refinement is not monotone: {cells}"


def test_clustering_beats_a_uniform_grid(metric):
    """The whole point: more cells on the wall than a uniform grid of equal N."""
    n = LADDER[0]
    graded = wall_cells_on_axis(metric, _axis(metric, n)).cells
    uniform = wall_cells_on_axis(metric, np.linspace(*BOUNDS[0], n)).cells
    # Measured gain is ~1.63x at N=80; assert comfortably inside that.
    assert graded > 1.4 * uniform, f"graded {graded:.2f} vs uniform {uniform:.2f}"


def test_ladder_cell_counts_are_pinned(metric):
    """Golden values for the corrected map, so a regression is loud.

    Pre-fix this ladder read 4.45 / 5.56 / 6.69 on the +x crossing and only
    3.27 / 4.04 / 4.87 on -x.
    """
    got = [wall_cells_on_axis(metric, _axis(metric, n)).cells for n in LADDER]
    assert got == pytest.approx([5.905, 7.578, 8.874], rel=1e-3), got
