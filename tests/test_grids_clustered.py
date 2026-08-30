"""The wall-clustered grid must actually refine the wall, on both sides.

These pin the two defects fixed in the second revision:

1. the coordinate map was not *anchored*, it clustered, but the densest
   sampling drifted off the wall (x = 1.266 for a wall at r = 1);
2. it clustered only toward ``+wall_radius``, so the ``-x`` crossing of the same
   spherical wall was measurably coarser and fell below the four-cell criterion
   at the coarsest ladder level.

Building one graded axis costs tens of seconds, so the ladder is built once and
every property is asserted over it.
"""

from __future__ import annotations

import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.grids import wall_cells_on_axis, wall_clustered

# One graded axis per ladder level costs tens of seconds, so the module fixture is
# built once. Keep the three cases on one xdist worker or each rebuilds it.
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
    """Exact endpoints, strictly increasing, densest node pair straddling |x| = R."""
    _, axes, _ = ladder
    for n, xs in axes.items():
        assert xs[0] == pytest.approx(BOUNDS[0][0], abs=1e-12), n
        assert xs[-1] == pytest.approx(BOUNDS[0][1], abs=1e-12), n
        assert np.all(np.diff(xs) > 0), f"N={n} coordinate map is not strictly increasing"

        mid = 0.5 * (xs[:-1] + xs[1:])
        densest = abs(mid[np.argmin(np.diff(xs))])
        # One grid interval of tolerance; the pre-fix map missed by 27% of R.
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
    """Golden ladder values for the corrected map, so a regression is loud.

    Pre-fix this ladder read 4.45 / 5.56 / 6.69 on the +x crossing and only
    3.27 / 4.04 / 4.87 on -x.
    """
    metric, axes, cells = ladder
    got = [cells[n].cells for n in LADDER]
    assert got == sorted(got), f"refinement is not monotone: {got}"
    assert got == pytest.approx([5.905, 7.578, 8.874], rel=1e-3), got

    n = LADDER[0]
    uniform = wall_cells_on_axis(metric, np.linspace(*BOUNDS[0], n)).cells
    # Measured gain is ~1.63x at N=80; assert comfortably inside that.
    assert got[0] > 1.4 * uniform, f"graded {got[0]:.2f} vs uniform {uniform:.2f}"
