"""Canonical wall-resolved benchmark grid (single source of truth).

The matched cross-metric benchmark holds the physical parameters common
(R_b = 1, sigma = 8) and evaluates every retained metric on ONE graded-grid
family so that "common parameters, common grid" is true by construction rather
than by coincidence across scattered literals. Because the tensor is
autodiff-exact pointwise, the wall-cell count governs only the sampling of the
discrete summaries, not the accuracy of the field.

Grid family (anchored two-sided sinh stretch, densest exactly at the wall and
symmetric about the centre; see grids/_clustered.py):
  - box   : +-3 R_b per axis
  - a*    : 2.0  (clustering strength)
  - ladder: N = [80, 100, 120]  -> 5.9 / 7.6 / 8.9 cells across the 10-90% wall,
    worst case over both axial crossings (grids/_resolution.py), a 1.6x gain over
    a uniform grid of the same N (3.6 / 4.5 / 5.4), every level clearing the
    four-cell criterion (WALL_CELL_FLOOR).

Extrema (min NEC/DEC margin, max|Im lambda|) are polished to the continuum with
warpax.analysis.extrema.refine_extremum (seeded from the deepest ladder sample),
so they are resolution-independent; only the (non-smooth) volume fractions carry a
grid stability spread across the ladder.
"""

from __future__ import annotations

import numpy as np

from warpax.grids import wall_cells_on_axis, wall_clustered

BOX = 3.0
BOUNDS = [(-BOX, BOX)] * 3
CLUSTER_A = 2.0
N_DEFAULT = 100
N_LADDER = [80, 100, 120]
WALL_CELL_FLOOR = 4


def benchmark_grid(metric, N: int = N_DEFAULT):
    """The canonical wall-resolved grid for ``metric`` at resolution ``N``."""
    return wall_clustered(metric, BOUNDS, (N, N, N), a=CLUSTER_A)


def wall_cells(metric, N: int) -> tuple[float, float]:
    """(worst-case cells across the 10-90% wall, the spacing that produced it).

    Delegates to the single shared witness, :func:`warpax.grids.wall_cells_on_axis`,
    measured on the grid this module actually builds. Worst case over every wall
    crossing the axis makes, see that module for why the previous best-case,
    single-crossing, asymptotic-width variant read high.
    """
    xs = np.asarray(benchmark_grid(metric, N).axes[0], dtype=float)
    res = wall_cells_on_axis(metric, xs)
    return float(res.cells), float(res.spacing)
