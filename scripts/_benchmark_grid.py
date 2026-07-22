"""Canonical wall-resolved benchmark grid (single source of truth).

The matched cross-metric benchmark holds the physical parameters common
(R_b = 1, sigma = 8) and evaluates every retained metric on ONE graded-grid
family so that "common parameters, common grid" is true by construction rather
than by coincidence across scattered literals. Because the tensor is
autodiff-exact pointwise, the wall-cell count governs only the sampling of the
discrete summaries, not the accuracy of the field.

Grid family (sinh cosh-stretch, densest at the wall; see grids/_clustered.py):
  - box   : +-3 R_b per axis
  - a*    : 2.0  (clustering strength)
  - ladder: N = [80, 100, 120]  -> 4.5 / 5.6 / 6.7 cells across the 10-90% wall on a
    radial crossing (width / Delta x_wall; Delta x_wall = 0.061 / 0.049 / 0.041),
    up to a fivefold gain over the 1.35 cells of a uniform grid, every level
    clearing the four-cell criterion (WALL_CELL_FLOOR).

Extrema (min NEC/DEC margin, max|Im lambda|) are polished to the continuum with
warpax.analysis.extrema.refine_extremum (seeded from the deepest ladder sample),
so they are resolution-independent; only the (non-smooth) volume fractions carry a
grid stability spread across the ladder.
"""
from __future__ import annotations

import math

import numpy as np

import jax
import jax.numpy as jnp

from warpax.grids import wall_clustered
from warpax.grids._clustered import _cosh_stretch, _infer_wall_radius

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
    """(effective cells across the 10-90% wall on a radial crossing, min wall spacing).

    Matches the uniform-grid definition of the wall-resolution table: the 10-90%
    transition width divided by the local node spacing at the wall. Counted on a
    SINGLE radial crossing (the +x wall), using the [0.1, 0.9] band, so it is
    directly comparable to the uniform-grid ``width / Delta x`` figure and is not
    inflated by summing both axial wall crossings or by a wider band.
    """
    lo, hi = BOUNDS[0]
    wall_r = _infer_wall_radius(metric, tuple(BOUNDS))
    u_wall = float(jnp.clip((wall_r - lo) / (hi - lo), 0.05, 0.95))
    u = jnp.linspace(0.0, 1.0, N)
    xs = np.asarray(lo + (hi - lo) * np.asarray(_cosh_stretch(u, u_wall, CLUSTER_A)))
    coords = jnp.stack([jnp.zeros(N), jnp.asarray(xs), jnp.zeros(N), jnp.zeros(N)], axis=1)
    f = np.asarray(jax.vmap(metric.shape_function_value)(coords))
    # +x radial crossing only
    pos = xs > 1e-6
    xp, fp = xs[pos], f[pos]
    fn = (fp - fp.min()) / (np.ptp(fp) + 1e-30)  # ~1 inside -> ~0 outside across +x wall
    band = (fn > 0.1) & (fn < 0.9)
    bx = np.sort(xp[band])
    dxw = float(np.min(np.diff(bx))) if bx.size >= 2 else float("nan")
    # analytic 10-90% width for a tanh wall (matches the uniform-grid table); fall
    # back to the empirical band span if the metric exposes no sigma.
    sigma = float(getattr(metric, "sigma", 0.0)) or None
    width = (2.0 * math.atanh(0.8) / sigma) if sigma else (
        float(bx.max() - bx.min()) if bx.size >= 2 else float("nan"))
    cells = width / dxw if (dxw and np.isfinite(dxw)) else float("nan")
    return float(cells), dxw
