"""Wall-resolution witness: cells across the 10-90% wall, worst case.

One definition, used everywhere. A wall is a closed surface, so a coordinate
axis crosses it more than once (twice for a bubble, four times for a shell), and
the resolution criterion is only met if *every* crossing is resolved. This module
therefore reports the **minimum** over crossings, measured on the grid actually
used.

Three things it deliberately does not do, each of which inflates the number:

- it does not divide by the *smallest* node spacing in the band (that is the
  best case, at the one crossing the grid happens to favour);
- it does not sum the ``+x`` and ``-x`` crossings into a single count (that
  doubles the figure for a bubble);
- it does not use the asymptotic ``2 atanh(0.8)/sigma`` width, which is not
  valid at ``sigma*R ~ 1``, the band edges are located by interpolation on the
  actual shape function.

It also does not renormalise the shape function by its peak-to-peak range: that
is meaningless for a shell profile, whose scalar is non-monotone and whose
disconnected transitions the normalisation would merge.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["WallResolution", "wall_cells_on_axis"]

_DENSE = 20001


class WallResolution(NamedTuple):
    """Resolution witness for one axis.

    Attributes
    ----------
    cells : worst-case cells across the 10-90% band, over all crossings.
    spacing : the node spacing that produced ``cells`` (the worst one).
    width : the 10-90% width of the crossing that produced ``cells``.
    n_crossings : how many wall crossings the axis makes.
    per_crossing : ``(centre, width, spacing, cells)`` for every crossing.
    """

    cells: float
    spacing: float
    width: float
    n_crossings: int
    per_crossing: tuple[tuple[float, float, float, float], ...]


def _shape_along_axis(metric, xs: np.ndarray, t: float = 0.0) -> np.ndarray:
    coords = jnp.stack(
        [jnp.full(xs.shape, t), jnp.asarray(xs), jnp.zeros(xs.shape), jnp.zeros(xs.shape)],
        axis=1,
    )
    return np.asarray(jax.vmap(metric.shape_function_value)(coords), dtype=float)


def _segments(x: np.ndarray, f: np.ndarray, lo: float, hi: float):
    """Contiguous ``lo <= f <= hi`` runs on a dense sampling, with exact edges."""
    inside = (f >= lo) & (f <= hi)
    if not inside.any():
        return []
    n = inside.size
    runs, i = [], 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1

    out = []
    for i, j in runs:
        # Refine each end by linear interpolation onto the crossed level.
        def _edge(k: int, step: int) -> float:
            m = k + step
            if m < 0 or m >= n:
                return float(x[k])
            level = hi if abs(f[k] - hi) < abs(f[k] - lo) else lo
            df = f[m] - f[k]
            if df == 0.0:
                return float(x[k])
            s = (level - f[k]) / df
            s = min(max(s, 0.0), 1.0)
            return float(x[k] + s * (x[m] - x[k]))

        a, b = _edge(i, -1), _edge(j, +1)
        if b < a:
            a, b = b, a
        if b > a:
            out.append((a, b))
    return out


def wall_cells_on_axis(
    metric, xs, *, f_low: float = 0.1, f_high: float = 0.9, t: float = 0.0
) -> WallResolution:
    """Worst-case cells across the 10-90% wall on the axis sampled by ``xs``.

    Parameters
    ----------
    metric : exposes ``shape_function_value(coords)``.
    xs : the *actual* 1-D node positions of the axis under test. Pass the grid
        that will be used; reconstructing the coordinate map by hand lets the
        witness drift out of sync with the grid generator.
    """
    xs = np.asarray(xs, dtype=float)
    dense = np.linspace(float(xs[0]), float(xs[-1]), _DENSE)
    f_dense = _shape_along_axis(metric, dense, t=t)

    per = []
    for a, b in _segments(dense, f_dense, f_low, f_high):
        width = b - a
        # Worst node spacing over exactly the intervals that overlap the band.
        # Selecting by overlap (rather than by searchsorted offsets) keeps the
        # witness mirror-symmetric, so the two crossings of a sphere report the
        # same number when the grid is in fact symmetric.
        overlaps = (xs[:-1] < b) & (xs[1:] > a)
        if not overlaps.any():
            continue
        spacing = float(np.max(np.diff(xs)[overlaps]))
        per.append((0.5 * (a + b), width, spacing, width / spacing if spacing else 0.0))

    if not per:
        return WallResolution(0.0, float("nan"), float("nan"), 0, ())

    worst = min(per, key=lambda p: p[3])
    return WallResolution(
        cells=float(worst[3]),
        spacing=float(worst[2]),
        width=float(worst[1]),
        n_crossings=len(per),
        per_crossing=tuple(per),
    )
