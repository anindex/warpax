"""Non-uniform grid generators for warpax.

Provides:

- :func:`axisymmetric_grid` - exact (r, mu) reduction of an axisymmetric slice.
- :func:`wall_clustered` - anchored sinh-stretched grid around the wall radius.
- :func:`wall_refined` - 2-level AMR patch (coarse base + fine wall patch).
- :func:`wall_cells_on_axis` - the single wall-resolution witness (worst case
  over every wall crossing, measured on the grid actually used).

All returned grids are JIT-safe (static) and, for ``wall_clustered``,
carry ``volume_weights`` for non-uniform stats.
"""
from __future__ import annotations

from warpax.geometry import GridSpec

from ._axisymmetric import AxisymmetricGrid, axisymmetric_grid
from ._clustered import wall_clustered
from ._refined import RefinedGrid, wall_refined
from ._resolution import WallResolution, wall_cells_on_axis

__all__ = [
    "AxisymmetricGrid",
    "GridSpec",
    "RefinedGrid",
    "WallResolution",
    "axisymmetric_grid",
    "wall_cells_on_axis",
    "wall_clustered",
    "wall_refined",
]
