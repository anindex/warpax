"""Non-uniform grid generators for warpax.

Provides:

- :func:`axisymmetric_grid` - exact (r, mu) reduction of an axisymmetric slice.
- :func:`wall_clustered` - anchored sinh-stretched grid around the wall radius.
- :func:`wall_cells_on_axis` - the single wall-resolution witness (worst case
  over every wall crossing, measured on the grid actually used).

All returned grids are JIT-safe (static) and, for ``wall_clustered``,
carry ``volume_weights`` for non-uniform stats.
"""
from __future__ import annotations

from warpax.geometry import GridSpec

from ._axisymmetric import AxisymmetricGrid, axisymmetric_grid
from ._clustered import wall_clustered
from ._resolution import WallResolution, wall_cells_on_axis
from ._volume_weights import proper_volume_weights

__all__ = [
    "AxisymmetricGrid",
    "GridSpec",
    "WallResolution",
    "axisymmetric_grid",
    "proper_volume_weights",
    "wall_cells_on_axis",
    "wall_clustered",
]
