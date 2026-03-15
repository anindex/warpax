"""Non-uniform grid generators for warpax.

Subpackage landed in (, ). Provides:

- :func:`wall_clustered` - cosh-stretched radial grid around the wall radius.
- :func:`wall_refined` - 2-level AMR patch (coarse base + fine wall patch).

All returned grids are JIT-safe (static per ) and, for
``wall_clustered``, carry ``volume_weights`` for non-uniform stats
.
"""
from __future__ import annotations

from warpax.geometry import GridSpec

from ._clustered import wall_clustered
from ._refined import RefinedGrid, wall_refined

__all__ = [
    "GridSpec",
    "RefinedGrid",
    "wall_clustered",
    "wall_refined",
]
