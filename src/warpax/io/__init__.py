"""External-metric I/O readers for warpax.

Provides the shared :class:`InterpolatedADMMetric` base class and reader
functions for third-party numerical-metric formats.

- :func:`load_warpfactory` - WarpFactory ``.mat`` exports 
- :func:`load_einfield` - EinFields Flax/Orbax checkpoints 
- :func:`load_cactus_slice` - Cactus / Einstein Toolkit HDF5 slices 

All loaders return an :class:`InterpolatedADMMetric` instance so downstream
pipelines (curvature chain, EC verification) treat external data identically
to analytic metrics.
"""
from __future__ import annotations

from ._interpolated import InterpolatedADMMetric
from .cactus import load_cactus_slice
from .einfields import load_einfield
from .warpfactory import load_warpfactory

__all__ = [
    "InterpolatedADMMetric",
    "load_cactus_slice",
    "load_einfield",
    "load_warpfactory",
]
