"""Darmois / Israel junction-condition checker for shell metrics.

Darmois 1927 and Israel 1966 (*Nuovo Cimento* B 44) give the junction
conditions across a timelike or spacelike boundary hypersurface Sigma:

- **First form** - the induced 3-metric ``h_{ab}`` must be continuous:
  ``[[h_{ab}]] = 0``.
- **Second form** - if ``[[K_{ab}]] = 0`` (extrinsic curvature continuous),
  the junction is smooth (no surface layer). If ``[[K_{ab}]] != 0``, a
  surface stress-energy tensor (Israel thin shell) is present.

The `darmois` function pattern-matches an input metric and level-set
boundary function against these two conditions, returning raw discontinuity
magnitudes plus a physicality verdict.
"""
from __future__ import annotations

from .darmois import DarmoisResult, darmois

__all__ = [
    "DarmoisResult",
    "darmois",
]
