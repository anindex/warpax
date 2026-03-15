"""ANEC and AWEC averaged energy-condition line integrals along geodesics.

See the source code for
geodesic-completeness policy  and tangent-norm renormalisation
semantics .

Public API:

- ``anec(metric, geodesic, tangent_norm='renormalized')`` - Averaged
  Null Energy Condition line integral along a null geodesic.
- ``awec(metric, geodesic, tangent_norm='renormalized')`` - Averaged
  Weak Energy Condition line integral along a timelike geodesic.

Both return a NamedTuple with ``line_integral``,
``geodesic_complete: bool``, and ``termination_reason: str``.
"""
from __future__ import annotations

from .anec import ANECResult, anec
from .awec import AWECResult, awec

__all__ = [
    "ANECResult",
    "AWECResult",
    "anec",
    "awec",
]
