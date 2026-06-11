"""ANEC and AWEC averaged energy-condition line integrals along geodesics.

Public API:

- ``anec(metric, geodesic, tangent_norm='null_projected')`` - Averaged
  Null Energy Condition line integral along a null geodesic.
- ``awec(metric, geodesic, tangent_norm='renormalized')`` - Averaged
  Weak Energy Condition line integral along a timelike geodesic.
- ``anec_rigorous(metric, x0, n_spatial)`` - symplectically integrated ANEC
  with on-cone witness ``max|g(k,k)|``.

Both return a NamedTuple with ``line_integral``,
``geodesic_complete: bool``, and ``termination_reason: str``.
"""
from __future__ import annotations

from .anec import ANECResult, RigorousANEC, anec, anec_rigorous
from .awec import AWECResult, awec

__all__ = [
    "ANECResult",
    "AWECResult",
    "RigorousANEC",
    "anec",
    "anec_rigorous",
    "awec",
]
