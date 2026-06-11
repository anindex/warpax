"""Quantum-inequality evaluators for warp drive spacetimes.

Contains:

- ``ford_roman(metric, worldline, tau0, sampling='lorentzian')`` --
  Ford-Roman QI per Pfenning & Ford 1997 eq. (9), with constant
  ``C = 3 / (32 pi^2)`` for the massless scalar field.
"""
from __future__ import annotations

from .ford_roman import QIResult, ford_roman

__all__ = [
    "QIResult",
    "ford_roman",
]
