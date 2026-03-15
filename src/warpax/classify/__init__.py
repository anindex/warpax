"""Warp drive metric classification against the Bobrick-Martire taxonomy.

Bobrick & Martire 2021 (Class Quantum Grav 38 105009, arXiv:2102.08443)
propose a three-class taxonomy of physically-interesting spacetimes:

- Class I : Killing-field structure; trivial or constant shape function
             (Minkowski, Schwarzschild).
- Class II : Alcubierre-family shape-function-supported metrics with
             comoving stress-energy (Alcubierre, Rodal, Natario, Van den
             Broeck, Lentz).
- Class III: Matter-shell / junction-structured metrics (WarpShell,
             Morris-Thorne wormhole).

The `bobrick_martire` function pattern-matches an input metric against
these three classes using a three-test cascade: stationarity, comoving-fluid
sign, and shape-function support.
"""
from __future__ import annotations

from .bobrick_martire import ClassifiedMetric, bobrick_martire

__all__ = [
    "ClassifiedMetric",
    "bobrick_martire",
]
