"""Fuchs et al. constant-velocity subluminal warp shell metric.

Reference: Fuchs, Helmerich, Bobrick, Sellers, Melcher, Martire (2024).
"Constant velocity physical warp drive solution."
Classical and Quantum Gravity 41 (2024), DOI: 10.1088/1361-6382/ad26aa
arXiv: 2405.02709

This module provides a pre-configured ``FuchsMetric`` class and a
factory function ``fuchs_default()`` returning the canonical parameter
set from the Fuchs et al. CQG paper.

The Fuchs metric is a constant-velocity subluminal warp drive with:
- Flat Minkowski interior (passenger volume, v_s shift)
- Schwarzschild-like shell (positive energy density, curved spatial metric)
- Flat Minkowski exterior (no shift)

This is structurally identical to the ``WarpShellPhysical`` class in
``warpax.metrics.warpshell``, with parameters chosen to match the Fuchs
et al. paper:
- v_s = 0.01  (subluminal)
- R_1 = 10    (inner shell boundary)
- R_2 = 20    (outer shell boundary)
- r_s = 5.0   (Schwarzschild radius parameter in shell)
"""
from __future__ import annotations

from ..metrics.warpshell import WarpShellPhysical


class FuchsMetric(WarpShellPhysical):
    """Fuchs et al. constant-velocity warp shell metric.

    Inherits from WarpShellPhysical with documentation linking to the
    Fuchs et al. (2024) paper. All parameters are identical to
    WarpShellPhysical.

    Default parameters match the canonical configuration from the paper.
    """

    def name(self) -> str:
        return "Fuchs-CQG2024"


def fuchs_default() -> FuchsMetric:
    """Return the canonical Fuchs metric with paper-matched parameters.

    Returns
    -------
    FuchsMetric
        v_s=0.01, R_1=10, R_2=20, r_s_param=5.0, C2 transition.
    """
    return FuchsMetric(
        v_s=0.01,
        R_1=10.0,
        R_2=20.0,
        R_b=1.0,
        r_s_param=5.0,
        transition_order=2,
    )
