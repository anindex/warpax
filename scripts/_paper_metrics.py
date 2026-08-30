"""The four compact constructions the paper sweeps, and their instantiation.

``analysis.construction_adapter.construction_registry`` carries the same four
plus grid bounds, wall radii and claims, and is not a drop-in for this table.
"""

from __future__ import annotations

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (VanDenBroeckMetric, {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "Rodal": (RodalMetric, {}),
}

METRIC_ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def instantiate(name: str, v_s: float, R: float = 1.0, sigma: float = 8.0):
    """One construction at ``(v_s, R, sigma)``."""
    cls, extra = METRICS[name]
    return cls(v_s=v_s, R=R, sigma=sigma, **extra)
