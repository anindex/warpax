"""Uniform adapter for auditing heterogeneous positive-energy warp constructions.

The warp-drive metrics in :mod:`warpax.metrics` do not share a constructor: the
compact family takes ``cls(v_s=, R=1, sigma=8)`` while the source-prescribed
shells take ``R_1, R_2, ...`` through factory functions and (for the T-shell) a
matter tilt ``v_0`` rather than a shift speed ``v_s``. This module wraps each
construction behind a single :class:`ConstructionSpec` so one audit pipeline can flow
all of them through the frame-independent certifier
(:func:`..energy_conditions.frame_free.certify_grid_frame_free`) and the
all-observer verification (:mod:`.invariant_verification`) at matched, wall-
resolved settings.

A resolution gate (:func:`wall_cells`) operationalises the paper's
"never report an unresolved wall" rule: a construction whose wall spans fewer than
``MIN_WALL_CELLS`` grid cells is flagged ``resolved=False`` and its certification
numbers are withheld.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import numpy as np

from ..benchmarks import AlcubierreMetric
from ..geometry.metric import MetricSpecification
from ..metrics import (
    GarattiniMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellPhysical,
    fuchs_default,
    sshell_default,
    tshell_default,
)

MIN_WALL_CELLS = 4.0


@dataclass(frozen=True)
class ConstructionSpec:
    """A positive-energy warp construction wrapped for uniform auditing."""

    name: str
    build: Callable[[float], MetricSpecification]
    default_speed: float
    speed_param: str  # "v_s" or "v_0"
    bounds: tuple[tuple[float, float], ...]
    grid_n: int
    is_comoving: bool
    # Published energy-condition claim (community summary), for the
    # certified-vs-claimed agreement column. Free-text, not a refutation.
    claim: str = ""
    extra: dict = field(default_factory=dict)

    def metric(self, speed: float | None = None) -> MetricSpecification:
        return self.build(self.default_speed if speed is None else speed)


def _alcubierre(v_s):
    return AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)


def _rodal(v_s):
    return RodalMetric(v_s=v_s, R=1.0, sigma=8.0)


def _natario(v_s):
    return NatarioMetric(v_s=v_s, R=1.0, sigma=8.0)


def _vdb(v_s):
    return VanDenBroeckMetric(
        v_s=v_s, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0
    )


def _garattini(v_s):
    # de Sitter background warp bubble; H is matched so that the audit speed
    # v_s = H * R sits at the Garattini-Zatrimaylov averaged-condition regime.
    return GarattiniMetric(v_s=v_s, R=1.0, sigma=8.0, H=v_s)


def construction_registry() -> dict[str, ConstructionSpec]:
    """All audit constructions keyed by name (compact references + shells)."""
    specs = [
        # Compact references (baseline + the irrotational global-Type-I claim).
        ConstructionSpec(
            "Alcubierre", _alcubierre, 0.5, "v_s", ((-3.0, 3.0),) * 3, 50,
            is_comoving=True,
            claim="baseline; NEC/WEC violated for all observers",
        ),
        ConstructionSpec(
            "Rodal", _rodal, 0.5, "v_s", ((-3.0, 3.0),) * 3, 50,
            is_comoving=True,
            claim="global Hawking-Ellis Type I; reduced (not eliminated) violations",
        ),
        # Additional positive-energy / source-prescribed constructions.
        ConstructionSpec(
            "Fuchs", lambda v: fuchs_default(v_s=v), 0.02, "v_s",
            ((-25.0, 25.0),) * 3, 60, is_comoving=True,
            claim="constant-velocity shell satisfying all energy conditions "
                  "(arXiv:2405.02709)",
        ),
        ConstructionSpec(
            "WarpShell", lambda v: WarpShellPhysical(v_s=v), 0.02, "v_s",
            ((-25.0, 25.0),) * 3, 60, is_comoving=True,
            claim="Bobrick-Martire / Fell-Heisenberg shell; WEC/NEC/SEC at wall, "
                  "DEC violated",
        ),
        ConstructionSpec(
            "Garattini", _garattini, 0.1, "v_s",
            ((-3.0, 3.0),) * 3, 50, is_comoving=True,
            claim="de Sitter background; averaged ANEC/AWEC satisfied at the "
                  "matched speed v_s = H R, pointwise NEC/WEC violated at the wall "
                  "(arXiv:2502.13153)",
        ),
        ConstructionSpec(
            "S-shell", lambda v: sshell_default(v_s=v), 0.02, "v_s",
            ((-25.0, 25.0),) * 3, 60, is_comoving=True,
            claim="source-first Class-I positive-density shell",
        ),
        ConstructionSpec(
            "T-shell", lambda v: tshell_default(v_0=v), 0.1, "v_0",
            ((-25.0, 25.0),) * 3, 60, is_comoving=False,
            claim="origin-static transport shell (matter tilt v_0)",
        ),
    ]
    return {s.name: s for s in specs}


def wall_cells(spec: ConstructionSpec, speed: float | None = None,
               n: int | None = None) -> float:
    """Number of grid cells spanning the wall transition along +x.

    Samples the metric's ``shape_function_value`` on a uniform 1-D radial line
    at the audit grid spacing and counts points in the active transition band
    ``f in [0.05, 0.95]``. This is the operational resolution witness behind the
    "never report an unresolved wall" rule.
    """
    metric = spec.metric(speed)
    n = spec.grid_n if n is None else n
    xs = np.linspace(spec.bounds[0][0], spec.bounds[0][1], n)
    fvals = []
    for x in xs:
        c = jnp.array([0.0, float(x), 0.0, 0.0])
        fvals.append(float(metric.shape_function_value(c)))
    f = np.array(fvals)
    fn = (f - f.min()) / (np.ptp(f) + 1e-30)  # normalise to [0,1]
    in_band = (fn > 0.05) & (fn < 0.95)
    return float(np.sum(in_band))


def is_resolved(spec: ConstructionSpec, speed: float | None = None,
                n: int | None = None) -> tuple[bool, float]:
    cells = wall_cells(spec, speed=speed, n=n)
    return cells >= MIN_WALL_CELLS, cells
