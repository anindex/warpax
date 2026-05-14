"""Source-first shell optimization framework.

Compact-support Bernstein basis for source profile parameterization,
multi-objective loss (constraints + EC + tidal + transport + mass),
EC constraint enforcement (soft penalty + hard feasibility),
and parameter space sweep for transport utility maximization.
"""
from __future__ import annotations

from .basis import (
    ShellCoeffs,
    bernstein_basis,
    bernstein_eval,
    clamp_endpoints,
    coeffs_to_profiles_sshell,
    coeffs_to_profiles_tshell,
    default_theta,
    pack_theta,
    unpack_theta,
)
from .ec_constraints import (
    ECFeasibilityResult,
    ec_feasibility_check,
    ec_penalty,
)
from .loss import (
    LossComponents,
    LossWeights,
    evaluate_loss,
)
from .optimizer import (
    OptimizationResult,
    optimize_shell,
)
from .sweep import (
    SweepPoint,
    SweepResult,
    sweep_transport,
)

__all__ = [
    "ECFeasibilityResult",
    "LossComponents",
    "LossWeights",
    "OptimizationResult",
    "ShellCoeffs",
    "SweepPoint",
    "SweepResult",
    "bernstein_basis",
    "bernstein_eval",
    "clamp_endpoints",
    "coeffs_to_profiles_sshell",
    "coeffs_to_profiles_tshell",
    "default_theta",
    "ec_feasibility_check",
    "ec_penalty",
    "evaluate_loss",
    "optimize_shell",
    "pack_theta",
    "sweep_transport",
    "unpack_theta",
]
