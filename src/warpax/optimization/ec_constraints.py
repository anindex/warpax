"""EC constraint enforcement for shell optimization.

Soft penalty: sum of softplus(-margin)^2 for gradient-based optimization.
Hard feasibility: binary certification with the full warpax EC pipeline.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..energy_conditions.optimization import optimize_nec, optimize_wec, optimize_dec
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


class ECFeasibilityResult(NamedTuple):
    """Result of a hard EC feasibility check.

    feasible : True iff all conditions satisfied at all probe points.
    margins : per-condition margin arrays, shape (N_probes,).
    worst_point_idx : index of the worst-violating probe point.
    worst_condition : name of the worst-violated condition.
    worst_margin : value of the worst margin (negative = violated).
    """

    feasible: bool
    margins: dict[str, Float[Array, "N"]]
    worst_point_idx: int
    worst_condition: str
    worst_margin: float


_OPTIMIZER_MAP = {"nec": optimize_nec, "wec": optimize_wec, "dec": optimize_dec}


def ec_penalty(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
    *,
    conditions: tuple[str, ...] = ("nec", "wec", "dec"),
    n_starts: int = 4,
) -> Float[Array, ""]:
    """Soft EC penalty: sum of softplus(-margin)^2 over probe points.

    Uses a Python loop over probes because the EC optimizers contain
    internal Python control flow (DEC subconditions) not compatible with vmap.
    """
    total_penalty = jnp.float64(0.0)

    for i in range(r_probes.shape[0]):
        coords = jnp.array([0.0, r_probes[i], 0.0, 0.0])
        cc = compute_curvature_chain(metric, coords)
        T = jnp.where(jnp.isnan(cc.stress_energy), 0.0, cc.stress_energy)
        g = cc.metric
        for cond in conditions:
            res = _OPTIMIZER_MAP[cond](T, g, n_starts=n_starts)
            total_penalty = total_penalty + jax.nn.softplus(-res.margin) ** 2

    return total_penalty


def ec_feasibility_check(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
    *,
    conditions: tuple[str, ...] = ("nec", "wec", "dec"),
    n_starts: int = 16,
) -> ECFeasibilityResult:
    """Hard EC check with per-point, per-condition margins.

    Uses n_starts=16 by default for certification quality.
    """
    margins = {}
    for cond in conditions:
        opt_fn = _OPTIMIZER_MAP[cond]
        cond_margins_list = []

        for i in range(r_probes.shape[0]):
            coords = jnp.array([0.0, r_probes[i], 0.0, 0.0])
            cc = compute_curvature_chain(metric, coords)
            T = jnp.where(jnp.isnan(cc.stress_energy), 0.0, cc.stress_energy)
            g = cc.metric
            res = opt_fn(T, g, n_starts=n_starts)
            cond_margins_list.append(res.margin)

        margins[cond] = jnp.stack(cond_margins_list)

    worst_condition = conditions[0]
    worst_margin = float("inf")
    worst_idx = 0

    for cond in conditions:
        min_val = float(jnp.min(margins[cond]))
        if min_val < worst_margin:
            worst_margin = min_val
            worst_condition = cond
            worst_idx = int(jnp.argmin(margins[cond]))

    feasible = all(float(jnp.min(margins[c])) >= 0.0 for c in conditions)

    return ECFeasibilityResult(
        feasible=feasible,
        margins=margins,
        worst_point_idx=worst_idx,
        worst_condition=worst_condition,
        worst_margin=worst_margin,
    )
