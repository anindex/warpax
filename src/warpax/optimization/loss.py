"""Multi-objective loss for source-first shell optimization.

    L = w_H |H|^2 + w_M |M|^2 + w_E sum softplus(-m_EC)^2
      + w_T A_geo(interior) - w_U max|beta^x| + w_R M_ADM

Evaluated on a radial probe grid exploiting spherical symmetry.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .basis import unpack_theta, coeffs_to_profiles_sshell, coeffs_to_profiles_tshell


class LossWeights(NamedTuple):
    """Multi-objective loss weights."""

    w_constraint: float = 1.0
    w_ec: float = 10.0
    w_tidal: float = 0.1
    w_transport: float = -1.0
    w_mass: float = 0.01


class LossComponents(NamedTuple):
    """Per-term breakdown of the multi-objective loss."""

    total: float
    constraint: float
    ec_penalty: float
    tidal: float
    transport: float
    mass: float


def evaluate_loss(
    theta: Float[Array, "D"],
    *,
    ansatz: str = "sshell",
    R_1: float = 10.0,
    R_2: float = 20.0,
    n_density: int = 4,
    n_velocity: int = 4,
    n_grid: int = 512,
    n_probes: int = 20,
    weights: LossWeights | None = None,
    n_ec_starts: int = 4,
    skip_ec: bool = False,
    fast: bool = False,
) -> tuple[Float[Array, ""], LossComponents]:
    """Evaluate multi-objective loss for a parameter vector.

    Parameters
    ----------
    theta : flat parameter vector from pack_theta.
    ansatz : "sshell" or "tshell".
    R_1, R_2 : shell radii.
    n_density, n_velocity : free coefficient counts.
    n_grid : constraint solver grid resolution.
    n_probes : radial probe points for EC/constraint evaluation.
    weights : loss weights.
    n_ec_starts : BFGS multi-start count for EC margin.
    skip_ec : if True, skip EC optimization (set ec_loss=0).
    fast : if True, skip both EC and constraint probing. Only
        transport + mass are computed. Use for the inner
        optimization loop when constraint residuals are not
        needed (the constraint solver already enforces them).
    """
    if weights is None:
        weights = LossWeights()

    coeffs = unpack_theta(theta, n_density, n_velocity)

    if ansatz == "sshell":
        profiles = coeffs_to_profiles_sshell(coeffs, R_1, R_2)
        from ..metrics.sshell import sshell_from_profiles
        metric = sshell_from_profiles(profiles, v_s=0.0, n_grid=n_grid)
        max_beta = 0.0
    elif ansatz == "tshell":
        profiles = coeffs_to_profiles_tshell(coeffs, R_1, R_2)
        from ..metrics.tshell import tshell_from_profiles
        metric = tshell_from_profiles(profiles, n_grid=n_grid)
        max_beta = float(jnp.max(jnp.abs(metric._beta_x_grid)))
    else:
        raise ValueError(f"Unknown ansatz: {ansatz!r}")

    transport = jnp.asarray(max_beta)
    mass = jnp.asarray(float(metric.total_mass))

    if fast:
        total = weights.w_transport * transport + weights.w_mass * mass
        return total, LossComponents(
            total=float(total),
            constraint=0.0,
            ec_penalty=0.0,
            tidal=0.0,
            transport=float(transport),
            mass=float(mass),
        )

    r_probes = jnp.linspace(R_1, R_2, n_probes)

    from ..constraints.residuals import normalized_residuals

    def eval_constraint(r_val):
        coords = jnp.array([0.0, r_val, 0.0, 0.0])
        res = normalized_residuals(metric, coords)
        return res["epsilon_H"]**2 + res["epsilon_M"]**2

    constraint_vals = jax.vmap(eval_constraint)(r_probes)
    constraint_loss = jnp.mean(constraint_vals)

    if skip_ec:
        ec_loss = jnp.float64(0.0)
    else:
        from ..energy_conditions.optimization import optimize_nec, optimize_wec, optimize_dec
        from ..geometry.geometry import compute_curvature_chain

        ec_penalties = []
        for i in range(n_probes):
            r_val = r_probes[i]
            coords = jnp.array([0.0, r_val, 0.0, 0.0])
            cc = compute_curvature_chain(metric, coords)
            T = jnp.where(jnp.isnan(cc.stress_energy), 0.0, cc.stress_energy)
            g = cc.metric

            nec_res = optimize_nec(T, g, n_starts=n_ec_starts)
            wec_res = optimize_wec(T, g, n_starts=n_ec_starts)
            dec_res = optimize_dec(T, g, n_starts=n_ec_starts)

            penalty = (
                jax.nn.softplus(-nec_res.margin)**2
                + jax.nn.softplus(-wec_res.margin)**2
                + jax.nn.softplus(-dec_res.margin)**2
            )
            ec_penalties.append(penalty)

        ec_loss = jnp.mean(jnp.stack(ec_penalties))

    from ..transport.diagnostics import geodesic_deviation_diagnostic
    tidal = geodesic_deviation_diagnostic(
        metric, jnp.array([0.0, R_1 * 0.5, 0.0, 0.0]),
    )

    total = (
        weights.w_constraint * constraint_loss
        + weights.w_ec * ec_loss
        + weights.w_tidal * tidal
        + weights.w_transport * transport
        + weights.w_mass * mass
    )

    components = LossComponents(
        total=float(total),
        constraint=float(constraint_loss),
        ec_penalty=float(ec_loss),
        tidal=float(tidal),
        transport=float(transport),
        mass=float(mass),
    )

    return total, components
