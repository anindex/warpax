"""Derivative-free optimizer for source-first shell optimization.

Uses scipy.optimize.minimize (Nelder-Mead/Powell) on the multi-objective
loss. The forward pass includes interpax interpolation construction which
is not JIT-compilable, so derivative-free methods are used.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import minimize as scipy_minimize

from .basis import ShellCoeffs, default_theta, unpack_theta
from .ec_constraints import ECFeasibilityResult, ec_feasibility_check
from .loss import LossComponents, LossWeights, evaluate_loss


class OptimizationResult(NamedTuple):
    """Result of optimize_shell."""

    theta_opt: Float[Array, "D"]
    coeffs: ShellCoeffs
    loss_final: float
    loss_components: LossComponents
    ec_feasibility: ECFeasibilityResult | None
    n_evals: int
    converged: bool


def optimize_shell(
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
    theta_init: Float[Array, "D"] | None = None,
    method: str = "Nelder-Mead",
    maxiter: int = 200,
    certify_ec: bool = True,
    callback=None,
) -> OptimizationResult:
    """Optimize shell source profiles to minimize multi-objective loss.

    Parameters
    ----------
    ansatz : "sshell" or "tshell".
    R_1, R_2 : shell radii.
    n_density, n_velocity : free coefficient counts.
    n_grid : constraint solver resolution.
    n_probes : radial probe count.
    weights : loss weights.
    n_ec_starts : EC multi-start for optimization loop.
    theta_init : initial parameter vector (default: default_theta).
    method : scipy.optimize.minimize method.
    maxiter : maximum iterations.
    certify_ec : run full EC feasibility check post-optimization.
    callback : optional callback(theta) per iteration.
    """
    if theta_init is None:
        theta_init = default_theta(n_density, n_velocity)
    if weights is None:
        weights = LossWeights()

    n_evals = 0

    def objective(theta_np):
        nonlocal n_evals
        n_evals += 1
        theta_jax = jnp.asarray(theta_np)
        try:
            loss_val, _ = evaluate_loss(
                theta_jax,
                ansatz=ansatz,
                R_1=R_1,
                R_2=R_2,
                n_density=n_density,
                n_velocity=n_velocity,
                n_grid=n_grid,
                n_probes=n_probes,
                weights=weights,
                n_ec_starts=n_ec_starts,
            )
            return float(loss_val)
        except Exception:
            return 1e10

    import numpy as np
    x0 = np.asarray(theta_init)

    result = scipy_minimize(
        objective,
        x0,
        method=method,
        options={"maxiter": maxiter, "maxfev": maxiter * 4},
        callback=callback,
    )

    theta_opt = jnp.asarray(result.x)
    coeffs = unpack_theta(theta_opt, n_density, n_velocity)

    _, loss_components = evaluate_loss(
        theta_opt,
        ansatz=ansatz,
        R_1=R_1,
        R_2=R_2,
        n_density=n_density,
        n_velocity=n_velocity,
        n_grid=n_grid,
        n_probes=n_probes,
        weights=weights,
        n_ec_starts=n_ec_starts,
    )

    ec_result = None
    if certify_ec:
        metric = _build_metric(coeffs, ansatz, R_1, R_2, n_grid)
        r_cert = jnp.linspace(R_1 - 1.0, R_2 + 1.0, n_probes)
        ec_result = ec_feasibility_check(metric, r_cert, n_starts=16)

    return OptimizationResult(
        theta_opt=theta_opt,
        coeffs=coeffs,
        loss_final=float(result.fun),
        loss_components=loss_components,
        ec_feasibility=ec_result,
        n_evals=n_evals,
        converged=result.success,
    )


def _build_metric(coeffs, ansatz, R_1, R_2, n_grid):
    """Build a metric from ShellCoeffs for EC certification."""
    from .basis import coeffs_to_profiles_sshell, coeffs_to_profiles_tshell

    if ansatz == "sshell":
        from ..metrics.sshell import sshell_from_profiles
        profiles = coeffs_to_profiles_sshell(coeffs, R_1, R_2)
        return sshell_from_profiles(profiles, v_s=0.0, n_grid=n_grid)
    elif ansatz == "tshell":
        from ..metrics.tshell import tshell_from_profiles
        profiles = coeffs_to_profiles_tshell(coeffs, R_1, R_2)
        return tshell_from_profiles(profiles, n_grid=n_grid)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz!r}")
