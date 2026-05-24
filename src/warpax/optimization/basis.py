"""Compact-support Bernstein basis for shell source profile parameterization.

Bernstein polynomials on t = (r - R_1) / (R_2 - R_1), with structural
endpoint clamping (c_0 = c_n = 0) for compact support. Density coefficients
use softplus reparameterization for positivity; velocity is unconstrained.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float


class ShellCoeffs(NamedTuple):
    """Unpacked optimization coefficients for a shell.

    density_coeffs : full Bernstein coefficients including clamped endpoints.
    velocity_coeffs : full Bernstein coefficients including clamped endpoints.
    v_0 : velocity scale factor.
    rho_scale : density scale factor.
    """

    density_coeffs: Float[Array, "N"]
    velocity_coeffs: Float[Array, "M"]
    v_0: float
    rho_scale: float


def bernstein_basis(n: int, t: Float[Array, ""]) -> Float[Array, "n1"]:
    """Evaluate all n+1 Bernstein basis functions B_{n,k}(t) at t."""
    k_idx = jnp.arange(n + 1, dtype=jnp.float64)
    log_binom = (
        gammaln(n + 1.0)
        - gammaln(k_idx + 1.0)
        - gammaln(n - k_idx + 1.0)
    )
    binom = jnp.exp(log_binom)
    return binom * (t ** k_idx) * ((1.0 - t) ** (n - k_idx))


def bernstein_eval(
    coeffs: Float[Array, "N"], t: Float[Array, ""]
) -> Float[Array, ""]:
    """Evaluate sum_k c_k B_{n,k}(t)."""
    n = coeffs.shape[0] - 1
    basis = bernstein_basis(n, t)
    return jnp.sum(coeffs * basis)


def clamp_endpoints(
    theta: Float[Array, "K"],
    positive: bool = False,
) -> Float[Array, "K2"]:
    """Prepend/append zeros to interior coefficients for compact support.

    If positive=True, applies softplus to enforce non-negative interior values.
    """
    if positive:
        interior = jax.nn.softplus(theta)
    else:
        interior = theta
    return jnp.concatenate([jnp.array([0.0]), interior, jnp.array([0.0])])


def pack_theta(
    density_theta: Float[Array, "Kd"],
    velocity_theta: Float[Array, "Kv"],
    v_0: float,
    rho_scale: float,
) -> Float[Array, "D"]:
    """Pack free parameters into flat vector [density | velocity | v_0 | rho_scale]."""
    return jnp.concatenate([
        density_theta,
        velocity_theta,
        jnp.array([v_0, rho_scale]),
    ])


def unpack_theta(
    theta: Float[Array, "D"],
    n_density: int,
    n_velocity: int,
    positive_density: bool = True,
) -> ShellCoeffs:
    """Unpack flat parameter vector into ShellCoeffs with endpoint clamping."""
    density_theta = theta[:n_density]
    velocity_theta = theta[n_density:n_density + n_velocity]
    v_0 = theta[n_density + n_velocity]
    rho_scale = theta[n_density + n_velocity + 1]

    density_coeffs = clamp_endpoints(density_theta, positive=positive_density)
    velocity_coeffs = clamp_endpoints(velocity_theta, positive=False)

    # rho(r) = softplus(rho_scale_raw) * sum c_k B(t)
    density_coeffs = jax.nn.softplus(rho_scale) * density_coeffs

    return ShellCoeffs(
        density_coeffs=density_coeffs,
        velocity_coeffs=velocity_coeffs,
        v_0=float(v_0),
        rho_scale=float(jax.nn.softplus(rho_scale)),
    )


def coeffs_to_profiles_sshell(
    coeffs: ShellCoeffs,
    R_1: float = 10.0,
    R_2: float = 20.0,
):
    """Convert ShellCoeffs to SShellSourceProfiles via bernstein_density_profiles."""
    from ..metrics.sshell_profiles import bernstein_density_profiles
    return bernstein_density_profiles(
        R_1=R_1, R_2=R_2, coeffs=coeffs.density_coeffs,
    )


def coeffs_to_profiles_tshell(
    coeffs: ShellCoeffs,
    R_1: float = 10.0,
    R_2: float = 20.0,
):
    """Convert ShellCoeffs to TShellSourceProfiles via bernstein_velocity_profiles."""
    from ..metrics.tshell_profiles import bernstein_velocity_profiles
    return bernstein_velocity_profiles(
        R_1=R_1,
        R_2=R_2,
        density_coeffs=coeffs.density_coeffs,
        velocity_coeffs=coeffs.velocity_coeffs,
        v_0=coeffs.v_0,
    )


def default_theta(
    n_density: int = 4,
    n_velocity: int = 4,
    v_0: float = 0.1,
    rho_scale: float = 1e-4,
) -> Float[Array, "D"]:
    """Construct a default initial parameter vector with uniform interior coefficients."""
    density_theta = jnp.ones(n_density) * 0.5
    velocity_theta = jnp.ones(n_velocity) * 0.5

    # Stable inverse softplus: log(expm1(x)) for any positive x; ``expm1``
    # avoids catastrophic cancellation when ``rho_scale`` is small and
    # clipping guards the overflow tail at large arguments.
    rho_scale_raw = jnp.log(jnp.expm1(jnp.clip(rho_scale, 1e-300, 700.0)))

    return pack_theta(density_theta, velocity_theta, v_0, float(rho_scale_raw))
