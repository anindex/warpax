"""Constraint solver for static spherically symmetric shells.

Implements the forward map from source profiles (rho, p_r) to metric
potentials (Phi, Lambda, m) by solving the Einstein constraint equations
on a radial grid.

For a static, spherically symmetric, flow-orthogonal shell:

    Hamiltonian:  e^{2Lambda(r)} = 1 / (1 - 2m(r)/r)
                  where m(r) = 4pi int_0^r rho(r') r'^2 dr'

    TOV / lapse:  dPhi/dr = (m(r) + 4pi r^3 p_r(r)) / (r(r - 2m(r)))
                  with Phi(r_max) = 0.5 * ln(1 - 2M/r_max)

Metric potentials are returned as cubic spline interpolants (interpax)
for C2-smooth evaluation through the curvature chain.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class SShellPotentials(NamedTuple):
    """Metric potentials solved from source profiles.

    Attributes
    ----------
    Phi_fn : callable r -> Phi(r)
        Lapse potential: alpha = e^{Phi(r)}.
    Lambda_fn : callable r -> Lambda(r)
        Radial metric potential: gamma_rr = e^{2 Lambda(r)}.
    m_fn : callable r -> m(r)
        Cumulative mass function.
    total_mass : float
        Total shell mass M = m(R_2).
    r_grid : Float[Array, "N"]
        Radial grid used for integration.
    Phi_grid : Float[Array, "N"]
        Phi values on the grid.
    Lambda_grid : Float[Array, "N"]
        Lambda values on the grid.
    m_grid : Float[Array, "N"]
        Cumulative mass values on the grid.
    """

    Phi_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    Lambda_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    m_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    total_mass: float
    r_grid: Float[Array, "N"]
    Phi_grid: Float[Array, "N"]
    Lambda_grid: Float[Array, "N"]
    m_grid: Float[Array, "N"]


def solve_sshell_potentials(
    rho: Callable[[Float[Array, ""]], Float[Array, ""]],
    p_r: Callable[[Float[Array, ""]], Float[Array, ""]],
    R_1: float,
    R_2: float,
    n_grid: int = 1024,
    r_pad_factor: float = 1.5,
) -> SShellPotentials:
    """Solve Einstein constraints for S-shell metric potentials.

    Given radial source profiles rho(r) and p_r(r) with compact support
    in [R_1, R_2], integrates the Hamiltonian constraint and TOV/lapse
    equation to produce the metric potentials Phi(r) and Lambda(r).

    The resulting potentials define a static spherically symmetric
    spacetime::

        ds^2 = -e^{2Phi} dt^2 + e^{2Lambda} dr^2 + r^2 dOmega^2

    Parameters
    ----------
    rho : callable r -> rho(r)
        Energy density profile with compact support in [R_1, R_2].
    p_r : callable r -> p_r(r)
        Radial pressure profile.
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    n_grid : number of radial grid points (default: 1024).
    r_pad_factor : extend grid to r_pad_factor * R_2 (default: 1.5).

    Returns
    -------
    SShellPotentials

    Raises
    ------
    ValueError
        If 2m(r)/r >= 1 anywhere (trapped surface).
    """
    r_max = r_pad_factor * R_2
    r_min = 1e-6
    r_grid = jnp.linspace(r_min, r_max, n_grid)
    dr = r_grid[1] - r_grid[0]

    import interpax  # noqa: PLC0415  (optional dep; only needed here)

    # Cumulative mass: m(r) = 4pi int_0^r rho(r') r'^2 dr'
    rho_grid = jax.vmap(rho)(r_grid)
    integrand = 4.0 * jnp.pi * rho_grid * r_grid**2
    m_grid = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dr),
    ])

    total_mass = float(m_grid[-1])

    compactness_max = float(jnp.max(2.0 * m_grid / jnp.maximum(r_grid, 1e-30)))
    if compactness_max >= 1.0:
        raise ValueError(
            f"Shell compactness 2m(r)/r reaches {compactness_max:.4f} >= 1. "
            "Reduce rho_0 or widen the shell to avoid a trapped surface."
        )

    # Radial metric potential: Lambda = -0.5 * ln(1 - 2m/r)
    compactness = 2.0 * m_grid / jnp.maximum(r_grid, 1e-30)
    compactness_safe = jnp.minimum(compactness, 1.0 - 1e-12)
    Lambda_grid = -0.5 * jnp.log(1.0 - compactness_safe)

    # Lapse potential via inward integration from Schwarzschild boundary
    p_r_grid = jax.vmap(p_r)(r_grid)

    numerator = m_grid + 4.0 * jnp.pi * r_grid**3 * p_r_grid
    denominator = r_grid * (r_grid - 2.0 * m_grid)
    denom_safe = jnp.where(
        jnp.abs(denominator) < 1e-30,
        jnp.where(denominator >= 0.0, 1e-30, -1e-30),
        denominator,
    )
    dPhi_dr = numerator / denom_safe
    dPhi_dr = jnp.where(r_grid < R_1 * 0.5, 0.0, dPhi_dr)

    Phi_boundary = 0.5 * jnp.log(1.0 - 2.0 * total_mass / r_max)
    forward_integral = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (dPhi_dr[:-1] + dPhi_dr[1:]) * dr),
    ])
    Phi_grid = Phi_boundary - (forward_integral[-1] - forward_integral)

    # Interpolated callables via interpax cubic splines
    def Phi_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        return interpax.interp1d(
            jnp.clip(r, r_min, r_max), r_grid, Phi_grid, method="cubic",
        )

    def Lambda_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        return interpax.interp1d(
            jnp.clip(r, r_min, r_max), r_grid, Lambda_grid, method="cubic",
        )

    def m_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        return interpax.interp1d(
            jnp.clip(r, r_min, r_max), r_grid, m_grid, method="cubic",
        )

    return SShellPotentials(
        Phi_fn=Phi_fn,
        Lambda_fn=Lambda_fn,
        m_fn=m_fn,
        total_mass=total_mass,
        r_grid=r_grid,
        Phi_grid=Phi_grid,
        Lambda_grid=Lambda_grid,
        m_grid=m_grid,
    )
