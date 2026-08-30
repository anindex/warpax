"""Hamiltonian mass integral and TOV lapse on a uniform radial grid.

Shared by the S-shell and T-shell solvers. The S-shell passes the rest-frame
``(rho, p_r)``; the T-shell passes the Eulerian ``(E, p_eff)`` of the tilted
flow. The quadrature and the boundary condition are the same either way.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def _cumulative_trapezoid(integrand: Float[Array, "N"], dr: Float[Array, ""]):
    return jnp.concatenate(
        [jnp.array([0.0]), jnp.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dr)]
    )


def hamiltonian_and_lapse(
    energy_grid: Float[Array, "N"],
    pressure_grid: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    dr: Float[Array, ""],
    R_1: float,
    r_max: float,
):
    """``(m, Lambda, Phi, dPhi/dr, compactness)`` on ``r_grid``.

    ``m(r) = 4pi int E r'^2 dr'`` by trapezoid, ``Lambda = -ln(1 - 2m/r)/2``, and
    ``Phi`` from the TOV lapse equation integrated inward from the Schwarzschild
    boundary value at ``r_max``. Raises if ``2m/r`` reaches 1, which would put a
    trapped surface inside the grid.
    """
    m_grid = _cumulative_trapezoid(4.0 * jnp.pi * energy_grid * r_grid**2, dr)

    compactness = 2.0 * m_grid / jnp.maximum(r_grid, 1e-30)
    compactness_max = float(jnp.max(compactness))
    if compactness_max >= 1.0:
        raise ValueError(
            f"Shell compactness 2m(r)/r reaches {compactness_max:.4f} >= 1. "
            "Reduce rho_0 or widen the shell to avoid a trapped surface."
        )
    compactness_safe = jnp.minimum(compactness, 1.0 - 1e-12)
    Lambda_grid = -0.5 * jnp.log(1.0 - compactness_safe)

    denominator = r_grid * (r_grid - 2.0 * m_grid)
    denom_safe = jnp.where(
        jnp.abs(denominator) < 1e-30,
        jnp.where(denominator >= 0.0, 1e-30, -1e-30),
        denominator,
    )
    dPhi_dr = (m_grid + 4.0 * jnp.pi * r_grid**3 * pressure_grid) / denom_safe
    dPhi_dr = jnp.where(r_grid < R_1 * 0.5, 0.0, dPhi_dr)

    Phi_boundary = 0.5 * jnp.log(1.0 - 2.0 * m_grid[-1] / r_max)
    forward = _cumulative_trapezoid(dPhi_dr, dr)
    Phi_grid = Phi_boundary - (forward[-1] - forward)

    return m_grid, Lambda_grid, Phi_grid, dPhi_dr, compactness_safe
