"""Constraint solver for T-shell spacetimes with tilted matter flow.

Solves the Einstein constraint equations for a static spherically
symmetric shell with x-directed tilted matter flow:

    Hamiltonian: m(r) = 4pi int E(r') r'^2 dr',  Lambda = -0.5 ln(1 - 2m/r)
    TOV/lapse:   dPhi/dr = (m + 4pi r^3 p_eff) / (r(r - 2m))
    Momentum:    beta'' + A(r) beta' + B(r) beta = 8pi alpha S_x

where A = 2/r + 2 Phi' - 2 Lambda', B = -2/r^2. The momentum constraint
is solved as a tridiagonal BVP with BCs: beta'(0)=0, beta(r_max)=0.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.linalg import solve_banded


class TShellPotentials(NamedTuple):
    """Metric potentials solved from tilted-flow source profiles.

    Attributes
    ----------
    Phi_fn : callable r -> Phi(r)
        Lapse potential: alpha = e^{Phi(r)}.
    Lambda_fn : callable r -> Lambda(r)
        Radial metric potential: gamma_rr = e^{2 Lambda(r)}.
    m_fn : callable r -> m(r)
        Cumulative mass function (Eulerian).
    beta_x_fn : callable r -> beta^x(r)
        Shift x-component from momentum constraint.
    total_mass : float
        Total shell mass M = m(R_2).
    r_grid : Float[Array, "N"]
        Radial grid.
    Phi_grid : Float[Array, "N"]
        Phi values on the grid.
    Lambda_grid : Float[Array, "N"]
        Lambda values on the grid.
    m_grid : Float[Array, "N"]
        Cumulative mass on the grid.
    beta_x_grid : Float[Array, "N"]
        Shift beta^x on the grid.
    """

    Phi_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    Lambda_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    m_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    beta_x_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    total_mass: float
    r_grid: Float[Array, "N"]
    Phi_grid: Float[Array, "N"]
    Lambda_grid: Float[Array, "N"]
    m_grid: Float[Array, "N"]
    beta_x_grid: Float[Array, "N"]


def solve_tshell_potentials(
    rho: Callable[[Float[Array, ""]], Float[Array, ""]],
    p_r: Callable[[Float[Array, ""]], Float[Array, ""]],
    v_x: Callable[[Float[Array, ""]], Float[Array, ""]],
    R_1: float,
    R_2: float,
    n_grid: int = 1024,
    r_pad_factor: float = 1.5,
) -> TShellPotentials:
    """Solve Einstein constraints for T-shell metric potentials.

    Parameters
    ----------
    rho : comoving energy density rho(r).
    p_r : comoving radial pressure p_r(r).
    v_x : spatial velocity v^x(r).
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    n_grid : radial grid points.
    r_pad_factor : extend grid to r_pad_factor * R_2.
    """
    r_max = r_pad_factor * R_2
    r_min = 1e-6
    r_grid = jnp.linspace(r_min, r_max, n_grid)
    dr = r_grid[1] - r_grid[0]

    rho_grid = jax.vmap(rho)(r_grid)
    p_r_grid = jax.vmap(p_r)(r_grid)
    v_x_grid = jax.vmap(v_x)(r_grid)

    v_max = float(jnp.max(jnp.abs(v_x_grid)))
    if v_max >= 1.0:
        raise ValueError(
            f"Velocity |v^x| reaches {v_max:.4f} >= 1. "
            "Reduce v_0 to maintain subluminal flow."
        )

    # Eulerian projections
    v_sq_grid = v_x_grid**2
    Gamma_sq_grid = 1.0 / jnp.maximum(1.0 - v_sq_grid, 1e-30)
    E_grid = Gamma_sq_grid * (rho_grid + p_r_grid * v_sq_grid)
    S_x_grid = Gamma_sq_grid * (rho_grid + p_r_grid) * v_x_grid

    # --- Hamiltonian constraint: m(r) from Eulerian E(r) ---
    integrand_m = 4.0 * jnp.pi * E_grid * r_grid**2
    m_grid = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand_m[:-1] + integrand_m[1:]) * dr),
    ])
    total_mass = float(m_grid[-1])

    compactness_max = float(jnp.max(2.0 * m_grid / jnp.maximum(r_grid, 1e-30)))
    if compactness_max >= 1.0:
        raise ValueError(
            f"Shell compactness 2m(r)/r reaches {compactness_max:.4f} >= 1. "
            "Reduce rho_0 or widen the shell to avoid a trapped surface."
        )

    compactness = 2.0 * m_grid / jnp.maximum(r_grid, 1e-30)
    compactness_safe = jnp.minimum(compactness, 1.0 - 1e-12)
    Lambda_grid = -0.5 * jnp.log(1.0 - compactness_safe)

    # --- TOV/lapse ODE with tilted-flow effective pressure ---
    # p_eff = Gamma^2 (rho + p) v^2 + p
    p_eff_grid = Gamma_sq_grid * (rho_grid + p_r_grid) * v_sq_grid + p_r_grid

    numerator = m_grid + 4.0 * jnp.pi * r_grid**3 * p_eff_grid
    denominator = r_grid * (r_grid - 2.0 * m_grid)
    denom_safe = jnp.where(
        jnp.abs(denominator) < 1e-30,
        jnp.sign(denominator + 1e-60) * 1e-30,
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

    # --- Momentum constraint BVP for beta^x(r) ---
    alpha_grid = jnp.exp(Phi_grid)
    r_safe = jnp.maximum(r_grid, 1e-30)
    factor_1m2mr = jnp.maximum(1.0 - compactness_safe, 1e-30)

    dm_dr_grid = 4.0 * jnp.pi * E_grid * r_grid**2
    dLambda_dr = (dm_dr_grid * r_safe - m_grid) / (r_safe**2 * factor_1m2mr)
    dLambda_dr = jnp.where(r_grid < 1e-6, 0.0, dLambda_dr)

    # Source: 8pi alpha S_x (the e^{2Lambda} factors cancel in index raising)
    source_beta = 8.0 * jnp.pi * alpha_grid * S_x_grid

    # ODE coefficients: A(r) = 2/r + 2 Phi' - 2 Lambda',  B(r) = -2/r^2
    A_coeff = 2.0 / r_safe + 2.0 * dPhi_dr - 2.0 * dLambda_dr
    A_coeff = jnp.where(r_grid < 1e-6, 0.0, A_coeff)
    B_coeff = -2.0 / r_safe**2
    B_coeff = jnp.where(r_grid < 1e-6, 0.0, B_coeff)

    # Tridiagonal finite-difference BVP solve
    beta_x_grid = _solve_shift_bvp(
        np.asarray(A_coeff), np.asarray(B_coeff),
        np.asarray(source_beta), float(dr), n_grid,
    )

    # Interpolated callables
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

    def beta_x_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        return interpax.interp1d(
            jnp.clip(r, r_min, r_max), r_grid, beta_x_grid, method="cubic",
        )

    return TShellPotentials(
        Phi_fn=Phi_fn,
        Lambda_fn=Lambda_fn,
        m_fn=m_fn,
        beta_x_fn=beta_x_fn,
        total_mass=total_mass,
        r_grid=r_grid,
        Phi_grid=Phi_grid,
        Lambda_grid=Lambda_grid,
        m_grid=m_grid,
        beta_x_grid=beta_x_grid,
    )


def _solve_shift_bvp(
    A: np.ndarray,
    B: np.ndarray,
    source: np.ndarray,
    dr: float,
    n_pts: int,
) -> Float[Array, "N"]:
    """Solve beta'' + A beta' + B beta = source via tridiagonal FD.

    BCs: beta'(0) = 0 (regularity), beta(r_max) = 0 (asymptotic flatness).
    Uses second-order central differences and scipy solve_banded.
    """
    inv_dr2 = 1.0 / dr**2
    inv_2dr = 1.0 / (2.0 * dr)

    # Vectorized coefficient assembly (no Python loop)
    idx = np.arange(1, n_pts - 1)
    lower = np.zeros(n_pts)
    main = np.zeros(n_pts)
    upper = np.zeros(n_pts)
    rhs_vec = np.zeros(n_pts)

    lower[idx] = inv_dr2 - A[idx] * inv_2dr
    main[idx] = -2.0 * inv_dr2 + B[idx]
    upper[idx] = inv_dr2 + A[idx] * inv_2dr
    rhs_vec[idx] = source[idx]

    # BC at i=0: regularity (beta'=0 => beta_{-1} = beta_1)
    main[0] = -2.0 * inv_dr2 + B[0]
    upper[0] = 2.0 * inv_dr2
    rhs_vec[0] = source[0]

    # BC at i=n-1: beta(r_max) = 0
    main[-1] = 1.0
    lower[-1] = 0.0
    rhs_vec[-1] = 0.0

    ab = np.zeros((3, n_pts))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]

    return jnp.array(solve_banded((1, 1), ab, rhs_vec))
