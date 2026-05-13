"""Source profile library for the S-shell ansatz.

Provides factory functions for matter source profiles (rho, p_r, p_t)
with compact support in a spherical shell [R_1, R_2]. Each factory
returns an SShellSourceProfiles NamedTuple with callable density,
pressure, and cumulative mass functions.

Three density families:

1. Constant density: uniform rho_0 with C2 smoothstep transitions.
2. Parabolic density: rho(r) = rho_max * (1 - ((r-r_c)/Dr)^2),
   naturally vanishing at shell boundaries.
3. Bernstein-polynomial density: rho(r) = sum_k c_k B_{n,k}(t),
   endpoint coefficients fixed to 0 for compact support.
   Differentiable w.r.t. control coefficients.

All families use isotropic pressure (p_t = p_r) from TOV integration.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.transitions import smoothstep_c2


class SShellSourceProfiles(NamedTuple):
    """Source profiles for the S-shell ansatz.

    Attributes
    ----------
    density : callable r -> rho(r)
        Energy density with compact support in [R_1, R_2].
    radial_pressure : callable r -> p_r(r)
        Radial pressure from TOV integration.
    tangential_pressure : callable r -> p_t(r)
        Tangential pressure (= p_r for isotropic case).
    cumulative_mass : callable r -> m(r)
        Cumulative mass: m(r) = 4pi int_0^r rho(r') r'^2 dr'.
    total_mass : float
        Total shell mass M = m(R_2).
    R_1 : float
        Inner shell radius.
    R_2 : float
        Outer shell radius.
    """

    density: Callable[[Float[Array, ""]], Float[Array, ""]]
    radial_pressure: Callable[[Float[Array, ""]], Float[Array, ""]]
    tangential_pressure: Callable[[Float[Array, ""]], Float[Array, ""]]
    cumulative_mass: Callable[[Float[Array, ""]], Float[Array, ""]]
    total_mass: float
    R_1: float
    R_2: float


def _integrate_tov_pressure(
    rho_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    m_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    R_1: float,
    R_2: float,
    n_grid: int = 512,
) -> Callable[[Float[Array, ""]], Float[Array, ""]]:
    """Integrate the isotropic TOV equation inward from R_2.

    dp_r/dr = -(rho + p_r)(m + 4pi r^3 p_r) / (r(r - 2m))

    Boundary condition: p_r(R_2) = 0. Uses forward Euler via jax.lax.scan
    on a fine inward grid; sufficient for the smooth profiles here.

    Parameters
    ----------
    rho_fn : energy density callable.
    m_fn : cumulative mass callable.
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    n_grid : number of grid points.

    Returns
    -------
    callable r -> p_r(r)
    """
    import interpax

    r_grid = jnp.linspace(R_2, R_1, n_grid)
    dr = r_grid[1] - r_grid[0]

    rho_vals = jax.vmap(rho_fn)(r_grid)
    m_vals = jax.vmap(m_fn)(r_grid)

    def tov_rhs(r_val, p_val, rho_val, m_val):
        r_safe = jnp.maximum(r_val, 1e-30)
        denom = r_safe * (r_safe - 2.0 * m_val)
        denom_safe = jnp.where(jnp.abs(denom) < 1e-30, 1e-30, denom)
        numer = -(rho_val + p_val) * (m_val + 4.0 * jnp.pi * r_safe**3 * p_val)
        return numer / denom_safe

    def scan_step(p_current, inputs):
        r_val, rho_val, m_val = inputs
        dp = tov_rhs(r_val, p_current, rho_val, m_val) * dr
        p_next = jnp.maximum(p_current + dp, 0.0)
        return p_next, p_next

    _, p_scan = jax.lax.scan(scan_step, jnp.float64(0.0), (r_grid[:-1], rho_vals[:-1], m_vals[:-1]))
    p_all = jnp.concatenate([jnp.array([0.0]), p_scan])

    r_ascending = jnp.flip(r_grid)
    p_ascending = jnp.flip(p_all)

    def p_r_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        in_shell = (r >= R_1) & (r <= R_2)
        r_clamped = jnp.clip(r, R_1, R_2)
        p_val = interpax.interp1d(r_clamped, r_ascending, p_ascending, method="cubic")
        return jnp.where(in_shell, jnp.maximum(p_val, 0.0), 0.0)

    return p_r_fn


def _compute_cumulative_mass(
    rho_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    R_2: float,
    n_grid: int = 512,
) -> tuple[Callable[[Float[Array, ""]], Float[Array, ""]], float]:
    """Compute cumulative mass m(r) = 4pi int_0^r rho r'^2 dr'.

    Returns an interpolated callable and the total mass M.
    """
    import interpax

    r_max = R_2 * 1.1
    r_grid = jnp.linspace(0.0, r_max, n_grid)
    dr = r_grid[1] - r_grid[0]

    rho_vals = jax.vmap(rho_fn)(r_grid)
    integrand = 4.0 * jnp.pi * rho_vals * r_grid**2

    m_vals = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dr),
    ])

    total_mass = float(m_vals[-1])

    def m_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        return interpax.interp1d(
            jnp.clip(r, 0.0, r_max), r_grid, m_vals, method="cubic",
        )

    return m_fn, total_mass


def constant_density_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    rho_0: float = 1e-4,
    smooth_width: float | None = None,
) -> SShellSourceProfiles:
    """Constant-density shell with C2 smoothstep transitions.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    rho_0 : constant density in the shell.
    smooth_width : transition width (default: 0.05 * (R_2 - R_1)).

    Returns
    -------
    SShellSourceProfiles
    """
    sw = smooth_width if smooth_width is not None else 0.05 * (R_2 - R_1)

    def density(r: Float[Array, ""]) -> Float[Array, ""]:
        ramp_in = smoothstep_c2((r - (R_1 - sw)) / jnp.maximum(sw, 1e-12))
        ramp_out = smoothstep_c2(((R_2 + sw) - r) / jnp.maximum(sw, 1e-12))
        return rho_0 * ramp_in * ramp_out

    m_fn, total_mass = _compute_cumulative_mass(density, R_2)
    p_r_fn = _integrate_tov_pressure(density, m_fn, R_1, R_2)

    return SShellSourceProfiles(
        density=density,
        radial_pressure=p_r_fn,
        tangential_pressure=p_r_fn,
        cumulative_mass=m_fn,
        total_mass=total_mass,
        R_1=R_1,
        R_2=R_2,
    )


def parabolic_density_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    rho_max: float = 1e-4,
) -> SShellSourceProfiles:
    """Parabolic density: rho = rho_max * (1 - ((r-r_c)/Dr)^2).

    Naturally vanishes at R_1 and R_2, no smoothstep needed.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    rho_max : peak density at shell center.

    Returns
    -------
    SShellSourceProfiles
    """
    r_c = 0.5 * (R_1 + R_2)
    Delta_r = 0.5 * (R_2 - R_1)

    def density(r: Float[Array, ""]) -> Float[Array, ""]:
        t = (r - r_c) / Delta_r
        in_shell = (r >= R_1) & (r <= R_2)
        rho = rho_max * jnp.maximum(1.0 - t**2, 0.0)
        return jnp.where(in_shell, rho, 0.0)

    m_fn, total_mass = _compute_cumulative_mass(density, R_2)
    p_r_fn = _integrate_tov_pressure(density, m_fn, R_1, R_2)

    return SShellSourceProfiles(
        density=density,
        radial_pressure=p_r_fn,
        tangential_pressure=p_r_fn,
        cumulative_mass=m_fn,
        total_mass=total_mass,
        R_1=R_1,
        R_2=R_2,
    )


def bernstein_density_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    coeffs: Float[Array, "N"] | None = None,
) -> SShellSourceProfiles:
    """Bernstein-polynomial density with compact support.

    rho(r) = sum_k c_k B_{n,k}(t) where t = (r - R_1) / (R_2 - R_1).
    Endpoint coefficients c_0 = c_n = 0 enforce compact support.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    coeffs : Bernstein control coefficients. Default: a 6-point
        parabolic-like profile with compact support.

    Returns
    -------
    SShellSourceProfiles
    """
    from jax.scipy.special import gammaln

    if coeffs is None:
        coeffs = jnp.array([0.0, 0.5e-4, 1.0e-4, 1.0e-4, 0.5e-4, 0.0])
    else:
        coeffs = jnp.asarray(coeffs)
        coeffs = coeffs.at[0].set(0.0)
        coeffs = coeffs.at[-1].set(0.0)

    n = coeffs.shape[0] - 1
    k_idx = jnp.arange(n + 1, dtype=jnp.float64)

    def density(r: Float[Array, ""]) -> Float[Array, ""]:
        t = jnp.clip((r - R_1) / (R_2 - R_1), 0.0, 1.0)
        in_shell = (r >= R_1) & (r <= R_2)

        log_binom = (
            gammaln(n + 1.0)
            - gammaln(k_idx + 1.0)
            - gammaln(n - k_idx + 1.0)
        )
        binom = jnp.exp(log_binom)
        basis_vals = binom * (t ** k_idx) * ((1.0 - t) ** (n - k_idx))
        rho = jnp.sum(coeffs * basis_vals)
        return jnp.where(in_shell, jnp.maximum(rho, 0.0), 0.0)

    m_fn, total_mass = _compute_cumulative_mass(density, R_2)
    p_r_fn = _integrate_tov_pressure(density, m_fn, R_1, R_2)

    return SShellSourceProfiles(
        density=density,
        radial_pressure=p_r_fn,
        tangential_pressure=p_r_fn,
        cumulative_mass=m_fn,
        total_mass=total_mass,
        R_1=R_1,
        R_2=R_2,
    )
