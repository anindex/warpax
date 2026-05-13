"""Source profiles for the T-shell ansatz.

Extends S-shell profiles with a spatial velocity field v^x(r) that
tilts the matter 4-velocity relative to the hypersurface normal.

Tilted fluid decomposition (u^a = Gamma (n^a + v^a), v^a n_a = 0):

    E   = Gamma^2 (rho + p v^2)                       -- Eulerian energy
    S_i = Gamma^2 (rho + p) v_i                       -- momentum density
    S_{ij} = Gamma^2 (rho + p) v_i v_j + p gamma_{ij} -- spatial stress

Three velocity profile families:
1. Constant velocity with C2 smoothstep compact support
2. Parabolic velocity vanishing at boundaries
3. Bernstein-polynomial velocity with fixed endpoint coefficients
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.transitions import smoothstep_c2
from .sshell_profiles import (
    SShellSourceProfiles,
    _compute_cumulative_mass,
    constant_density_profiles,
    parabolic_density_profiles,
    bernstein_density_profiles,
)


class TShellSourceProfiles(NamedTuple):
    """Source profiles for the T-shell ansatz.

    Attributes
    ----------
    density : callable r -> rho(r)
        Comoving energy density with compact support in [R_1, R_2].
    radial_pressure : callable r -> p_r(r)
        Comoving radial pressure from TOV integration.
    tangential_pressure : callable r -> p_t(r)
        Comoving tangential pressure (= p_r for isotropic case).
    cumulative_mass : callable r -> m(r)
        Cumulative mass from Eulerian energy density.
    total_mass : float
        Total shell mass M = m(R_2).
    R_1 : float
        Inner shell radius.
    R_2 : float
        Outer shell radius.
    velocity_x : callable r -> v^x(r)
        Spatial velocity profile (x-directed) with compact support.
    lorentz_factor : callable r -> Gamma(r)
        Lorentz factor Gamma = 1/sqrt(1 - v^2).
    eulerian_energy : callable r -> E(r)
        Eulerian energy density E = Gamma^2 (rho + p v^2).
    momentum_density_x : callable r -> S_x(r)
        Eulerian momentum density S_x = Gamma^2 (rho + p) v^x.
    """

    density: Callable[[Float[Array, ""]], Float[Array, ""]]
    radial_pressure: Callable[[Float[Array, ""]], Float[Array, ""]]
    tangential_pressure: Callable[[Float[Array, ""]], Float[Array, ""]]
    cumulative_mass: Callable[[Float[Array, ""]], Float[Array, ""]]
    total_mass: float
    R_1: float
    R_2: float
    velocity_x: Callable[[Float[Array, ""]], Float[Array, ""]]
    lorentz_factor: Callable[[Float[Array, ""]], Float[Array, ""]]
    eulerian_energy: Callable[[Float[Array, ""]], Float[Array, ""]]
    momentum_density_x: Callable[[Float[Array, ""]], Float[Array, ""]]


def _build_eulerian_projections(
    rho_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    p_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    v_x_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
) -> tuple[
    Callable[[Float[Array, ""]], Float[Array, ""]],
    Callable[[Float[Array, ""]], Float[Array, ""]],
    Callable[[Float[Array, ""]], Float[Array, ""]],
]:
    """Build Eulerian projection functions from comoving quantities.

    Returns (lorentz_fn, E_fn, S_x_fn).
    """

    def lorentz_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        v = v_x_fn(r)
        return 1.0 / jnp.sqrt(jnp.maximum(1.0 - v**2, 1e-30))

    def E_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        rho = rho_fn(r)
        p = p_fn(r)
        v = v_x_fn(r)
        v_sq = v**2
        Gamma_sq = 1.0 / jnp.maximum(1.0 - v_sq, 1e-30)
        return Gamma_sq * (rho + p * v_sq)

    def S_x_fn(r: Float[Array, ""]) -> Float[Array, ""]:
        rho = rho_fn(r)
        p = p_fn(r)
        v = v_x_fn(r)
        Gamma_sq = 1.0 / jnp.maximum(1.0 - v**2, 1e-30)
        return Gamma_sq * (rho + p) * v

    return lorentz_fn, E_fn, S_x_fn


def _tshell_from_sshell(
    sshell: SShellSourceProfiles,
    v_x_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
) -> TShellSourceProfiles:
    """Promote an S-shell profile to a T-shell profile with a velocity field.

    Recomputes cumulative mass using Eulerian E(r) instead of comoving rho(r).
    """
    lorentz_fn, E_fn, S_x_fn = _build_eulerian_projections(
        sshell.density, sshell.radial_pressure, v_x_fn,
    )
    m_fn, total_mass = _compute_cumulative_mass(E_fn, sshell.R_2)

    return TShellSourceProfiles(
        density=sshell.density,
        radial_pressure=sshell.radial_pressure,
        tangential_pressure=sshell.tangential_pressure,
        cumulative_mass=m_fn,
        total_mass=total_mass,
        R_1=sshell.R_1,
        R_2=sshell.R_2,
        velocity_x=v_x_fn,
        lorentz_factor=lorentz_fn,
        eulerian_energy=E_fn,
        momentum_density_x=S_x_fn,
    )


def constant_velocity_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    rho_0: float = 1e-4,
    v_0: float = 0.1,
    smooth_width: float | None = None,
) -> TShellSourceProfiles:
    """Constant density + constant velocity T-shell profiles.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    rho_0 : constant density in the shell.
    v_0 : peak spatial velocity (x-directed). Must satisfy |v_0| < 1.
    smooth_width : transition width (default: 0.05 * (R_2 - R_1)).
    """
    if abs(v_0) >= 1.0:
        raise ValueError(
            f"Velocity v_0={v_0} is superluminal. Must satisfy |v_0| < 1."
        )

    sshell = constant_density_profiles(
        R_1=R_1, R_2=R_2, rho_0=rho_0, smooth_width=smooth_width,
    )
    sw = smooth_width if smooth_width is not None else 0.05 * (R_2 - R_1)

    def velocity_x(r: Float[Array, ""]) -> Float[Array, ""]:
        ramp_in = smoothstep_c2((r - (R_1 - sw)) / jnp.maximum(sw, 1e-12))
        ramp_out = smoothstep_c2(((R_2 + sw) - r) / jnp.maximum(sw, 1e-12))
        return v_0 * ramp_in * ramp_out

    return _tshell_from_sshell(sshell, velocity_x)


def parabolic_velocity_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    rho_max: float = 1e-4,
    v_0: float = 0.1,
) -> TShellSourceProfiles:
    """Parabolic density + parabolic velocity T-shell profiles.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    rho_max : peak density at shell center.
    v_0 : peak spatial velocity at shell center.
    """
    if abs(v_0) >= 1.0:
        raise ValueError(
            f"Velocity v_0={v_0} is superluminal. Must satisfy |v_0| < 1."
        )

    sshell = parabolic_density_profiles(R_1=R_1, R_2=R_2, rho_max=rho_max)
    r_c = 0.5 * (R_1 + R_2)
    Delta_r = 0.5 * (R_2 - R_1)

    def velocity_x(r: Float[Array, ""]) -> Float[Array, ""]:
        t = (r - r_c) / Delta_r
        in_shell = (r >= R_1) & (r <= R_2)
        v = v_0 * jnp.maximum(1.0 - t**2, 0.0)
        return jnp.where(in_shell, v, 0.0)

    return _tshell_from_sshell(sshell, velocity_x)


def bernstein_velocity_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    density_coeffs: Float[Array, "N"] | None = None,
    velocity_coeffs: Float[Array, "M"] | None = None,
    v_0: float = 0.1,
) -> TShellSourceProfiles:
    """Bernstein-polynomial density + velocity T-shell profiles.

    Parameters
    ----------
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    density_coeffs : Bernstein density coefficients (default: 6-point parabolic).
    velocity_coeffs : Bernstein velocity coefficients (default: 6-point parabolic).
    v_0 : peak velocity scale factor.
    """
    from jax.scipy.special import gammaln

    if abs(v_0) >= 1.0:
        raise ValueError(
            f"Velocity v_0={v_0} is superluminal. Must satisfy |v_0| < 1."
        )

    sshell = bernstein_density_profiles(R_1=R_1, R_2=R_2, coeffs=density_coeffs)

    if velocity_coeffs is None:
        velocity_coeffs = jnp.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    else:
        velocity_coeffs = jnp.asarray(velocity_coeffs)
        velocity_coeffs = velocity_coeffs.at[0].set(0.0)
        velocity_coeffs = velocity_coeffs.at[-1].set(0.0)

    m = velocity_coeffs.shape[0] - 1
    k_idx = jnp.arange(m + 1, dtype=jnp.float64)

    def velocity_x(r: Float[Array, ""]) -> Float[Array, ""]:
        t = jnp.clip((r - R_1) / (R_2 - R_1), 0.0, 1.0)
        in_shell = (r >= R_1) & (r <= R_2)
        log_binom = (
            gammaln(m + 1.0)
            - gammaln(k_idx + 1.0)
            - gammaln(m - k_idx + 1.0)
        )
        binom = jnp.exp(log_binom)
        basis_vals = binom * (t ** k_idx) * ((1.0 - t) ** (m - k_idx))
        v = v_0 * jnp.sum(velocity_coeffs * basis_vals)
        return jnp.where(in_shell, v, 0.0)

    return _tshell_from_sshell(sshell, velocity_x)
