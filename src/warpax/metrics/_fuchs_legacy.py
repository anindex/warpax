"""Legacy Fuchs analytical shell profiles (retained for backward compat).

The canonical FuchsMetric class is now in ``fuchs_construction.py``.
This module provides only the analytical shell profile helpers:

- ``FuchsShellProfiles``: source profiles (density, pressure, mass) as callables
- ``fuchs_shell_profiles()``: factory with paper-default parameters
- ``fuchs_input_stress_energy()``: construct T_input from profiles
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..metrics.warpshell import WarpShellPhysical


# ---------------------------------------------------------------------------
# Shell source physics
# ---------------------------------------------------------------------------


class FuchsShellProfiles(NamedTuple):
    """Radial source profiles for the Fuchs shell.

    These profiles describe the matter content of the shell as
    functions of radial coordinate r. All functions map a scalar
    r to a scalar output.

    Attributes
    ----------
    density : callable r -> rho(r)
        Energy density profile. Zero outside [R_1, R_2].
    radial_pressure : callable r -> p_r(r)
        Radial pressure. Zero outside shell and at r=R_2.
    tangential_pressure : callable r -> p_t(r)
        Tangential (hoop) pressure. Differs from p_r near R_1.
    cumulative_mass : callable r -> m(r)
        Cumulative mass: m(r) = 4pi int_0^r rho(r') r'^2 dr'.
    total_mass : float
        Total shell mass M (geometric units).
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


def _constant_density_shell_profiles(
    R_1: float,
    R_2: float,
    r_s_param: float,
) -> FuchsShellProfiles:
    """Construct analytical shell profiles for the Schwarzschild-shell model.

    This implements the constant-density approximation from Section 3.1
    of the Fuchs paper, with the analytical Schwarzschild-shell form.
    The cumulative mass profile is:

        m(r) = 0               for r < R_1
        m(r) = M * (r^3 - R_1^3) / (R_2^3 - R_1^3)  for R_1 <= r <= R_2
        m(r) = M               for r > R_2

    where M = r_s_param / 2 (geometric units, G=c=1).

    The density is the derivative: rho(r) = (1/4pi r^2) dm/dr.

    The radial pressure is obtained from the TOV equation with boundary
    condition P(R_2) = 0. For a constant-density shell this gives the
    interior solution (Carroll Eq. 5.157 adapted):

        dp_r/dr = -(rho + p_r)(m + 4pi r^3 p_r) / (r(r - 2m))

    integrated inward from r = R_2.

    The tangential pressure differs from radial by the anisotropic
    TOV term (Bowers-Liang): p_t = p_r + (r/2) dp_r/dr + ...

    For the analytical model, we use p_t = p_r as a first approximation
    (isotropic). The full anisotropic profiles from the paper's iterative
    smoothing are needed for source-consistency checks.

    Parameters
    ----------
    R_1 : inner shell radius
    R_2 : outer shell radius
    r_s_param : Schwarzschild radius parameter (2M in geometric units)

    Returns
    -------
    FuchsShellProfiles
        Named tuple with density, pressure, and mass profile callables.
    """
    M_total = r_s_param / 2.0  # M = r_s / 2 in geometric units
    shell_volume_factor = R_2**3 - R_1**3  # proportional to shell volume

    # Constant density in the shell (Eq. 18 of paper)
    rho_0 = 3.0 * M_total / (4.0 * jnp.pi * shell_volume_factor)

    def density(r: Float[Array, ""]) -> Float[Array, ""]:
        in_shell = (r >= R_1) & (r <= R_2)
        return jnp.where(in_shell, rho_0, 0.0)

    def cumulative_mass(r: Float[Array, ""]) -> Float[Array, ""]:
        r_clamp = jnp.clip(r, R_1, R_2)
        m_shell = M_total * (r_clamp**3 - R_1**3) / shell_volume_factor
        return jnp.where(r < R_1, 0.0, jnp.where(r > R_2, M_total, m_shell))

    # Radial pressure via TOV (isotropic approximation)
    # For constant density, integrated analytically (Oppenheimer-Volkoff):
    #   P(r) = rho * [sqrt(1-2m(r)/r) - sqrt(1-2M/R_2)] /
    #          [sqrt(1-2M/R_2) - sqrt(1-2m(r)/r)]
    # This is a simplified form; the full paper uses numerical integration.
    # We use a simpler approximation: linear falloff with P(R_2) = 0.
    def radial_pressure(r: Float[Array, ""]) -> Float[Array, ""]:
        in_shell = (r >= R_1) & (r <= R_2)
        m_r = cumulative_mass(r)
        r_safe = jnp.maximum(r, 1e-30)
        # Newtonian hydrostatic equilibrium approximation:
        # P(r) ~ (2pi/3) rho_0^2 (R_2^2 - r^2) / (1 - 2*m_r/r_safe)
        # with relativistic correction factor
        compactness = 2.0 * m_r / r_safe
        compactness_safe = jnp.minimum(compactness, 0.99)
        factor = 1.0 / (1.0 - compactness_safe)
        p_hydro = (2.0 * jnp.pi / 3.0) * rho_0**2 * (R_2**2 - r**2) * factor
        p_hydro = jnp.maximum(p_hydro, 0.0)
        return jnp.where(in_shell, p_hydro, 0.0)

    # Tangential pressure = radial pressure (isotropic approximation)
    # The paper's anisotropic corrections arise from iterative smoothing
    # and differ mainly near R_1 (hoop stress).
    def tangential_pressure(r: Float[Array, ""]) -> Float[Array, ""]:
        return radial_pressure(r)

    return FuchsShellProfiles(
        density=density,
        radial_pressure=radial_pressure,
        tangential_pressure=tangential_pressure,
        cumulative_mass=cumulative_mass,
        total_mass=M_total,
        R_1=R_1,
        R_2=R_2,
    )


# _FuchsAnalytical -- retained for backward compatibility in tests


class _FuchsAnalytical(WarpShellPhysical):
    """Legacy analytical Schwarzschild-shell approximation.

    Retained for backward compatibility. The canonical FuchsMetric is
    in fuchs_construction.py.
    """

    def name(self) -> str:
        return "Fuchs-Analytical"

    def shell_profiles(self) -> FuchsShellProfiles:
        return _constant_density_shell_profiles(
            self.R_1, self.R_2, self.r_s_param
        )


def _fuchs_analytical_default() -> _FuchsAnalytical:
    """Legacy factory, returns analytical Fuchs metric."""
    return _FuchsAnalytical(
        v_s=0.02,
        R_1=10.0,
        R_2=20.0,
        R_b=1.0,
        r_s_param=5.0,
        transition_order=2,
    )


def fuchs_shell_profiles(
    R_1: float = 10.0,
    R_2: float = 20.0,
    r_s_param: float = 5.0,
) -> FuchsShellProfiles:
    """Factory for Fuchs shell source profiles.

    Convenience wrapper around ``_constant_density_shell_profiles`` with
    paper-default parameters. Use these profiles for source-consistency
    checks (comparing T_input against G_{ab}/8pi).

    Parameters
    ----------
    R_1 : inner shell radius (default: 10.0)
    R_2 : outer shell radius (default: 20.0)
    r_s_param : Schwarzschild radius parameter (default: 5.0)

    Returns
    -------
    FuchsShellProfiles
    """
    return _constant_density_shell_profiles(R_1, R_2, r_s_param)


# ---------------------------------------------------------------------------
# Input stress-energy from shell profiles
# ---------------------------------------------------------------------------


def fuchs_input_stress_energy(
    metric: "_FuchsAnalytical",
    coords: Float[Array, "4"],
) -> Float[Array, "4 4"]:
    """Construct the claimed anisotropic-fluid T_input at a spacetime point.

    T_{ab} = (rho + p_r) u_a u_b + p_r g_{ab} + (p_t - p_r) s_a s_b

    where u^a = (1/alpha, 0, 0, 0) is the static-observer 4-velocity
    and s^a is the unit radial spacelike vector orthogonal to u^a.

    Parameters
    ----------
    metric : FuchsMetric
    coords : Float[Array, "4"]
        (t, x, y, z).

    Returns
    -------
    Float[Array, "4 4"]
        Covariant T_input_{ab}.
    """
    profiles = metric.shell_profiles()
    g = metric(coords)

    t, x, y, z = coords
    x_rel = x - metric.v_s * t
    r = jnp.sqrt(x_rel**2 + y**2 + z**2)

    rho = profiles.density(r)
    p_r = profiles.radial_pressure(r)
    p_t = profiles.tangential_pressure(r)

    alpha = metric.lapse(coords)

    # Static observer
    u_up = jnp.array([1.0 / alpha, 0.0, 0.0, 0.0])
    u_down = g @ u_up

    # Radial unit spacelike vector, orthogonalised against u
    r_safe = jnp.maximum(r, 1e-30)
    e_r = jnp.array([0.0, x_rel / r_safe, y / r_safe, z / r_safe])
    e_r_dot_u = jnp.dot(g @ e_r, u_up)
    s_up = e_r - e_r_dot_u * u_up
    s_norm_sq = jnp.maximum(jnp.dot(g @ s_up, s_up), 1e-30)
    s_up = s_up / jnp.sqrt(s_norm_sq)
    s_down = g @ s_up

    T_input = (
        (rho + p_r) * jnp.outer(u_down, u_down)
        + p_r * g
        + (p_t - p_r) * jnp.outer(s_down, s_down)
    )

    return 0.5 * (T_input + T_input.T)

