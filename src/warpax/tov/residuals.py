"""Anisotropic TOV equilibrium checker.

Implements the generalized Bowers-Liang anisotropic TOV equation:

    dp_r/dr + (rho + p_r) Phi' - 2(p_t - p_r)/r = 0

where Phi' = dPhi/dr for g_tt = -e^{2Phi}, or more generally is derived
from the metric lapse function.

Uses JAX autodiff for derivatives instead of finite differences.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def tov_residual(
    r: Float[Array, ""],
    rho: Callable[[Float[Array, ""], ], Float[Array, ""]],
    p_r: Callable[[Float[Array, ""], ], Float[Array, ""]],
    p_t: Callable[[Float[Array, ""], ], Float[Array, ""]],
    Phi_prime: Float[Array, ""],
) -> Float[Array, ""]:
    """Anisotropic TOV equilibrium residual.

    Residual = dp_r/dr + (rho + p_r) Phi' - 2(p_t - p_r)/r

    For a valid equilibrium shell, this should be ~0.

    Parameters
    ----------
    r : radial coordinate
    rho : energy density function rho(r)
    p_r : radial pressure function p_r(r)
    p_t : tangential pressure function p_t(r)
    Phi_prime : dPhi/dr where g_tt = -exp(2Phi)

    Returns
    -------
    TOV residual (should be ~0 for equilibrium)
    """
    r_arr = jnp.asarray(r, dtype=jnp.float64)

    dp_r = jax.grad(p_r)(r_arr)

    rho_val = rho(r_arr)
    p_r_val = p_r(r_arr)
    p_t_val = p_t(r_arr)

    r_safe = jnp.maximum(r_arr, 1e-10)
    residual = dp_r + (rho_val + p_r_val) * Phi_prime - 2.0 * (p_t_val - p_r_val) / r_safe
    return residual


def tov_residual_from_metric(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    r: Float[Array, ""],
    rho: Callable[[Float[Array, ""], ], Float[Array, ""]],
    p_r: Callable[[Float[Array, ""], ], Float[Array, ""]],
    p_t: Callable[[Float[Array, ""], ], Float[Array, ""]],
) -> Float[Array, ""]:
    """Anisotropic TOV residual with Phi' extracted from the metric.

    For a static spherically symmetric metric g_tt = -e^{2Phi(r)},
    computes Phi'(r) = d/dr [0.5 ln(-g_tt)] via autodiff.

    Parameters
    ----------
    metric_fn : spacetime metric callable
    r : radial coordinate
    rho, p_r, p_t : matter profile functions

    Returns
    -------
    TOV residual
    """
    r_arr = jnp.asarray(r, dtype=jnp.float64)

    # Probe ``g_tt`` at the requested radius to fail loud on signature
    # flips: the (-+++) convention requires ``g_tt < 0`` everywhere
    # outside the horizon; if a caller passes a metric in (+---) or a
    # CTC region the residual identity below silently produces garbage.
    probe = metric_fn(jnp.array([0.0, r_arr, 0.0, 0.0], dtype=jnp.float64))
    if not bool(probe[0, 0] < 0.0):
        raise ValueError(
            f"tov_residual_from_metric: g_tt({float(r_arr)}) = {float(probe[0, 0])} "
            "is not negative; the routine assumes the (-+++) signature. "
            "Pass a metric in this convention or move the probe outside "
            "any horizon / CTC region."
        )

    def Phi_of_r(rr: Float[Array, ""]) -> Float[Array, ""]:
        coords = jnp.array([0.0, rr, 0.0, 0.0], dtype=jnp.float64)
        g = metric_fn(coords)
        return 0.5 * jnp.log(jnp.maximum(-g[0, 0], 1e-30))

    Phi_prime = jax.grad(Phi_of_r)(r_arr)
    return tov_residual(r_arr, rho, p_r, p_t, Phi_prime)
