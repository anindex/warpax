"""Invariant transport diagnostics for warp shell spacetimes.

Three gauge-invariant observables for admissibility assessment:

1. Dtau_gamma -- null round-trip time asymmetry
2. A_geo      -- geodesic deviation diagnostic
3. B          -- blueshift hazard functional

All use the existing warpax.geodesics infrastructure (Diffrax integration,
Jacobi deviation, blueshift observables).
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geodesics.initial_conditions import null_ic
from ..geodesics.integrator import integrate_geodesic
from ..geodesics.deviation import tidal_eigenvalues


def null_round_trip_asymmetry(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    emitter: Float[Array, "4"],
    receiver: Float[Array, "4"],
    *,
    tau_max: float = 50.0,
    num_points: int = 500,
    dt0: float = 0.01,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Float[Array, ""]:
    """Compute null round-trip time asymmetry between emitter and receiver.

    Deltatau_gamma = tau_forward - tau_backward

    where tau_forward is the coordinate time for a null geodesic from emitter
    to receiver, and tau_backward is the return trip.

    For static, symmetric spacetimes, Deltatau_gamma = 0.
    For warp bubbles with genuine transport, Dtau_gamma != 0.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    emitter : emitter coordinates (t, x, y, z)
    receiver : receiver coordinates (t, x, y, z)
    tau_max : maximum affine parameter for integration
    num_points : saved points per leg
    dt0 : initial step size
    rtol, atol : integration tolerances

    Returns
    -------
    Deltatau_gamma : round-trip asymmetry (0 for symmetric spacetimes)
    """
    # Spatial direction from emitter to receiver
    dx = receiver[1:] - emitter[1:]
    dist = jnp.linalg.norm(dx)

    # Handle zero-distance case
    is_zero_dist = dist < 1e-15
    dx_safe = jnp.where(is_zero_dist, jnp.array([1.0, 0.0, 0.0]), dx / jnp.maximum(dist, 1e-15))

    # Forward leg: emitter -> receiver
    _, k0_fwd = null_ic(metric_fn, emitter, dx_safe)
    sol_fwd = integrate_geodesic(
        metric_fn, emitter, k0_fwd,
        tau_span=(0.0, tau_max),
        num_points=num_points,
        dt0=dt0, rtol=rtol, atol=atol,
    )

    # Find arrival: closest approach to receiver position
    spatial_dist_fwd = jnp.linalg.norm(sol_fwd.positions[:, 1:] - receiver[1:], axis=1)
    idx_fwd = jnp.argmin(spatial_dist_fwd)
    t_arrive_fwd = sol_fwd.positions[idx_fwd, 0]
    tau_fwd = t_arrive_fwd - emitter[0]

    # Backward leg: receiver -> emitter
    arrival_pos = sol_fwd.positions[idx_fwd]  # Start from where forward leg arrived
    dx_back = emitter[1:] - arrival_pos[1:]
    dx_back_safe = jnp.where(
        jnp.linalg.norm(dx_back) < 1e-15,
        -dx_safe,
        dx_back / jnp.maximum(jnp.linalg.norm(dx_back), 1e-15),
    )
    _, k0_bwd = null_ic(metric_fn, arrival_pos, dx_back_safe)
    sol_bwd = integrate_geodesic(
        metric_fn, arrival_pos, k0_bwd,
        tau_span=(0.0, tau_max),
        num_points=num_points,
        dt0=dt0, rtol=rtol, atol=atol,
    )

    # Find return: closest approach to emitter position
    spatial_dist_bwd = jnp.linalg.norm(sol_bwd.positions[:, 1:] - emitter[1:], axis=1)
    idx_bwd = jnp.argmin(spatial_dist_bwd)
    t_arrive_bwd = sol_bwd.positions[idx_bwd, 0]
    tau_bwd = t_arrive_bwd - arrival_pos[0]

    # Asymmetry
    delta_tau = jnp.where(is_zero_dist, 0.0, tau_fwd - tau_bwd)
    return delta_tau


def geodesic_deviation_diagnostic(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    emitter: Float[Array, "4"],
    *,
    velocity: Float[Array, "4"] | None = None,
) -> Float[Array, ""]:
    """Compute maximum tidal acceleration magnitude at a point.

    A_geo = max eigenvalue of |K^mu_rho| = |R^mu_{nu rho sigma} v^nu v^sigma|

    For Minkowski, A_geo = 0 (no tidal forces).

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    emitter : spacetime position
    velocity : optional 4-velocity (defaults to static observer)

    Returns
    -------
    A_geo : maximum tidal eigenvalue magnitude
    """
    if velocity is None:
        # Static observer: u^mu = (1/sqrt(-g_00), 0, 0, 0)
        g = metric_fn(emitter)
        u_t = 1.0 / jnp.sqrt(jnp.maximum(-g[0, 0], 1e-30))
        velocity = jnp.array([u_t, 0.0, 0.0, 0.0], dtype=emitter.dtype)

    eigs = tidal_eigenvalues(metric_fn, emitter, velocity)
    return jnp.max(jnp.abs(eigs))


def blueshift_hazard(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    emitter: Float[Array, "4"],
    *,
    n_directions: int = 6,
    tau_max: float = 20.0,
    num_points: int = 200,
) -> Float[Array, ""]:
    """Compute maximum blueshift hazard over null geodesic directions.

    B = max_gamma |log(omega_received / omega_emitted)|

    Fires null geodesics in multiple directions from the emitter and
    computes the maximum frequency ratio encountered.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    emitter : emission point
    n_directions : number of spatial directions to probe
    tau_max : maximum affine parameter
    num_points : saved points per geodesic

    Returns
    -------
    B : blueshift hazard scalar
    """
    g = metric_fn(emitter)
    u_t = 1.0 / jnp.sqrt(jnp.maximum(-g[0, 0], 1e-30))
    u_observer = jnp.array([u_t, 0.0, 0.0, 0.0], dtype=emitter.dtype)

    # Probe directions: +/-x, +/-y, +/-z (or fewer)
    directions = [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([-1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, -1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([0.0, 0.0, -1.0]),
    ][:n_directions]

    max_log_shift = jnp.float64(0.0)

    for d in directions:
        _, k0 = null_ic(metric_fn, emitter, d)

        sol = integrate_geodesic(
            metric_fn, emitter, k0,
            tau_span=(0.0, tau_max),
            num_points=num_points,
            dt0=0.01, rtol=1e-8, atol=1e-8,
        )

        # Compute blueshift at each point relative to emission
        # omega = -g_{ab} k^a u^b for a static observer
        omega_emit = -jnp.einsum("ab,a,b", g, k0, u_observer)

        def omega_at(x, k):
            g_at = metric_fn(x)
            u_t_at = 1.0 / jnp.sqrt(jnp.maximum(-g_at[0, 0], 1e-30))
            u_at = jnp.array([u_t_at, 0.0, 0.0, 0.0], dtype=x.dtype)
            return -jnp.einsum("ab,a,b", g_at, k, u_at)

        omegas = jax.vmap(omega_at)(sol.positions, sol.velocities)
        log_shifts = jnp.abs(jnp.log(jnp.maximum(jnp.abs(omegas / omega_emit), 1e-30)))
        max_this = jnp.max(log_shifts)
        max_log_shift = jnp.maximum(max_log_shift, max_this)

    return max_log_shift
