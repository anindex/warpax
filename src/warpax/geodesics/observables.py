"""Post-processing observables for geodesic trajectories.

Provides physical observables extracted from geodesic integration results:
- velocity_norm: g_ab v^a v^b conservation check (should be -1 or 0)
- monitor_conservation: norm drift along entire trajectory
- compute_blueshift: frequency ratio from 4-velocity dot products
- blueshift_along_trajectory: blueshift profile along a null geodesic
- proper_time_elapsed: cumulative proper time via trapezoidal integration
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Velocity norm and conservation monitoring
# ---------------------------------------------------------------------------


def velocity_norm(
    metric_fn: object,
    x: Float[Array, "4"],
    v: Float[Array, "4"],
) -> Float[Array, ""]:
    """Compute g_ab v^a v^b at a single spacetime point.

    The result should be -1 for timelike geodesics and 0 for null geodesics.
    Deviation from these values indicates numerical integration error.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable: coords (4,) -> g_ab (4,4).
    x : Float[Array, "4"]
        Spacetime position (t, x, y, z).
    v : Float[Array, "4"]
        4-velocity at x.

    Returns
    -------
    Float[Array, ""]
        Scalar norm g_ab v^a v^b.
    """
    g = metric_fn(x)  # (4, 4)
    return jnp.einsum("ab,a,b", g, v, v)


def monitor_conservation(
    metric_fn: object,
    sol: object,
) -> Float[Array, "N"]:
    """Check 4-velocity norm conservation along a geodesic trajectory.

    Computes velocity_norm at each saved point using vmap. For timelike
    geodesics, the norm should stay near -1 throughout integration.
    For null geodesics, it should stay near 0.

    With rtol=atol=1e-10, maximum drift should be < 1e-8 for moderate
    integration lengths.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    sol : GeodesicResult or DeviationResult
        Integration result with ``.positions`` (N, 4) and ``.velocities`` (N, 4).

    Returns
    -------
    Float[Array, "N"]
        Array of g_ab v^a v^b at each saved point.
    """
    def norm_at_point(x: Float[Array, "4"], v: Float[Array, "4"]) -> Float[Array, ""]:
        return velocity_norm(metric_fn, x, v)

    return jax.vmap(norm_at_point)(sol.positions, sol.velocities)


# ---------------------------------------------------------------------------
# Blueshift computation
# ---------------------------------------------------------------------------


def compute_blueshift(
    metric_fn: object,
    k_emit: Float[Array, "4"],
    u_emit: Float[Array, "4"],
    x_emit: Float[Array, "4"],
    k_recv: Float[Array, "4"],
    u_recv: Float[Array, "4"],
    x_recv: Float[Array, "4"],
) -> Float[Array, ""]:
    """Compute blueshift factor between emission and reception events.

    The observed frequency of a photon by an observer with 4-velocity u^a is:
        omega = -g_{ab} k^a u^b

    The blueshift factor (1 + z) is the ratio of received to emitted frequency:
        1 + z = omega_recv / omega_emit

    z > 0 indicates blueshift (higher frequency at receiver).
    z < 0 indicates redshift (lower frequency at receiver).

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    k_emit : Float[Array, "4"]
        Photon 4-momentum at emission event.
    u_emit : Float[Array, "4"]
        Observer 4-velocity at emission event.
    x_emit : Float[Array, "4"]
        Spacetime coordinates at emission event.
    k_recv : Float[Array, "4"]
        Photon 4-momentum at reception event.
    u_recv : Float[Array, "4"]
        Observer 4-velocity at reception event.
    x_recv : Float[Array, "4"]
        Spacetime coordinates at reception event.

    Returns
    -------
    Float[Array, ""]
        Blueshift factor (1 + z) = omega_recv / omega_emit.
    """
    g_emit = metric_fn(x_emit)
    g_recv = metric_fn(x_recv)

    omega_emit = -jnp.einsum("ab,a,b", g_emit, k_emit, u_emit)
    omega_recv = -jnp.einsum("ab,a,b", g_recv, k_recv, u_recv)

    return omega_recv / omega_emit


def blueshift_along_trajectory(
    metric_fn: object,
    null_sol: object,
    observer_velocity_fn: Callable[[Float[Array, "4"]], Float[Array, "4"]],
) -> Float[Array, "N"]:
    """Compute blueshift profile along a null geodesic trajectory.

    At each saved point along the null geodesic, computes the blueshift factor
    relative to the first point (emission event). The observer 4-velocity at
    each point is provided by observer_velocity_fn.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    null_sol : GeodesicResult
        Null geodesic integration result with ``.positions`` (N, 4) and
        ``.velocities`` (N, 4). The velocities are the photon 4-momenta k^mu.
    observer_velocity_fn : callable
        Function mapping position (4,) -> observer 4-velocity (4,).
        For example, a static observer: lambda x: jnp.array([1/sqrt(-g00), 0, 0, 0]).

    Returns
    -------
    Float[Array, "N"]
        Blueshift factor (1 + z) at each saved point relative to the first point.
    """
    # Emission event (first saved point)
    x_emit = null_sol.positions[0]
    k_emit = null_sol.velocities[0]
    u_emit = observer_velocity_fn(x_emit)

    g_emit = metric_fn(x_emit)
    omega_emit = -jnp.einsum("ab,a,b", g_emit, k_emit, u_emit)

    def blueshift_at_point(
        x: Float[Array, "4"], k: Float[Array, "4"]
    ) -> Float[Array, ""]:
        g = metric_fn(x)
        u = observer_velocity_fn(x)
        omega = -jnp.einsum("ab,a,b", g, k, u)
        return omega / omega_emit

    return jax.vmap(blueshift_at_point)(null_sol.positions, null_sol.velocities)


# ---------------------------------------------------------------------------
# Proper time computation
# ---------------------------------------------------------------------------


def proper_time_elapsed(
    metric_fn: object,
    sol: object,
) -> Float[Array, "N"]:
    """Compute cumulative proper time along a timelike geodesic trajectory.

    Uses the trapezoidal rule to integrate:
        dtau = sqrt(-g_ab dx^a dx^b)

    between successive saved points. Returns a cumulative array starting at 0.

    For geodesics parameterized by proper time, this should match the affine
    parameter ts (up to numerical integration error from the trapezoidal rule
    on the saved points).

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    sol : GeodesicResult or DeviationResult
        Integration result with ``.positions`` (N, 4) and ``.ts`` (N,).

    Returns
    -------
    Float[Array, "N"]
        Cumulative proper time at each saved point. First element is 0.
    """
    positions = sol.positions  # (N, 4)
    N = positions.shape[0]

    # Coordinate differences between successive saved points
    dx = positions[1:] - positions[:-1]  # (N-1, 4)

    # Midpoint positions for metric evaluation (trapezoidal approximation)
    x_mid = 0.5 * (positions[1:] + positions[:-1])  # (N-1, 4)

    def ds_segment(x: Float[Array, "4"], delta: Float[Array, "4"]) -> Float[Array, ""]:
        """Proper time increment for one segment."""
        g = metric_fn(x)
        ds_sq = -jnp.einsum("ab,a,b", g, delta, delta)
        # Clamp to avoid sqrt of negative (should not happen for timelike)
        return jnp.sqrt(jnp.maximum(ds_sq, 0.0))

    # Vectorize over all segments
    dtau_segments = jax.vmap(ds_segment)(x_mid, dx)  # (N-1,)

    # Cumulative sum with leading zero
    cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dtau_segments)])

    return cumulative
