"""Initial condition builders for timelike and null geodesics.

Provides helpers that construct properly normalized initial 4-velocities
for geodesic integration. Each function evaluates the metric at the initial
point and solves the appropriate norm constraint:
    - Timelike: g_ab v^a v^b = -1
    - Null:     g_ab k^a k^b = 0

Also provides Schwarzschild-specific IC builders for circular orbits and
radial infall in isotropic coordinates.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# General IC builders (work with any MetricSpecification)
# ---------------------------------------------------------------------------


def timelike_ic(
    metric_fn: object,
    x0: Float[Array, "4"],
    v_spatial: Float[Array, "3"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Construct timelike initial conditions satisfying g_ab v^a v^b = -1.

    Given an initial position and desired spatial 3-velocity components,
    solves for the temporal component v^0 (future-directed) using the
    quadratic formula on the norm constraint.

    The norm constraint g_ab v^a v^b = -1 expands to:
        g_00 (v^0)^2 + 2 g_0i v^0 v^i + g_ij v^i v^j = -1

    This is a quadratic in v^0:
        a (v^0)^2 + b v^0 + c = 0
    with a = g_00, b = 2 g_0i v^i, c = g_ij v^i v^j + 1.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable: coords (4,) -> g_ab (4,4).
    x0 : Float[Array, "4"]
        Initial spacetime position (t, x, y, z).
    v_spatial : Float[Array, "3"]
        Spatial 3-velocity components (v^x, v^y, v^z).

    Returns
    -------
    tuple[Float[Array, "4"], Float[Array, "4"]]
        (x0, v0) where v0 = [v^0, v^x, v^y, v^z] with g_ab v^a v^b = -1.

    Notes
    -----
    Returns NaN for v^0 if no real root exists (superluminal velocity).
    Chooses the future-directed root (positive v^0 for -+++ signature).
    """
    g = metric_fn(x0)  # (4, 4)

    # Quadratic coefficients
    a = g[0, 0]
    b = 2.0 * jnp.dot(g[0, 1:], v_spatial)
    c = jnp.einsum("ij,i,j->", g[1:, 1:], v_spatial, v_spatial) + 1.0

    # Discriminant
    disc = b**2 - 4.0 * a * c

    # Future-directed root: for g_00 < 0 (as in -+++ signature),
    # a < 0 so we need the root (-b + sqrt(disc)) / (2a)
    # which gives the larger (more positive) v^0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    v0_t = (-b + sqrt_disc) / (2.0 * a)

    # If discriminant was negative, v0_t will be NaN-like from sqrt(0)/...
    # Signal superluminal with NaN
    v0_t = jnp.where(disc >= 0.0, v0_t, jnp.nan)

    v0 = jnp.concatenate([jnp.array([v0_t]), v_spatial])
    return x0, v0


def null_ic(
    metric_fn: object,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Construct null initial conditions satisfying g_ab k^a k^b = 0.

    Given an initial position and spatial direction (need not be unit norm),
    solves for the temporal component k^0 (future-directed) using the
    null norm constraint.

    The constraint g_ab k^a k^b = 0 expands to:
        g_00 (k^0)^2 + 2 g_0i k^0 n^i + g_ij n^i n^j = 0

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable: coords (4,) -> g_ab (4,4).
    x0 : Float[Array, "4"]
        Initial spacetime position (t, x, y, z).
    n_spatial : Float[Array, "3"]
        Spatial direction for the null ray (v^x, v^y, v^z).
        Need not be normalized.

    Returns
    -------
    tuple[Float[Array, "4"], Float[Array, "4"]]
        (x0, k0) where k0 = [k^0, n^x, n^y, n^z] with g_ab k^a k^b = 0.

    Notes
    -----
    The affine parameter normalization is free for null geodesics.
    The spatial components are kept as-is (not renormalized).
    """
    g = metric_fn(x0)  # (4, 4)

    # Quadratic coefficients (same as timelike but c has no +1 term)
    a = g[0, 0]
    b = 2.0 * jnp.dot(g[0, 1:], n_spatial)
    c = jnp.einsum("ij,i,j->", g[1:, 1:], n_spatial, n_spatial)

    # Discriminant
    disc = b**2 - 4.0 * a * c

    # Future-directed root
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    k0_t = (-b + sqrt_disc) / (2.0 * a)

    k0_t = jnp.where(disc >= 0.0, k0_t, jnp.nan)

    k0 = jnp.concatenate([jnp.array([k0_t]), n_spatial])
    return x0, k0


# ---------------------------------------------------------------------------
# Schwarzschild-specific IC builders (isotropic coordinates)
# ---------------------------------------------------------------------------


def _schw_r_to_iso(r_schw: float | Float[Array, ""], M: float = 1.0) -> Float[Array, ""]:
    """Convert Schwarzschild radial coordinate to isotropic radial coordinate.

    r_iso = (r_schw - M + sqrt(r_schw^2 - 2 M r_schw)) / 2

    Valid for r_schw > 2M (outside the horizon).
    """
    return (r_schw - M + jnp.sqrt(r_schw**2 - 2.0 * M * r_schw)) / 2.0


def circular_orbit_ic(
    metric_fn: object,
    r_schw: float = 10.0,
    M: float = 1.0,
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Initial conditions for a circular orbit in Schwarzschild spacetime.

    Sets up a circular orbit at Schwarzschild radius ``r_schw`` in the
    equatorial (x-y) plane, starting on the positive x-axis with velocity
    in the y-direction.

    Uses the Schwarzschild energy-angular momentum approach:
        E/m = (1 - 2M/r) / sqrt(1 - 3M/r)
        L/m = sqrt(M r) / sqrt(1 - 3M/r)

    The velocity in isotropic Cartesian coordinates is then determined by
    converting the angular momentum to coordinate velocity and solving for
    v^0 via the timelike norm constraint.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Schwarzschild metric (must be in isotropic Cartesian coordinates).
    r_schw : float
        Desired circular orbit radius in Schwarzschild coordinates.
        Must be > 6M (ISCO) for stable orbits, > 3M for any circular orbit.
    M : float
        Mass parameter (default 1.0).

    Returns
    -------
    tuple[Float[Array, "4"], Float[Array, "4"]]
        (x0, v0) with x0 on the x-axis and v0 in the y-direction,
        satisfying g_ab v^a v^b = -1.
    """
    r_schw = jnp.asarray(r_schw, dtype=jnp.float64)
    M = jnp.asarray(M, dtype=jnp.float64)

    # Convert to isotropic radius
    r_iso = _schw_r_to_iso(r_schw, M)

    # Position: equatorial plane, on x-axis
    x0 = jnp.array([0.0, r_iso, 0.0, 0.0])

    # Compute 4-velocity directly from Schwarzschild conserved quantities.
    #
    # For a circular orbit at Schwarzschild radius r_schw:
    #   u^t = dt/dtau = 1 / sqrt(1 - 3M/r_schw)
    #   dphi/dtau = sqrt(M/r_schw) / (r_schw * sqrt(1 - 3M/r_schw))
    #
    # Since the time coordinate t and the azimuthal angle phi are the
    # same in both Schwarzschild and isotropic coordinates, these
    # proper-time derivatives transfer directly.
    #
    # At position (r_iso, 0, 0) on the x-axis, the y-component of
    # 4-velocity is u^y = dy/dtau = r_iso * dphi/dtau.
    #
    # We set u^t and u^y directly (no need for timelike_ic, which
    # expects spatial 3-velocity and solves for u^t).

    factor = jnp.sqrt(1.0 - 3.0 * M / r_schw)
    u_t = 1.0 / factor
    dphi_dtau = jnp.sqrt(M / r_schw) / (r_schw * factor)
    u_y = r_iso * dphi_dtau

    v0 = jnp.array([u_t, 0.0, u_y, 0.0])
    return x0, v0


def radial_infall_ic(
    metric_fn: object,
    r_start_schw: float = 10.0,
    M: float = 1.0,
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Initial conditions for radial infall from rest in Schwarzschild spacetime.

    Sets up a particle at rest (zero spatial velocity) at Schwarzschild radius
    ``r_start_schw``. The particle will fall inward under gravity.

    "At rest" means v^i = 0 in the coordinate frame at the starting point.
    The temporal component v^0 is determined by the timelike norm constraint
    g_ab v^a v^b = -1.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Schwarzschild metric (must be in isotropic Cartesian coordinates).
    r_start_schw : float
        Starting radius in Schwarzschild coordinates. Must be > 2M.
    M : float
        Mass parameter (default 1.0).

    Returns
    -------
    tuple[Float[Array, "4"], Float[Array, "4"]]
        (x0, v0) with zero spatial velocity, satisfying g_ab v^a v^b = -1.
    """
    r_start_schw = jnp.asarray(r_start_schw, dtype=jnp.float64)
    M = jnp.asarray(M, dtype=jnp.float64)

    # Convert to isotropic radius
    r_iso = _schw_r_to_iso(r_start_schw, M)

    # Position on x-axis in equatorial plane
    x0 = jnp.array([0.0, r_iso, 0.0, 0.0])

    # At rest: zero spatial velocity
    v_spatial = jnp.array([0.0, 0.0, 0.0])

    return timelike_ic(metric_fn, x0, v_spatial)
