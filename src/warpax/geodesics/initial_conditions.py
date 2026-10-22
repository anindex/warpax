"""Initial condition builders for timelike and null geodesics.

Provides helpers that construct properly normalized initial 4-velocities
for geodesic integration. Each function evaluates the metric at the initial
point and solves the appropriate norm constraint:
    - Timelike: g_ab v^a v^b = -1
    - Null: g_ab k^a k^b = 0

Also provides Schwarzschild-specific IC builders for circular orbits and
radial infall in isotropic coordinates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _future_time_component(a, b, c):
    r"""Future-directed root of ``a (u^0)^2 + b u^0 + c = 0``, valid at ``a = 0``.

    Both initial-condition helpers solve this normalization quadratic for the
    time component given a prescribed spatial direction. Where ``v_s f(r) = 1``
    the coordinate vector ``\partial_t`` is null, ``g_{00} = 0`` and the equation
    degenerates to the linear ``b u^0 + c = 0`` with the ordinary root
    ``-c/b``. The degeneracy test ``|a| |c| <= 1e-24 b^2`` is scale invariant,
    so a large spatial seed (``c`` of order ``1e14``) is not mistaken for it.

    Roots come from the stable form ``q = -(b + sign(b) sqrt(disc)) / 2`` as
    ``c/q`` (which tends to ``-c/b`` as ``a -> 0``) and ``q/a``; the larger
    positive one is future-directed when ``g_00 < 0``. Branchless
    (vmap/jit-safe). The radicand is floored so forward-mode gradients stay
    finite when the discriminant is negative (the superluminal case), which is
    signalled by returning NaN. Callers should test ``jnp.isfinite``.
    """
    degenerate = jnp.abs(a) * jnp.abs(c) <= 1e-24 * b**2
    disc = b**2 - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    sgn = jnp.where(b >= 0.0, 1.0, -1.0)
    q = -0.5 * (b + sgn * sqrt_disc)
    q_ok = q != 0.0
    r_lin = jnp.where(q_ok, c / jnp.where(q_ok, q, 1.0), jnp.nan)
    r_quad = jnp.where(degenerate, -jnp.inf, q / jnp.where(degenerate, 1.0, a))
    root = jnp.maximum(r_lin, r_quad)
    nan = jax.lax.stop_gradient(jnp.full_like(root, jnp.nan))
    ok = (disc >= 0.0) & (root > 0.0) & jnp.isfinite(root)
    return jnp.where(ok, root, nan)


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
    Selects the FUTURE-directed root (the larger ``v^0``; for the usual
    ``g_00 < 0`` exactly one root is positive and this picks it). The sign
    of ``v^0`` matters for time-dependent metrics: an Alcubierre bubble at
    ``x_s = v_s t`` is *not* symmetric under ``t -> -t``, so a past-directed
    velocity traces a physically different trajectory, not a time-reverse.
    """
    g = metric_fn(x0)

    a = g[0, 0]
    b = 2.0 * jnp.dot(g[0, 1:], v_spatial)
    c = jnp.einsum("ij,i,j->", g[1:, 1:], v_spatial, v_spatial) + 1.0

    v0_t = _future_time_component(a, b, c)
    v0 = jnp.concatenate([jnp.array([v0_t]), v_spatial])
    return x0, v0


def null_ic(
    metric_fn: object,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Construct null initial conditions satisfying g_ab k^a k^b = 0.

    Given an initial position and spatial direction (need not be unit norm),
    solves for the temporal component k^0 using the null norm constraint.
    Selects the FUTURE-directed root (the larger ``k^0``; for the usual
    ``g_00 < 0`` exactly one root is positive and this picks it). Although
    the ANEC integrand ``T_ab k^a k^b`` is even in ``k``, the *trajectory*
    is not: for time-dependent metrics (e.g. an Alcubierre bubble at
    ``x_s = v_s t``) a past-directed ray traces a different path through
    the bubble, so the sign of ``k^0`` changes the geodesic itself.

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
    The affine parameter normalization is free for null geodesics, and this
    function does NOT fix it: the spatial components are kept as-is. Any
    quantity that is not invariant under ``k -> c k`` (notably the ANEC line
    integral ``int T_ab k^a k^b dlambda``, which scales linearly in ``c``) is
    therefore undefined at this level. Use :func:`null_ic_killing_normalized`
    to fix the affine scale canonically before integrating such a quantity.
    """
    g = metric_fn(x0)

    a = g[0, 0]
    b = 2.0 * jnp.dot(g[0, 1:], n_spatial)
    c = jnp.einsum("ij,i,j->", g[1:, 1:], n_spatial, n_spatial)

    k0_t = _future_time_component(a, b, c)
    k0 = jnp.concatenate([jnp.array([k0_t]), n_spatial])
    return x0, k0


def killing_energy(
    metric_fn: object,
    x: Float[Array, "4"],
    k: Float[Array, "4"],
    v_s: float | Float[Array, ""],
) -> Float[Array, ""]:
    r"""Conserved quantity ``E_K = -g_ab k^a K^b`` of the metric's Killing vector.

    ``K`` is ``metric_fn.killing_vector()`` when the metric provides one, else
    the helical ``d/dt + v_s d/dx`` of the comoving ``x - v_s t`` drives. For the
    stationary Fuchs and T-shell constructions, which depend on ``x`` rather than
    ``x - v_s t``, the Killing vector is ``d/dt`` alone. ``E_K = -g(k, K)`` is
    exactly conserved along every geodesic of the metric that ``K`` generates.

    ``E_K`` is a handle on the otherwise free affine scale of a null geodesic:
    under ``k -> c k`` it scales as ``E_K -> c E_K``, so fixing ``E_K = 1`` fixes
    the affine parameter, which is what makes ANEC *magnitudes* comparable
    rather than only their signs.

    For ``v_s > 1`` the helical ``K`` is spacelike in the asymptotic region, so
    ``E_K`` is no longer an energy and can vanish: in Minkowski with
    ``K = (1, 2, 0, 0)`` the future null ``k = (1, 1/2, sqrt(3)/2, 0)`` has
    ``-K . k = 0``, and :func:`null_ic_killing_normalized` returns NaN there.

    Returns
    -------
    Float[Array, ""]
        The conserved quantity ``E_K``. Also usable as a numerical witness:
        its drift along an integrated geodesic bounds the integrator error
        independently of the on-cone witness ``g(k, k)``.
    """
    g = metric_fn(x)
    return -jnp.einsum("ab,a,b->", g, k, _killing_vector(metric_fn, v_s))


def _killing_vector(metric_fn: object, v_s: float | Float[Array, ""]) -> Float[Array, "4"]:
    if hasattr(metric_fn, "killing_vector"):
        return metric_fn.killing_vector()
    return jnp.array([1.0, v_s, 0.0, 0.0])


def eulerian_affine_scale(
    metric_fn: object,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"] | None = None,
) -> Float[Array, ""]:
    """Factor pinning the free null affine scale to ``-g(k, n) = 1`` at ``x0``.

    ``null_ic`` solves only ``k^0`` and passes the spatial direction through, so
    the affine parameter of a null geodesic is left free. The ANEC line integral
    scales linearly under ``k -> c k``, so its *magnitude* is undefined until this
    is fixed; only its sign is not. Pinning against the Eulerian normal works at
    every warp speed, since ``n`` stays unit timelike where ``d_t`` does not.

    For the corrected laboratory Natario metric, a unit spatial seed has
    limiting Eulerian frequency one at spatial infinity. At a finite initial
    point the factor is computed from the metric, as for every other profile.

    Use as ``k -> s * k`` with the affine window shrunk by the same factor, so the
    geodesic covers an identical coordinate path and only its parametrization
    changes.
    """
    if n_spatial is None:
        n_spatial = jnp.array([1.0, 0.0, 0.0])
    _, k = null_ic(metric_fn, x0, n_spatial)
    return 1.0 / eulerian_frequency(metric_fn, x0, k)


def eulerian_frequency(
    metric_fn: object,
    x: Float[Array, "4"],
    k: Float[Array, "4"],
) -> Float[Array, ""]:
    r"""Frequency ``-g_ab k^a n^b`` measured by the Eulerian normal observer.

    ``n`` is the unit normal to the ``t = const`` slice, which is unit timelike
    at every warp speed (it is not the coordinate-stationary ``d_t``), so this
    is well defined through and beyond the luminal transition.
    """
    from ..energy_conditions.observer import eulerian_normal

    g = metric_fn(x)
    _, n_up = eulerian_normal(jnp.linalg.inv(g))
    return -jnp.einsum("ab,a,b->", g, k, n_up)


def null_ic_eulerian_normalized(
    metric_fn: object,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
    freq_target: float = 1.0,
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    r"""Null initial conditions with ``-g(k, n) = freq_target`` at ``x0``.

    This is the common initial affine normalization for cross-metric ANEC
    magnitudes: it pins the free scale ``k -> c k`` against the Eulerian normal
    at a stated initial surface. Unlike the Killing normalization it does not
    degenerate as ``v_s -> 1``, so it is the default for the reported line
    integrals; :func:`killing_energy` is then carried alongside as an exactly
    conserved witness.

    The laboratory Natario shift now vanishes at infinity, so its unit spatial
    seed has limiting frequency one. This function also handles finite initial
    surfaces and profiles with different lapse and spatial metric.
    """
    x0, k0 = null_ic(metric_fn, x0, n_spatial)
    freq = eulerian_frequency(metric_fn, x0, k0)
    scale = jnp.where(jnp.abs(freq) > 0.0, freq_target / freq, jnp.nan)
    return x0, k0 * scale


def null_ic_killing_normalized(
    metric_fn: object,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
    v_s: float | Float[Array, ""],
    e_target: float = 1.0,
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    r"""Null initial conditions with the affine scale fixed by ``E_K = e_target``.

    Wraps :func:`null_ic` and rescales the tangent, which is exactly an affine
    reparametrization ``lambda -> lambda / c`` of the same geodesic: the path is
    unchanged, only its parametrization is pinned. This removes the
    ``k -> c k`` ambiguity that otherwise leaves cross-metric ANEC magnitudes
    undefined.

    Returns NaN for the tangent where ``E_K`` vanishes (a ray with zero Killing
    energy admits no such normalization).
    """
    x0, k0 = null_ic(metric_fn, x0, n_spatial)
    e_k = killing_energy(metric_fn, x0, k0, v_s)
    g = metric_fn(x0)
    K = _killing_vector(metric_fn, v_s)
    floor = 1e-12 * jnp.max(jnp.abs(g)) * jnp.max(jnp.abs(k0)) * jnp.max(jnp.abs(K))
    scale = jnp.where(
        jnp.abs(e_k) > floor, e_target / jnp.where(jnp.abs(e_k) > floor, e_k, 1.0), jnp.nan
    )
    return x0, k0 * scale


def _schw_r_to_iso(
    r_schw: float | Float[Array, ""], M: float | Float[Array, ""] = 1.0
) -> Float[Array, ""]:
    """Convert Schwarzschild radial coordinate to isotropic radial coordinate.

    r_iso = (r_schw - M + sqrt(r_schw^2 - 2 M r_schw)) / 2

    Valid for r_schw > 2M (outside the horizon). The floor inside ``sqrt``
    is autodiff-safe: at ``r_schw = 2M`` the value collapses to ``M/2``
    and the gradient remains finite.
    """
    disc = jnp.maximum(r_schw**2 - 2.0 * M * r_schw, 1e-60)
    return (r_schw - M + jnp.sqrt(disc)) / 2.0


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
    v^0 via the timelike norm constraint: ``u^t = 1/sqrt(1 - 3M/r_schw)``
    and ``dphi/dtau = sqrt(M/r_schw) / (r_schw sqrt(1 - 3M/r_schw))``; t and
    phi agree in both charts, so ``u^y = r_iso dphi/dtau`` on the x-axis.
    The radicand ``1 - 3M/r_schw`` is floored at ``1e-30`` so radii below
    ``3M`` (where ``u^t`` would be complex) give a finite NaN-free result;
    the caller checks the domain.

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

    r_iso = _schw_r_to_iso(r_schw, M)

    x0 = jnp.array([0.0, r_iso, 0.0, 0.0])

    factor = jnp.sqrt(jnp.maximum(1.0 - 3.0 * M / r_schw, 1e-30))
    u_t = 1.0 / factor
    dphi_dtau = jnp.sqrt(jnp.maximum(M / r_schw, 0.0)) / (r_schw * factor)
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

    r_iso = _schw_r_to_iso(r_start_schw, M)

    x0 = jnp.array([0.0, r_iso, 0.0, 0.0])

    v_spatial = jnp.array([0.0, 0.0, 0.0])

    return timelike_ic(metric_fn, x0, v_spatial)
