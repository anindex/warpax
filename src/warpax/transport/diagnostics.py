"""Transport diagnostics for warp shell spacetimes.

Three observables for admissibility assessment:

1. delta_t_coord, null round-trip *coordinate-time* asymmetry;
   gauge-dependent (it lives in the chosen time slicing).
2. A_geo        , geodesic deviation diagnostic (gauge-invariant).
3. B            , blueshift hazard functional (gauge-invariant for
   a chosen observer worldline).

Built on warpax.geodesics (Diffrax integration, Jacobi deviation,
blueshift observables).
"""

from __future__ import annotations

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..geodesics.deviation import tidal_eigenvalues
from ..geodesics.initial_conditions import null_ic
from ..geodesics.integrator import integrate_geodesic


def _first_local_min_idx(d: Float[Array, "N"]) -> Int[Array, ""]:
    """Index of the first interior local minimum of ``d`` (else global argmin).

    Closest-approach detection for a round-trip null ray must pick the FIRST
    time the geodesic reaches the target, not a later re-approach: a warp bubble
    can carry a ray past the receiver and back, so a global ``argmin`` over the
    whole saved trajectory may select a spurious second pass and report the
    wrong arrival time. We take the first ``d[i] <= d[i-1] and d[i] <= d[i+1]``;
    if none exists (still approaching at the integration horizon) we fall back
    to the global minimum. vmap/JIT-safe (no data-dependent Python control).
    """
    interior = (d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:])
    has_local = jnp.any(interior)
    first_local = jnp.argmax(interior) + 1
    return jnp.where(has_local, first_local, jnp.argmin(d))


def _static_observer(g: Float[Array, "4 4"], dtype) -> Float[Array, "4"]:
    """Static 4-velocity (1/sqrt(-g_00), 0, 0, 0); NaN where g_00 >= 0 (no static observer)."""
    g00 = g[0, 0]
    u_t = jnp.where(g00 < 0, 1.0 / jnp.sqrt(-g00), jnp.nan)
    return jnp.array([u_t, 0.0, 0.0, 0.0], dtype=dtype)


@eqx.filter_jit
def _tidal_max_jit(metric_fn, x: Float[Array, "4"], u: Float[Array, "4"]) -> Float[Array, ""]:
    return jnp.max(jnp.abs(tidal_eigenvalues(metric_fn, x, u)))


def _tidal_max(metric_fn, x: Float[Array, "4"], u: Float[Array, "4"]) -> Float[Array, ""]:
    """Maximum |eigenvalue| of the tidal tensor at ``x`` for 4-velocity ``u``.

    Python-float leaves of ``metric_fn`` are promoted to arrays so the jitted
    kernel is traced once per metric class, not once per parameter value.
    """
    metric_fn = jax.tree_util.tree_map(
        lambda v: jnp.asarray(v) if isinstance(v, float) else v, metric_fn
    )
    return _tidal_max_jit(metric_fn, x, u)


def _closest_approach(sol, target: Float[Array, "3"], tol: Float[Array, ""]):
    """Index of first closest approach to ``target`` and whether it is within ``tol``.

    ``tol`` is widened to half the spatial distance traveled between saved
    samples, so a ray passing through the target is not rejected because no
    sample landed on it.
    """
    d = jnp.linalg.norm(sol.positions[:, 1:] - target, axis=1)
    idx = _first_local_min_idx(d)
    step = sol.ts[1] - sol.ts[0]
    tol = jnp.maximum(tol, 0.5 * step * jnp.linalg.norm(sol.velocities[idx, 1:]))
    return idx, d[idx] <= tol


def null_coord_time_asymmetry(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    emitter: Float[Array, "4"],
    receiver: Float[Array, "4"],
    *,
    tau_max: float = 50.0,
    num_points: int = 500,
    dt0: float = 0.01,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    arrival_rtol: float = 0.01,
) -> Float[Array, ""]:
    """Null round-trip *coordinate-time* asymmetry between emitter and receiver.

    .. math::
        \\Delta t_{coord} = t_{forward} - t_{backward}

    where ``t_forward`` is the coordinate time elapsed for a null geodesic
    from emitter to receiver, and ``t_backward`` is the return trip.

    For static, symmetric spacetimes, :math:`\\Delta t_{coord} = 0`.
    For warp bubbles with net transport, :math:`\\Delta t_{coord} \\neq 0`.

    .. warning::

       Gauge-dependent: invariant under :math:`t \\to t + c` but not under
       general re-slicings :math:`t \\to t + \\phi(\\mathbf{x})`. For a
       coordinate-free observable, compare proper-time intervals on a
       fixed worldline (round-trip Shapiro delay).

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    emitter : emitter coordinates (t, x, y, z)
    receiver : receiver coordinates (t, x, y, z)
    tau_max : maximum affine parameter for integration
    num_points : saved points per leg
    dt0 : initial step size
    rtol, atol : integration tolerances
    arrival_rtol : each leg must pass within ``arrival_rtol * |receiver - emitter|``
        (or half the inter-sample spacing, whichever is larger) of its
        target, else the result is NaN

    Returns
    -------
    delta_t_coord : round-trip coordinate-time asymmetry (0 for symmetric
        spacetimes in the same slicing); NaN if either leg failed to
        integrate successfully or never reached its target.
    """
    dx = receiver[1:] - emitter[1:]
    dist = jnp.linalg.norm(dx)

    is_zero_dist = dist < 1e-15
    dx_safe = jnp.where(is_zero_dist, jnp.array([1.0, 0.0, 0.0]), dx / jnp.maximum(dist, 1e-15))

    _, k0_fwd = null_ic(metric_fn, emitter, dx_safe)
    sol_fwd = integrate_geodesic(
        metric_fn,
        emitter,
        k0_fwd,
        tau_span=(0.0, tau_max),
        num_points=num_points,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
    )

    idx_fwd, reached_fwd = _closest_approach(sol_fwd, receiver[1:], arrival_rtol * dist)
    t_arrive_fwd = sol_fwd.positions[idx_fwd, 0]
    dt_coord_fwd = t_arrive_fwd - emitter[0]

    arrival_pos = sol_fwd.positions[idx_fwd]
    dx_back = emitter[1:] - arrival_pos[1:]
    dx_back_safe = jnp.where(
        jnp.linalg.norm(dx_back) < 1e-15,
        -dx_safe,
        dx_back / jnp.maximum(jnp.linalg.norm(dx_back), 1e-15),
    )
    _, k0_bwd = null_ic(metric_fn, arrival_pos, dx_back_safe)
    sol_bwd = integrate_geodesic(
        metric_fn,
        arrival_pos,
        k0_bwd,
        tau_span=(0.0, tau_max),
        num_points=num_points,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
    )

    idx_bwd, reached_bwd = _closest_approach(sol_bwd, emitter[1:], arrival_rtol * dist)
    t_arrive_bwd = sol_bwd.positions[idx_bwd, 0]
    dt_coord_bwd = t_arrive_bwd - arrival_pos[0]

    reached = reached_fwd & reached_bwd
    solved = (sol_fwd.result == diffrax.RESULTS.successful) & (
        sol_bwd.result == diffrax.RESULTS.successful
    )
    valid = is_zero_dist | (reached & solved)
    return jnp.where(valid, jnp.where(is_zero_dist, 0.0, dt_coord_fwd - dt_coord_bwd), jnp.nan)


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
        velocity = _static_observer(metric_fn(emitter), emitter.dtype)
    return _tidal_max(metric_fn, emitter, velocity)


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
    n_directions : number of spatial directions to probe (1..6)
    tau_max : maximum affine parameter
    num_points : saved points per geodesic

    Returns
    -------
    B : blueshift hazard scalar
    """
    if not 1 <= n_directions <= 6:
        raise ValueError(f"n_directions must be in [1, 6], got {n_directions}")
    g = metric_fn(emitter)
    u_observer = _static_observer(g, emitter.dtype)

    directions = jnp.stack(
        [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([-1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, -1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([0.0, 0.0, -1.0]),
        ]
    )[:n_directions]

    def omega_at(x, k):
        g_at = metric_fn(x)
        return -jnp.einsum("ab,a,b", g_at, k, _static_observer(g_at, x.dtype))

    def _hazard_one(d):
        _, k0 = null_ic(metric_fn, emitter, d)
        sol = integrate_geodesic(
            metric_fn,
            emitter,
            k0,
            tau_span=(0.0, tau_max),
            num_points=num_points,
            dt0=0.01,
            rtol=1e-8,
            atol=1e-8,
        )
        omega_emit = -jnp.einsum("ab,a,b", g, k0, u_observer)
        omegas = jax.vmap(omega_at)(sol.positions, sol.velocities)
        log_shifts = jnp.abs(jnp.log(jnp.maximum(jnp.abs(omegas / omega_emit), 1e-30)))
        return jnp.max(log_shifts)

    max_log_shift = jnp.max(jax.vmap(_hazard_one)(directions))
    return max_log_shift
