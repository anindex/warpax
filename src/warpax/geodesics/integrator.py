"""Core geodesic ODE integration via Diffrax.

Integrates the geodesic equation as a first-order ODE system using Diffrax
(JAX-native adaptive solver). The state vector y = [x^mu (4,), v^mu (4,)]
has shape (8,), where x are spacetime coordinates and v are 4-velocities.

The geodesic equation:
    dx^mu / dtau = v^mu
    dv^mu / dtau = -Gamma^mu_{alpha beta} v^alpha v^beta

Christoffel symbols are computed at each integration step via exact JAX autodiff.
"""
from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from warpax.geometry import christoffel_symbols


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class GeodesicResult(NamedTuple):
    """Result of geodesic integration.

    Wraps the Diffrax Solution fields for type clarity and stable API.

    Attributes
    ----------
    ts : Float[Array, "N"]
        Saved affine parameter values.
    positions : Float[Array, "N 4"]
        Coordinate positions x^mu at each saved point.
    velocities : Float[Array, "N 4"]
        4-velocities v^mu at each saved point.
    result : int
        Diffrax result code (0 = success, 1 = max_steps_reached, 2 = event).
    event_mask : Array or None
        Which event triggered (if any). Shape depends on number of events.
    """

    ts: Float[Array, "N"]
    positions: Float[Array, "N 4"]
    velocities: Float[Array, "N 4"]
    result: int
    event_mask: Array | None


# ---------------------------------------------------------------------------
# Geodesic vector field (ODE right-hand side)
# ---------------------------------------------------------------------------


def geodesic_vector_field(
    tau: Float[Array, ""],
    y: Float[Array, "8"],
    args: object,
) -> Float[Array, "8"]:
    """Right-hand side of the geodesic ODE: dy/dtau = f(tau, y, args).

    Computes the geodesic acceleration from the Christoffel symbols of the
    metric at the current spacetime position.

    Parameters
    ----------
    tau : Float[Array, ""]
        Affine parameter (proper time for timelike, affine lambda for null).
    y : Float[Array, "8"]
        State vector [x^mu (4,), v^mu (4,)].
    args : MetricSpecification
        The spacetime metric (Equinox module, passed as Diffrax args pytree).

    Returns
    -------
    Float[Array, "8"]
        Time derivative [v^mu (4,), a^mu (4,)] where
        a^mu = -Gamma^mu_{alpha beta} v^alpha v^beta.
    """
    metric_fn = args
    x = y[:4]  # position x^mu
    v = y[4:]  # velocity v^mu = dx^mu/dtau

    # Christoffel symbols at current position: Gamma^lam_{mu nu}
    gamma = christoffel_symbols(metric_fn, x)  # (4, 4, 4)

    # Geodesic acceleration: a^lam = -Gamma^lam_{mu nu} v^mu v^nu
    a = -jnp.einsum("lab,a,b->l", gamma, v, v)  # (4,)

    return jnp.concatenate([v, a])  # (8,)


# ---------------------------------------------------------------------------
# Event detection functions
# ---------------------------------------------------------------------------


def bounding_box_event(
    t: Float[Array, ""],
    y: Float[Array, "8"],
    args: object,
    **kwargs: object,
) -> Float[Array, ""]:
    """Terminate when geodesic leaves a spherical bounding box.

    Returns R_max - r, which crosses zero when r exceeds R_max.

    Parameters
    ----------
    t : Float[Array, ""]
        Current affine parameter.
    y : Float[Array, "8"]
        State vector [x^mu, v^mu].
    args : object
        Metric (unused here).
    **kwargs : object
        Optional ``R_max`` (default 100.0).

    Returns
    -------
    Float[Array, ""]
        Positive inside domain, negative outside.
    """
    x = y[:4]
    r = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    R_max = kwargs.get("R_max", 100.0)
    return R_max - r


def horizon_event(
    t: Float[Array, ""],
    y: Float[Array, "8"],
    args: object,
    **kwargs: object,
) -> Float[Array, ""]:
    """Terminate when approaching a coordinate singularity (Schwarzschild horizon).

    Returns r_iso - margin * r_horizon, which crosses zero near the horizon
    in isotropic coordinates.

    Parameters
    ----------
    t : Float[Array, ""]
        Current affine parameter.
    y : Float[Array, "8"]
        State vector [x^mu, v^mu].
    args : object
        Metric. If it has attribute ``M``, uses that mass parameter.
    **kwargs : object
        Optional ``margin`` (default 1.1), ``M`` (default 1.0).

    Returns
    -------
    Float[Array, ""]
        Positive outside horizon margin, negative inside.
    """
    x = y[:4]
    r_iso = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    margin = kwargs.get("margin", 1.1)
    # Try to get M from the metric args, fall back to kwarg or default
    M = getattr(args, "M", kwargs.get("M", 1.0))
    r_horizon_iso = M / 2.0  # Schwarzschild horizon in isotropic coords
    return r_iso - margin * r_horizon_iso


def make_event(*cond_fns: object) -> diffrax.Event:
    """Combine multiple event condition functions into a single Diffrax Event.

    Each condition function should have signature ``(t, y, args, **kwargs) -> scalar``
    and cross zero when the event occurs. The Event uses an optimistix Newton
    root finder for exact zero-crossing detection.

    Parameters
    ----------
    *cond_fns : callable
        One or more event condition functions.

    Returns
    -------
    diffrax.Event
        Combined event for use in ``diffrax.diffeqsolve``.
    """
    root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
    if len(cond_fns) == 1:
        return diffrax.Event(cond_fn=cond_fns[0], root_finder=root_finder)
    return diffrax.Event(cond_fn=list(cond_fns), root_finder=root_finder)


# ---------------------------------------------------------------------------
# Single geodesic integration
# ---------------------------------------------------------------------------


def integrate_geodesic(
    metric: object,
    x0: Float[Array, "4"],
    v0: Float[Array, "4"],
    tau_span: tuple[float, float],
    *,
    num_points: int = 1000,
    dt0: float = 0.01,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 16384,
    event: diffrax.Event | None = None,
) -> GeodesicResult:
    """Integrate a single geodesic through a spacetime.

    Uses Diffrax Tsit5 (5th-order adaptive Runge-Kutta) with PID step size
    control. The metric is evaluated pointwise at each integration step via
    exact JAX autodiff Christoffel symbols.

    Parameters
    ----------
    metric : MetricSpecification
        Spacetime metric (Equinox module, pytree-compatible).
    x0 : Float[Array, "4"]
        Initial spacetime position (t, x, y, z).
    v0 : Float[Array, "4"]
        Initial 4-velocity (v^t, v^x, v^y, v^z).
    tau_span : tuple[float, float]
        (tau_start, tau_end) integration interval for the affine parameter.
    num_points : int
        Number of equally-spaced save points (default 1000).
    dt0 : float
        Initial step size (default 0.01).
    rtol : float
        Relative tolerance for adaptive stepping (default 1e-10).
    atol : float
        Absolute tolerance for adaptive stepping (default 1e-10).
    max_steps : int
        Maximum number of integration steps (default 16384).
    event : diffrax.Event or None
        Optional event for early termination.

    Returns
    -------
    GeodesicResult
        Named tuple with ts, positions, velocities, result code, event_mask.
    """
    y0 = jnp.concatenate([x0, v0])  # (8,)

    term = diffrax.ODETerm(geodesic_vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    save_ts = jnp.linspace(tau_span[0], tau_span[1], num_points)
    saveat = diffrax.SaveAt(ts=save_ts)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=tau_span[0],
        t1=tau_span[1],
        dt0=dt0,
        y0=y0,
        args=metric,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,
        event=event,
    )

    # Unpack positions and velocities from state
    positions = sol.ys[:, :4]  # (N, 4)
    velocities = sol.ys[:, 4:]  # (N, 4)

    return GeodesicResult(
        ts=sol.ts,
        positions=positions,
        velocities=velocities,
        result=sol.result,
        event_mask=sol.event_mask,
    )


# ---------------------------------------------------------------------------
# Batched geodesic families via vmap
# ---------------------------------------------------------------------------


def integrate_geodesic_family(
    metric: object,
    x0_batch: Float[Array, "N 4"],
    v0_batch: Float[Array, "N 4"],
    tau_span: tuple[float, float],
    **kwargs: object,
) -> GeodesicResult:
    """Integrate a batch of geodesics in parallel via vmap.

    All geodesics share the same metric and integration parameters. Only the
    initial conditions (positions and velocities) vary across the batch.

    Parameters
    ----------
    metric : MetricSpecification
        Single spacetime metric (shared across batch, not vmapped).
    x0_batch : Float[Array, "N 4"]
        Batch of initial positions.
    v0_batch : Float[Array, "N 4"]
        Batch of initial 4-velocities.
    tau_span : tuple[float, float]
        Integration interval (shared across batch).
    **kwargs : object
        Additional keyword arguments passed to ``integrate_geodesic``.

    Returns
    -------
    GeodesicResult
        Batched result with shapes (N, num_points, 4) for positions/velocities.
    """

    def solve_one(x0: Float[Array, "4"], v0: Float[Array, "4"]) -> GeodesicResult:
        return integrate_geodesic(metric, x0, v0, tau_span, **kwargs)

    return jax.vmap(solve_one)(x0_batch, v0_batch)
