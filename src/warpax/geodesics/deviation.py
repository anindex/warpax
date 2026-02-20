"""Jacobi geodesic deviation co-integration and tidal tensor computation.

Co-integrates the Jacobi deviation equation alongside the geodesic ODE as a
single 16-component Diffrax system. The state vector is:

    y = [x^mu (4), v^mu (4), xi^mu (4), w^mu (4)]

where:
    - x^mu: spacetime coordinates (position)
    - v^mu: 4-velocity
    - xi^mu: deviation vector (connecting nearby geodesics)
    - w^mu = D xi^mu / D tau: covariant derivative of deviation along geodesic

The deviation equations use the tidal tensor K^mu_rho = R^mu_{nu rho sigma} v^nu v^sigma
and Christoffel transport terms, giving:

    d(xi)/dtau = w - Gamma^mu_{ab} v^a xi^b
    d(w)/dtau  = -K^mu_rho xi^rho - Gamma^mu_{ab} v^a w^b

The Riemann tensor R^lam_{mu nu rho} is computed at each integration step via
exact nested JAX autodiff (riemann_tensor from geometry.py).
"""
from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.geometry import christoffel_symbols, riemann_tensor


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class DeviationResult(NamedTuple):
    """Result of geodesic integration with Jacobi deviation.

    Extends the geodesic result with deviation vector and its covariant
    derivative along the trajectory.

    Attributes
    ----------
    ts : Float[Array, "N"]
        Saved affine parameter values.
    positions : Float[Array, "N 4"]
        Coordinate positions x^mu at each saved point.
    velocities : Float[Array, "N 4"]
        4-velocities v^mu at each saved point.
    deviations : Float[Array, "N 4"]
        Deviation vectors xi^mu at each saved point.
    deviation_velocities : Float[Array, "N 4"]
        Covariant deviation derivatives w^mu = D xi^mu / D tau at each saved point.
    result : int
        Diffrax result code (0 = success, 1 = max_steps_reached, 2 = event).
    event_mask : Array or None
        Which event triggered (if any).
    """

    ts: Float[Array, "N"]
    positions: Float[Array, "N 4"]
    velocities: Float[Array, "N 4"]
    deviations: Float[Array, "N 4"]
    deviation_velocities: Float[Array, "N 4"]
    result: int
    event_mask: Array | None


# ---------------------------------------------------------------------------
# Coupled geodesic + deviation vector field (ODE right-hand side)
# ---------------------------------------------------------------------------


def geodesic_deviation_vector_field(
    tau: Float[Array, ""],
    y: Float[Array, "16"],
    args: object,
) -> Float[Array, "16"]:
    """Right-hand side of the coupled geodesic + Jacobi deviation ODE.

    Computes the time derivatives of all 16 state components:
    - Geodesic: dx/dtau = v, dv/dtau = -Gamma v v
    - Deviation: dxi/dtau = w - Gamma v xi, dw/dtau = -K xi - Gamma v w

    Parameters
    ----------
    tau : Float[Array, ""]
        Affine parameter (proper time for timelike).
    y : Float[Array, "16"]
        State vector [x^mu (4), v^mu (4), xi^mu (4), w^mu (4)].
    args : MetricSpecification
        The spacetime metric (Equinox module, passed as Diffrax args pytree).

    Returns
    -------
    Float[Array, "16"]
        Time derivative [v, a, dxi, dw] of shape (16,).
    """
    metric_fn = args
    x = y[:4]      # position x^mu
    v = y[4:8]     # velocity v^mu
    xi = y[8:12]   # deviation vector xi^mu
    w = y[12:16]   # covariant deviation velocity w^mu

    # Christoffel symbols at current position: Gamma^lam_{mu nu} (4,4,4)
    gamma = christoffel_symbols(metric_fn, x)

    # Riemann tensor at current position: R^lam_{mu nu rho} (4,4,4,4)
    R = riemann_tensor(metric_fn, x)

    # --- Geodesic equations ---
    # a^lam = -Gamma^lam_{mu nu} v^mu v^nu
    a = -jnp.einsum("lab,a,b->l", gamma, v, v)

    # --- Tidal tensor ---
    # K^mu_rho = R^mu_{nu rho sigma} v^nu v^sigma
    K = jnp.einsum("mnrs,n,s->mr", R, v, v)  # (4,4)

    # --- Deviation equations ---
    # dxi^mu/dtau = w^mu - Gamma^mu_{ab} v^a xi^b
    dxi = w - jnp.einsum("lab,a,b->l", gamma, v, xi)

    # dw^mu/dtau = -K^mu_rho xi^rho - Gamma^mu_{ab} v^a w^b
    dw = -jnp.einsum("mr,r->m", K, xi) - jnp.einsum("lab,a,b->l", gamma, v, w)

    return jnp.concatenate([v, a, dxi, dw])


# ---------------------------------------------------------------------------
# Integration with deviation
# ---------------------------------------------------------------------------


def integrate_geodesic_with_deviation(
    metric: object,
    x0: Float[Array, "4"],
    v0: Float[Array, "4"],
    xi0: Float[Array, "4"],
    w0: Float[Array, "4"],
    tau_span: tuple[float, float],
    *,
    num_points: int = 1000,
    dt0: float = 0.01,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 32768,
    event: diffrax.Event | None = None,
) -> DeviationResult:
    """Integrate a geodesic with co-integrated Jacobi deviation equation.

    Uses Diffrax Tsit5 (5th-order adaptive Runge-Kutta) with PID step size
    control. Both Christoffel symbols and Riemann tensor are computed at each
    step via exact JAX autodiff.

    The 16-component state is [x^mu, v^mu, xi^mu, w^mu] where xi is the
    deviation vector and w = D xi / D tau.

    Note: First JIT compilation may take several minutes due to nested jacfwd
    tracing for the Riemann tensor. Subsequent calls use cached compilation.

    Parameters
    ----------
    metric : MetricSpecification
        Spacetime metric (Equinox module, pytree-compatible).
    x0 : Float[Array, "4"]
        Initial spacetime position (t, x, y, z).
    v0 : Float[Array, "4"]
        Initial 4-velocity.
    xi0 : Float[Array, "4"]
        Initial deviation vector.
    w0 : Float[Array, "4"]
        Initial covariant deviation velocity (D xi / D tau at tau=0).
    tau_span : tuple[float, float]
        (tau_start, tau_end) integration interval.
    num_points : int
        Number of equally-spaced save points (default 1000).
    dt0 : float
        Initial step size (default 0.01).
    rtol : float
        Relative tolerance (default 1e-10).
    atol : float
        Absolute tolerance (default 1e-10).
    max_steps : int
        Maximum integration steps (default 32768, doubled from bare geodesic).
    event : diffrax.Event or None
        Optional event for early termination.

    Returns
    -------
    DeviationResult
        Named tuple with ts, positions, velocities, deviations,
        deviation_velocities, result code, event_mask.
    """
    y0 = jnp.concatenate([x0, v0, xi0, w0])  # (16,)

    term = diffrax.ODETerm(geodesic_deviation_vector_field)
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

    # Unpack the 16-component state into 4 fields
    positions = sol.ys[:, :4]          # (N, 4)
    velocities = sol.ys[:, 4:8]        # (N, 4)
    deviations = sol.ys[:, 8:12]       # (N, 4)
    deviation_velocities = sol.ys[:, 12:16]  # (N, 4)

    return DeviationResult(
        ts=sol.ts,
        positions=positions,
        velocities=velocities,
        deviations=deviations,
        deviation_velocities=deviation_velocities,
        result=sol.result,
        event_mask=sol.event_mask,
    )


# ---------------------------------------------------------------------------
# Tidal tensor and eigenvalues (pointwise)
# ---------------------------------------------------------------------------


def tidal_tensor(
    metric_fn: object,
    x: Float[Array, "4"],
    v: Float[Array, "4"],
) -> Float[Array, "4 4"]:
    """Compute the tidal tensor at a single spacetime point.

    The tidal tensor K^mu_rho = R^mu_{nu rho sigma} v^nu v^sigma describes
    the relative acceleration of nearby geodesics (tidal forces).

    For Schwarzschild radial infall at radius r:
        - Radial eigenvalue: -2M/r^3 (stretching)
        - Transverse eigenvalues: +M/r^3 (compression) x2

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    x : Float[Array, "4"]
        Spacetime position (t, x, y, z).
    v : Float[Array, "4"]
        4-velocity at x.

    Returns
    -------
    Float[Array, "4 4"]
        Tidal tensor K^mu_rho of shape (4, 4).
    """
    R = riemann_tensor(metric_fn, x)  # R^lam_{mu nu rho} (4,4,4,4)
    return jnp.einsum("mnrs,n,s->mr", R, v, v)  # K^mu_rho (4,4)


def tidal_eigenvalues(
    metric_fn: object,
    x: Float[Array, "4"],
    v: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Compute the principal tidal accelerations (eigenvalues of tidal tensor).

    The eigenvalues of K^mu_rho give the principal stretching and compression
    rates. For Schwarzschild, these should show the characteristic -2M/r^3
    (stretching) and +M/r^3 (compression x2) pattern.

    Note: The tidal tensor K^mu_rho is not generally symmetric, so we use
    ``jnp.linalg.eigvals`` and take real parts rather than ``eigvalsh``
    (which assumes Hermitian symmetry).

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric callable.
    x : Float[Array, "4"]
        Spacetime position (t, x, y, z).
    v : Float[Array, "4"]
        4-velocity at x.

    Returns
    -------
    Float[Array, "4"]
        Real parts of the eigenvalues of the tidal tensor, sorted ascending.
    """
    K = tidal_tensor(metric_fn, x, v)
    eigs = jnp.linalg.eigvals(K)
    # Take real parts and sort
    eigs_real = jnp.real(eigs)
    return jnp.sort(eigs_real)
