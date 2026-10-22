"""ADM mass and asymptotic falloff computations.

Implements the ADM mass surface integral:

    M_ADM = (1/16pi) oint_S (dj gamma_{ij} - di gamma_{jj}) n^i dA

evaluated on coordinate spheres at large radii using angular quadrature
and JAX autodiff for spatial derivatives.

Also provides Richardson extrapolation for convergence verification.

.. warning::

   The ADM surface integral converges to a finite ``M_ADM`` only for
   metrics whose stress-energy has sufficiently rapid asymptotic
   falloff (formally ``T_{ab}`` of compact support, or
   ``rho ~ O(1/r^4)``). For thick-shell warp constructions where the
   density profile decays only as ``1/r^2`` or slower, the surface
   integral grows logarithmically (or worse) with ``r_surface``; the
   Richardson extrapolation in :func:`adm_mass_richardson` will then
   report a divergent ``M_extrapolated`` and a ``convergence_order``
   below 1. Treat such reports as evidence the model is **not**
   asymptotically flat in the ADM sense rather than as a numerical
   issue. See the Fuchs canonical regression test in
   ``tests/test_physics_validation.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.geometry.metric import MetricSpecification


def _spatial_metric_derivs(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> Float[Array, "3 3 3"]:
    """Compute dk gamma_{ij} at a single point via autodiff.

    Returns
    -------
    dgamma[i, j, k] = d_ gamma_{ij} / d_ x^k
    """
    t = coords[0]
    spatial = coords[1:]

    def spatial_metric(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g = metric_fn(full)
        return g[1:, 1:]

    return jax.jacfwd(spatial_metric)(spatial)


def adm_mass(
    metric: MetricSpecification,
    r_surface: float = 100.0,
    n_theta: int = 16,
    n_phi: int = 32,
) -> Float[Array, ""]:
    """Compute ADM mass via surface integral on a coordinate sphere.

    M_ADM = (1/16pi) oint_S (dj gamma_{ij} - di tr(gamma)) n^i dA

    Uses Gauss-Legendre quadrature in theta and uniform quadrature in phi.
    Gauss-Legendre on ``u = cos(theta)`` absorbs the ``sin(theta)`` Jacobian
    into the weights, so the surface integral is a plain weighted sum over
    ``(theta, phi)``; the nodes are numpy compile-time constants.

    .. note::

       The value returned is the integral on a coordinate sphere of
       radius ``r_surface``. It converges to the ADM mass only when the
       metric satisfies the ADM falloff conditions
       (``g_ij - delta_ij ~ O(1/r)``, ``dg ~ O(1/r^2)``). Use
       :func:`adm_mass_richardson` or :func:`asymptotic_flatness_report`
       to verify that ``M_ADM`` has converged with increasing
       ``r_surface`` before quoting the value. Thick-shell warp metrics
       with extended-support density typically have a divergent
       coordinate-sphere integral - this is a physical statement about
       the spacetime, not a numerical artifact.

    Parameters
    ----------
    metric : MetricSpecification
    r_surface : radius of evaluation sphere (should be large)
    n_theta : number of theta quadrature points
    n_phi : number of phi quadrature points

    Returns
    -------
    M_ADM : ADM mass scalar (only physically meaningful when convergent
        with ``r_surface``; see note above)
    """
    r = jnp.float64(r_surface)

    import numpy as np

    cos_nodes, weights_theta = np.polynomial.legendre.leggauss(n_theta)
    cos_nodes = jnp.array(cos_nodes, dtype=jnp.float64)
    weights_theta = jnp.array(weights_theta, dtype=jnp.float64)
    theta_nodes = jnp.arccos(cos_nodes)

    phi_nodes = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False, dtype=jnp.float64)
    dphi = 2 * jnp.pi / n_phi

    def integrand_at_angle(theta: Float[Array, ""], phi: Float[Array, ""]) -> Float[Array, ""]:
        """ADM integrand at a single angular point on the sphere."""
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)

        x = r * sin_theta * cos_phi
        y = r * sin_theta * sin_phi
        z = r * cos_theta
        coords = jnp.array([0.0, x, y, z], dtype=jnp.float64)

        n_hat = jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])

        dgamma = _spatial_metric_derivs(metric, coords)

        div_gamma = jnp.einsum("ijj->i", dgamma)

        grad_trace = jnp.einsum("jji->i", dgamma)

        f = jnp.dot(div_gamma - grad_trace, n_hat)

        return f * r**2

    integrand_grid = jax.vmap(
        lambda th: jax.vmap(lambda ph: integrand_at_angle(th, ph))(phi_nodes)
    )(theta_nodes)
    phi_sums = jnp.sum(integrand_grid, axis=1) * dphi
    total = jnp.dot(weights_theta, phi_sums)

    M_ADM = total / (16.0 * jnp.pi)
    return M_ADM


def _richardson_order(radii: list[float], M_values: list[float], steps: int = 60) -> float:
    """Solve (M0-M1)/(M1-M2) = (r0^-p - r1^-p)/(r1^-p - r2^-p) for p by bisection on [0.1, 8]."""
    r0, r1, r2 = (float(r) for r in radii[-3:])
    M0, M1, M2 = (float(M) for M in M_values[-3:])
    target = (M0 - M1) / (M1 - M2)

    def residual(p: float) -> float:
        return (r0**-p - r1**-p) / (r1**-p - r2**-p) - target

    lo, hi = 0.1, 8.0
    f_lo = residual(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid * f_lo > 0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def adm_mass_richardson(
    metric: MetricSpecification,
    radii: list[float] | None = None,
    n_theta: int = 16,
    n_phi: int = 32,
) -> dict[str, object]:
    """ADM mass with Richardson extrapolation for convergence verification.

    Computes M_ADM at multiple radii and extrapolates to r -> inf assuming
    M(r) = M_inf + C r^-p. With three or more radii p is solved from the
    last three values (no geometric spacing assumed) and the extrapolation
    uses that p; with exactly two radii p = 1 is assumed and reported as NaN.
    The result is ``valid`` when the last two masses agree to 1e-12 relative
    (p reported as inf; two exact zeros count as agreement), or when
    |p - 1| < 0.25 and the last pair of masses is closer than the previous
    pair.

    Parameters
    ----------
    metric : MetricSpecification
    radii : list of evaluation radii (default: [50, 100, 200, 400])
    n_theta, n_phi : angular quadrature points

    Returns
    -------
    dict with 'M_values' (mass at each radius), 'radii', 'M_extrapolated',
    'convergence_order', 'valid'
    """
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0]

    M_values = [float(adm_mass(metric, r_surface=r, n_theta=n_theta, n_phi=n_phi)) for r in radii]

    if len(radii) < 2:
        return {
            "M_values": M_values,
            "radii": radii,
            "M_extrapolated": M_values[-1],
            "convergence_order": float("nan"),
            "valid": False,
        }

    r1, r2 = float(radii[-2]), float(radii[-1])
    M1, M2 = M_values[-2], M_values[-1]
    converged = abs(M1 - M2) <= 1e-12 * max(abs(M1), abs(M2))

    if len(radii) < 3:
        conv_order = float("nan")
        p = 1.0
        valid = True
    elif converged:
        conv_order = float("inf")
        p = 1.0
        valid = True
    else:
        conv_order = _richardson_order(radii, M_values)
        p = conv_order
        shrinking = abs(M1 - M2) < abs(M_values[-3] - M1)
        valid = abs(conv_order - 1.0) < 0.25 and shrinking

    M_extrap = (M2 * r2**p - M1 * r1**p) / (r2**p - r1**p) if valid else float("nan")

    return {
        "M_values": M_values,
        "radii": radii,
        "M_extrapolated": M_extrap,
        "convergence_order": conv_order,
        "valid": valid,
    }


_COMPONENT_INDICES = [
    (i, j, f"g_{a}{b}") for i, a in enumerate("txyz") for j, b in enumerate("txyz") if j >= i
]
_DIRECTIONS = [
    jnp.array([0.0, 1.0, 0.0, 0.0]),
    jnp.array([0.0, -1.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0, 0.0]),
    jnp.array([0.0, 0.0, -1.0, 0.0]),
    jnp.array([0.0, 0.0, 0.0, 1.0]),
    jnp.array([0.0, 0.0, 0.0, -1.0]),
]
_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))


def _deviations(metric: MetricSpecification, radii: list[float]) -> jnp.ndarray:
    """|g - eta| sampled along +-x, +-y, +-z at each radius, shape (6, n_radii, 4, 4)."""
    return jnp.stack(
        [jnp.stack([jnp.abs(metric(float(r) * d) - _ETA) for r in radii]) for d in _DIRECTIONS]
    )


def falloff_check(
    metric: MetricSpecification,
    r_test: float = 100.0,
    expected_order: int = 1,
    tol: float = 1e-2,
) -> dict[str, bool]:
    """Check asymptotic falloff of metric components.

    For asymptotically flat metrics, g_{munu} - eta_{munu} ~ O(1/r^n).
    Tests by comparing deviations at r and r/2 along +-x, +-y, +-z; a
    component passes only if it passes in every direction.

    Parameters
    ----------
    metric : MetricSpecification
    r_test : radius to evaluate falloff
    expected_order : expected power-law decay (1 for standard asymptotic flatness)
    tol : tolerance for falloff verification

    Returns
    -------
    dict mapping the ten component names g_tt, g_tx, ..., g_zz to bool
    """
    dev = _deviations(metric, [r_test / 2.0, r_test])
    dev_near, dev_far = dev[:, 0], dev[:, 1]
    eps = 1e-30
    measured_order = -jnp.log((dev_far + eps) / (dev_near + eps)) / jnp.log(2.0)
    passed = (dev_far <= 1e-15) | (measured_order >= expected_order - tol)
    return {name: bool(jnp.all(passed[:, i, j])) for i, j, name in _COMPONENT_INDICES}


def asymptotic_flatness_report(
    metric: MetricSpecification,
    radii: list[float] | None = None,
    expected_order: int = 1,
    tol: float = 0.1,
) -> dict[str, object]:
    """Full asymptotic flatness diagnostic.

    Evaluates |g - eta| at multiple radii along +-x, +-y, +-z, fits the
    power-law falloff exponent of every component from the last two radii
    in each direction, and passes a component only if it is exactly flat or
    falls off at least as fast as ``expected_order`` in every direction.

    Parameters
    ----------
    metric : MetricSpecification
    radii : list of evaluation radii (default: [50, 100, 200, 400])
    expected_order : expected falloff power (default: 1 for 1/r)
    tol : tolerance on measured exponent (default: 0.1)

    Returns
    -------
    dict with keys:
        'components' : dict per component name with 'passed',
                       'measured_order' (worst direction) and
                       'deviations' (list over directions of lists over radii)
        'diagonal', 'shift' : the g_tt, g_xx, g_yy, g_zz and g_tx, g_ty, g_tz
                       entries of 'components'
        'directions' : the six sampled unit directions
        'is_asymptotically_flat' : bool (all components pass)
        'radii' : list of radii used
    """
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0]

    dev = _deviations(metric, radii)
    results = {}
    for i, j, name in _COMPONENT_INDICES:
        d = dev[:, :, i, j]
        orders = []
        passed = True
        for k in range(len(_DIRECTIONS)):
            dk = [float(v) for v in d[k]]
            if all(v < 1e-12 for v in dk):
                measured = float("inf")
            elif len(radii) >= 2 and dk[-2] > 1e-15 and dk[-1] > 1e-15:
                measured = float(jnp.log(dk[-2] / dk[-1]) / jnp.log(radii[-1] / radii[-2]))
            else:
                measured = 0.0
            orders.append(measured)
            passed = passed and measured >= expected_order - tol
        results[name] = {
            "passed": bool(passed),
            "measured_order": min(orders),
            "deviations": [[float(v) for v in row] for row in d],
        }

    return {
        "components": results,
        "diagonal": {k: results[k] for k in ("g_tt", "g_xx", "g_yy", "g_zz")},
        "shift": {k: results[k] for k in ("g_tx", "g_ty", "g_tz")},
        "directions": ["+x", "-x", "+y", "-y", "+z", "-z"],
        "is_asymptotically_flat": all(r["passed"] for r in results.values()),
        "radii": radii,
    }
