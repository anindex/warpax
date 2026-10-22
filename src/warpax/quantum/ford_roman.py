"""Ford-Roman Quantum Inequality.

Citations:

- Ford, L. H. & Roman, T. A. (1995). *Phys. Rev. D* 51, 4277.
- Pfenning, M. J. & Ford, L. H. (1997). gr-qc/9702026, eq. (9).
- Fewster, C. J. (2012). "Lectures on quantum energy inequalities,"
  arXiv:1208.5399.

Definitions (massless scalar field, 4D, flat-space form):

.. math::

    \\int \\rho_T(\\tau) f(\\tau) \\, d\\tau \\ge - \\frac{\\hbar C}{\\tau_0^4}

with ``C = 3 / (32 pi^2)`` and the normalized Lorentzian sampling kernel
``f(\\tau) = (\\tau_0 / \\pi) / (\\tau^2 + \\tau_0^2)``.

Units: the stress-energy is geometric (``G = c = 1``, dimension ``L^-2``)
while ``tau0^-4`` has dimension ``L^-4``, so the bound carries ``hbar``;
this module sets ``hbar = 1`` (Planck units, lengths in Planck lengths).

Applicability: this is the flat-space massless-scalar free-field bound.
Pfenning & Ford (1997) restrict its use in curved spacetime to sampling
times ``tau0`` short compared with the local curvature scale; the margin
reported here is a diagnostic under that assumption, not a theorem for
the warp spacetime.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification

FORD_ROMAN_CONSTANT_C: float = 3.0 / (32.0 * jnp.pi**2)


class QIResult(NamedTuple):
    """Ford-Roman quantum-inequality result.

    Attributes
    ----------
    margin : Float[Array, ""]
        Signed QI margin:
        ``integral(rho * f d_tau) - (- hbar C / tau0^4)`` with ``hbar = 1``.
        Positive => QI satisfied along the worldline.
    bound : Float[Array, ""]
        The Ford-Roman bound value ``- hbar C / tau0^4`` with ``hbar = 1``.
    C : Float[Array, ""]
        The Ford-Roman constant ``3 / (32 pi^2)`` for the massless
        scalar field.
    kernel_weight : Float[Array, ""]
        Fraction of the Lorentzian kernel mass covered by the sampled
        proper-time window, ``(arctan(tau_max/tau0) - arctan(tau_min/tau0)) / pi``
        (about 0.99936 for the default window reaching ``+-1000 tau0``).
    """

    margin: Float[Array, ""]
    bound: Float[Array, ""]
    C: Float[Array, ""]
    kernel_weight: Float[Array, ""] = jnp.asarray(1.0)


def _lorentzian_kernel(tau: Float[Array, "N"], tau0: float) -> Float[Array, "N"]:
    """Normalized Lorentzian temporal sampling kernel.

    :math:`f(\\tau) = (\\tau_0 / \\pi) / (\\tau^2 + \\tau_0^2)`.
    """
    return (tau0 / jnp.pi) / (tau**2 + tau0**2)


def _rate(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    lam: Float[Array, ""],
) -> Float[Array, ""]:
    """``dtau/dlambda`` at worldline parameter ``lam``.

    NaN where the tangent is not timelike, inf where it overflows.
    """
    u_raw = jax.jacfwd(worldline)(lam)
    u_sq = jnp.einsum("a,ab,b->", u_raw, metric(worldline(lam)), u_raw)
    u_sq = jnp.where(jnp.all(jnp.isfinite(u_raw)), u_sq, -jnp.inf)
    return jnp.sqrt(jnp.where(u_sq < 0, -u_sq, jnp.nan))


def _rho_and_rate(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    lam: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """``(rho, dtau/dlambda)`` at worldline parameter ``lam``.

    ``rate`` is NaN where the tangent is not timelike.
    """
    coords = worldline(lam)
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy
    g = curv.metric

    u_raw = jax.jacfwd(worldline)(lam)
    u_sq = jnp.einsum("a,ab,b->", u_raw, g, u_raw)
    rate = jnp.sqrt(jnp.where(u_sq < 0, -u_sq, jnp.nan))
    u = u_raw / rate

    return jnp.einsum("ab,a,b->", T_ab, u, u), rate


def _rho_at_tau(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    tau: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute ``rho = T_{ab} u^a u^b`` at worldline parameter ``tau``."""
    return _rho_and_rate(metric, worldline, tau)[0]


def _proper_time_map(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    core: float,
    far: float,
    n_core: int,
) -> tuple[Float[Array, "M"], Float[Array, "M"]]:
    """Parameter grid ``lam`` and proper time ``tau(lam)`` with ``tau(0) = 0``.

    The grid is a uniform core plus log-spaced tails in the parameter. Its
    extents are rescaled until the core reaches about ``+-core`` and the
    tails about ``+-far`` in proper time: an extent the tabulated map already
    covers is read off by inverting the map, an extent it falls short of is
    doubled relative to the reach, and an extent where the rate overflows is
    pulled back below the first overflowing node.
    """
    h_core, h_far = core, far
    for _ in range(8):
        tail = jnp.logspace(jnp.log10(h_core), jnp.log10(h_far), 257)[1:]
        lam = jnp.concatenate([-tail[::-1], jnp.linspace(-h_core, h_core, n_core), tail])
        rate = jax.vmap(lambda t: _rate(metric, worldline, t))(lam)
        if bool(jnp.any(jnp.isnan(rate))):
            raise ValueError("worldline not timelike")
        finite = jnp.isfinite(rate)
        if not bool(jnp.all(finite)):
            h_far = 0.5 * float(jnp.min(jnp.abs(lam[~finite])))
            h_core = min(h_core, h_far / 1.5)
            continue
        tau = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (rate[1:] + rate[:-1]) * jnp.diff(lam))]
        )
        tau = tau - jnp.interp(0.0, lam, tau)
        n_tail = tail.shape[0]
        reach_core = float(jnp.minimum(-tau[n_tail], tau[-n_tail - 1]))
        reach_far = float(jnp.minimum(-tau[0], tau[-1]))
        ok_core = core * (1.0 - 1e-6) <= reach_core <= 2.0 * core
        ok_far = far * (1.0 - 1e-6) <= reach_far <= 2.0 * far
        if ok_core and ok_far:
            break

        def extent(target: float, h: float) -> float:
            if reach_far >= target:
                return float(
                    jnp.maximum(jnp.interp(target, tau, lam), -jnp.interp(-target, tau, lam))
                )
            return h * 2.0 * target / max(reach_far, 1e-300)

        h_core = extent(core, h_core)
        h_far = max(extent(far, h_far), 1.5 * h_core)
    return lam, tau


@jaxtyped(typechecker=beartype)
def ford_roman(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    tau0: float,
    sampling: str = "lorentzian",
    n_samples: int = 256,
) -> QIResult:
    """Evaluate the Ford-Roman quantum inequality along a timelike worldline.

    The bound is ``-hbar C / tau0^4`` with ``hbar = 1`` (Planck units) for
    ``G = c = 1`` geometric stress-energy. It is the flat-space massless
    scalar free-field bound; Pfenning & Ford (1997) restrict its use in
    curved spacetime to ``tau0`` short compared with the curvature scale.

    The kernel width and measure are proper time, but the worldline carries
    the caller's parameter ``lambda``. Proper time ``tau(lambda)`` is first
    tabulated by integrating ``dtau/dlambda`` along the worldline (zero at
    ``lambda = 0``) and inverted by interpolation, so the sample grid is
    built in proper time and the result does not depend on the
    parameterization. The grid is a uniform core of ``n_samples`` points on
    ``[-10 tau0, 10 tau0]`` plus 64 log-spaced nodes per side out to
    ``+-1000 tau0`` (or as far as the worldline stays timelike); the
    captured kernel mass is reported as ``kernel_weight``.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime.
    worldline : Callable[[lam], Float[Array, "4"]]
        Timelike worldline; the parameter need not be proper time.
    tau0 : float
        Characteristic sampling width (inverse sampling frequency).
    sampling : str
        Temporal sampling kernel; only ``'lorentzian'`` is supported
        (the kernel of the original Ford-Roman bound).
    n_samples : int
        Number of uniform samples in the ``[-10 tau0, 10 tau0]`` core.

    Returns
    -------
    QIResult
        NamedTuple with ``margin`` (positive => QI satisfied), the
        Ford-Roman ``bound`` value, the constant ``C`` and ``kernel_weight``.

    Raises
    ------
    ValueError
        If ``sampling`` is not ``'lorentzian'`` or the worldline is not
        timelike on the sampled window.
    """
    if sampling != "lorentzian":
        raise ValueError(f"sampling must be 'lorentzian' (only supported kernel), got {sampling!r}")

    core = 10.0 * tau0
    far = 100.0 * core
    lam_map, tau_map = _proper_time_map(metric, worldline, core, far, 4 * n_samples + 1)
    reach = jnp.minimum(far, jnp.minimum(-tau_map[0], tau_map[-1]))
    tail = jnp.logspace(jnp.log10(core), jnp.log10(reach), 65)[1:]
    tau = jnp.concatenate([-tail[::-1], jnp.linspace(-core, core, n_samples), tail])
    lam = jnp.interp(tau, tau_map, lam_map)
    rho_vals = jax.vmap(lambda t: _rho_and_rate(metric, worldline, t)[0])(lam)
    integral = jnp.trapezoid(rho_vals * _lorentzian_kernel(tau, tau0), x=tau)
    kernel_weight = (jnp.arctan(tau[-1] / tau0) - jnp.arctan(tau[0] / tau0)) / jnp.pi

    C = jnp.asarray(FORD_ROMAN_CONSTANT_C)
    bound = -C / tau0**4
    margin = integral - bound

    return QIResult(
        margin=margin,
        bound=bound,
        C=C,
        kernel_weight=kernel_weight,
    )
