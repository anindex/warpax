"""Ford-Roman Quantum Inequality.

Citations:

- Ford, L. H. & Roman, T. A. (1995). *Phys. Rev. D* 51, 4277.
- Pfenning, M. J. & Ford, L. H. (1997). gr-qc/9702026, eq. (9).
- Fewster, C. J. (2012). "Lectures on quantum energy inequalities,"
  arXiv:1208.5399.

Definitions (massless scalar field, 4D, flat-space form):

.. math::

    \\int \\rho_T(\\tau) f(\\tau) \\, d\\tau \\ge - \\frac{C}{\\tau_0^4}

with ``C = 3 / (32 pi^2)`` and the normalized Lorentzian sampling kernel
``f(\\tau) = (\\tau_0 / \\pi) / (\\tau^2 + \\tau_0^2)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification

# Ford-Roman constant for the massless scalar field, 4D
# (Ford & Roman 1995; Pfenning & Ford 1997 eq. 9).
FORD_ROMAN_CONSTANT_C: float = 3.0 / (32.0 * jnp.pi**2)


class QIResult(NamedTuple):
    """Ford-Roman quantum-inequality result.

    Attributes
    ----------
    margin : Float[Array, ""]
        Signed QI margin:
        ``integral(rho * f d_tau) - (- C / tau0^4)``.
        Positive => QI satisfied along the worldline.
    bound : Float[Array, ""]
        The Ford-Roman bound value ``- C / tau0^4``.
    C : Float[Array, ""]
        The Ford-Roman constant ``3 / (32 pi^2)`` for the massless
        scalar field.
    """

    margin: Float[Array, ""]
    bound: Float[Array, ""]
    C: Float[Array, ""]


def _lorentzian_kernel(tau: Float[Array, "N"], tau0: float) -> Float[Array, "N"]:
    """Normalized Lorentzian temporal sampling kernel.

    :math:`f(\\tau) = (\\tau_0 / \\pi) / (\\tau^2 + \\tau_0^2)`.
    """
    return (tau0 / jnp.pi) / (tau**2 + tau0**2)


def _rho_and_rate(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    lam: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """``(rho, dtau/dlambda)`` at worldline parameter ``lam``."""
    coords = worldline(lam)
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy  # covariant (lower indices)
    g = curv.metric

    # 4-velocity from worldline derivative; renormalize to g(u,u) = -1
    u_raw = jax.jacfwd(worldline)(lam)
    u_sq = jnp.einsum("a,ab,b->", u_raw, g, u_raw)
    rate = jnp.sqrt(jnp.abs(u_sq) + 1e-30)
    u = u_raw / rate

    return jnp.einsum("ab,a,b->", T_ab, u, u), rate


def _rho_at_tau(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    tau: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute ``rho = T_{ab} u^a u^b`` at worldline parameter ``tau``."""
    return _rho_and_rate(metric, worldline, tau)[0]


@jaxtyped(typechecker=beartype)
def ford_roman(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    tau0: float,
    sampling: str = "lorentzian",
    n_samples: int = 256,
) -> QIResult:
    """Evaluate the Ford-Roman quantum inequality along a timelike worldline.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime.
    worldline : Callable[[tau], Float[Array, "4"]]
        Timelike worldline parametrized by proper time ``tau``.
    tau0 : float
        Characteristic sampling width (inverse sampling frequency).
    sampling : str
        Temporal sampling kernel; only ``'lorentzian'`` is supported
        (the kernel of the original Ford-Roman bound).
    n_samples : int
        Number of proper-time samples for the QI line integral.
        Default ``256`` - span ``[-10 tau0, +10 tau0]`` captures ~94% of
        the Lorentzian kernel weight.

    Returns
    -------
    QIResult
        NamedTuple with ``margin`` (positive => QI satisfied), the
        Ford-Roman ``bound`` value, and the constant ``C``.

    Raises
    ------
    ValueError
        If ``sampling`` is not ``'lorentzian'``.
    """
    if sampling != "lorentzian":
        raise ValueError(f"sampling must be 'lorentzian' (only supported kernel), got {sampling!r}")

    # The kernel width, measure and span are PROPER time, but the worldline carries the
    # caller's parameter, so integrate there and carry dtau/dlambda. A coordinate-static
    # observer in Alcubierre has rate = sqrt(1 - v_s^2 f^2) < 1 inside the wall.
    span = 10.0 * tau0
    half = span
    for _ in range(3):
        lam = jnp.linspace(-half, half, n_samples)
        rho_vals, rate = jax.vmap(lambda t: _rho_and_rate(metric, worldline, t))(lam)
        dlam = lam[1] - lam[0]
        tau = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (rate[1:] + rate[:-1]) * dlam)])
        tau = tau - tau[n_samples // 2]
        reach = float(jnp.minimum(-tau[0], tau[-1]))
        if reach >= span * (1.0 - 1e-6):
            break
        half += span - reach  # rate -> 1 outside the wall, so this converges
    integrand = rho_vals * _lorentzian_kernel(tau, tau0) * rate
    integral = jnp.trapezoid(integrand, dx=dlam)

    C = jnp.asarray(FORD_ROMAN_CONSTANT_C)
    bound = -C / tau0**4
    margin = integral - bound

    return QIResult(
        margin=margin,
        bound=bound,
        C=C,
    )
