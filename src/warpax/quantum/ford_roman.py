"""Ford-Roman Quantum Inequality.

Citations:

- Fewster, C. J. (2012). "Quantum Inequalities: recent developments"
  eq. (2.1) - pinned at

- Pretto, A. et al. (2024). "Quantum Inequalities and Sampling
  Prescriptions." *Phys. Rev. D* 110, 024023 - Lorentzian sampling
  justification.

Definitions (massless scalar field, 4D):

.. math::

    \\int \\rho_T(\\tau) f(\\tau)^2 \\, d\\tau \\ge - \\frac{C}{\\tau_0^4}

with ``C = 3 / (32 pi^2)`` and the Lorentzian sampling kernel
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

# Ford-Roman constant for the massless scalar field, 4D (Fewster 2012
# eq. 2.1). Pinned for see
#
FORD_ROMAN_CONSTANT_C: float = 3.0 / (32.0 * jnp.pi ** 2)


class QIResult(NamedTuple):
    """Ford-Roman quantum-inequality result.

    Attributes
    ----------
    margin : Float[Array, ""]
        Signed QI margin:
        ``integral(rho * f^2 d_tau) - (- C / tau0^4)``.
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lorentzian_kernel(
    tau: Float[Array, "N"], tau0: float
) -> Float[Array, "N"]:
    """Lorentzian temporal sampling kernel (Pretto 2024).

    :math:`f(\\tau) = (\\tau_0 / \\pi) / (\\tau^2 + \\tau_0^2)`.
    """
    return (tau0 / jnp.pi) / (tau ** 2 + tau0 ** 2)


def _rho_at_tau(
    metric: MetricSpecification,
    worldline: Callable[[Float[Array, ""]], Float[Array, "4"]],
    tau: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute ``rho = T_{ab} u^a u^b`` at proper-time ``tau``."""
    coords = worldline(tau)
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy  # covariant (lower indices)
    g = curv.metric

    # 4-velocity from worldline derivative; renormalise to g(u,u) = -1
    u_raw = jax.jacfwd(worldline)(tau)
    u_sq = jnp.einsum("a,ab,b->", u_raw, g, u_raw)
    scale = jnp.sqrt(jnp.abs(u_sq) + 1e-30)
    u = u_raw / scale

    # rho = T_{ab} u^a u^b
    rho = jnp.einsum("ab,a,b->", T_ab, u, u)
    return rho


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        Temporal sampling kernel; only ``'lorentzian'`` is supported in
        this release (per Pretto 2024). See research anchor

    n_samples : int
        Number of proper-time samples for the QI line integral.
        Default ``256`` - span ``[-10 tau0, +10 tau0]`` captures ~99% of
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

    Notes
    -----
    The QI bound for a massless scalar field (Fewster 2012 eq. 2.1):

    .. math::

        \\int \\rho_T(\\tau) f(\\tau)^2 \\, d\\tau \\ge - \\frac{C}{\\tau_0^4}

    where ``C = 3 / (32 pi^2)`` and ``f(tau) = (tau0 / pi) / (tau^2 +
    tau0^2)`` is the Lorentzian sampling kernel.
    """
    if sampling != "lorentzian":
        raise ValueError(
            f"sampling must be 'lorentzian' (only supported kernel), got {sampling!r}"
        )

    tau = jnp.linspace(-10.0 * tau0, 10.0 * tau0, n_samples)
    dtau = tau[1] - tau[0]
    f_vals = _lorentzian_kernel(tau, tau0)

    rho_vals = jax.vmap(lambda t: _rho_at_tau(metric, worldline, t))(tau)
    integrand = rho_vals * f_vals ** 2
    integral = jnp.sum(integrand) * dtau

    C = jnp.asarray(FORD_ROMAN_CONSTANT_C)
    bound = -C / tau0 ** 4
    margin = integral - bound

    return QIResult(
        margin=margin,
        bound=bound,
        C=C,
    )
