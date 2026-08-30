"""Averaged Weak Energy Condition (AWEC) line integral.

AWEC:

.. math::

    \\mathrm{AWEC}[\\gamma] = \\int_{\\gamma} T_{ab} u^a u^b \\, d\\tau

where ``u^a`` is the timelike 4-velocity of a timelike geodesic
``gamma`` and ``tau`` is proper time.

Shares the ``geodesic_complete`` + ``termination_reason`` flags with
``anec``; tangent-norm renormalization projects ``u`` onto the
timelike-unit hyperboloid at every saved step (``g(u,u) = -1``).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geodesics._result_codes import (
    RESULT_SUCCESS,
    result_code_to_int,
    termination_reason,
)
from ..geodesics.integrator import GeodesicResult
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


_VALID_TANGENT_NORM = frozenset({"renormalized", "fixed"})


class AWECResult(NamedTuple):
    """Result of an AWEC line-integral evaluation.

    Attributes
    ----------
    line_integral : Float[Array, ""]
        :math:`\\int T_{ab} u^a u^b \\, d\\tau` along the timelike
        geodesic. Positive => WEC-respecting along the path; negative
        => WEC-violating integrated.
    geodesic_complete : bool
        True iff the Diffrax integrator completed without early
        termination.
    termination_reason : str
        Human-readable reason: ``'complete'`` on success; otherwise a
        Diffrax failure mode (e.g. ``'max_steps'``, ``'nonfinite'``,
        ``'dt_min_reached'``, ``'event_occurred'``) or ``'unknown'``
        for an unrecognized result code.
    timelike_preserved : bool
        True iff every sampled tangent is timelike. AWEC is defined on a
        timelike curve; without this a spacelike one integrates silently.
    max_u_sq : Float[Array, ""]
        Worst (least negative) ``g_{ab} u^a u^b`` over the samples.
    """

    line_integral: Float[Array, ""]
    geodesic_complete: bool
    termination_reason: str
    timelike_preserved: bool = True
    max_u_sq: Float[Array, ""] = None


def _tangent_renormalized_timelike(
    g_ab: Float[Array, "4 4"],
    u: Float[Array, "4"],
) -> tuple[Float[Array, "4"], Float[Array, ""], Float[Array, ""]]:
    """Rescale ``u`` so that ``g_{ab} u^a u^b = -1``.

    Returns the rescaled tangent, ``dtau/dlambda = sqrt(-g(u,u))``, and the
    unnormalised ``g(u,u)`` as a causal witness. Taking ``sqrt(abs(.))``
    instead accepted a spacelike curve and returned it with ``g(u,u) = +1``.
    """
    u_sq = jnp.einsum("a,ab,b->", u, g_ab, u)
    rate = jnp.sqrt(jnp.clip(-u_sq, min=0.0))
    scale = jnp.where(rate > 0.0, rate, 1.0)
    return u / scale, rate, u_sq


def _awec_integrand_at_point(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
    u: Float[Array, "4"],
    tangent_norm: str,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """``T_{ab} u^a u^b``, ``dtau/dlambda``, and ``g(u,u)`` at a single sample."""
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy
    u_sq = jnp.einsum("a,ab,b->", u, curv.metric, u)
    if tangent_norm == "renormalized":
        u_final, rate, u_sq = _tangent_renormalized_timelike(curv.metric, u)
    else:
        u_final, rate = u, jnp.ones_like(u_sq)
    return jnp.einsum("ab,a,b->", T_ab, u_final, u_final), rate, u_sq


def _extract_trajectory(
    metric: MetricSpecification,
    geodesic: Union[
        GeodesicResult,
        Callable[[Float[Array, ""]], Float[Array, "4"]],
    ],
    n_samples: int,
    affine_bounds: tuple[float, float],
) -> tuple[
    Float[Array, "N"], Float[Array, "N 4"], Float[Array, "N 4"], int
]:
    """Return (ts, positions, velocities, result_code)."""
    if isinstance(geodesic, GeodesicResult):
        # Robust conversion (diffrax 0.7.x EnumerationItem is not
        # int()-convertible); never defaults to success.
        return (
            geodesic.ts,
            geodesic.positions,
            geodesic.velocities,
            result_code_to_int(geodesic.result),
        )
    lam_min, lam_max = affine_bounds
    lam = jnp.linspace(lam_min, lam_max, n_samples)

    def _pos(l):
        return geodesic(l)

    positions = jax.vmap(_pos)(lam)
    velocities = jax.vmap(jax.jacfwd(_pos))(lam)
    return lam, positions, velocities, RESULT_SUCCESS


@jaxtyped(typechecker=beartype)
def awec(
    metric: MetricSpecification,
    geodesic: Union[
        GeodesicResult,
        Callable[[Float[Array, ""]], Float[Array, "4"]],
    ],
    tangent_norm: str = "renormalized",
    n_samples: int = 256,
    affine_bounds: tuple[float, float] = (-5.0, 5.0),
) -> AWECResult:
    """Evaluate the Averaged Weak Energy Condition line integral.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime metric.
    geodesic : GeodesicResult or Callable[[Float[Array, ""]], Float[Array, "4"]]
        Timelike geodesic path, either as a ``GeodesicResult`` from
        ``warpax.geodesics.integrator.integrate_geodesic`` or as a
        callable ``geodesic(tau) -> coords``.
    tangent_norm : str
        ``'renormalized'`` (default): rescale ``u`` so ``g(u,u) = -1``.
        ``'fixed'``: raw velocity without renormalization.
    n_samples : int
        Number of proper-time samples when ``geodesic`` is callable.
    affine_bounds : tuple[float, float]
        ``(tau_min, tau_max)`` when ``geodesic`` is callable.

    Returns
    -------
    AWECResult
        NamedTuple with ``line_integral``, ``geodesic_complete``, and
        ``termination_reason``.

    Raises
    ------
    ValueError
        If ``tangent_norm`` not in ``{'renormalized', 'fixed'}``.
    """
    if tangent_norm not in _VALID_TANGENT_NORM:
        raise ValueError(
            f"tangent_norm must be one of {sorted(_VALID_TANGENT_NORM)}, "
            f"got {tangent_norm!r}"
        )

    lam, positions, velocities, result_code = _extract_trajectory(
        metric, geodesic, n_samples, affine_bounds
    )

    integrand, rate, u_sq = jax.vmap(
        lambda c, u: _awec_integrand_at_point(metric, c, u, tangent_norm)
    )(positions, velocities)

    # Integrate in proper time. Renormalising the tangent means lambda is no
    # longer tau, so the measure needs dtau/dlambda; ford_roman.py carries it,
    # this did not.
    line_integral = jnp.trapezoid(integrand * rate, lam)

    # Causal witness on the unnormalised tangent, relative to the sample scale.
    u_scale = jnp.max(jnp.abs(u_sq))
    timelike = bool(jnp.all(u_sq < -1e-10 * jnp.where(u_scale > 0.0, u_scale, 1.0)))

    geodesic_complete = result_code == RESULT_SUCCESS

    return AWECResult(
        line_integral=line_integral,
        geodesic_complete=geodesic_complete,
        termination_reason=termination_reason(result_code),
        timelike_preserved=timelike,
        max_u_sq=jnp.max(u_sq),
    )
