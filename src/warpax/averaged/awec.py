"""Averaged Weak Energy Condition (AWEC) line integral.

AWEC:

.. math::

    \\mathrm{AWEC}[\\gamma] = \\int_{\\gamma} T_{ab} u^a u^b \\, d\\tau

where ``u^a`` is the timelike 4-velocity of a timelike geodesic
``gamma`` and ``tau`` is proper time.

Shares the ``geodesic_complete`` + ``termination_reason`` flags with
``anec``; tangent-norm renormalisation projects ``u`` onto the
timelike-unit hyperboloid at every saved step (``g(u,u) = -1``).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geodesics.integrator import GeodesicResult
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


_VALID_TANGENT_NORM = frozenset({"renormalized", "fixed"})
_SUCCESS_CODE = 0
_TERMINATION_REASONS: dict[int, str] = {
    0: "complete",
    1: "max_steps",
    2: "nan",
    3: "out_of_bounds",
    4: "singularity",
}


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
        termination .
    termination_reason : str
        Human-readable reason: ``'complete'``, ``'max_steps'``,
        ``'nan'``, ``'out_of_bounds'``, or ``'singularity'``.
    """

    line_integral: Float[Array, ""]
    geodesic_complete: bool
    termination_reason: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tangent_renormalized_timelike(
    g_ab: Float[Array, "4 4"],
    u: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Rescale ``u`` so that ``g_{ab} u^a u^b = -1``.

    Robust against non-strictly-timelike numerical drift via the
    absolute-value norm + small epsilon floor.
    """
    u_sq = jnp.einsum("a,ab,b->", u, g_ab, u)
    scale = jnp.sqrt(jnp.abs(u_sq) + 1e-30)
    return u / scale


def _awec_integrand_at_point(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
    u: Float[Array, "4"],
    tangent_norm: str,
) -> Float[Array, ""]:
    """Compute ``T_{ab} u^a u^b`` at a single sample."""
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy
    if tangent_norm == "renormalized":
        u_final = _tangent_renormalized_timelike(curv.metric, u)
    else:
        u_final = u
    return jnp.einsum("ab,a,b->", T_ab, u_final, u_final)


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
        return (
            geodesic.ts,
            geodesic.positions,
            geodesic.velocities,
            int(geodesic.result),
        )
    lam_min, lam_max = affine_bounds
    lam = jnp.linspace(lam_min, lam_max, n_samples)

    def _pos(l):
        return geodesic(l)

    positions = jax.vmap(_pos)(lam)
    velocities = jax.vmap(jax.jacfwd(_pos))(lam)
    return lam, positions, velocities, _SUCCESS_CODE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        ``'fixed'``: raw velocity without renormalisation.
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

    integrand = jax.vmap(
        lambda c, u: _awec_integrand_at_point(metric, c, u, tangent_norm)
    )(positions, velocities)

    dlam = lam[1] - lam[0]
    line_integral = jnp.sum(integrand) * dlam

    geodesic_complete = result_code == _SUCCESS_CODE
    termination_reason = _TERMINATION_REASONS.get(result_code, "unknown")

    return AWECResult(
        line_integral=line_integral,
        geodesic_complete=geodesic_complete,
        termination_reason=termination_reason,
    )
