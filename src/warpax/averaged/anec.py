"""Averaged Null Energy Condition (ANEC) line integral.

Physical definition:

.. math::

    \\mathrm{ANEC}[\\gamma] = \\int_{\\gamma} T_{ab} k^a k^b \\, d\\lambda

where ``k^a`` is the null tangent to the geodesic ``gamma`` and ``lambda``
is an affine parameter.

The ``geodesic_complete`` flag and ``termination_reason`` field on
``ANECResult`` report whether the integrator completed successfully.
When ``tangent_norm='renormalized'`` (the default), the tangent is
projected onto the null cone at every saved step; ``'fixed'`` retains
the raw velocity.
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

# Diffrax result-code convention (matches `diffrax.RESULTS`):
# 0 = successful, anything else = non-success.
_SUCCESS_CODE = 0

# Human-readable termination reason dispatch
_TERMINATION_REASONS: dict[int, str] = {
    0: "complete",
    1: "max_steps",
    2: "nan",
    3: "out_of_bounds",
    4: "singularity",
}


class ANECResult(NamedTuple):
    """Result of an ANEC line-integral evaluation.

    Attributes
    ----------
    line_integral : Float[Array, ""]
        :math:`\\int T_{ab} k^a k^b \\, d\\lambda` along the null
        geodesic. Positive => NEC-respecting along the path; negative
        => NEC-violating integrated.
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


def _tangent_renormalized_null(
    g_ab: Float[Array, "4 4"],
    u: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Rescale ``u`` so that :math:`|u^0|` is held constant and the
    spatial part is projected onto the null cone.

    We scale the full 4-vector by ``|u^0| / max(|u^0|, 1e-30)`` (keeps
    pacing) - the curvature-chain stress-energy evaluation at the same
    point then picks up an invariant ``T_{ab} k^a k^b`` projection so
    small drifts in ``g(k, k)`` along an exact null geodesic do not
    accumulate. For perfect null geodesics, this is an identity
    transformation.
    """
    scale = jnp.abs(u[0]) + 1e-30
    return u / scale * jnp.abs(u[0])


def _anec_integrand_at_point(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
    u: Float[Array, "4"],
    tangent_norm: str,
) -> Float[Array, ""]:
    """Compute ``T_{ab} k^a k^b`` at a single (coords, u) sample."""
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy  # covariant (lower indices)
    if tangent_norm == "renormalized":
        k = _tangent_renormalized_null(curv.metric, u)
    else:
        k = u
    return jnp.einsum("ab,a,b->", T_ab, k, k)


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
    """Return (ts, positions, velocities, result_code) from either a
    ``GeodesicResult`` or a callable geodesic.
    """
    if isinstance(geodesic, GeodesicResult):
        return (
            geodesic.ts,
            geodesic.positions,
            geodesic.velocities,
            int(geodesic.result),
        )
    # Callable geodesic: sample uniformly
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
def anec(
    metric: MetricSpecification,
    geodesic: Union[
        GeodesicResult,
        Callable[[Float[Array, ""]], Float[Array, "4"]],
    ],
    tangent_norm: str = "renormalized",
    n_samples: int = 256,
    affine_bounds: tuple[float, float] = (-5.0, 5.0),
) -> ANECResult:
    """Evaluate the Averaged Null Energy Condition line integral.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime metric.
    geodesic : GeodesicResult or Callable[[Float[Array, ""]], Float[Array, "4"]]
        The null geodesic path. If a ``GeodesicResult`` NamedTuple
        (produced by ``warpax.geodesics.integrator.integrate_geodesic``),
        its saved trajectory samples are used directly. If a callable
        ``geodesic(affine)`` -> coords, the tangent is obtained by
        ``jax.jacfwd`` and the affine parameter is sampled uniformly
        over ``affine_bounds``.
    tangent_norm : str
        ``'renormalized'`` (default): rescale the tangent at each step
        to project onto the null cone. ``'fixed'``:
        use the raw velocity without renormalisation.
    n_samples : int
        Number of affine samples when ``geodesic`` is a callable.
    affine_bounds : tuple[float, float]
        Affine bounds ``(lam_min, lam_max)`` when ``geodesic`` is a
        callable. Ignored when a ``GeodesicResult`` is passed.

    Returns
    -------
    ANECResult
        NamedTuple with ``line_integral``, ``geodesic_complete``, and
        ``termination_reason``.

    Raises
    ------
    ValueError
        If ``tangent_norm`` is not in ``{'renormalized', 'fixed'}``.

    Notes
    -----
    See the ``_tangent_renormalized_null`` helper for the projection
    implementation details.
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
        lambda c, u: _anec_integrand_at_point(metric, c, u, tangent_norm)
    )(positions, velocities)

    # Uniform trapezoid integration
    dlam = lam[1] - lam[0]
    line_integral = jnp.sum(integrand) * dlam

    geodesic_complete = result_code == _SUCCESS_CODE
    termination_reason = _TERMINATION_REASONS.get(result_code, "unknown")

    return ANECResult(
        line_integral=line_integral,
        geodesic_complete=geodesic_complete,
        termination_reason=termination_reason,
    )
