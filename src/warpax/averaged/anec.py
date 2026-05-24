"""Averaged Null Energy Condition (ANEC) line integral.

Physical definition:

.. math::

    \\mathrm{ANEC}[\\gamma] = \\int_{\\gamma} T_{ab} k^a k^b \\, d\\lambda

where ``k^a`` is the null tangent to the geodesic ``gamma`` and ``lambda``
is an affine parameter.

The ``geodesic_complete`` flag and ``termination_reason`` field on
``ANECResult`` report whether the integrator completed successfully.

.. note::

   ``tangent_norm='renormalized'`` (the default) is a near-identity
   rescale that protects against zero-``u^0`` numerical degeneracies;
   it does **not** project the tangent onto the null cone. The
   line integral is therefore evaluated along the geodesic tangent
   produced by Diffrax, which can drift from exact null by integrator
   tolerance. ``'fixed'`` selects the unmodified tangent. Treat the
   resulting ANEC value as a coordinate-ray integral; for an exact
   null-cone observable, decrease ``rtol``/``atol`` on the geodesic
   integrator. See :func:`_tangent_renormalized_null` for the
   implementation detail.
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


_VALID_TANGENT_NORM = frozenset({"renormalized", "fixed", "null_projected"})

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


def _tangent_renormalized_null(
    g_ab: Float[Array, "4 4"],
    u: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Identity rescale guarded against ``u[0] = 0`` division.

    Historical misnomer: this does **not** project onto the null cone -
    use ``tangent_norm='null_projected'`` for that. The integrand
    ``T_{ab} k^a k^b`` is evaluated on the unprojected tangent.
    """
    scale = jnp.abs(u[0]) + 1e-30
    return u / scale * jnp.abs(u[0])


def _project_to_null(
    g_ab: Float[Array, "4 4"],
    u: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Rescale spatial part of ``u`` so ``g(k, k) = 0`` exactly.

    Splits ``u = u_t e_0 + u_s`` (along the coordinate ``t``-axis) and
    solves the quadratic for ``lambda`` in ``k = u_t e_0 + lambda u_s``.
    Exact on Minkowski-like metrics; stable Lagrange projection elsewhere.
    """
    e0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=u.dtype)
    u_t = u[0]
    u_s = u.at[0].set(0.0)
    A_t = jnp.einsum("ab,a,b->", g_ab, u_t * e0, u_t * e0)
    A_s = jnp.einsum("ab,a,b->", g_ab, u_s, u_s)
    cross = 2.0 * jnp.einsum("ab,a,b->", g_ab, u_t * e0, u_s)
    disc = cross**2 - 4.0 * A_s * A_t
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    lam = (-cross + sqrt_disc) / (2.0 * A_s + 1e-30)
    return u_t * e0 + lam * u_s


def _anec_integrand_at_point(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
    u: Float[Array, "4"],
    tangent_norm: str,
) -> Float[Array, ""]:
    """Compute ``T_{ab} k^a k^b`` at a single (coords, u) sample."""
    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy
    if tangent_norm == "renormalized":
        k = _tangent_renormalized_null(curv.metric, u)
    elif tangent_norm == "null_projected":
        k = _project_to_null(curv.metric, u)
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
        # Diffrax ``result`` may be a typed enumeration; coerce to plain int
        # via the underlying numeric value (.value if present, raw int else).
        raw = geodesic.result
        try:
            result_code = int(getattr(raw, "value", raw))
        except (TypeError, ValueError):
            result_code = _SUCCESS_CODE
        return (
            geodesic.ts,
            geodesic.positions,
            geodesic.velocities,
            result_code,
        )
    # Callable geodesic: sample uniformly
    lam_min, lam_max = affine_bounds
    lam = jnp.linspace(lam_min, lam_max, n_samples)

    def _pos(l):
        return geodesic(l)

    positions = jax.vmap(_pos)(lam)
    velocities = jax.vmap(jax.jacfwd(_pos))(lam)
    return lam, positions, velocities, _SUCCESS_CODE


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
        - ``'renormalized'`` (default): legacy historical option; preserves
          direction up to a numerical floor and is the identity for null
          tangents (see :func:`_tangent_renormalized_null`).
        - ``'null_projected'``: rescale the spatial part of the tangent
          so that ``g_{ab} k^a k^b = 0`` exactly, via a Lagrange-style
          quadratic solve. Use this when the underlying geodesic
          integrator drifts off the null cone enough to bias the
          integral.
        - ``'fixed'``: pass the raw velocity through; integral is then a
          coordinate-ray average rather than a null average.
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
        If ``tangent_norm`` is not in
        ``{'renormalized', 'fixed', 'null_projected'}``.
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

    # trapezoid over affine parameter (non-uniform lam OK)
    line_integral = jnp.trapezoid(integrand, lam)

    geodesic_complete = result_code == _SUCCESS_CODE
    termination_reason = _TERMINATION_REASONS.get(result_code, "unknown")

    return ANECResult(
        line_integral=line_integral,
        geodesic_complete=geodesic_complete,
        termination_reason=termination_reason,
    )
