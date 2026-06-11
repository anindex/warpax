"""Averaged Null Energy Condition (ANEC) line integral.

Physical definition:

.. math::

    \\mathrm{ANEC}[\\gamma] = \\int_{\\gamma} T_{ab} k^a k^b \\, d\\lambda

where ``k^a`` is the null tangent to the geodesic ``gamma`` and ``lambda``
is an affine parameter.

The ``geodesic_complete`` flag and ``termination_reason`` field on
``ANECResult`` report whether the integrator completed successfully.

.. note::

   ``tangent_norm='null_projected'`` (the default) rescales the spatial
   part of each sampled tangent so ``g(k, k) = 0`` exactly, making the
   line integral a genuine null-cone average even when the underlying
   integrator drifts off the cone. ``'renormalized'`` (the legacy
   default) is a near-identity rescale that protects against
   zero-``u^0`` numerical degeneracies but does **not** project onto
   the null cone; ``'fixed'`` selects the unmodified tangent. For
   certification use ``'null_projected'`` or the symplectic integrator
   (:func:`anec_rigorous`).
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
from ..geodesics.observables import velocity_norm
from ..geodesics.symplectic import (
    SymplecticGeodesicResult,
    integrate_geodesic_symplectic,
    null_ic_canonical,
)
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


_VALID_TANGENT_NORM = frozenset({"renormalized", "fixed", "null_projected"})


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
        termination.
    termination_reason : str
        Human-readable reason: ``'complete'`` on success; otherwise a
        Diffrax failure mode (e.g. ``'max_steps'``, ``'nonfinite'``,
        ``'dt_min_reached'``, ``'event_occurred'``) or ``'unknown'``
        for an unrecognized result code.
    max_abs_g_kk : Float[Array, ""]
        Rigor witness: the worst off-cone deviation ``max_n |g_{ab} k^a k^b|``
        over the sampled tangents. For a genuine null average this should be
        tiny; a large value flags that the integrated tangent drifted off the
        null cone (use the symplectic integrator or the ``null_projected``
        tangent norm). 0.0 by construction when ``tangent_norm='null_projected'``.
    null_preserved : bool
        True iff ``max_abs_g_kk < null_tol`` -- i.e. the line integral is a
        defensible null-geodesic average rather than a coordinate-ray diagnostic.
    """

    line_integral: Float[Array, ""]
    geodesic_complete: bool
    termination_reason: str
    max_abs_g_kk: Float[Array, ""]
    null_preserved: bool


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
    Of the two roots, picks the one closest to 1 (jit-safe), so an
    already-null tangent is returned unchanged. The naive ``+sqrt`` root
    selects the reflected null branch wherever ``g_00 > 0`` (e.g. inside
    a superluminal Alcubierre bubble), mangling exactly-null input.
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
    denom = 2.0 * A_s + 1e-30
    lam_plus = (-cross + sqrt_disc) / denom
    lam_minus = (-cross - sqrt_disc) / denom
    lam = jnp.where(
        jnp.abs(lam_plus - 1.0) <= jnp.abs(lam_minus - 1.0),
        lam_plus,
        lam_minus,
    )
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
    # Duck-type on the trajectory interface so this accepts both the Diffrax
    # GeodesicResult (carries an int ``result``) and the SymplecticGeodesicResult
    # (carries a bool ``complete`` + ``termination_reason``).
    if hasattr(geodesic, "positions") and hasattr(geodesic, "velocities"):
        if hasattr(geodesic, "result"):
            # Robust conversion (diffrax 0.7.x EnumerationItem carries
            # ``._value``, not ``.value``); never defaults to success.
            result_code = result_code_to_int(geodesic.result)
        else:
            # Symplectic result: map completeness to the diffrax code
            # convention (9 = nonfinite, its only failure mode).
            result_code = RESULT_SUCCESS if bool(geodesic.complete) else 9
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
    return lam, positions, velocities, RESULT_SUCCESS


@jaxtyped(typechecker=beartype)
def anec(
    metric: MetricSpecification,
    geodesic: Union[
        GeodesicResult,
        SymplecticGeodesicResult,
        Callable[[Float[Array, ""]], Float[Array, "4"]],
    ],
    tangent_norm: str = "null_projected",
    n_samples: int = 256,
    affine_bounds: tuple[float, float] = (-5.0, 5.0),
    null_tol: float = 1e-8,
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
        - ``'null_projected'`` (default): rescale the spatial part of the
          tangent so that ``g_{ab} k^a k^b = 0`` exactly, via a
          Lagrange-style quadratic solve. Identity for already-null
          tangents; corrects integrator drift off the null cone.
        - ``'renormalized'``: legacy option; preserves
          direction up to a numerical floor and is the identity for null
          tangents (see :func:`_tangent_renormalized_null`). Does NOT
          project onto the null cone.
        - ``'fixed'``: pass the raw velocity through; integral is then a
          coordinate-ray average rather than a null average.

        .. admonition:: Recommendation

           For certification-grade ANEC values use ``'null_projected'``
           (the default) or, better, the symplectic geodesic integrator
           via :func:`anec_rigorous`, which keeps the raw tangent on the
           null cone to ~machine precision and carries an explicit
           on-cone witness. ``'renormalized'`` and ``'fixed'`` evaluate
           coordinate-ray averages whenever the integrated tangent has
           drifted off the cone.
    n_samples : int
        Number of affine samples when ``geodesic`` is a callable.
    affine_bounds : tuple[float, float]
        Affine bounds ``(lam_min, lam_max)`` when ``geodesic`` is a
        callable. Ignored when a ``GeodesicResult`` is passed.
    null_tol : float
        Threshold on the rigor witness ``max|g(k,k)|`` below which the
        integral is certified as a genuine null-geodesic average
        (``null_preserved=True``). Default ``1e-8``.

    Returns
    -------
    ANECResult
        NamedTuple with ``line_integral``, ``geodesic_complete``,
        ``termination_reason``, ``max_abs_g_kk`` (rigor witness), and
        ``null_preserved``.

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

    # Rigor witness: worst off-cone deviation of the SAMPLED tangents. With
    # 'null_projected' this is ~0 by construction; otherwise it reports how
    # well the integrated geodesic stayed on the null cone.
    if tangent_norm == "null_projected":
        g_kk = jax.vmap(
            lambda c, u: velocity_norm(metric, c, _project_to_null(metric(c), u))
        )(positions, velocities)
    else:
        g_kk = jax.vmap(lambda c, u: velocity_norm(metric, c, u))(
            positions, velocities
        )
    max_abs_g_kk = jnp.max(jnp.abs(g_kk))
    null_preserved = bool(max_abs_g_kk < null_tol)

    geodesic_complete = result_code == RESULT_SUCCESS

    return ANECResult(
        line_integral=line_integral,
        geodesic_complete=geodesic_complete,
        termination_reason=termination_reason(result_code),
        max_abs_g_kk=max_abs_g_kk,
        null_preserved=null_preserved,
    )


class RigorousANEC(NamedTuple):
    """Dual-path rigorous ANEC: symplectic primary + projection fallback.

    Attributes
    ----------
    symplectic : ANECResult
        ANEC along the structure-preserving symplectic null geodesic, with its
        ``max_abs_g_kk`` witness. This is the rigorous, geodesic-integrated
        value -- defensible whenever ``symplectic.null_preserved`` is True.
    projection : ANECResult or None
        Populated only when the symplectic witness exceeds ``null_tol`` (or the
        geodesic is incomplete): the same trajectory re-evaluated with the
        per-sample ``null_projected`` tangent, a clearly-labeled safety net for
        the strongest-shift cases. ``None`` when the symplectic value already
        passed.
    method_used : str
        ``'symplectic'`` or ``'symplectic+projection_fallback'``.
    """

    symplectic: ANECResult
    projection: ANECResult | None
    method_used: str


def anec_rigorous(
    metric: MetricSpecification,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
    *,
    affine_bounds: tuple[float, float] = (-30.0, 30.0),
    num_steps: int = 8192,
    order: int = 4,
    omega: float = 1.0,
    null_tol: float = 1e-8,
) -> RigorousANEC:
    """Rigorous geodesic-integrated ANEC with an on-cone witness.

    Integrates the *actual* null geodesic from ``x0`` in spatial direction
    ``n_spatial`` using the structure-preserving symplectic integrator (which
    holds ``g(k,k)`` to ~machine precision where adaptive RK drifts off the
    cone for long bubble crossings), then evaluates
    :math:`\\int T_{ab} k^a k^b \\, d\\lambda`. The returned
    :class:`RigorousANEC` carries the rigor witness ``max|g(k,k)|`` and, where
    that witness exceeds ``null_tol``, an additionally projection-corrected
    fallback value.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime.
    x0 : Float[Array, "4"]
        Geodesic starting point (should lie in the asymptotically-flat region).
    n_spatial : Float[Array, "3"]
        Spatial direction of the null ray (need not be unit norm).
    affine_bounds, num_steps, order, omega
        Forwarded to :func:`warpax.geodesics.integrate_geodesic_symplectic`.
    null_tol : float
        On-cone witness threshold for certifying the symplectic value.

    Returns
    -------
    RigorousANEC
    """
    x0c, p0 = null_ic_canonical(metric, x0, n_spatial)
    geo = integrate_geodesic_symplectic(
        metric, x0c, p0, affine_bounds,
        num_steps=num_steps, order=order, omega=omega,
    )
    # 'fixed' tangent: do not mask the witness -- the symplectic integrator is
    # supposed to keep the raw tangent on the cone.
    sym = anec(metric, geo, tangent_norm="fixed", null_tol=null_tol)
    if sym.null_preserved and geo.complete:
        return RigorousANEC(symplectic=sym, projection=None,
                            method_used="symplectic")
    proj = anec(metric, geo, tangent_norm="null_projected", null_tol=null_tol)
    return RigorousANEC(symplectic=sym, projection=proj,
                        method_used="symplectic+projection_fallback")
