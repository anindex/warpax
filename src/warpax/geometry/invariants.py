"""Pointwise curvature invariants via JAX einsum contractions.

Scalars computed at a single point:

- Kretschmann scalar ``K = R_{abcd} R^{abcd}`` (Kretschmann 1915).
- Ricci-squared ``R_{ab} R^{ab}``.
- Weyl tensor ``C_{abcd}`` (trace-free part of Riemann) and Weyl-squared
  ``C_{abcd} C^{abcd}`` contracted directly from it, which avoids the
  catastrophic cancellation of ``K - 2 R_{ab} R^{ab} + R^2/3`` when the
  Ricci part dominates.
- Chern-Pontryagin ``*R_{abcd} R^{abcd}`` (Hodge-dual contraction);
  vanishes identically for conformally flat spacetimes.

``_EVEN_PERMS`` and ``_ODD_PERMS`` list the 24 permutations of (0,1,2,3) by
sign; ``_EVEN_IDX`` and ``_ODD_IDX`` are their transposed index tuples (4 tuples
of 12 ints) so the Levi-Civita symbol is built by a static scatter under JIT.
``_DET_RTOL`` is the degeneracy floor on the determinant of the
scale-normalized metric, so it is dimensionless.

Index conventions match ``geometry.py``: Riemann ``R^a_{bcd}``, Ricci
``R_{ab}``, metric ``g_{ab}``, inverse metric ``g^{ab}``. These functions
are pointwise; use ``grid.evaluate_curvature_grid`` for batched evaluation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import CurvatureResult

_EVEN_PERMS = (
    (0, 1, 2, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (1, 0, 3, 2),
    (1, 2, 0, 3),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 2, 1),
    (3, 1, 0, 2),
    (3, 2, 1, 0),
)
_ODD_PERMS = (
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 2, 3, 0),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
    (2, 1, 0, 3),
    (2, 3, 1, 0),
    (3, 0, 1, 2),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
)

_EVEN_IDX = tuple(zip(*_EVEN_PERMS, strict=True))
_ODD_IDX = tuple(zip(*_ODD_PERMS, strict=True))

_DET_RTOL = 1e-30


def kretschmann_scalar(
    riemann: Float[Array, "4 4 4 4"],
    metric: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Kretschmann scalar ``K = R_{abcd} R^{abcd}`` at a single point.

    Parameters
    ----------
    riemann : Float[Array, "4 4 4 4"]
        Riemann tensor ``R^a_{bcd}`` (upper first index).
    metric : Float[Array, "4 4"]
        Metric tensor ``g_{ab}``.
    metric_inv : Float[Array, "4 4"]
        Inverse metric ``g^{ab}``.

    Returns
    -------
    Float[Array, ""]
        Kretschmann scalar.
    """
    R_down = jnp.einsum("ae,ebcd->abcd", metric, riemann)
    R_up_all = jnp.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        metric_inv,
        metric_inv,
        metric_inv,
        metric_inv,
        R_down,
    )
    return jnp.einsum("abcd,abcd->", R_down, R_up_all)


def ricci_squared(
    ricci: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Ricci-squared ``R_{ab} R^{ab}`` at a single point."""
    R_up = jnp.einsum("ac,bd,cd->ab", metric_inv, metric_inv, ricci)
    return jnp.einsum("ab,ab->", ricci, R_up)


def weyl_tensor(
    riemann: Float[Array, "4 4 4 4"],
    ricci: Float[Array, "4 4"],
    ricci_scalar: Float[Array, ""],
    metric: Float[Array, "4 4"],
) -> Float[Array, "4 4 4 4"]:
    """Weyl tensor ``C_{abcd}`` (all indices down) at a single point.

    ``C_abcd = R_abcd - 1/2 (g_ac R_bd - g_ad R_bc - g_bc R_ad + g_bd R_ac)
    + R/6 (g_ac g_bd - g_ad g_bc)`` with ``R_abcd = g_ae R^e_bcd``.
    """
    R_down = jnp.einsum("ae,ebcd->abcd", metric, riemann)
    ricci_part = 0.5 * (
        jnp.einsum("ac,bd->abcd", metric, ricci)
        - jnp.einsum("ad,bc->abcd", metric, ricci)
        - jnp.einsum("bc,ad->abcd", metric, ricci)
        + jnp.einsum("bd,ac->abcd", metric, ricci)
    )
    scalar_part = (ricci_scalar / 6.0) * (
        jnp.einsum("ac,bd->abcd", metric, metric) - jnp.einsum("ad,bc->abcd", metric, metric)
    )
    return R_down - ricci_part + scalar_part


def weyl_squared(
    riemann: Float[Array, "4 4 4 4"],
    ricci: Float[Array, "4 4"],
    ricci_scalar: Float[Array, ""],
    metric: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Weyl-squared ``C_{abcd} C^{abcd}`` contracted from ``weyl_tensor``.

    Not clamped: the Lorentzian contraction can be negative.
    """
    C_down = weyl_tensor(riemann, ricci, ricci_scalar, metric)
    C_up = jnp.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        metric_inv,
        metric_inv,
        metric_inv,
        metric_inv,
        C_down,
    )
    return jnp.einsum("abcd,abcd->", C_down, C_up)


def _levi_civita_4d(
    metric: Float[Array, "4 4"],
) -> Float[Array, "4 4 4 4"]:
    """Lorentzian Levi-Civita tensor ``epsilon_{abcd} = sqrt(-det g) [abcd]``.

    The permutation symbol is built once via static index tuples so the
    function stays trace-friendly. The metric is divided by its largest
    entry before the determinant, because ``scale**4`` underflows below
    ``|g| ~ 1e-77``; ``_DET_RTOL`` then floors the normalized determinant.
    """
    eps = jnp.zeros((4, 4, 4, 4))
    eps = eps.at[_EVEN_IDX].set(1.0)
    eps = eps.at[_ODD_IDX].set(-1.0)
    tiny = jnp.finfo(metric.dtype).tiny
    scale = jnp.maximum(jnp.max(jnp.abs(metric)), tiny)
    neg_det_hat = -jnp.linalg.det(metric / scale)
    sqrt_neg_det = scale**2 * jnp.sqrt(jnp.maximum(neg_det_hat, _DET_RTOL))
    return sqrt_neg_det * eps


def chern_pontryagin(
    riemann: Float[Array, "4 4 4 4"],
    metric: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Chern-Pontryagin scalar ``*R_{abcd} R^{abcd}`` at a single point.

    Contracts the left Hodge dual of the Riemann tensor with the fully
    contravariant Riemann tensor. Detects parity violation; vanishes for
    conformally flat or maximally symmetric spacetimes.
    """
    R_down = jnp.einsum("ae,ebcd->abcd", metric, riemann)
    R_up_cd = jnp.einsum("ae,bf,efcd->abcd", metric_inv, metric_inv, R_down)
    epsilon = _levi_civita_4d(metric)
    R_dual = 0.5 * jnp.einsum("abef,efcd->abcd", epsilon, R_up_cd)
    R_up_all = jnp.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        metric_inv,
        metric_inv,
        metric_inv,
        metric_inv,
        R_down,
    )
    return jnp.einsum("abcd,abcd->", R_dual, R_up_all)


def compute_invariants(
    result: CurvatureResult,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute all four curvature invariants from a ``CurvatureResult``.

    Returns ``(K, R2, W2, CP)``: Kretschmann, Ricci-squared, Weyl-squared,
    and Chern-Pontryagin.
    """
    K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
    R2 = ricci_squared(result.ricci, result.metric_inv)
    W2 = weyl_squared(
        result.riemann, result.ricci, result.ricci_scalar, result.metric, result.metric_inv
    )
    CP = chern_pontryagin(result.riemann, result.metric, result.metric_inv)
    return (K, R2, W2, CP)
