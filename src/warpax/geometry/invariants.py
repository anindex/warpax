"""Pointwise curvature invariants via JAX einsum contractions.

Scalars computed at a single point:

- Kretschmann scalar ``K = R_{abcd} R^{abcd}`` (Kretschmann 1915).
- Ricci-squared ``R_{ab} R^{ab}``.
- Weyl-squared via the 4-D Gauss-Bonnet identity:
  ``C^2 = K - 2 R_{ab} R^{ab} + (1/3) R^2``.
- Chern-Pontryagin ``*R_{abcd} R^{abcd}`` (Hodge-dual contraction);
  vanishes identically for conformally flat spacetimes.

Index conventions match ``geometry.py``: Riemann ``R^a_{bcd}``, Ricci
``R_{ab}``, metric ``g_{ab}``, inverse metric ``g^{ab}``. These functions
are pointwise; use ``grid.evaluate_curvature_grid`` for batched evaluation.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import CurvatureResult


# All 24 permutations of (0,1,2,3) with their signs.
# Even permutations (+1):
_EVEN_PERMS = (
    (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
    (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
    (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
)
# Odd permutations (-1):
_ODD_PERMS = (
    (0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 2, 3, 0), (1, 3, 0, 2),
    (2, 0, 3, 1), (2, 1, 0, 3), (2, 3, 1, 0),
    (3, 0, 1, 2), (3, 1, 2, 0), (3, 2, 0, 1),
)

# Pre-build index arrays for JIT-compatible construction
_EVEN_IDX = tuple(zip(*_EVEN_PERMS))  # 4 tuples of 12 ints each
_ODD_IDX = tuple(zip(*_ODD_PERMS))


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
        metric_inv, metric_inv, metric_inv, metric_inv, R_down,
    )
    return jnp.einsum("abcd,abcd->", R_down, R_up_all)


def ricci_squared(
    ricci: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Ricci-squared ``R_{ab} R^{ab}`` at a single point."""
    R_up = jnp.einsum("ac,bd,cd->ab", metric_inv, metric_inv, ricci)
    return jnp.einsum("ab,ab->", ricci, R_up)


def weyl_squared(
    kretschmann: Float[Array, ""],
    ricci_sq: Float[Array, ""],
    ricci_scalar: Float[Array, ""],
) -> Float[Array, ""]:
    """Weyl-squared ``C_{abcd} C^{abcd}`` via the 4-D Gauss-Bonnet identity.

    ``C^2 = K - 2 R_{ab} R^{ab} + (1/3) R^2`` in four dimensions.
    """
    return kretschmann - 2.0 * ricci_sq + (1.0 / 3.0) * ricci_scalar ** 2


def _levi_civita_4d(
    metric: Float[Array, "4 4"],
) -> Float[Array, "4 4 4 4"]:
    """Lorentzian Levi-Civita tensor ``epsilon_{abcd} = sqrt(-det g) [abcd]``.

    The permutation symbol is built once via static index tuples so the
    function stays trace-friendly.
    """
    eps = jnp.zeros((4, 4, 4, 4))
    eps = eps.at[_EVEN_IDX].set(1.0)
    eps = eps.at[_ODD_IDX].set(-1.0)
    sqrt_neg_det = jnp.sqrt(jnp.maximum(-jnp.linalg.det(metric), 1e-30))
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
        metric_inv, metric_inv, metric_inv, metric_inv, R_down,
    )
    return jnp.einsum("abcd,abcd->", R_dual, R_up_all)


def compute_invariants(
    result: CurvatureResult,
) -> tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]
]:
    """Compute all four curvature invariants from a ``CurvatureResult``.

    Returns ``(K, R2, W2, CP)``: Kretschmann, Ricci-squared, Weyl-squared,
    and Chern-Pontryagin.
    """
    K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
    R2 = ricci_squared(result.ricci, result.metric_inv)
    W2 = weyl_squared(K, R2, result.ricci_scalar)
    CP = chern_pontryagin(result.riemann, result.metric, result.metric_inv)
    return (K, R2, W2, CP)
