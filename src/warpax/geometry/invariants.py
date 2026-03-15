"""Pointwise curvature invariants via JAX einsum contractions.

Computes gauge-invariant curvature scalars at a single spacetime point:
    - Kretschner scalar: K = R_{abcd} R^{abcd}
    - Ricci-squared: R_{ab} R^{ab}
    - Weyl-squared: C_{abcd} C^{abcd} = K - 2 R_{ab} R^{ab} + (1/3) R^2
    - Chern-Pontryagin: *R_{abcd} R^{abcd} (Hodge dual contraction)

These invariants are useful for identifying physical singularities (vs.
coordinate singularities), classifying spacetimes, and validating numerical
pipelines. All three quadratic invariants are non-negative in Riemannian
geometry; in Lorentzian geometry Kretschner is always real but may take any
sign. The Chern-Pontryagin scalar detects parity violation and vanishes
identically for conformally flat spacetimes.

Index conventions (matching geometry.py):
    - Riemann: R^a_{bcd} as (4,4,4,4) array [upper, lower, lower, lower]
    - Ricci: R_{ab} as (4,4) array [lower, lower]
    - Metric: g_{ab} as (4,4) array [lower, lower]
    - Inverse metric: g^{ab} as (4,4) array [upper, upper]

These are pointwise functions operating on raw JAX arrays (no grid/batch
dimensions). For grid-level evaluation, use the vmap-based pipeline in
``grid.py``.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import CurvatureResult


# ---------------------------------------------------------------------------
# Even/odd permutation indices for the 4D Levi-Civita tensor
# ---------------------------------------------------------------------------

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


def kretschner_scalar(
    riemann: Float[Array, "4 4 4 4"],
    metric: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Kretschner scalar K = R_{abcd} R^{abcd} at a single point.

    Parameters
    ----------
    riemann : Float[Array, "4 4 4 4"]
        Riemann tensor R^a_{bcd} (upper first index).
    metric : Float[Array, "4 4"]
        Metric tensor g_{ab}.
    metric_inv : Float[Array, "4 4"]
        Inverse metric g^{ab}.

    Returns
    -------
    Float[Array, ""]
        Kretschner scalar (dimensionless).
    """
    # Lower first index: R_{abcd} = g_{ae} R^e_{bcd}
    R_down = jnp.einsum("ae,ebcd->abcd", metric, riemann)

    # Raise all four indices: R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
    R_up_all = jnp.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        metric_inv, metric_inv, metric_inv, metric_inv, R_down,
    )

    # Contract: K = R_{abcd} R^{abcd}
    return jnp.einsum("abcd,abcd->", R_down, R_up_all)


def ricci_squared(
    ricci: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Ricci-squared R_{ab} R^{ab} at a single point.

    Parameters
    ----------
    ricci : Float[Array, "4 4"]
        Ricci tensor R_{ab}.
    metric_inv : Float[Array, "4 4"]
        Inverse metric g^{ab}.

    Returns
    -------
    Float[Array, ""]
        Ricci-squared scalar.
    """
    # Raise both indices: R^{ab} = g^{ac} g^{bd} R_{cd}
    R_up = jnp.einsum("ac,bd,cd->ab", metric_inv, metric_inv, ricci)

    # Contract: R_{ab} R^{ab}
    return jnp.einsum("ab,ab->", ricci, R_up)


def weyl_squared(
    kretschner: Float[Array, ""],
    ricci_sq: Float[Array, ""],
    ricci_scalar: Float[Array, ""],
) -> Float[Array, ""]:
    """Weyl-squared C_{abcd} C^{abcd} at a single point.

    Uses the 4D Gauss-Bonnet identity:
        C^2 = K - 2 R_{ab} R^{ab} + (1/3) R^2

    Parameters
    ----------
    kretschner : Float[Array, ""]
        Kretschner scalar K = R_{abcd} R^{abcd}.
    ricci_sq : Float[Array, ""]
        Ricci-squared R_{ab} R^{ab}.
    ricci_scalar : Float[Array, ""]
        Ricci scalar R.

    Returns
    -------
    Float[Array, ""]
        Weyl-squared scalar.
    """
    return kretschner - 2.0 * ricci_sq + (1.0 / 3.0) * ricci_scalar ** 2


def _levi_civita_4d(
    metric: Float[Array, "4 4"],
) -> Float[Array, "4 4 4 4"]:
    """4D Levi-Civita tensor epsilon_{abcd} = sqrt(|det(g)|) * [abcd].

    Constructs the totally antisymmetric Levi-Civita tensor from the
    metric determinant and the permutation symbol. The permutation
    symbol is built as a static array using hardcoded index tuples for
    all 24 permutations, ensuring JIT compatibility (no Python loops
    at trace time).

    Parameters
    ----------
    metric : Float[Array, "4 4"]
        Metric tensor g_{ab}.

    Returns
    -------
    Float[Array, "4 4 4 4"]
        Levi-Civita tensor with all indices lowered.
    """
    # Build permutation symbol [abcd] via index scatter
    eps = jnp.zeros((4, 4, 4, 4))
    eps = eps.at[_EVEN_IDX].set(1.0)
    eps = eps.at[_ODD_IDX].set(-1.0)

    # Multiply by sqrt(|det(g)|); use -det for Lorentzian signature
    sqrt_neg_det = jnp.sqrt(jnp.maximum(-jnp.linalg.det(metric), 1e-30))
    return sqrt_neg_det * eps


def chern_pontryagin(
    riemann: Float[Array, "4 4 4 4"],
    metric: Float[Array, "4 4"],
    metric_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Chern-Pontryagin scalar *R_{abcd} R^{abcd} at a single point.

    Computes the contraction of the (left) Hodge dual of the Riemann
    tensor with the fully contravariant Riemann tensor. This scalar
    detects parity violation in the spacetime geometry and vanishes
    identically for conformally flat or maximally symmetric spacetimes.

    Parameters
    ----------
    riemann : Float[Array, "4 4 4 4"]
        Riemann tensor R^a_{bcd} (upper first index).
    metric : Float[Array, "4 4"]
        Metric tensor g_{ab}.
    metric_inv : Float[Array, "4 4"]
        Inverse metric g^{ab}.

    Returns
    -------
    Float[Array, ""]
        Chern-Pontryagin scalar.
    """
    # Step 1: Lower first index R_{abcd} = g_{ae} R^e_{bcd}
    R_down = jnp.einsum("ae,ebcd->abcd", metric, riemann)

    # Step 2: Raise first two indices R^{ab}_{cd} = g^{ae} g^{bf} R_{efcd}
    R_up_cd = jnp.einsum("ae,bf,efcd->abcd", metric_inv, metric_inv, R_down)

    # Step 3: Levi-Civita tensor
    epsilon = _levi_civita_4d(metric)

    # Step 4: Hodge dual *R_{abcd} = (1/2) epsilon_{abef} R^{ef}_{cd}
    R_dual = 0.5 * jnp.einsum("abef,efcd->abcd", epsilon, R_up_cd)

    # Step 5: Fully raise R_down for contraction
    # R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
    R_up_all = jnp.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        metric_inv, metric_inv, metric_inv, metric_inv, R_down,
    )

    # Step 6: Contract *R_{abcd} R^{abcd}
    return jnp.einsum("abcd,abcd->", R_dual, R_up_all)


def compute_invariants(
    result: CurvatureResult,
) -> tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]
]:
    """Compute all four curvature invariants from a CurvatureResult.

    Convenience function that extracts the relevant fields from a
    CurvatureResult NamedTuple and calls the individual invariant functions.

    Parameters
    ----------
    result : CurvatureResult
        Output of ``compute_curvature_chain`` at a single point.

    Returns
    -------
    tuple of (kretschner, ricci_sq, weyl_sq, chern_pontryagin)
        The four curvature invariants as scalar arrays:
        - K: Kretschner scalar R_{abcd} R^{abcd}
        - R2: Ricci-squared R_{ab} R^{ab}
        - Weyl-squared C_{abcd} C^{abcd}
        - CP: Chern-Pontryagin *R_{abcd} R^{abcd}
    """
    K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
    R2 = ricci_squared(result.ricci, result.metric_inv)
    W2 = weyl_squared(K, R2, result.ricci_scalar)
    CP = chern_pontryagin(result.riemann, result.metric, result.metric_inv)
    return (K, R2, W2, CP)
