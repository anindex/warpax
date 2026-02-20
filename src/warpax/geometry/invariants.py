"""Pointwise curvature invariants via JAX einsum contractions.

Computes gauge-invariant curvature scalars at a single spacetime point:
    - Kretschner scalar: K = R_{abcd} R^{abcd}
    - Ricci-squared: R_{ab} R^{ab}
    - Weyl-squared: C_{abcd} C^{abcd} = K - 2 R_{ab} R^{ab} + (1/3) R^2

These invariants are useful for identifying physical singularities (vs.
coordinate singularities), classifying spacetimes, and validating numerical
pipelines.  All three are non-negative in Riemannian geometry; in Lorentzian
geometry Kretschner is always real but may take any sign.

Index conventions (matching geometry.py):
    - Riemann: R^a_{bcd} as (4,4,4,4) array [upper, lower, lower, lower]
    - Ricci: R_{ab} as (4,4) array [lower, lower]
    - Metric: g_{ab} as (4,4) array [lower, lower]
    - Inverse metric: g^{ab} as (4,4) array [upper, upper]

These are pointwise functions operating on raw JAX arrays (no grid/batch
dimensions).  For grid-level evaluation, use the vmap-based pipeline in
``grid.py``.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import CurvatureResult


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


def compute_invariants(
    result: CurvatureResult,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute all three curvature invariants from a CurvatureResult.

    Convenience function that extracts the relevant fields from a
    CurvatureResult NamedTuple and calls the individual invariant functions.

    Parameters
    ----------
    result : CurvatureResult
        Output of ``compute_curvature_chain`` at a single point.

    Returns
    -------
    tuple of (kretschner, ricci_sq, weyl_sq)
        The three curvature invariants as scalar arrays.
    """
    K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
    R2 = ricci_squared(result.ricci, result.metric_inv)
    W2 = weyl_squared(K, R2, result.ricci_scalar)
    return (K, R2, W2)
