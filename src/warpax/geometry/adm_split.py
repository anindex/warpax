"""3+1 ADM decomposition utilities.

Extracts (alpha, beta^i, gamma_{ij}, K_{ij}) from a full spacetime metric g_{munu},
or computes K_{ij} from the ADM evolution formula when time derivatives
are available.

Key formulas (MTW / Baumgarte-Shapiro conventions):
    alpha^2 = beta_i beta^i - g_{00}
    beta_i = g_{0i}
    gamma_{ij} = g_{ij}
    beta^i = gamma^{ij} beta_j
    K_{ij} = -(1/2alpha)(d__t gamma_{ij} - D_i beta_j - D_j beta_i)

For a static slice (d__t gamma_{ij} = 0):
    K_{ij} = (1/2alpha)(D_i beta_j + D_j beta_i)

where D_i is the spatial covariant derivative compatible with gamma_{ij}.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class ADMSplit(NamedTuple):
    """3+1 ADM decomposition at a single spacetime point.

    Attributes
    ----------
    lapse : Float[Array, ""]
        Lapse function alpha.
    shift_lower : Float[Array, "3"]
        Covariant shift vector beta_i = g_{0i}.
    shift_upper : Float[Array, "3"]
        Contravariant shift vector beta^i = gamma^{ij} beta_j.
    spatial_metric : Float[Array, "3 3"]
        Spatial metric gamma_{ij} = g_{ij}.
    spatial_metric_inv : Float[Array, "3 3"]
        Inverse spatial metric gamma^{ij}.
    extrinsic_curvature : Float[Array, "3 3"]
        Extrinsic curvature K_{ij}.
    """

    lapse: Float[Array, ""]
    shift_lower: Float[Array, "3"]
    shift_upper: Float[Array, "3"]
    spatial_metric: Float[Array, "3 3"]
    spatial_metric_inv: Float[Array, "3 3"]
    extrinsic_curvature: Float[Array, "3 3"]


def adm_split(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> ADMSplit:
    """Perform 3+1 ADM decomposition of a spacetime metric at a point.

    Extracts lapse, shift, spatial metric, and extrinsic curvature from
    a full 4D metric function. Extrinsic curvature is computed via the
    ADM formula using JAX autodiff for all derivatives.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4, 4)
        The spacetime metric g_{munu}(x).
    coords : Float[Array, "4"]
        Spacetime coordinates (t, x, y, z).

    Returns
    -------
    ADMSplit
        Named tuple with all ADM fields.
    """
    g = metric_fn(coords)

    # Extract spatial metric and shift
    gamma = g[1:, 1:]  # gamma_{ij} = g_{ij}
    beta_lower = g[0, 1:]  # beta_i = g_{0i}

    # Inverse spatial metric
    gamma_inv = jnp.linalg.inv(gamma)

    # Contravariant shift beta^i = gamma^{ij} beta_j
    beta_upper = gamma_inv @ beta_lower

    # Lapse: alpha^2 = beta_i beta^i - g_{00}
    beta_sq = jnp.dot(beta_lower, beta_upper)
    alpha_sq = beta_sq - g[0, 0]
    alpha = jnp.sqrt(jnp.maximum(alpha_sq, 1e-30))

    # Extrinsic curvature via K_{ij} = -(1/2alpha)(d__t gamma_{ij} - D_i beta_j - D_j beta_i)
    # We need d__t gamma_{ij} and the spatial covariant derivative of beta.
    # Use JAX autodiff to compute both.
    K = _extrinsic_curvature(metric_fn, coords, alpha, beta_upper, gamma, gamma_inv)

    return ADMSplit(
        lapse=alpha,
        shift_lower=beta_lower,
        shift_upper=beta_upper,
        spatial_metric=gamma,
        spatial_metric_inv=gamma_inv,
        extrinsic_curvature=K,
    )


def _extrinsic_curvature(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
    alpha: Float[Array, ""],
    beta_upper: Float[Array, "3"],
    gamma: Float[Array, "3 3"],
    gamma_inv: Float[Array, "3 3"],
) -> Float[Array, "3 3"]:
    """Compute extrinsic curvature K_{ij} via the ADM formula.

    K_{ij} = -(1/2alpha)(d__t gamma_{ij} - D_i beta_j - D_j beta_i)

    where D_i beta_j = d__i beta_j - Gamma^k_{ij} beta_k  (spatial covariant derivative).

    All derivatives are computed via JAX autodiff.
    """
    t = coords[0]
    spatial_coords = coords[1:]

    # d__t gamma_{ij}
    # Compute the time derivative of the spatial metric
    def gamma_at_t(t_val: Float[Array, ""]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t_val]), spatial_coords])
        g_at = metric_fn(full_coords)
        return g_at[1:, 1:]

    dt_gamma = jax.jacfwd(gamma_at_t)(t)  # shape (3, 3)

    # Spatial Christoffel symbols Gamma^k_{ij}
    def spatial_metric_at(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g_at = metric_fn(full_coords)
        return g_at[1:, 1:]

    # dgamma[i, j, k] = d_ gamma_{ij} / d_ x^k (derivative index last per JAX)
    dgamma = jax.jacfwd(spatial_metric_at)(spatial_coords)

    # Gamma^l_{ij} = 0.5 gamma^{lk} (d__i gamma_{jk} + d__j gamma_{ik} - d__k gamma_{ij})
    # dgamma indexing: dgamma[a, b, c] = d_ gamma_{ab} / d_ x^c
    # So d__i gamma_{jk} = dgamma[j, k, i]
    term1 = jnp.einsum("lk,jki->lij", gamma_inv, dgamma)  # gamma^{lk} d__i gamma_{jk}
    term2 = jnp.einsum("lk,ikj->lij", gamma_inv, dgamma)  # gamma^{lk} d__j gamma_{ik}
    term3 = jnp.einsum("lk,ijk->lij", gamma_inv, dgamma)  # gamma^{lk} d__k gamma_{ij}
    christoffel_3d = 0.5 * (term1 + term2 - term3)  # Gamma^l_{ij}

    # Spatial covariant derivative of beta
    # beta_j = gamma_{jk} beta^k (lower the shift index)
    beta_lower_local = gamma @ beta_upper

    # d__i beta_j: derivative of beta_j w.r.t. spatial coords
    def beta_lower_at(xyz: Float[Array, "3"]) -> Float[Array, "3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g_at = metric_fn(full_coords)
        return g_at[0, 1:]

    # d_beta[j, i] = d_ beta_j / d_ x^i (index i is the derivative direction)
    d_beta = jax.jacfwd(beta_lower_at)(spatial_coords)  # (3, 3)

    # D_i beta_j = d__i beta_j - Gamma^k_{ij} beta_k
    # d_beta[j, i] = d__i beta_j  ->  transpose to get (i, j) indexing
    partial_i_beta_j = d_beta.T  # (i, j)
    christoffel_term = jnp.einsum("kij,k->ij", christoffel_3d, beta_lower_local)

    D_beta = partial_i_beta_j - christoffel_term  # D_i beta_j with shape (i, j)

    # Symmetrize: D_i beta_j + D_j beta_i
    D_beta_sym = D_beta + D_beta.T

    # K_{ij} = -(1/2alpha)(d__t gamma_{ij} - D_i beta_j - D_j beta_i)
    K = -(1.0 / (2.0 * alpha)) * (dt_gamma - D_beta_sym)

    return K
