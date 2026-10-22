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

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import christoffel_from_jacobian


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
    a full 4D metric function: gamma_{ij} = g_{ij}, beta_i = g_{0i},
    beta^i = gamma^{ij} beta_j and alpha^2 = beta_i beta^i - g_{00}.
    Extrinsic curvature is computed via the ADM formula using JAX
    autodiff for all derivatives.

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

    gamma = g[1:, 1:]
    beta_lower = g[0, 1:]

    gamma_inv = jnp.linalg.inv(gamma)

    beta_upper = gamma_inv @ beta_lower

    beta_sq = jnp.dot(beta_lower, beta_upper)
    alpha_sq = beta_sq - g[0, 0]
    alpha = jnp.sqrt(jnp.where(alpha_sq > 0.0, alpha_sq, jnp.nan))

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

    where D_i beta_j = d__i beta_j - Gamma^k_{ij} beta_k  (spatial covariant derivative)
    and Gamma^l_{ij} = 0.5 gamma^{lk} (d__i gamma_{jk} + d__j gamma_{ik} - d__k gamma_{ij}).

    All derivatives are computed via JAX autodiff, whose jacfwd puts the
    derivative index last: dgamma[a, b, c] = d_ gamma_{ab} / d_ x^c, so
    d__i gamma_{jk} = dgamma[j, k, i], and d_beta[j, i] = d__i beta_j, so the
    Jacobian of beta_j is transposed to (i, j) order before subtracting the
    Christoffel term.
    """
    t = coords[0]
    spatial_coords = coords[1:]

    def gamma_at_t(t_val: Float[Array, ""]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t_val]), spatial_coords])
        g_at = metric_fn(full_coords)
        return g_at[1:, 1:]

    dt_gamma = jax.jacfwd(gamma_at_t)(t)

    def spatial_metric_at(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g_at = metric_fn(full_coords)
        return g_at[1:, 1:]

    christoffel_3d = christoffel_from_jacobian(
        gamma_inv, jax.jacfwd(spatial_metric_at)(spatial_coords)
    )

    beta_lower_local = gamma @ beta_upper

    def beta_lower_at(xyz: Float[Array, "3"]) -> Float[Array, "3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g_at = metric_fn(full_coords)
        return g_at[0, 1:]

    d_beta = jax.jacfwd(beta_lower_at)(spatial_coords)

    partial_i_beta_j = d_beta.T
    christoffel_term = jnp.einsum("kij,k->ij", christoffel_3d, beta_lower_local)

    D_beta = partial_i_beta_j - christoffel_term

    D_beta_sym = D_beta + D_beta.T

    K = -(1.0 / (2.0 * alpha)) * (dt_gamma - D_beta_sym)

    return K
