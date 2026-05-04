"""ADM constraint residual computations.

Implements the Hamiltonian and momentum constraint equations for the
3+1 ADM decomposition of general relativity. Uses JAX autodiff for all
spatial derivatives (Christoffel symbols, Ricci tensor).

Key equations (Baumgarte-Shapiro conventions):

Hamiltonian constraint:
    H = R + K^2 - K_{ij} K^{ij} - 16pi E = 0

Momentum constraint:
    M_i = D_j(K^j_i - delta^j_i K) - 8pi S_i = 0

Normalized residuals use scale-invariant normalization:
    eps_H = |H| / (|R| + |K^2| + |K_{ij}K^{ij}| + |16piE|)
    eps_M = |M| / (|DK| + |8piS|)
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.adm_split import adm_split


def _spatial_ricci_scalar(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> Float[Array, ""]:
    """Compute spatial Ricci scalar R(gamma) via autodiff at a single point.

    Computes spatial Christoffel symbols and Riemann tensor from the spatial
    metric gamma_{ij}(x) using nested jax.jacfwd, then contracts to the Ricci
    scalar R = gamma^{ij} R_{ij}.
    """
    t = coords[0]
    spatial_coords = coords[1:]

    def spatial_metric_fn(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g = metric_fn(full_coords)
        return g[1:, 1:]

    gamma = spatial_metric_fn(spatial_coords)
    gamma_inv = jnp.linalg.inv(gamma)

    # Spatial Christoffel symbols via autodiff
    def christoffel_fn(x: Float[Array, "3"]) -> Float[Array, "3 3 3"]:
        g_at = spatial_metric_fn(x)
        g_inv_at = jnp.linalg.inv(g_at)
        # dg[a, b, c] = d_ gamma_{ab} / d_ x^c
        dg = jax.jacfwd(spatial_metric_fn)(x)
        # Gamma^l_{ij} = 0.5 gamma^{lk} (d__i gamma_{jk} + d__j gamma_{ik} - d__k gamma_{ij})
        t1 = jnp.einsum("lk,jki->lij", g_inv_at, dg)
        t2 = jnp.einsum("lk,ikj->lij", g_inv_at, dg)
        t3 = jnp.einsum("lk,ijk->lij", g_inv_at, dg)
        return 0.5 * (t1 + t2 - t3)

    christoffel = christoffel_fn(spatial_coords)
    dchristoffel = jax.jacfwd(christoffel_fn)(spatial_coords)

    # Riemann tensor R^l_{ijk} from Christoffel derivatives + quadratic terms
    deriv_term = jnp.swapaxes(dchristoffel, 2, 3) - dchristoffel
    quad_pos = jnp.einsum("lsn,smr->lmnr", christoffel, christoffel)
    quad_neg = jnp.einsum("lsr,smn->lmnr", christoffel, christoffel)
    riemann = deriv_term + quad_pos - quad_neg

    # Ricci tensor: R_{ij} = R^k_{ikj}
    ricci = jnp.einsum("kikj->ij", riemann)

    # Ricci scalar: R = gamma^{ij} R_{ij}
    R = jnp.einsum("ij,ij->", gamma_inv, ricci)
    return R


def hamiltonian_constraint(
    gamma: Float[Array, "3 3"],
    K: Float[Array, "3 3"],
    energy_density: Float[Array, ""],
    R: Float[Array, ""] | None = None,
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]] | None = None,
    coords: Float[Array, "4"] | None = None,
) -> Float[Array, ""]:
    """ADM Hamiltonian constraint residual.

    H = R + K^2 - K_{ij} K^{ij} - 16piE

    Parameters
    ----------
    gamma : spatial metric gamma_{ij}
    K : extrinsic curvature K_{ij}
    energy_density : energy density E = T_{ab} n^a n^b
    R : optional precomputed spatial Ricci scalar
    metric_fn : optional, used to compute R via autodiff if R is None
    coords : optional, required when metric_fn is provided

    Returns
    -------
    H : Hamiltonian constraint residual (should be ~0 for valid initial data)
    """
    gamma_inv = jnp.linalg.inv(gamma)

    if R is None and metric_fn is not None and coords is not None:
        R = _spatial_ricci_scalar(metric_fn, coords)
    elif R is None:
        # Detect flat metric
        is_flat = jnp.all(jnp.abs(gamma - jnp.eye(3, dtype=gamma.dtype)) <= 1e-12)
        R = jnp.where(is_flat, 0.0, jnp.nan)

    K_trace = jnp.einsum("ij,ij->", gamma_inv, K)
    K_sq = jnp.einsum("ij,kl,ik,jl", gamma_inv, gamma_inv, K, K)
    H = R + K_trace**2 - K_sq - 16.0 * jnp.pi * energy_density
    return H


def _spatial_div_tensor(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
    A_upper_lower: Float[Array, "3 3"],
) -> Float[Array, "3"]:
    """Compute spatial covariant divergence D_j A^j_i via autodiff.

    A^j_i is a mixed (1,1) spatial tensor. The divergence is:
        D_j A^j_i = d__j A^j_i + Gamma^j_{jk} A^k_i - Gamma^k_{ji} A^j_k
    """
    t = coords[0]
    spatial_coords = coords[1:]

    def spatial_metric_fn(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g = metric_fn(full_coords)
        return g[1:, 1:]

    # Christoffel symbols
    gamma_val = spatial_metric_fn(spatial_coords)
    gamma_inv_val = jnp.linalg.inv(gamma_val)
    dg = jax.jacfwd(spatial_metric_fn)(spatial_coords)
    t1 = jnp.einsum("lk,jki->lij", gamma_inv_val, dg)
    t2 = jnp.einsum("lk,ikj->lij", gamma_inv_val, dg)
    t3 = jnp.einsum("lk,ijk->lij", gamma_inv_val, dg)
    christoffel = 0.5 * (t1 + t2 - t3)

    # Compute A^j_i as a function of spatial coords for autodiff
    # For now, we pass A_upper_lower as constant (evaluated at the point)
    # and compute the partial derivative d__j A^j_i

    # Reconstruct A^j_i from the metric at each point
    def A_at(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full_coords = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g_at = metric_fn(full_coords)
        gamma_at = g_at[1:, 1:]
        gamma_inv_at = jnp.linalg.inv(gamma_at)
        # Re-extract K from the full ADM split
        adm = adm_split(metric_fn, full_coords)
        K_at = adm.extrinsic_curvature
        K_trace_at = jnp.einsum("ij,ij->", gamma_inv_at, K_at)
        # A^j_i = K^j_i - delta^j_i K
        K_mixed = gamma_inv_at @ K_at  # K^j_i = gamma^{jk} K_{ki}
        return K_mixed - K_trace_at * jnp.eye(3, dtype=gamma_at.dtype)

    # dA[j, i, k] = d_ A^j_i / d_ x^k
    dA = jax.jacfwd(A_at)(spatial_coords)  # (3, 3, 3)

    # d__j A^j_i = dA[j, i, j] -> trace over first and third indices
    partial_div = jnp.einsum("jij->i", dA)

    # Connection terms:
    # + Gamma^j_{jk} A^k_i
    trace_christoffel = jnp.einsum("jjk->k", christoffel)  # Gamma^j_{jk}
    conn_pos = jnp.einsum("k,ki->i", trace_christoffel, A_upper_lower)

    # - Gamma^k_{ji} A^j_k
    conn_neg = jnp.einsum("kji,jk->i", christoffel, A_upper_lower)

    return partial_div + conn_pos - conn_neg


def momentum_constraint(
    gamma: Float[Array, "3 3"],
    K: Float[Array, "3 3"],
    momentum_density: Float[Array, "3"],
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]] | None = None,
    coords: Float[Array, "4"] | None = None,
) -> Float[Array, "3"]:
    """ADM momentum constraint residual.

    M_i = D_j(K^j_i - delta^j_i K) - 8piS_i

    Parameters
    ----------
    gamma : spatial metric gamma_{ij}
    K : extrinsic curvature K_{ij}
    momentum_density : momentum density S_i = -T_{ab} n^a h^b_i
    metric_fn : optional, needed for covariant divergence computation
    coords : optional, needed for covariant divergence computation

    Returns
    -------
    M_i : momentum constraint residual (shape (3,))
    """
    gamma_inv = jnp.linalg.inv(gamma)
    K_trace = jnp.einsum("ij,ij->", gamma_inv, K)
    K_mixed = gamma_inv @ K  # K^j_i = gamma^{jk} K_{ki}
    A_mixed = K_mixed - K_trace * jnp.eye(3, dtype=gamma.dtype)

    if metric_fn is not None and coords is not None:
        div_A = _spatial_div_tensor(metric_fn, coords, A_mixed)
    else:
        # Fall back to zero divergence for flat metric with zero K
        is_K_zero = jnp.all(jnp.abs(K) <= 1e-12)
        is_flat = jnp.all(jnp.abs(gamma - jnp.eye(3, dtype=gamma.dtype)) <= 1e-12)
        div_A = jnp.where(
            is_K_zero | is_flat,
            jnp.zeros(3, dtype=gamma.dtype),
            jnp.full(3, jnp.nan, dtype=gamma.dtype),
        )

    M_i = div_A - 8.0 * jnp.pi * momentum_density
    return M_i


def normalized_residuals(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
    energy_density: Float[Array, ""] | None = None,
    momentum_density: Float[Array, "3"] | None = None,
) -> dict[str, Float[Array, ""]]:
    """Compute normalized Hamiltonian and momentum constraint residuals.

    Performs a full 3+1 ADM split of the metric, computes constraint
    residuals, and normalizes them with scale-invariant denominators:

    eps_H = |H| / (|R| + |K^2| + |K_{ij}K^{ij}| + |16piE| + eps)
    eps_M = |M| / (|div_A| + |8piS| + eps)

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    coords : spacetime coordinates (t, x, y, z)
    energy_density : optional, defaults to 0 (vacuum)
    momentum_density : optional, defaults to 0

    Returns
    -------
    dict with keys 'epsilon_H', 'epsilon_M', plus raw diagnostics
    """
    # Full 3+1 decomposition
    adm = adm_split(metric_fn, coords)
    gamma = adm.spatial_metric
    gamma_inv = adm.spatial_metric_inv
    K = adm.extrinsic_curvature

    E = jnp.array(0.0, dtype=gamma.dtype) if energy_density is None else energy_density
    S_i = jnp.zeros(3, dtype=gamma.dtype) if momentum_density is None else momentum_density

    # Compute spatial Ricci scalar
    R = _spatial_ricci_scalar(metric_fn, coords)

    # Hamiltonian constraint
    K_trace = jnp.einsum("ij,ij->", gamma_inv, K)
    K_sq = jnp.einsum("ij,kl,ik,jl", gamma_inv, gamma_inv, K, K)
    H = R + K_trace**2 - K_sq - 16.0 * jnp.pi * E

    # Momentum constraint
    M_i = momentum_constraint(gamma, K, S_i, metric_fn=metric_fn, coords=coords)

    # Scale-invariant normalization:
    # eps_H = |H| / (|R| + |K^2| + |K_{ij}K^{ij}| + |16piE| + floor)
    # The floor of 1.0 prevents spurious large eps when all constraint terms
    # are near zero (vacuum), where floating-point noise would otherwise
    # dominate the ratio.
    component_scale = jnp.abs(R) + jnp.abs(K_trace**2) + jnp.abs(K_sq) + jnp.abs(16.0 * jnp.pi * E)
    norm_H = jnp.maximum(component_scale, 1.0)
    epsilon_H = jnp.abs(H) / norm_H

    # eps_M = |M| / max(|component scales|, 1.0)
    M_scale = jnp.linalg.norm(M_i + 8.0 * jnp.pi * S_i) + jnp.linalg.norm(8.0 * jnp.pi * S_i)
    norm_M = jnp.maximum(M_scale, 1.0)
    epsilon_M = jnp.linalg.norm(M_i) / norm_M

    return {
        "epsilon_H": epsilon_H,
        "epsilon_M": epsilon_M,
        "H_raw": H,
        "M_raw": M_i,
        "R_spatial": R,
        "K_trace": K_trace,
        "K_sq": K_sq,
    }
