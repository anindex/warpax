"""Kinematic scalars for the Eulerian congruence.

Computes the expansion scalar theta, shear scalar sigma^2, and
vorticity scalar omega^2 at each point, characterizing the
kinematics of the Eulerian (hypersurface-orthogonal) observer
congruence u^a = n^a.

For Eulerian observers:
- theta = -K  (expansion = negative trace of extrinsic curvature)
- sigma^2 = K_{ij} K^{ij} - (1/3) K^2  (shear scalar)
- omega^2 = 0  (vorticity vanishes identically by Frobenius theorem)

K_{ij} is computed directly from the 4D Christoffel symbols:
K_{ij} = -alpha * Gamma^0_{ij}.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from warpax.geometry import christoffel_symbols
from warpax.geometry.grid import build_coord_batch
from warpax.geometry.types import GridSpec


def compute_kinematic_scalars(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute expansion, shear-squared, vorticity-squared at a point.

    For the Eulerian congruence u^a = n^a (normal to t=const slices):
        theta = -K  (trace of extrinsic curvature)
        sigma^2 = K_{ij} K^{ij} - (1/3) K^2
        omega^2 = 0  (hypersurface-orthogonal)

    K_{ij} = -alpha * Gamma^0_{ij} where Gamma are the 4D Christoffel
    symbols and alpha = 1/sqrt(-g^{00}).

    Parameters
    ----------
    metric_fn : callable
        Maps coords ``(4,)`` to metric tensor ``(4, 4)``.
    coords : Float[Array, "4"]
        Spacetime coordinates ``(t, x, y, z)``.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(theta, sigma_sq, omega_sq)`` expansion, shear scalar
        squared, vorticity scalar squared.
    """
    # Metric and its inverse
    g = metric_fn(coords)
    g_inv = jnp.linalg.inv(g)

    # Christoffel symbols Gamma^lam_{mu nu}
    gamma = christoffel_symbols(metric_fn, coords)

    # ADM lapse: alpha = 1 / sqrt(-g^{00})
    alpha = 1.0 / jnp.sqrt(-g_inv[0, 0])

    # Extrinsic curvature: K_{ij} = -alpha * Gamma^0_{ij} for spatial i,j
    # gamma[0, 1:, 1:] = Gamma^0_{ij} for i,j = 1,2,3
    K_ij = -alpha * gamma[0, 1:, 1:]  # (3, 3)

    # Spatial metric gamma_{ij} and its inverse
    gamma_ij = g[1:, 1:]  # (3, 3) spatial part of full metric
    gamma_inv = jnp.linalg.inv(gamma_ij)  # gamma^{ij}

    # Trace: K = gamma^{ij} K_{ij}
    K_trace = jnp.einsum("ij,ij->", gamma_inv, K_ij)

    # Expansion: theta = -K
    theta = -K_trace

    # K^{ij} = gamma^{ik} gamma^{jl} K_{kl}
    K_up = jnp.einsum("ik,jl,kl->ij", gamma_inv, gamma_inv, K_ij)

    # Shear-squared: sigma^2 = K_{ij} K^{ij} - (1/3) K^2
    K_ij_K_up = jnp.einsum("ij,ij->", K_ij, K_up)
    sigma_sq = K_ij_K_up - (1.0 / 3.0) * K_trace**2

    # Vorticity: identically zero for Eulerian observers
    omega_sq = jnp.array(0.0)

    return theta, sigma_sq, omega_sq


def compute_kinematic_scalars_grid(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    grid_spec: GridSpec,
    t: float = 0.0,
    batch_size: int | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Compute kinematic scalars on a full 3D spatial grid.

    Lifts the pointwise ``compute_kinematic_scalars`` to a grid using
    the flatten-vmap-reshape pattern (same as ``evaluate_curvature_grid``).

    Parameters
    ----------
    metric_fn : callable
        Maps coords ``(4,)`` to metric tensor ``(4, 4)``.
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    t : float
        Time coordinate for the constant-time slice.
    batch_size : int or None
        If set, use ``lax.map`` with this batch size for memory safety.
        If None, use full ``jax.vmap``.

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
        ``(theta_grid, sigma_sq_grid, omega_sq_grid)`` each with
        shape ``(*grid_shape,)``.
    """
    coords_flat = build_coord_batch(grid_spec, t=t)  # (N, 4)
    grid_shape = grid_spec.shape

    def point_fn(c: Float[Array, "4"]):
        theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric_fn, c)
        return jnp.stack([theta, sigma_sq, omega_sq])

    if batch_size is not None:
        results_flat = lax.map(point_fn, coords_flat, batch_size=batch_size)
    else:
        results_flat = jax.vmap(point_fn)(coords_flat)

    # results_flat has shape (N, 3)
    theta_flat = results_flat[:, 0]
    sigma_sq_flat = results_flat[:, 1]
    omega_sq_flat = results_flat[:, 2]

    theta_grid = theta_flat.reshape(grid_shape)
    sigma_sq_grid = sigma_sq_flat.reshape(grid_shape)
    omega_sq_grid = omega_sq_flat.reshape(grid_shape)

    return theta_grid, sigma_sq_grid, omega_sq_grid
