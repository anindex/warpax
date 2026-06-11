"""Kinematic decomposition of the ADM shift vector field.

Splits the spatial gradient of the shift one-form ``beta_i = g_{0i}`` on a
constant-time slice into its irreducible parts:

    D_i beta_j = (1/3) theta_beta gamma_{ij} + sigma^beta_{ij} + omega^beta_{ij}

with

    theta_beta      = gamma^{ij} D_i beta_j                  (expansion / divergence)
    sigma^beta_{ij} = D_{(i} beta_{j)} - (1/3) theta_beta gamma_{ij}   (shear)
    omega^beta_{ij} = D_{[i} beta_{j]}                       (vorticity / curl)

The vorticity part is the exterior derivative of the shift one-form,
``omega^beta_{ij} = (1/2)(d_i beta_j - d_j beta_i)``: the connection terms cancel,
so it is exact from first metric derivatives and invariant under spatial
coordinate changes.

The shift vorticity decides the Hawking-Ellis algebraic type of the warp-drive
wall: an irrotational shift (``omega^2_beta = 0``, e.g. the Rodal drive) yields
Type I matter, the sufficient direction proved by Santiago, Schuster and
Visser; nonzero vorticity drives the wall to Type IV. The zero-expansion
Natario drive (``theta_beta = 0`` yet ``omega^2_beta > 0``) isolates vorticity
from expansion as the operative obstruction.

See ``kinematic_scalars`` for the analogous decomposition of the Eulerian
*congruence* (whose vorticity vanishes identically by hypersurface
orthogonality); the shift field carries the rotation that the Eulerian normal
cannot.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.geometry.grid import build_coord_batch
from warpax.geometry.types import GridSpec


def compute_shift_kinematics(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Expansion, shear-squared and vorticity-squared of the ADM shift at a point.

    Parameters
    ----------
    metric_fn : callable
        Maps coords ``(4,)`` to the metric tensor ``(4, 4)``.
    coords : Float[Array, "4"]
        Spacetime coordinates ``(t, x, y, z)``.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
        ``(theta_beta, sigma_sq_beta, omega_sq_beta)``: shift divergence, shear
        scalar squared and vorticity scalar squared, all raised with the
        spatial metric ``gamma^{ij}``.
    """
    g = metric_fn(coords)
    gamma = g[1:, 1:]
    gamma_inv = jnp.linalg.inv(gamma)
    t = coords[0]
    xyz = coords[1:]

    def beta_lower_at(p: Float[Array, "3"]) -> Float[Array, "3"]:
        c = jnp.concatenate([jnp.array([t], dtype=coords.dtype), p])
        return metric_fn(c)[0, 1:]

    # d_beta[j, i] = d beta_j / d x^i  ->  P[i, j] = d_i beta_j
    d_beta = jax.jacfwd(beta_lower_at)(xyz)
    partial_i_beta_j = d_beta.T

    # Spatial Christoffel symbols Gamma^k_{ij} compatible with gamma_{ij}.
    def spatial_metric_at(p: Float[Array, "3"]) -> Float[Array, "3 3"]:
        c = jnp.concatenate([jnp.array([t], dtype=coords.dtype), p])
        return metric_fn(c)[1:, 1:]

    dgamma = jax.jacfwd(spatial_metric_at)(xyz)  # dgamma[a, b, c] = d_c gamma_{ab}
    term1 = jnp.einsum("lk,jki->lij", gamma_inv, dgamma)
    term2 = jnp.einsum("lk,ikj->lij", gamma_inv, dgamma)
    term3 = jnp.einsum("lk,ijk->lij", gamma_inv, dgamma)
    christoffel = 0.5 * (term1 + term2 - term3)  # Gamma^l_{ij}

    beta_lower = g[0, 1:]
    christoffel_term = jnp.einsum("kij,k->ij", christoffel, beta_lower)
    cov = partial_i_beta_j - christoffel_term  # D_i beta_j

    theta = jnp.einsum("ij,ij->", gamma_inv, cov)
    cov_sym = 0.5 * (cov + cov.T)
    cov_anti = 0.5 * (cov - cov.T)
    shear = cov_sym - (theta / 3.0) * gamma
    sigma_sq = jnp.einsum("ik,jl,ij,kl->", gamma_inv, gamma_inv, shear, shear)
    omega_sq = jnp.einsum("ik,jl,ij,kl->", gamma_inv, gamma_inv, cov_anti, cov_anti)
    return theta, sigma_sq, omega_sq


def rotationality(
    theta_beta: Float[Array, "..."],
    sigma_sq_beta: Float[Array, "..."],
    omega_sq_beta: Float[Array, "..."],
    eps: float = 1e-300,
) -> Float[Array, "..."]:
    """Dimensionless vorticity fraction of the shift gradient, in ``[0, 1]``.

    ``rotationality = omega^2 / ((1/3) theta^2 + sigma^2 + omega^2)``.

    Equals the share of the shift-gradient norm carried by its rotational part.
    It is zero for an irrotational shift and, because the shift scales linearly
    with the warp speed, is independent of ``v_s`` for constant-velocity drives,
    so it is a per-drive geometric fingerprint.
    """
    norm_sq = theta_beta**2 / 3.0 + sigma_sq_beta + omega_sq_beta
    return omega_sq_beta / (norm_sq + eps)


def compute_shift_kinematics_grid(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    grid_spec: GridSpec,
    t: float = 0.0,
    batch_size: int | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Grid version of :func:`compute_shift_kinematics`.

    Returns ``(theta_grid, sigma_sq_grid, omega_sq_grid)``, each of shape
    ``grid_spec.shape``.
    """
    coords_flat = build_coord_batch(grid_spec, t=t)
    grid_shape = grid_spec.shape

    def point_fn(c: Float[Array, "4"]) -> Float[Array, "3"]:
        theta, sigma_sq, omega_sq = compute_shift_kinematics(metric_fn, c)
        return jnp.stack([theta, sigma_sq, omega_sq])

    if batch_size is not None:
        results = jax.lax.map(point_fn, coords_flat, batch_size=batch_size)
    else:
        results = jax.vmap(point_fn)(coords_flat)

    theta_grid = results[:, 0].reshape(grid_shape)
    sigma_sq_grid = results[:, 1].reshape(grid_shape)
    omega_sq_grid = results[:, 2].reshape(grid_shape)
    return theta_grid, sigma_sq_grid, omega_sq_grid
