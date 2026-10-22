"""Kinematic scalars (theta, sigma^2, omega^2) for the Eulerian congruence.

For ``u^a = n^a``:

- ``theta = -K`` (trace of the extrinsic curvature),
- ``sigma^2 = K_{ij} K^{ij} - (1/3) K^2``,
- ``omega^2 = 0`` (vanishes by Frobenius for hypersurface-orthogonal n^a).

``K_{ij} = -alpha * Gamma^0_{ij}``.

For unit-lapse metrics (Alcubierre, Lentz, Natario, Van den Broeck,
Rodal) ``n^a`` is geodesic and ``theta``/``sigma^2`` feed directly into
Raychaudhuri. For WarpShell (``alpha != 1`` in the shell) the congruence
is accelerated; the focusing interpretation needs the ``-D_a a^a`` term
not computed here (Wald, *General Relativity*, eq. 9.2.11).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from warpax.energy_conditions.observer import eulerian_normal
from warpax.geometry import christoffel_symbols
from warpax.geometry.grid import build_coord_batch
from warpax.geometry.types import GridSpec


def compute_kinematic_scalars(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute expansion, shear-squared, vorticity-squared at a point.

    For the Eulerian congruence u^a = n^a (normal to t=const slices):
        theta = -K (trace of extrinsic curvature)
        sigma^2 = K_{ij} K^{ij} - (1/3) K^2
        omega^2 = 0 (hypersurface-orthogonal)

    K_{ij} = -alpha * Gamma^0_{ij} where Gamma are the 4D Christoffel
    symbols and ``alpha = 1/sqrt(-g^{00})`` from :func:`eulerian_normal`, so
    every scalar is NaN where the slice is not spacelike.

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
    g = metric_fn(coords)
    g_inv = jnp.linalg.inv(g)

    gamma = christoffel_symbols(metric_fn, coords)

    alpha, _ = eulerian_normal(g_inv)

    K_ij = -alpha * gamma[0, 1:, 1:]

    gamma_ij = g[1:, 1:]
    gamma_inv = jnp.linalg.inv(gamma_ij)

    K_trace = jnp.einsum("ij,ij->", gamma_inv, K_ij)

    theta = -K_trace

    K_up = jnp.einsum("ik,jl,kl->ij", gamma_inv, gamma_inv, K_ij)

    K_ij_K_up = jnp.einsum("ij,ij->", K_ij, K_up)
    sigma_sq = K_ij_K_up - (1.0 / 3.0) * K_trace**2

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
    coords_flat = build_coord_batch(grid_spec, t=t)
    results_flat = _kinematic_points(metric_fn, coords_flat, batch_size)
    grid_shape = grid_spec.shape
    theta_grid = results_flat[:, 0].reshape(grid_shape)
    sigma_sq_grid = results_flat[:, 1].reshape(grid_shape)
    omega_sq_grid = results_flat[:, 2].reshape(grid_shape)
    return theta_grid, sigma_sq_grid, omega_sq_grid


@eqx.filter_jit
def _kinematic_points(metric_fn, coords, batch_size):
    """Jitted point-map of :func:`compute_kinematic_scalars` over ``(N, 4)`` coords."""

    def point_fn(c: Float[Array, "4"]):
        return jnp.stack(compute_kinematic_scalars(metric_fn, c))

    if batch_size is not None:
        return lax.map(point_fn, coords, batch_size=batch_size)
    return jax.vmap(point_fn)(coords)
