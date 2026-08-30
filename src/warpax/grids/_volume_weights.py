"""Non-uniform grid volume-weight helpers.

:func:`warpax.energy_conditions.compute_wall_restricted_stats` and other
integration-based diagnostics must multiply each cell's integrand by its
volume weight when the grid is non-uniform. This module provides the
pointwise Jacobian calculation.

Private module; used by :func:`warpax.grids.wall_clustered`.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["compute_volume_weights", "proper_volume_weights"]


def proper_volume_weights(
    coord_weights: Float[Array, "..."],
    g_field: Float[Array, "... 4 4"],
) -> Float[Array, "..."]:
    """Attach the slice Jacobian: ``dV = sqrt(det gamma) d^3x``.

    :func:`compute_volume_weights` returns *coordinate* cell volumes, which are
    the proper volumes only on a flat slice. Alcubierre, Natário and Rodal carry
    ``gamma_ij = delta_ij``; Van den Broeck carries ``B^2 delta_ij``, so
    ``sqrt(det gamma) = B^3`` runs over ``[1, 3.4]`` across its wall and moved
    its wall Type-I fraction from 0.1372 to 0.1449.

    ``gamma_ij`` is the spatial block of ``g_ab``, so pass the metric field the
    curvature pass already returned.

    Parameters
    ----------
    coord_weights : Float[Array, "..."]
        Coordinate cell volumes, shape ``grid_shape`` (or flattened).
    g_field : Float[Array, "... 4 4"]
        Four-metric on the same points, shape ``(*coord_weights.shape, 4, 4)``.

    Returns
    -------
    Float[Array, "..."]
        ``coord_weights * sqrt(|det gamma|)``, same shape as ``coord_weights``.
    """
    if coord_weights is None:
        return None  # uniform grid: equal weights, the documented None path
    if g_field.shape[:-2] != coord_weights.shape:
        raise ValueError(
            f"metric grid shape {g_field.shape[:-2]} != weight shape "
            f"{coord_weights.shape}; a broadcast here reweights the wrong points"
        )
    gamma = g_field[..., 1:, 1:]
    # Clip, not |det|: a degenerate point gets zero weight, not a plausible one.
    return coord_weights * jnp.sqrt(jnp.clip(jnp.linalg.det(gamma), min=0.0))


def _cell_widths(coords: Float[Array, "N"]) -> Float[Array, "N"]:
    """Central-difference cell widths with half-differences at endpoints."""
    n = coords.shape[0]
    if n == 1:
        return jnp.ones(1, dtype=coords.dtype)
    return jnp.concatenate(
        [
            0.5 * (coords[1:2] - coords[0:1]),  # first endpoint
            0.5 * (coords[2:] - coords[:-2]),  # interior
            0.5 * (coords[n - 1 : n] - coords[n - 2 : n - 1]),  # last endpoint
        ]
    )


def compute_volume_weights(
    x_coords: Float[Array, "Nx"],
    y_coords: Float[Array, "Ny"],
    z_coords: Float[Array, "Nz"],
) -> Float[Array, "Nx Ny Nz"]:
    """Compute per-cell 3D volume weights from 1D non-uniform coord arrays.

    Uses trapezoidal cell widths: ``dx[i] = 0.5 * (x[i+1] - x[i-1])`` with
    end-cell halving. Result: ``sum(volume_weights)`` approximates the
    total bounded volume.

    Parameters
    ----------
    x_coords, y_coords, z_coords : Float[Array, "N*"]
        Non-uniform 1D coordinate arrays per spatial axis.

    Returns
    -------
    Float[Array, "Nx Ny Nz"]
        Per-cell volume weight tensor.
    """
    dx = _cell_widths(x_coords)  # (Nx,)
    dy = _cell_widths(y_coords)  # (Ny,)
    dz = _cell_widths(z_coords)  # (Nz,)
    return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
