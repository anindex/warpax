"""JIT-safe interpolation helpers for :class:`InterpolatedADMMetric`.

Uses :func:`jax.scipy.ndimage.map_coordinates` with ``mode="nearest"`` for
out-of-bounds safety (extrapolation clamped to the grid boundary).

Private module; consumers should use :class:`warpax.io.InterpolatedADMMetric`.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

from warpax.geometry import GridSpec

__all__ = [
    "_interpolate_scalar",
    "_interpolate_tensor",
    "_interpolate_vector",
]

_CUBIC_FALLBACK_WARNED = False


def _coords_to_indices(
    coords: Float[Array, "4"], grid_spec: GridSpec
) -> Float[Array, "4"]:
    """Map physical coords ``(t, x, y, z)`` to fractional grid indices."""
    bounds = jnp.asarray(grid_spec.bounds)
    shape = jnp.asarray(grid_spec.shape)
    low = bounds[:, 0]
    high = bounds[:, 1]
    frac = (coords - low) / (high - low)
    indices = frac * (shape - 1)
    return indices


def _order_for_method(method: str) -> int:
    """Map interpolation-method string to ``map_coordinates`` ``order``."""
    global _CUBIC_FALLBACK_WARNED
    if method == "linear":
        return 1
    if method == "cubic":
        if not _CUBIC_FALLBACK_WARNED:
            warnings.warn(
                "interp_method='cubic' is not supported by jax.scipy.ndimage "
                "map_coordinates; falling back to linear interpolation.",
                stacklevel=3,
            )
            _CUBIC_FALLBACK_WARNED = True
        return 1
    raise ValueError(
        f"Unknown interp_method: {method!r}; expected 'linear' or 'cubic'."
    )


def _interpolate_scalar(
    grid: Float[Array, "Nt Nx Ny Nz"],
    coords: Float[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, ""]:
    """Interpolate a scalar field at coords. Returns a 0-d array."""
    order = _order_for_method(method)
    idx = _coords_to_indices(coords, grid_spec)[:, None]
    value = map_coordinates(grid, idx, order=order, mode="nearest")
    return value[0]


def _interpolate_vector(
    grid: Float[Array, "Nt Nx Ny Nz 3"],
    coords: Float[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3"]:
    """Interpolate a 3-vector field via a single batched ``map_coordinates``.

    Stacks the 3 spatial-component grids into a leading channel axis and
    issues one batched call to :func:`map_coordinates`.
    """
    order = _order_for_method(method)
    idx = _coords_to_indices(coords, grid_spec)[:, None]  # (4, 1)
    # ``moveaxis`` puts the channel axis (originally last) up front so
    # shape becomes (3, Nt, Nx, Ny, Nz). Padding the index by zeros along
    # the new channel axis lets a single map_coordinates evaluate all
    # components in lock-step.
    stacked = jnp.moveaxis(grid, -1, 0)  # (3, Nt, Nx, Ny, Nz)
    channel_idx = jnp.arange(3, dtype=idx.dtype)[None, :]  # (1, 3)
    spatial_idx = jnp.broadcast_to(idx, (4, 3))  # (4, 3)
    full_idx = jnp.concatenate([channel_idx, spatial_idx], axis=0)  # (5, 3)
    return map_coordinates(stacked, full_idx, order=order, mode="nearest")


def _interpolate_tensor(
    grid: Float[Array, "Nt Nx Ny Nz 3 3"],
    coords: Float[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3 3"]:
    """Interpolate a 3x3 tensor field via a single batched ``map_coordinates``.

    Reshapes the trailing ``(3, 3)`` axes into a flat 9-channel axis and
    issues one call instead of nine; the result is then reshaped back to
    ``(3, 3)``.
    """
    order = _order_for_method(method)
    idx = _coords_to_indices(coords, grid_spec)[:, None]  # (4, 1)
    Nt, Nx, Ny, Nz, _, _ = grid.shape
    flat = grid.reshape(Nt, Nx, Ny, Nz, 9)
    stacked = jnp.moveaxis(flat, -1, 0)  # (9, Nt, Nx, Ny, Nz)
    channel_idx = jnp.arange(9, dtype=idx.dtype)[None, :]  # (1, 9)
    spatial_idx = jnp.broadcast_to(idx, (4, 9))  # (4, 9)
    full_idx = jnp.concatenate([channel_idx, spatial_idx], axis=0)  # (5, 9)
    flat_values = map_coordinates(stacked, full_idx, order=order, mode="nearest")
    return flat_values.reshape(3, 3)
