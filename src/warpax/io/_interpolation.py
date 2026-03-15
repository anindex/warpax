"""JIT-safe interpolation helpers for :class:`InterpolatedADMMetric`.

Uses :func:`jax.scipy.ndimage.map_coordinates` with ``mode="nearest"`` for
out-of-bounds safety (mitigation in

Private module; consumers should use :class:`warpax.io.InterpolatedADMMetric`.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

from warpax.geometry import GridSpec

__all__ = [
    "_interpolate_scalar",
    "_interpolate_tensor",
    "_interpolate_vector",
]


def _coords_to_indices(
    coords: Float[Array, "4"], grid_spec: GridSpec
) -> Float[Array, "4"]:
    """Map physical coords ``(t, x, y, z)`` to fractional grid indices.

    Parameters
    ----------
    coords : Float[Array, "4"]
        Spacetime coordinates in physical units.
    grid_spec : GridSpec
        4D grid specification with ``bounds`` of length 4 and ``shape`` of
        length 4 (``(Nt, Nx, Ny, Nz)``).

    Returns
    -------
    Float[Array, "4"]
        Fractional grid indices in ``[0, N-1]`` range for each axis.
    """
    bounds = jnp.asarray(grid_spec.bounds)  # shape (4, 2)
    shape = jnp.asarray(grid_spec.shape)  # shape (4,)
    low = bounds[:, 0]
    high = bounds[:, 1]
    frac = (coords - low) / (high - low)  # in [0, 1] for in-bounds
    indices = frac * (shape - 1)
    return indices


def _order_for_method(method: str) -> int:
    """Map interpolation-method string to ``map_coordinates`` ``order``."""
    if method == "linear":
        return 1
    if method == "cubic":
        # jax.scipy.ndimage.map_coordinates currently supports order 0 and 1
        # only; fall back to linear for "cubic" when cubic is unavailable.
        # TODO: revisit if / when JAX adds native order=3 support.
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
    idx = _coords_to_indices(coords, grid_spec)[:, None]  # shape (4, 1)
    value = map_coordinates(grid, idx, order=order, mode="nearest")
    return value[0]  # unpack 0-d


def _interpolate_vector(
    grid: Float[Array, "Nt Nx Ny Nz 3"],
    coords: Float[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3"]:
    """Interpolate a 3-vector field component-wise. Returns shape ``(3,)``."""
    return jnp.stack(
        [
            _interpolate_scalar(grid[..., k], coords, grid_spec, method)
            for k in range(3)
        ]
    )


def _interpolate_tensor(
    grid: Float[Array, "Nt Nx Ny Nz 3 3"],
    coords: Float[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3 3"]:
    """Interpolate a 3x3 tensor field component-wise. Returns shape ``(3, 3)``."""
    rows = []
    for i in range(3):
        cols = []
        for j in range(3):
            cols.append(
                _interpolate_scalar(grid[..., i, j], coords, grid_spec, method)
            )
        rows.append(jnp.stack(cols))
    return jnp.stack(rows)
