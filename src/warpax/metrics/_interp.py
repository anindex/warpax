"""Cubic interpolation on a stored radial grid, shared by the shell metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def cubic_on_grid(
    r: Float[Array, ""],
    r_grid: Float[Array, "N"],
    grid_vals: Float[Array, "N"],
) -> Float[Array, ""]:
    """``grid_vals`` at ``r``, clamped to the grid span so extrapolation cannot run."""
    import interpax

    r_clamped = jnp.clip(r, r_grid[0], r_grid[-1])
    return interpax.interp1d(r_clamped, r_grid, grid_vals, method="cubic")
