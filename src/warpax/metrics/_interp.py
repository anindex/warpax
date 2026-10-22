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
    from ..numerics import cubic_interp

    return cubic_interp(jnp.clip(r, r_grid[0], r_grid[-1]), r_grid, grid_vals)
