"""Uniform-grid sanity check, gated behind ``WARPAX_STRICT``."""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._constants import strict_mode_enabled


def assert_uniform_grid(
    r_grid: Float[Array, "N"],
    *,
    name: str = "r_grid",
    rtol: float = 1e-6,
) -> None:
    """Host-side check that ``r_grid`` has constant spacing.

    A no-op unless ``WARPAX_STRICT=1``. Used by TOV / Fuchs / sshell code
    paths that rely on ``dr = r_grid[1] - r_grid[0]`` and trapezoidal /
    Gaussian-convolution semantics.
    """
    if not strict_mode_enabled():
        return
    arr = jnp.asarray(r_grid)
    if arr.shape[0] < 3:
        return
    diffs = jnp.diff(arr)
    span = float(jnp.max(diffs) - jnp.min(diffs))
    base = float(jnp.abs(diffs[0]))
    if base == 0.0:
        raise ValueError(f"{name}: zero spacing between consecutive samples.")
    if span / base > rtol:
        raise ValueError(
            f"{name}: non-uniform grid spacing detected "
            f"(max-min diff = {span:.3e}, base spacing = {base:.3e}, "
            f"relative drift = {span / base:.3e} > {rtol}). Set "
            "WARPAX_STRICT=0 to silence or pass a uniformly-spaced grid."
        )
