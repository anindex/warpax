"""Shared shape function utilities for warp drive metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def alcubierre_shape(
    r: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Standard Alcubierre top-hat shape function.

    f(r) = [tanh(sigma*(r+R)) - tanh(sigma*(r-R))] / [2*tanh(sigma*R)]

    f(0) ~ 1, f(inf) ~ 0.

    Parameters
    ----------
    r : Float[Array, "..."]
        Radial distance (e.g., r_s from bubble center).
    R : float
        Bubble radius.
    sigma : float
        Wall thickness parameter (inverse thickness).

    Returns
    -------
    Float[Array, "..."]
        Shape function value in [0, 1].
    """
    return (jnp.tanh(sigma * (r + R)) - jnp.tanh(sigma * (r - R))) / (
        2.0 * jnp.tanh(sigma * R)
    )
