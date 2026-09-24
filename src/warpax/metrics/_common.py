"""Shared shape function utilities for warp drive metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def alcubierre_shape(
    r: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Standard Alcubierre top-hat shape function.

    f(r) = [tanh(sigma*(r+R)) - tanh(sigma*(r-R))] / [2*tanh(sigma*R)]

    f(0) = 1 exactly, f -> 0 as r -> inf.

    Parameters
    ----------
    r : Float[Array, "..."]
        Radial distance (e.g., r_s from bubble center).
    R : float
        Bubble radius.
    sigma : float
        Wall steepness parameter: larger sigma => thinner wall
        (inverse wall thickness).

    Returns
    -------
    Float[Array, "..."]
        Shape function value in [0, 1].
    """
    # Near r=0, differentiating the tanh difference loses f'(r) to cancellation.
    # The identical even form retains f''(0)=-2*sigma**2*sech(sigma*R)**2,
    # including when r is the regularized Cartesian radius used by the metrics.
    x = sigma * r
    exp_a = jnp.exp(-jnp.abs(sigma * R))
    sech_a = 2 * exp_a / (1 + exp_a**2)
    center = 1 / (1 + (sech_a * jnp.sinh(jnp.clip(x, -0.5, 0.5))) ** 2)
    outer = (jnp.tanh(sigma * (r + R)) - jnp.tanh(sigma * (r - R))) / (2.0 * jnp.tanh(sigma * R))
    return jnp.where(jnp.abs(x) < 0.5, center, outer)
