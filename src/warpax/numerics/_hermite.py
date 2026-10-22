"""Cubic Hermite interpolation on a one-dimensional knot grid.

``local_cubic_slopes`` reproduces interpax's ``method="cubic"`` (C1, averaged
secant slopes) and ``not_a_knot_slopes`` its ``method="cubic2"`` (C2, not-a-knot
end conditions). ``hermite_cubic`` evaluates the piecewise Hermite polynomial
and returns NaN outside the knot span.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def local_cubic_slopes(x: Float[Array, "N"], f: Float[Array, "N"]) -> Float[Array, "N"]:
    """Knot slopes from averaged neighboring secants, one-sided at the ends."""
    dx = jnp.diff(x)
    df = jnp.where(dx == 0, 0.0, jnp.diff(f) / jnp.where(dx == 0, 1.0, dx))
    return jnp.concatenate([df[:1], 0.5 * (df[:-1] + df[1:]), df[-1:]])


def not_a_knot_slopes(x: Float[Array, "N"], f: Float[Array, "N"]) -> Float[Array, "N"]:
    """Knot slopes of the C2 cubic spline with not-a-knot end conditions."""
    n = x.shape[0]
    dx = jnp.diff(x)
    df = jnp.diff(f) / dx
    if n == 2:
        return jnp.concatenate([df, df])
    if n == 3:
        a = jnp.array(
            [[1.0, 1.0, 0.0], [dx[1], 2.0 * (dx[0] + dx[1]), dx[0]], [0.0, 1.0, 1.0]],
            dtype=f.dtype,
        )
        b = jnp.array(
            [2.0 * df[0], 3.0 * (dx[0] * df[1] + dx[1] * df[0]), 2.0 * df[1]], dtype=f.dtype
        )
        return jnp.linalg.solve(a, b)
    diag = jnp.zeros(n, dtype=f.dtype).at[1:-1].set(2.0 * (dx[:-1] + dx[1:]))
    upper = jnp.zeros(n, dtype=f.dtype).at[1:-1].set(dx[:-1])
    lower = jnp.zeros(n, dtype=f.dtype).at[1:-1].set(dx[1:])
    b = jnp.zeros(n, dtype=f.dtype).at[1:-1].set(3.0 * (dx[1:] * df[:-1] + dx[:-1] * df[1:]))
    d0 = x[2] - x[0]
    diag = diag.at[0].set(dx[1])
    upper = upper.at[0].set(d0)
    b = b.at[0].set(((dx[0] + 2.0 * d0) * dx[1] * df[0] + dx[0] ** 2 * df[1]) / d0)
    d1 = x[-1] - x[-3]
    diag = diag.at[-1].set(dx[-2])
    lower = lower.at[-1].set(d1)
    b = b.at[-1].set((dx[-1] ** 2 * df[-2] + (2.0 * d1 + dx[-1]) * dx[-2] * df[-1]) / d1)
    return jax.lax.linalg.tridiagonal_solve(lower, diag, upper, b[:, None])[:, 0]


def hermite_cubic(
    xq: float | Float[Array, ""],
    x: Float[Array, "N"],
    f: Float[Array, "N"],
    fx: Float[Array, "N"],
) -> Float[Array, ""]:
    """Piecewise cubic Hermite value at ``xq``; NaN outside ``[x[0], x[-1]]``."""
    i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, x.shape[0] - 1)
    dx = x[i] - x[i - 1]
    t = (xq - x[i - 1]) / dx
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    val = h00 * f[i - 1] + h10 * dx * fx[i - 1] + h01 * f[i] + h11 * dx * fx[i]
    inside = (xq >= x[0]) & (xq <= x[-1])
    return jnp.where(inside, val, jnp.nan)


def cubic_interp(
    xq: float | Float[Array, ""],
    x: Float[Array, "N"],
    f: Float[Array, "N"],
    fx: Float[Array, "N"] | None = None,
) -> Float[Array, ""]:
    """Local C1 cubic (interpax ``cubic``) unless explicit knot slopes are given."""
    if fx is None:
        fx = local_cubic_slopes(x, f)
    return hermite_cubic(xq, x, f, fx)
