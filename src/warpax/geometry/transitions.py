"""Smooth transition functions for warp drive metric construction.

Provides parameterized smoothstep functions with configurable continuity
order. The C1 cubic (3t^2 - 2t^3) and C2 quintic (6t^5 - 15t^4 + 10t^3)
variants are the standard Hermite interpolants from computer graphics.

C1 cubic:
    f(0)=0, f(1)=1, f'(0)=f'(1)=0
    f''(0)=6, f''(1)=-6 (NOT zero C1 only)

C2 quintic:
    f(0)=0, f(1)=1, f'(0)=f'(1)=0, f''(0)=f''(1)=0
    First derivative:  30t^4 - 60t^3 + 30t^2 = 30t^2(t-1)^2
    Second derivative: 120t^3 - 180t^2 + 60t = 60t(2t-1)(t-1)

These functions are used by WarpShell's transition helpers
(_shell_indicator, _warpshell_transition) and can be reused by any
metric that needs smooth boundary transitions.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def smoothstep_c1(t: Float[Array, "..."]) -> Float[Array, "..."]:
    """C1-smooth cubic Hermite smoothstep: 3t^2 - 2t^3.

    Boundary conditions:
        f(0) = 0,  f(1) = 1
        f'(0) = 0, f'(1) = 0

    Second derivative (NOT zero at endpoints):
        f''(t) = 6 - 12t
        f''(0) = 6,  f''(1) = -6

    Parameters
    ----------
    t : array
        Input values. Clamped to [0, 1] before evaluation.

    Returns
    -------
    array
        Smoothstep values in [0, 1].
    """
    t_c = jnp.clip(t, 0.0, 1.0)
    return t_c * t_c * (3.0 - 2.0 * t_c)


def smoothstep_c2(t: Float[Array, "..."]) -> Float[Array, "..."]:
    """C2-smooth quintic Hermite smoothstep: 6t^5 - 15t^4 + 10t^3.

    Boundary conditions:
        f(0) = 0,   f(1) = 1
        f'(0) = 0,  f'(1) = 0
        f''(0) = 0, f''(1) = 0

    First derivative:
        f'(t) = 30t^4 - 60t^3 + 30t^2 = 30t^2(t - 1)^2

    Second derivative:
        f''(t) = 120t^3 - 180t^2 + 60t = 60t(2t - 1)(t - 1)

    Both derivatives vanish at t = 0 and t = 1, making this C2-smooth.
    Uses Horner form for numerical stability.

    Parameters
    ----------
    t : array
        Input values. Clamped to [0, 1] before evaluation.

    Returns
    -------
    array
        Smoothstep values in [0, 1].
    """
    t_c = jnp.clip(t, 0.0, 1.0)
    return t_c * t_c * t_c * (t_c * (t_c * 6.0 - 15.0) + 10.0)


def smoothstep(t: Float[Array, "..."], order: int = 2) -> Float[Array, "..."]:
    """Parameterized smoothstep with configurable continuity order.

    Parameters
    ----------
    t : array
        Input values. Clamped to [0, 1] before evaluation.
    order : int
        Continuity order: 1 for C1 cubic, 2 for C2 quintic.

    Returns
    -------
    array
        Smoothstep values in [0, 1].

    Raises
    ------
    ValueError
        If order is not 1 or 2.

    Notes
    -----
    Uses Python ``if``/``elif`` dispatch (not ``jax.lax.cond``) because
    the order is a structural choice, not a data-dependent branch. This
    means different orders produce different JAX traces, which is correct
    for an Equinox module field.
    """
    if order == 1:
        return smoothstep_c1(t)
    elif order == 2:
        return smoothstep_c2(t)
    else:
        raise ValueError(
            f"Unsupported smoothstep order: {order}. Use 1 (C1) or 2 (C2)."
        )
