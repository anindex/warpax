"""Vectorised eigenvalue-based energy condition checks for Type I stress-energy (JAX).

For Type I tensors with eigenvalues {-rho, p1, p2, p3}:

- **WEC:** rho >= 0  AND  rho + p_i >= 0   for all i
- **NEC:** rho + p_i >= 0                   for all i
- **DEC:** rho >= |p_i|                     for all i
- **SEC:** rho + p_i >= 0 for all i  AND  rho + sum(p_i) >= 0

Each function returns a **signed margin scalar** (inequality slack):
positive means satisfied, negative means violated.  This slack certifies
the Boolean truth of the condition but is not, in general, equal to the
minimum observer-contracted energy density ``min_u T_{ab} u^a u^b``;
it measures proximity to the eigenvalue inequality boundary rather than
a physical extremum over the observer manifold.

The threshold for labelling a violation is ``margin < -atol``
(configurable, default 1e-10), following the convention that
violations are labelled only when the margin falls below ``-atol``.

All functions are pointwise on ``(rho, pressures)`` and intended to be
lifted to grids via ``jax.vmap``.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Individual condition checks
# ---------------------------------------------------------------------------


def check_wec(
    rho: Float[Array, ""],
    pressures: Float[Array, "3"],
) -> Float[Array, ""]:
    """Weak Energy Condition margin.

    WEC: rho >= 0 AND rho + p_i >= 0 for all i.
    Returns ``min(rho, rho+p1, rho+p2, rho+p3)``.
    """
    candidates = jnp.concatenate(
        [jnp.expand_dims(rho, -1), rho + pressures]
    )
    return jnp.min(candidates)


def check_nec(
    rho: Float[Array, ""],
    pressures: Float[Array, "3"],
) -> Float[Array, ""]:
    """Null Energy Condition margin.

    NEC: rho + p_i >= 0 for all i.
    Returns ``min(rho+p1, rho+p2, rho+p3)``.
    """
    return jnp.min(rho + pressures)


def check_dec(
    rho: Float[Array, ""],
    pressures: Float[Array, "3"],
) -> Float[Array, ""]:
    """Dominant Energy Condition margin.

    DEC: rho >= |p_i| for all i.
    Returns ``min(rho - |p1|, rho - |p2|, rho - |p3|)``.
    """
    return jnp.min(rho - jnp.abs(pressures))


def check_sec(
    rho: Float[Array, ""],
    pressures: Float[Array, "3"],
) -> Float[Array, ""]:
    """Strong Energy Condition margin.

    SEC: rho + p_i >= 0 for all i  AND  rho + sum(p_i) >= 0.
    Returns ``min(rho+p1, rho+p2, rho+p3, rho+p1+p2+p3)``.
    """
    trace = rho + jnp.sum(pressures)
    candidates = jnp.concatenate(
        [rho + pressures, jnp.expand_dims(trace, -1)]
    )
    return jnp.min(candidates)


# ---------------------------------------------------------------------------
# Combined check
# ---------------------------------------------------------------------------


def check_all(
    rho: Float[Array, ""],
    pressures: Float[Array, "3"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Check all four energy conditions, returning (nec, wec, sec, dec) margins."""
    return (
        check_nec(rho, pressures),
        check_wec(rho, pressures),
        check_sec(rho, pressures),
        check_dec(rho, pressures),
    )
