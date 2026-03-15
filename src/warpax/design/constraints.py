"""DSGN constraint functions - bubble-size / velocity / boundedness.

Each constraint returns a :class:`ConstraintResult` NamedTuple with a
signed ``margin`` (positive ``=>`` satisfied). The optimizer
in composes these via string-dispatch through
:data:`CONSTRAINT_REGISTRY`.

Reference: see module-level docstring.
§5 (Physics Constraints).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .shape_functions import ShapeFunction


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ConstraintResult(NamedTuple):
    """Signed-margin constraint evaluation result.

    Attributes
    ----------
    satisfied
        True iff ``margin >= 0``.
    margin
        Signed scalar margin (positive = satisfied). Flows through
        ``jax.grad`` for use as a differentiable penalty term in
        the optimizer.
    name
        Constraint-family identifier (``'bubble_size'`` / ``'velocity'`` /
        ``'boundedness'``).
    """
    satisfied: bool
    margin: Float[Array, ""]
    name: str


# ---------------------------------------------------------------------------
# Constraint functions
# ---------------------------------------------------------------------------


def bubble_size_constraint(
    sf: ShapeFunction, max_radius: float = 10.0
) -> ConstraintResult:
    """Shape function must decay below ``0.01`` amplitude at ``max_radius``.

    ``margin = 0.01 - |sf(max_radius)|``. Positive margin ``=>`` the
    shape function has decayed sufficiently beyond the bubble wall.

    Parameters
    ----------
    sf
        Differentiable shape function.
    max_radius
        Radial distance at which the shape must have decayed. Default
        ``10.0`` (geometric units).

    Returns
    -------
    ConstraintResult
        With ``name='bubble_size'``.
    """
    r_far = jnp.asarray(max_radius)
    amp_far = jnp.abs(sf(r_far))
    margin = jnp.asarray(0.01) - amp_far
    return ConstraintResult(
        satisfied=bool(margin >= 0.0),
        margin=margin,
        name="bubble_size",
    )


def velocity_constraint(
    v_s, max_v: float = 10.0
) -> ConstraintResult:
    """Bubble velocity must satisfy ``|v_s| <= max_v``.

    ``margin = max_v - |v_s|``. Default ``max_v=10.0`` in ``c=1``
    units (superluminal-warp regime is permitted by REQUIREMENTS per
    paper-scope invariance).

    Parameters
    ----------
    v_s
        Bubble velocity (scalar or 0-D array).
    max_v
        Upper bound on ``|v_s|``.

    Returns
    -------
    ConstraintResult
        With ``name='velocity'``.
    """
    v_s_arr = jnp.asarray(v_s)
    margin = jnp.asarray(max_v) - jnp.abs(v_s_arr)
    return ConstraintResult(
        satisfied=bool(margin >= 0.0),
        margin=margin,
        name="velocity",
    )


def boundedness_constraint(
    sf: ShapeFunction,
    amp_max: float = 1.0,
    probe_r: Float[Array, "..."] | None = None,
) -> ConstraintResult:
    """Shape amplitude bounded: ``max_r |sf(r)| <= amp_max`` on ``probe_r``.

    ``margin = amp_max - max_r |sf(r)|``.

    Parameters
    ----------
    sf
        Differentiable shape function.
    amp_max
        Upper bound on ``|sf(r)|``. Default ``1.0``.
    probe_r
        Probe radial grid; default ``jnp.linspace(0, 10, 128)``.

    Returns
    -------
    ConstraintResult
        With ``name='boundedness'``.
    """
    if probe_r is None:
        probe_r = jnp.linspace(0.0, 10.0, 128)
    amps = jax.vmap(sf)(probe_r)
    max_amp = jnp.max(jnp.abs(amps))
    margin = jnp.asarray(amp_max) - max_amp
    return ConstraintResult(
        satisfied=bool(margin >= 0.0),
        margin=margin,
        name="boundedness",
    )


# ---------------------------------------------------------------------------
# Registry + composition helpers
# ---------------------------------------------------------------------------


CONSTRAINT_REGISTRY: dict[str, Callable] = {
    "bubble_size": bubble_size_constraint,
    "velocity": velocity_constraint,
    "boundedness": boundedness_constraint,
}


def all_constraints_satisfied(results: list[ConstraintResult]) -> bool:
    """Composable helper: True iff every :class:`ConstraintResult` is satisfied."""
    return all(r.satisfied for r in results)
