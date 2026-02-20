"""Minkowski (flat) spacetime benchmark.

The simplest spacetime: ds^2 = -dt^2 + dx^2 + dy^2 + dz^2

Ground truth:
- All curvature tensors = 0
- All energy conditions trivially satisfied (T = 0)
- Kretschner scalar = 0
"""

from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import MetricSpecification, SymbolicMetric


class MinkowskiMetric(MetricSpecification):
    """Flat Minkowski spacetime in Cartesian coordinates.

    No parameters (empty pytree leaf set).  The metric is simply
    ``diag(-1, 1, 1, 1)`` everywhere.
    """

    @jaxtyped(typechecker=beartype)
    def __call__(self, coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Minkowski"


def minkowski_symbolic() -> SymbolicMetric:
    """Module-level convenience: symbolic Minkowski metric."""
    return MinkowskiMetric().symbolic()


# Ground truth for validation
GROUND_TRUTH = {
    "kretschner": 0.0,
    "ricci_scalar": 0.0,
    "energy_conditions": {"WEC": True, "NEC": True, "DEC": True, "SEC": True},
    "stress_energy_zero": True,
}
