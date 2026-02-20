"""Schwarzschild black hole benchmark in isotropic Cartesian coordinates.

Isotropic form:
    ds^2 = -((1 - M/(2r_iso))/(1 + M/(2r_iso)))^2 dt^2
           + (1 + M/(2r_iso))^4 (dx^2 + dy^2 + dz^2)

where r_iso = sqrt(x^2 + y^2 + z^2) is the isotropic radial coordinate.

Ground truth:
- Vacuum solution: T_{mu nu} = 0
- Kretschner scalar: K = 48 M^2 / r^6  (in Schwarzschild r, not isotropic r_iso)
  In isotropic coords: r = r_iso * (1 + M/(2*r_iso))^2,
  so K = 48 M^2 / [r_iso * (1 + M/(2*r_iso))^2]^6
- All energy conditions trivially satisfied (vacuum)
"""

from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import MetricSpecification, SymbolicMetric


class SchwarzschildMetric(MetricSpecification):
    """Schwarzschild metric in isotropic Cartesian coordinates.

    Full g_ab interface (NOT ADM subclass) because Schwarzschild in isotropic
    coordinates has zero shift.

    Parameters
    ----------
    M : float
        Mass parameter.  Dynamic field (no recompilation on change).
    """

    M: float = 1.0

    @jaxtyped(typechecker=beartype)
    def __call__(self, coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        t, x, y, z = coords
        r_iso = jnp.sqrt(x**2 + y**2 + z**2)
        r_iso = jnp.maximum(r_iso, 1e-10)
        ratio = self.M / (2.0 * r_iso)
        alpha = (1.0 - ratio) / (1.0 + ratio)
        psi4 = (1.0 + ratio) ** 4

        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-alpha**2)
        g = g.at[1, 1].set(psi4)
        g = g.at[2, 2].set(psi4)
        g = g.at[3, 3].set(psi4)
        return g

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation."""
        t, x, y, z = sp.symbols("t x y z")
        M = sp.Symbol("M", positive=True)

        r_iso = sp.sqrt(x**2 + y**2 + z**2)
        ratio = M / (2 * r_iso)
        alpha = (1 - ratio) / (1 + ratio)
        psi4 = (1 + ratio) ** 4

        g = sp.Matrix([
            [-alpha**2, 0, 0, 0],
            [0, psi4, 0, 0],
            [0, 0, psi4, 0],
            [0, 0, 0, psi4],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Schwarzschild"


def schwarzschild_symbolic(M: sp.Symbol | None = None) -> SymbolicMetric:
    """Module-level convenience: symbolic Schwarzschild metric in isotropic coords.

    Parameters
    ----------
    M : sp.Symbol or None
        Mass symbol.  If *None*, creates ``Symbol('M', positive=True)``.
    """
    t, x, y, z = sp.symbols("t x y z")
    if M is None:
        M = sp.Symbol("M", positive=True)

    r_iso = sp.sqrt(x**2 + y**2 + z**2)
    ratio = M / (2 * r_iso)
    alpha = (1 - ratio) / (1 + ratio)
    psi4 = (1 + ratio) ** 4

    g = sp.Matrix([
        [-alpha**2, 0, 0, 0],
        [0, psi4, 0, 0],
        [0, 0, psi4, 0],
        [0, 0, 0, psi4],
    ])
    return SymbolicMetric([t, x, y, z], g)


def kretschner_isotropic(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    M: float = 1.0,
) -> Float[Array, "..."]:
    """Analytical Kretschner scalar in isotropic coordinates.

    K = 48 M^2 / r_schw^6  where  r_schw = r_iso * (1 + M / (2 * r_iso))^2
    """
    r_iso = jnp.sqrt(x**2 + y**2 + z**2)
    r_iso = jnp.maximum(r_iso, 1e-10)
    r_schw = r_iso * (1 + M / (2 * r_iso)) ** 2
    return 48 * M**2 / r_schw**6


# Ground truth for validation
GROUND_TRUTH = {
    "stress_energy_zero": True,
    "energy_conditions": {"WEC": True, "NEC": True, "DEC": True, "SEC": True},
}
