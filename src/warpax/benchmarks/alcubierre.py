"""Alcubierre warp drive benchmark.

The Alcubierre metric in ADM form:
    ds^2 = -dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2

ADM components:
    alpha = 1  (lapse)
    beta^x = -v_s f(r_s),  beta^y = beta^z = 0  (shift)
    gamma_ij = delta_ij  (flat spatial metric)

Shape function (top-hat smoothed by tanh):
    f(r_s) = [tanh(sigma (r_s + R)) - tanh(sigma (r_s - R))] / [2 tanh(sigma R)]

where r_s = sqrt((x - x_s)^2 + y^2 + z^2) is distance from bubble center,
R is bubble radius, sigma controls wall thickness, x_s = v_s t is bubble position.

Ground truth for Eulerian observers:
    rho_Euler = -(v_s^2 / (32 pi)) * (df/dr_s)^2 * (y^2 + z^2) / r_s^2

This is always <= 0, confirming WEC/NEC violation.
"""

from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric


# ---------------------------------------------------------------------------
# Shape function helpers (pure JAX)
# ---------------------------------------------------------------------------


def _shape_function(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Alcubierre shape function f(r_s).

    f(r_s) = [tanh(sigma (r_s + R)) - tanh(sigma (r_s - R))] / [2 tanh(sigma R)]
    """
    return (jnp.tanh(sigma * (r_s + R)) - jnp.tanh(sigma * (r_s - R))) / (
        2 * jnp.tanh(sigma * R)
    )


def _shape_function_derivative(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Derivative df/dr_s of the shape function."""
    sech2_plus = 1.0 / jnp.cosh(sigma * (r_s + R)) ** 2
    sech2_minus = 1.0 / jnp.cosh(sigma * (r_s - R)) ** 2
    return sigma * (sech2_plus - sech2_minus) / (2 * jnp.tanh(sigma * R))


# ---------------------------------------------------------------------------
# AlcubierreMetric (ADM decomposition)
# ---------------------------------------------------------------------------


class AlcubierreMetric(ADMMetric):
    """Alcubierre warp drive metric via ADM 3+1 decomposition.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity.
    R : float
        Warp bubble radius.
    sigma : float
        Wall thickness parameter (larger = thinner wall).
    x_s : float
        Bubble center x-coordinate.
    """

    v_s: float = 0.5
    R: float = 1.0
    sigma: float = 8.0
    x_s: float = 0.0

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        r_s = jnp.sqrt((x - self.x_s) ** 2 + y**2 + z**2)
        f = _shape_function(r_s, self.R, self.sigma)
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)

        r_s = sp.sqrt(x**2 + y**2 + z**2)
        f = (
            sp.tanh(sigma_val * (r_s + R_val))
            - sp.tanh(sigma_val * (r_s - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        # Full metric: ds^2 = -(1 - v_s^2 f^2) dt^2 - 2 v_s f dx dt
        #                      + dx^2 + dy^2 + dz^2
        g = sp.Matrix([
            [-(1 - v_s**2 * f**2), -v_s * f, 0, 0],
            [-v_s * f, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Alcubierre"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def alcubierre_symbolic(
    v_s: sp.Symbol | None = None,
    R_val: sp.Symbol | None = None,
    sigma_val: sp.Symbol | None = None,
) -> SymbolicMetric:
    """Module-level convenience: symbolic Alcubierre metric.

    Parameters
    ----------
    v_s, R_val, sigma_val : sp.Symbol or None
        Symbolic parameters.  If *None*, fresh positive symbols are created.
    """
    t, x, y, z = sp.symbols("t x y z")
    if v_s is None:
        v_s = sp.Symbol("v_s", positive=True)
    if R_val is None:
        R_val = sp.Symbol("R", positive=True)
    if sigma_val is None:
        sigma_val = sp.Symbol("sigma", positive=True)

    r_s = sp.sqrt(x**2 + y**2 + z**2)
    f = (
        sp.tanh(sigma_val * (r_s + R_val))
        - sp.tanh(sigma_val * (r_s - R_val))
    ) / (2 * sp.tanh(sigma_val * R_val))

    g = sp.Matrix([
        [-(1 - v_s**2 * f**2), -v_s * f, 0, 0],
        [-v_s * f, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return SymbolicMetric([t, x, y, z], g)


def eulerian_energy_density(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    v_s: float = 0.5,
    R: float = 1.0,
    sigma: float = 8.0,
    x_s: float = 0.0,
) -> Float[Array, "..."]:
    """Analytical Eulerian energy density.

    rho = -(v_s^2 / (32 pi)) * (df/dr_s)^2 * (y^2 + z^2) / r_s^2
    """
    r_s = jnp.sqrt((x - x_s) ** 2 + y**2 + z**2)
    r_s = jnp.maximum(r_s, 1e-12)

    df = _shape_function_derivative(r_s, R, sigma)
    rho_perp_sq = (y**2 + z**2) / r_s**2

    return -(v_s**2 / (32 * jnp.pi)) * df**2 * rho_perp_sq


# Ground truth for validation
GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
    "eulerian_energy_negative": True,
}
