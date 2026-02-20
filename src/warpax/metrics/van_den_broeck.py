"""Van Den Broeck volume-modified Alcubierre warp drive metric.

The Van Den Broeck metric (arXiv:gr-qc/9905084) modifies the Alcubierre metric
by introducing a conformal factor B(r_s) on the spatial metric.  This expands
the internal volume of the warp bubble while keeping the external surface area
small, dramatically reducing the total negative energy requirement.

Line element:
    ds^2 = -dt^2 + B^2(r_s) * [(dx - v_s f(r_s) dt)^2 + dy^2 + dz^2]

ADM components:
    alpha = 1  (unit lapse)
    beta^i = (-v_s * f(r_s), 0, 0)  (SAME shift as Alcubierre)
    gamma_ij = B^2(r_s) * delta_ij  (conformal spatial metric)

where B(r_s) = 1 + alpha_vdb * f_B(r_s) is the conformal factor with
a second shape function f_B controlling the volume expansion region.

Still violates WEC/NEC (same mechanism as Alcubierre), but total negative
energy reduced by orders of magnitude.
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


def _alcubierre_shape(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Standard Alcubierre top-hat shape function.

    f(r_s) = [tanh(sigma*(r_s+R)) - tanh(sigma*(r_s-R))] / [2*tanh(sigma*R)]
    """
    return (jnp.tanh(sigma * (r_s + R)) - jnp.tanh(sigma * (r_s - R))) / (
        2.0 * jnp.tanh(sigma * R)
    )


def _van_den_broeck_B(
    r_s: Float[Array, "..."],
    R_tilde: float,
    alpha_vdb: float,
    sigma_B: float,
) -> Float[Array, "..."]:
    """Van Den Broeck conformal factor B(r_s).

    B(r_s) = 1 + alpha_vdb * f_B(r_s)

    where f_B is the standard Alcubierre top-hat shape function with radius
    R_tilde and wall thickness sigma_B.

    B(0) = 1 + alpha_vdb (expanded interior),
    B(inf) = 1 (flat exterior).
    """
    f_B = _alcubierre_shape(r_s, R_tilde, sigma_B)
    return 1.0 + alpha_vdb * f_B


# ---------------------------------------------------------------------------
# VanDenBroeckMetric (ADM decomposition)
# ---------------------------------------------------------------------------


class VanDenBroeckMetric(ADMMetric):
    """Van Den Broeck volume-modified Alcubierre metric via ADM 3+1 decomposition.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity.
    R : float
        Outer bubble radius (for shift shape function).
    sigma : float
        Wall thickness for shift shape function.
    R_tilde : float
        Inner radius for volume expansion (conformal factor).
    alpha_vdb : float
        Expansion factor (named alpha_vdb to avoid conflict with lapse alpha).
    sigma_B : float
        Wall thickness for conformal factor shape function.
    """

    v_s: float = 0.1
    R: float = 350.0
    sigma: float = 8.0
    R_tilde: float = 200.0
    alpha_vdb: float = 0.5
    sigma_B: float = 8.0

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        r_s = jnp.sqrt((x - self.v_s * t) ** 2 + y**2 + z**2)
        f = _alcubierre_shape(r_s, self.R, self.sigma)
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        t, x, y, z = coords
        r_s = jnp.sqrt((x - self.v_s * t) ** 2 + y**2 + z**2)
        B = _van_den_broeck_B(r_s, self.R_tilde, self.alpha_vdb, self.sigma_B)
        return B**2 * jnp.eye(3)

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)
        R_tilde = sp.Symbol("R_tilde", positive=True)
        alpha_vdb = sp.Symbol("alpha_vdb", positive=True)
        sigma_B = sp.Symbol("sigma_B", positive=True)

        r_s = sp.sqrt(x**2 + y**2 + z**2)

        # Shift shape function (outer bubble)
        f = (
            sp.tanh(sigma_val * (r_s + R_val))
            - sp.tanh(sigma_val * (r_s - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        # Conformal factor shape function (inner bubble)
        f_B = (
            sp.tanh(sigma_B * (r_s + R_tilde))
            - sp.tanh(sigma_B * (r_s - R_tilde))
        ) / (2 * sp.tanh(sigma_B * R_tilde))
        B = 1 + alpha_vdb * f_B

        beta_x = -v_s * f

        # Full 4x4 metric:
        # g_00 = -(alpha^2 - gamma_ij beta^i beta^j) = -(1 - B^2 beta_x^2)
        # g_0x = gamma_xj beta^j = B^2 * beta_x
        # g_ij = B^2 * delta_ij
        g = sp.Matrix([
            [-(1 - B**2 * beta_x**2), B**2 * beta_x, 0, 0],
            [B**2 * beta_x, B**2, 0, 0],
            [0, 0, B**2, 0],
            [0, 0, 0, B**2],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Van Den Broeck"


# ---------------------------------------------------------------------------
# Ground truth for validation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
}
