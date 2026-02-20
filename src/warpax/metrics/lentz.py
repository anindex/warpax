"""Lentz soliton warp drive metric.

Implements the Lentz soliton warp drive (arXiv:2006.07125) using the
WarpFactory-style piecewise diamond-pattern shift construction
(arXiv:2404.03095).  The original Lentz formulation is based on solving
a hyperbolic wave equation for a scalar shift potential; no closed-form
analytical solution exists.  The WarpFactory piecewise approach is used
instead because:

  (a) Lentz's original paper has known algebraic errors
      (Celmaster-Rubin 2025, arXiv:2511.18251).
  (b) The piecewise approach is numerically well-defined.
  (c) It is the community standard implementation.

The soliton creates a "flying formation" of shift blocks surrounding
a flat interior.  The shift vector pattern in the x-y plane forms a
diamond/rhombus shape using the L1 (Manhattan) distance.

ADM components:
    alpha = 1  (unit lapse)
    beta^x = -v_s * f_diamond(d),  beta^y = beta^z = 0  (shift)
    gamma_ij = delta_ij  (flat spatial metric)

Shape function (diamond geometry via L1 norm):
    d = |x - v_s*t| + sqrt(y^2 + z^2)
    f_diamond(d) = [tanh(sigma*(d + R)) - tanh(sigma*(d - R))]
                   / [2*tanh(sigma*R)]

This produces a diamond-shaped bubble instead of a spherical one,
matching the Lentz soliton geometry.

Both subluminal (v_s < 1) and superluminal (v_s > 1) regimes are
supported; only the value of v_s differs.

JAX-based implementation.
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


def _diamond_shape(
    d: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Diamond-pattern shape function using L1 (Manhattan) distance.

    f(d) = [tanh(sigma*(d + R)) - tanh(sigma*(d - R))] / [2*tanh(sigma*R)]

    Transitions smoothly from ~1 (d << R, interior) to ~0 (d >> R, exterior).

    Parameters
    ----------
    d : array
        L1 distance from bubble center: d = |x_rel| + rho_perp.
    R : float
        Central flat-interior region radius.
    sigma : float
        Smoothing parameter (larger = sharper transitions).
    """
    return (jnp.tanh(sigma * (d + R)) - jnp.tanh(sigma * (d - R))) / (
        2.0 * jnp.tanh(sigma * R)
    )


# ---------------------------------------------------------------------------
# LentzMetric (ADM decomposition)
# ---------------------------------------------------------------------------


class LentzMetric(ADMMetric):
    """Lentz soliton warp drive metric via ADM 3+1 decomposition.

    Uses the WarpFactory-style piecewise diamond-pattern shift
    construction with L1 (Manhattan) distance for the rhomboidal
    bubble geometry.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity.  Subluminal (< 1) or superluminal (> 1).
    R : float
        Central flat-interior region radius.
    sigma : float
        Smoothing parameter for transitions (larger = sharper wall).
    """

    v_s: float = 0.1
    R: float = 100.0
    sigma: float = 8.0

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords

        # Relative position to bubble center (moving at v_s along x)
        x_rel = x - self.v_s * t
        rho_perp = jnp.sqrt(y**2 + z**2)

        # L1 (Manhattan) distance for diamond/rhombus geometry
        d = jnp.abs(x_rel) + rho_perp

        # Diamond shape function: 1 inside, 0 outside
        f = _diamond_shape(d, self.R, self.sigma)

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

        # L1 distance (diamond geometry)
        x_rel = x - v_s * t
        rho_perp = sp.sqrt(y**2 + z**2)
        d = sp.Abs(x_rel) + rho_perp

        # Diamond shape function
        f = (
            sp.tanh(sigma_val * (d + R_val))
            - sp.tanh(sigma_val * (d - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        # Full metric from ADM with unit lapse, flat spatial, shift = (-v_s*f, 0, 0)
        # g_00 = -(alpha^2 - beta_i beta^i) = -(1 - v_s^2 f^2)
        # g_0x = beta_x = gamma_xj beta^j = -v_s f  (gamma = delta)
        # g_ij = delta_ij
        g = sp.Matrix([
            [-(1 - v_s**2 * f**2), -v_s * f, 0, 0],
            [-v_s * f, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Lentz"


# ---------------------------------------------------------------------------
# Ground truth for validation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {
        "WEC": "contested",
        "NEC": "contested",
    },
    "note": (
        "Celmaster-Rubin (2025) contests original Lentz WEC "
        "satisfaction claim"
    ),
}
