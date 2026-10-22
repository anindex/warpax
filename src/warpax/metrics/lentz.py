"""Lentz soliton warp drive metric.

Implements the Lentz soliton warp drive (arXiv:2006.07125) using the
WarpFactory-style piecewise diamond-pattern shift construction
(arXiv:2404.03095). The original Lentz formulation is based on solving
a hyperbolic wave equation for a scalar shift potential; no closed-form
analytical solution exists. The WarpFactory piecewise approach is used
instead because:

  (a) Lentz's original paper has known algebraic errors
      (Celmaster-Rubin 2025, arXiv:2511.18251).
  (b) The piecewise approach is numerically well-defined.
  (c) It is the community standard implementation.

ADM components:
    alpha = 1, gamma_ij = delta_ij
    beta^x = -v_s * f_diamond(d), beta^y = beta^z = 0

Shape function (diamond geometry via L1 norm):
    d = |x - v_s*t| + sqrt(y^2 + z^2)
    f_diamond(d) = [tanh(sigma*(d + R)) - tanh(sigma*(d - R))]
                   / [2*tanh(sigma*R)]

Both subluminal (v_s < 1) and superluminal (v_s > 1) regimes are
supported; only the value of v_s differs.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric, flat_x_shift_symbolic


def _diamond_shape(
    d: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
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
    return (jnp.tanh(sigma * (d + R)) - jnp.tanh(sigma * (d - R))) / (2.0 * jnp.tanh(sigma * R))


class LentzMetric(ADMMetric):
    """Lentz soliton warp drive metric via ADM 3+1 decomposition.

    Uses the WarpFactory-style piecewise diamond-pattern shift
    construction with L1 (Manhattan) distance for the rhomboidal
    bubble geometry.

    All parameters are dynamic fields (no recompilation on change). They are
    stored as array leaves, not Python floats: ``eqx.filter_jit`` partitions on
    the value, so a float field would be static and every distinct value would
    retrace the whole curvature chain.

    Parameters
    ----------
    v_s : float
        Warp bubble velocity. Subluminal (< 1) or superluminal (> 1).
    R : float
        Central flat-interior region radius.
    sigma : float
        Smoothing parameter for transitions (larger = sharper wall).
    distance : str
        Distance metric for the bubble shape: ``"L1"`` (default, the
        WarpFactory Manhattan/diamond construction, whose ``|x_rel|`` and
        on-axis ``rho_perp`` kinks make the metric only C^1 and seed the
        large boundary curvature spikes) or ``"L2"`` (Euclidean/spherical,
        C^infty away from the floor) for a smooth-distance control showing
        that those spikes are an implementation artifact of the L1 shape.
    """

    v_s: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=0.1)
    R: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=100.0)
    sigma: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=8.0)
    distance: str = eqx.field(static=True, default="L1")

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift beta^i = (-v_s f(d), 0, 0) on the L1 or L2 distance d.

        ``rho_perp`` and the L2 distance carry a 1e-60 safe-sqrt floor so the
        derivative of sqrt is not 0/0 at y = z = 0 (NaN Christoffels on-axis);
        the floor is numerically negligible.
        """
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        rho_perp = jnp.sqrt(y**2 + z**2 + 1e-60)

        if self.distance == "L2":
            d = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)
        else:
            d = jnp.abs(x_rel) + rho_perp

        f = _diamond_shape(d, self.R, self.sigma)
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Shape function, on the same distance the shift uses.

        It must honor ``self.distance`` exactly as :meth:`shift` does;
        otherwise the wall masks and grid clustering describe a different
        bubble from the metric.
        """
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        rho_perp = jnp.sqrt(y**2 + z**2 + 1e-60)
        if self.distance == "L2":
            d = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)
        else:
            d = jnp.abs(x_rel) + rho_perp
        return _diamond_shape(d, self.R, self.sigma)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)

        x_rel = x - v_s * t
        rho_perp = sp.sqrt(y**2 + z**2)
        d = sp.Abs(x_rel) + rho_perp

        f = (sp.tanh(sigma_val * (d + R_val)) - sp.tanh(sigma_val * (d - R_val))) / (
            2 * sp.tanh(sigma_val * R_val)
        )

        return flat_x_shift_symbolic([t, x, y, z], v_s, f)

    def name(self) -> str:
        return "Lentz"


GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {
        "WEC": "contested",
        "NEC": "contested",
    },
    "note": ("Celmaster-Rubin (2025) contests original Lentz WEC satisfaction claim"),
}
