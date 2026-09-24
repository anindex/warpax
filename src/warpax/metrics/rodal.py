"""Rodal irrotational warp drive metric.

Rodal, GRG 58:1, 2026 (arXiv:2512.18008). The ideal shift derives from
a scalar potential and has Hawking-Ellis Type-I stress-energy. Near the
origin, an even Taylor expansion avoids cancellation in its derivatives.

ADM: ``alpha = 1``, ``gamma_ij = delta_ij``, ``beta^i`` from radial
profile ``F(r)`` and angular profile ``G(r)`` (lab frame:
``F(0) = G(0) = 1``, both -> 0 at infinity):

    beta = -v_s * [G(r_s) * x_hat + (F(r_s) - G(r_s)) * n_x * n]

with ``n = (dx, y, z) / r_s``. Manifestly regular at ``r_s = 0`` since
``F - G = O(r_s**2)``. The Cartesian implementation evaluates
``(F-G)/r_s**2`` directly at the origin.

"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric
from ._common import alcubierre_shape


def _stable_logcosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable ln(cosh(x)).

    ln(cosh(x)) = |x| + ln(1 + exp(-2|x|)) - ln(2)

    Avoids overflow for large |x| where cosh(x) overflows.
    """
    abs_x = jnp.abs(x)
    return abs_x + jnp.log1p(jnp.exp(-2.0 * abs_x)) - jnp.log(2.0)


def _rodal_profiles(r_squared, R, sigma):
    """Return G and (F-G)/r², using d(rG)/dr=F at the removable origin.

    For |sigma*r| < 0.01, integrate the even F series through order eight.
    Evaluate the complementary expressions away from zero so inactive
    branches also have finite autodiff derivatives.
    """
    e = jnp.exp(-jnp.abs(sigma * R))
    s = (2 * e / (1 + e**2)) ** 2
    c2, c4 = -s, s**2 - s / 3
    c6, c8 = -(s**3) + 2 * s**2 / 3 - 2 * s / 45, s**4 - s**3 + s**2 / 5 - s / 315
    x2 = jnp.minimum(sigma**2 * r_squared, 1e-4)
    G_center = 1 + x2 * (c2 / 3 + x2 * (c4 / 5 + x2 * (c6 / 7 + x2 * c8 / 9)))
    H_center = sigma**2 * (2 * c2 / 3 + x2 * (4 * c4 / 5 + x2 * (6 * c6 / 7 + x2 * 8 * c8 / 9)))
    r = jnp.sqrt(jnp.maximum(r_squared, 1e-4 / sigma**2))
    G_outer = (_stable_logcosh(sigma * (r + R)) - _stable_logcosh(sigma * (r - R))) / (
        2 * sigma * r * jnp.tanh(sigma * R)
    )
    H_outer = (alcubierre_shape(r, R, sigma) - G_outer) / r**2
    center = sigma**2 * r_squared < 1e-4
    return jnp.where(center, G_center, G_outer), jnp.where(center, H_center, H_outer)


def _rodal_g_paper(
    r: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Paper-convention angular profile: g_paper(0)=0, g_paper(infinity)=1."""
    return 1.0 - _rodal_G(r, R, sigma)


def _rodal_G(
    r: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Lab-frame irrotational angular profile G(r) = 1 - g_paper(r).

    G(0) = 1, G(inf) = 0. Matches Alcubierre far-field convention.
    """
    return _rodal_profiles(r**2, R, sigma)[0]


class RodalMetric(ADMMetric):
    """Rodal irrotational warp drive metric via ADM 3+1 decomposition.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity.
    R : float
        Warp bubble radius (rho in paper notation).
    sigma : float
        Wall thickness parameter (inverse thickness).
    """

    # Array leaves, not Python floats: eqx.filter_jit partitions on the
    # VALUE, so a float field is static and every distinct value retraced
    # the whole curvature chain.
    v_s: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=0.1)
    R: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=100.0)
    sigma: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=0.03)

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_s_sq = dx**2 + y**2 + z**2
        G, H = _rodal_profiles(r_s_sq, self.R, self.sigma)
        return -self.v_s * (G * jnp.array([1.0, 0.0, 0.0]) + H * dx * jnp.array([dx, y, z]))

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Shape function f(r_s) for the Rodal metric."""
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_safe = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        return alcubierre_shape(r_safe, self.R, self.sigma)

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return the full analytic laboratory tensor for ``r_s > 0``.

        Rodal's Eqs. (8), (36), (40) and (42) give the rest-frame tensor
        with a minus-sign shift. Pulling it back by ``xi = x - v_s*t``
        gives ``beta_lab = -X - v_s*e_x`` in our plus-sign convention.
        The center value and derivatives require their analytic limits.

        The numerical implementation uses an even Taylor expansion near the
        removable origin; elsewhere it evaluates this analytic profile.
        """
        t, x, y, z = sp.symbols("t x y z", real=True)
        v_s = sp.Symbol("v_s", real=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)

        dx = x - v_s * t
        r_s = sp.sqrt(dx**2 + y**2 + z**2)

        F = (sp.tanh(sigma_val * (r_s + R_val)) - sp.tanh(sigma_val * (r_s - R_val))) / (
            2 * sp.tanh(sigma_val * R_val)
        )
        G = (
            sp.log(sp.cosh(sigma_val * (r_s + R_val))) - sp.log(sp.cosh(sigma_val * (r_s - R_val)))
        ) / (2 * sigma_val * r_s * sp.tanh(sigma_val * R_val))
        beta = -v_s * (G * sp.Matrix([1, 0, 0]) + (F - G) * dx * sp.Matrix([dx, y, z]) / r_s**2)
        g = sp.eye(4)
        g[0, 0] = -1 + beta.dot(beta)
        for i in range(3):
            g[0, i + 1] = g[i + 1, 0] = beta[i]
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Rodal"


GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
    "hawking_ellis_type": 1,
}
