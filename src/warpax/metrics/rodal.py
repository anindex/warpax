"""Rodal irrotational warp drive metric.

Rodal, GRG 58:1, 2026 (arXiv:2512.18008). The ideal shift derives from
a scalar potential and has Hawking-Ellis Type-I stress-energy. The numerical
radius regularizations below do not preserve exact irrotationality.

ADM: ``alpha = 1``, ``gamma_ij = delta_ij``, ``beta^i`` from radial
profile ``F(r)`` and angular profile ``G(r)`` (lab frame:
``F(0) = G(0) = 1``, both -> 0 at infinity):

    beta = -v_s * [G(r_s) * x_hat + (F(r_s) - G(r_s)) * n_x * n]

with ``n = (dx, y, z) / r_s``. Manifestly regular at ``r_s = 0`` since
``F - G -> 0``. The 0/0 form of ``g_paper(0)`` is handled by the analytic
limit ``Delta'(0) = -2*sigma*tanh(sigma*R)``.

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


def _rodal_g_paper(
    r: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Paper-convention irrotational angular profile g(r) from Rodal Eq. (42).

    Rewritten as:
        g_paper(r) = 1 + cosh(R*sigma) * (log_ratio / r) / (2*sigma*sinh(R*sigma))

    where log_ratio = stable_logcosh(sigma*(r-R)) - stable_logcosh(sigma*(r+R)).

    For small r, log_ratio/r has the 0/0 removable form. The analytic limit is
        lim_{r->0} log_ratio / r = Delta'(0) = -2*sigma*tanh(sigma*R)
    which gives g_paper(0) = 0 exactly for all R*sigma.

    g_paper(0) = 0, g_paper(inf) = 1. (Paper co-moving frame convention.)
    """
    # Radius floor for numerical evaluation; the small-radius branch follows below.
    r_safe = jnp.sqrt(r**2 + 1e-60)

    # Numerically stable log-cosh difference (Delta)
    a = sigma * (r_safe - R)
    b = sigma * (r_safe + R)
    log_ratio = _stable_logcosh(a) - _stable_logcosh(b)

    sinh_R_sigma = jnp.sinh(R * sigma)
    cosh_R_sigma = jnp.cosh(R * sigma)

    # Analytic limit: lim_{r->0} log_ratio / r = -2*sigma*tanh(sigma*R)
    limit_ratio = -2.0 * sigma * jnp.tanh(sigma * R)
    log_ratio_over_r = jnp.where(
        r < 1e-8,
        limit_ratio,
        log_ratio / r_safe,
    )

    # g_paper = 1 + cosh(R*sigma) * (log_ratio/r) / (2*sigma*sinh(R*sigma))
    g_paper = 1.0 + cosh_R_sigma * log_ratio_over_r / (2.0 * sigma * sinh_R_sigma)
    return g_paper


def _rodal_G(
    r: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Lab-frame irrotational angular profile G(r) = 1 - g_paper(r).

    G(0) = 1, G(inf) = 0. Matches Alcubierre far-field convention.
    """
    return 1.0 - _rodal_g_paper(r, R, sigma)


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
        # Tight floor for value precision; coarser floor in the divisor
        # below keeps ``\\partial_i n_j`` finite at the bubble center,
        # because ``\\partial n_x / \\partial dx = 1 / r_div`` blows up
        # as ``r_div \\to 0`` even though n_x itself is finite.
        r_safe = jnp.sqrt(r_s_sq + 1e-60)
        r_div = jnp.sqrt(r_s_sq + 1e-12)

        F_val = alcubierre_shape(r_safe, self.R, self.sigma)
        G_val = _rodal_G(r_safe, self.R, self.sigma)

        n_x = dx / r_div
        n_y = y / r_div
        n_z = z / r_div

        # Direct Cartesian shift: beta = -v_s * [G * x_hat + (F-G) * n_x * n].
        # F(0) = G(0) = 1 so the (F-G) * n_x * n_j contributions vanish
        # at the origin; the coarser ``r_div`` floor keeps autodiff
        # bounded while ``r_safe`` carries the physical radial value.
        diff_FG = F_val - G_val
        beta_x = -self.v_s * (G_val + diff_FG * n_x * n_x)
        beta_y = -self.v_s * (diff_FG * n_x * n_y)
        beta_z = -self.v_s * (diff_FG * n_x * n_z)

        return jnp.array([beta_x, beta_y, beta_z])

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

        This represents the ideal profile; it omits the radius floors and
        small-radius branch in the numerical implementation. Agreement away
        from that regularized region is numerical, not an exact identity
        between the symbolic and regularized metrics.
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
