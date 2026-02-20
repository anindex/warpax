"""Natario zero-expansion warp drive metric.

The Natario metric (arXiv:gr-qc/0110086) demonstrates that spatial expansion
and contraction are NOT essential features of warp drive operation.  The trace
of the extrinsic curvature K = div(beta) = 0 everywhere by construction,
meaning there is no volume change of spatial slices.

ADM components:
    alpha = 1  (unit lapse)
    gamma_ij = delta_ij  (flat spatial metric)
    beta^i: two-component shift with zero-expansion constraint

The shift in the co-moving bubble frame (direct Cartesian form):
    beta^x = -v_s * (2*n(r) + r*n'(r)*sin^2(theta))
    beta^y =  v_s * r*n'(r) * x*y/r^2
    beta^z =  v_s * r*n'(r) * x*z/r^2

Shape function:
    n(r_s) = (1/2) * (1 - f_Alc(r_s))
    n(0) = 0  (flat interior),  n(inf) = 1/2  (asymptotic)

At bubble center: n(0) = 0 and n'(0) = 0, so shift = 0 (flat Minkowski interior).
At far field: shift = -v_s * x_hat (uniform flow past bubble).

The zero-expansion condition div(beta) = 0 is satisfied exactly by construction.
This means K (trace of extrinsic curvature) vanishes identically.

Energy density (Eulerian observers):
    rho = -(v_s^2 / kappa) * [3*(dn/dr)^2*cos^2(theta)
           + (dn/dr + r/2*d2n/dr2)^2*sin^2(theta)]
    where kappa = 16*pi.  Strictly negative -> WEC/NEC violation everywhere.

Note: The Natario metric uses the co-moving bubble frame where the interior
is Minkowski and the exterior has a uniform flow. This differs from the
Alcubierre convention where the far field is Minkowski.
"""

from __future__ import annotations

import jax
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

    f_Alc(r_s) = [tanh(sigma*(r_s+R)) - tanh(sigma*(r_s-R))] / [2*tanh(sigma*R)]

    f_Alc(0) ~ 1, f_Alc(inf) ~ 0.
    """
    return (jnp.tanh(sigma * (r_s + R)) - jnp.tanh(sigma * (r_s - R))) / (
        2.0 * jnp.tanh(sigma * R)
    )


def _natario_n(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Natario shape function n(r_s).

    n(r_s) = (1/2) * (1 - f_Alc(r_s))

    n(0) = 0, n(inf) = 1/2.
    """
    return 0.5 * (1.0 - _alcubierre_shape(r_s, R, sigma))


def _natario_dn_dr(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Derivative dn/dr_s of the Natario shape function.

    dn/dr_s = -(1/2) * df_Alc/dr_s
            = -sigma * (sech^2(sigma*(r_s+R)) - sech^2(sigma*(r_s-R)))
              / (4 * tanh(sigma*R))
    """
    sech2_plus = 1.0 / jnp.cosh(sigma * (r_s + R)) ** 2
    sech2_minus = 1.0 / jnp.cosh(sigma * (r_s - R)) ** 2
    return -sigma * (sech2_plus - sech2_minus) / (4.0 * jnp.tanh(sigma * R))


def _natario_d2n_dr2(
    r_s: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Second derivative d2n/dr_s^2 of the Natario shape function.

    d2n/dr2 = -(1/2) * d2f_Alc/dr^2
    where d2f_Alc/dr^2 = sigma^2 * (-2*tanh(a)*sech^2(a) + 2*tanh(b)*sech^2(b))
                          / (2*tanh(sigma*R))
    with a = sigma*(r+R), b = sigma*(r-R).
    """
    a = sigma * (r_s + R)
    b = sigma * (r_s - R)
    sech2_a = 1.0 / jnp.cosh(a) ** 2
    sech2_b = 1.0 / jnp.cosh(b) ** 2
    tanh_a = jnp.tanh(a)
    tanh_b = jnp.tanh(b)

    d2f_dr2 = sigma**2 * (-2.0 * tanh_a * sech2_a + 2.0 * tanh_b * sech2_b) / (
        2.0 * jnp.tanh(sigma * R)
    )
    return -0.5 * d2f_dr2


# ---------------------------------------------------------------------------
# NatarioMetric (ADM decomposition)
# ---------------------------------------------------------------------------


class NatarioMetric(ADMMetric):
    """Natario zero-expansion warp drive metric via ADM 3+1 decomposition.

    Uses the co-moving bubble frame where the interior is Minkowski and
    the exterior has a uniform flow of -v_s in the x-direction.  This is
    the natural Natario convention with div(beta) = 0 exactly.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity.
    R : float
        Warp bubble radius.
    sigma : float
        Wall thickness parameter (inverse thickness).
    """

    v_s: float = 0.1
    R: float = 100.0
    sigma: float = 0.03

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_s_raw = jnp.sqrt(dx**2 + y**2 + z**2)
        r_s = jnp.sqrt(r_s_raw**2 + 1e-24)

        n_val = _natario_n(r_s, self.R, self.sigma)
        dn_val = _natario_dn_dr(r_s, self.R, self.sigma)

        # Direct Cartesian shift satisfying div(beta) = 0 exactly.
        #
        # Derived from coordinate-basis spherical shift:
        #   beta^r = -2*v_s*n(r)*cos(theta)
        #   beta^theta_coord = v_s*(2n + r*dn/dr)*sin(theta)
        # converted via Jacobian to Cartesian.
        sin_theta_sq = (y**2 + z**2) / r_s**2

        beta_x = -self.v_s * (2.0 * n_val + r_s * dn_val * sin_theta_sq)
        beta_y = self.v_s * dn_val * dx * y / r_s
        beta_z = self.v_s * dn_val * dx * z / r_s

        # Origin safety: n(0) = 0, dn(0) = 0, so shift -> 0 at center
        beta_x = jnp.where(r_s_raw < 1e-12, 0.0, beta_x)
        beta_y = jnp.where(r_s_raw < 1e-12, 0.0, beta_y)
        beta_z = jnp.where(r_s_raw < 1e-12, 0.0, beta_z)

        return jnp.array([beta_x, beta_y, beta_z])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation.

        Uses an intermediate symbol r for radial distance to compute
        dn/dr, then substitutes the full r_s expression.
        """
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)
        # Intermediate symbol for differentiation
        r = sp.Symbol("r", positive=True)

        dx_sym = x - v_s * t
        r_s_expr = sp.sqrt(dx_sym**2 + y**2 + z**2)

        # Shape function n(r) = 1/2 * (1 - f_Alc(r))
        f_alc_r = (
            sp.tanh(sigma_val * (r + R_val))
            - sp.tanh(sigma_val * (r - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))
        n_r = sp.Rational(1, 2) * (1 - f_alc_r)

        # dn/dr via differentiation w.r.t. the symbol r
        dn_dr_r = sp.diff(n_r, r)

        # Substitute r -> r_s_expr
        n_val = n_r.subs(r, r_s_expr)
        dn_dr_val = dn_dr_r.subs(r, r_s_expr)

        # Direct Cartesian shift (x-component only for symbolic tractability)
        sin2_theta = (y**2 + z**2) / r_s_expr**2
        beta_x_sym = -v_s * (2 * n_val + r_s_expr * dn_dr_val * sin2_theta)

        # Full 4x4 metric using only beta^x for symbolic tractability
        # (beta^y, beta^z are proportional to dn/dr * (cross terms))
        g = sp.Matrix([
            [-(1 - beta_x_sym**2), beta_x_sym, 0, 0],
            [beta_x_sym, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Natario"


# ---------------------------------------------------------------------------
# Analytical Eulerian energy density
# ---------------------------------------------------------------------------


def natario_eulerian_energy_density(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    v_s: float = 0.1,
    R: float = 100.0,
    sigma: float = 0.03,
) -> Float[Array, "..."]:
    """Analytical Eulerian energy density for Natario metric.

    rho = -(v_s^2 / kappa) * [3*(dn/dr)^2*cos^2(theta)
           + (dn/dr + r/2*d2n/dr2)^2*sin^2(theta)]

    where kappa = 16*pi (in geometric units).

    This is strictly negative (WEC violation everywhere where dn/dr != 0).

    Parameters
    ----------
    x, y, z : array-like
        Spatial coordinates (bubble center at origin, time slice t=0).
    v_s : float
        Bubble velocity.
    R : float
        Bubble radius.
    sigma : float
        Wall thickness parameter.

    Returns
    -------
    rho : array-like
        Eulerian energy density (strictly <= 0).
    """
    r_s = jnp.sqrt(x**2 + y**2 + z**2)
    r_s_safe = jnp.sqrt(r_s**2 + 1e-24)

    cos_theta_sq = x**2 / r_s_safe**2
    sin_theta_sq = (y**2 + z**2) / r_s_safe**2

    dn = _natario_dn_dr(r_s_safe, R, sigma)
    d2n = _natario_d2n_dr2(r_s_safe, R, sigma)

    kappa = 16.0 * jnp.pi
    term_radial = 3.0 * dn**2 * cos_theta_sq
    term_angular = (dn + r_s_safe / 2.0 * d2n) ** 2 * sin_theta_sq

    rho = -(v_s**2 / kappa) * (term_radial + term_angular)

    # Handle origin: rho = 0 where dn/dr = 0
    return jnp.where(r_s < 1e-12, 0.0, rho)


# ---------------------------------------------------------------------------
# Ground truth for validation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
}
