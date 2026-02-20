"""Rodal irrotational warp drive metric.

The Rodal metric (arXiv:2512.18008) modifies the Alcubierre metric to produce
an *irrotational* shift vector derived from a scalar potential.  This ensures
the shift one-form is exact (curl-free), guaranteeing globally Hawking-Ellis
Type I everywhere.

ADM components:
    alpha = 1  (unit lapse)
    gamma_ij = delta_ij  (flat spatial metric)
    beta^i: irrotational shift with radial profile f(r) and angular profile g(r)

The paper defines shape functions f_paper(r) (f_paper(0)=0, f_paper(inf)=1)
and g_paper(r) (g_paper(0)=0, g_paper(inf)=1) in the co-moving bubble frame.
For consistency with Alcubierre convention (far field = Minkowski), we use
the lab-frame forms:

    F(r) = f_Alc(r) = 1 - f_paper(r)   [F(0)=1, F(inf)=0]
    G(r) = 1 - g_paper(r)               [G(0)=1, G(inf)=0]

The shift in spherical tetrad components (theta from +x direction of travel):
    beta_r_hat     = -v_s * F(r_s) * cos(theta)
    beta_theta_hat =  v_s * G(r_s) * sin(theta)

At bubble center: beta = (-v_s, 0, 0), passengers carried along.
At far field: beta = (0, 0, 0), Minkowski spacetime.

The irrotational property (curl-free shift) is preserved by the coordinate
transformation since a uniform translation field is curl-free.

Peak NEC/WEC violation reduced by factor ~38 vs Alcubierre.
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
    r: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Standard Alcubierre top-hat shape function.

    f_Alc(r) = [tanh(sigma*(r+R)) - tanh(sigma*(r-R))] / [2*tanh(sigma*R)]

    f_Alc(0) ~ 1, f_Alc(inf) ~ 0.
    """
    return (jnp.tanh(sigma * (r + R)) - jnp.tanh(sigma * (r - R))) / (
        2.0 * jnp.tanh(sigma * R)
    )


def _stable_logcosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable ln(cosh(x)).

    ln(cosh(x)) = |x| + ln(1 + exp(-2|x|)) - ln(2)

    Avoids overflow for large |x| where cosh(x) overflows.
    """
    abs_x = jnp.abs(x)
    return abs_x + jnp.log1p(jnp.exp(-2.0 * abs_x)) - jnp.log(2.0)


def _rodal_g_paper(
    r: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Paper-convention irrotational angular profile g(r) from Rodal Eq. (42).

    g(r) = [2*r*sigma*sinh(R*sigma) + cosh(R*sigma)*log_ratio]
           / [2*r*sigma*sinh(R*sigma)]

    where log_ratio = stable_logcosh(sigma*(r-R)) - stable_logcosh(sigma*(r+R)).

    g_paper(0) = 0, g_paper(inf) = 1.  (Paper co-moving frame convention.)
    """
    # C-inf regularization at r=0: eps^2 = 1e-24 chosen so that
    # sqrt(r^2 + eps^2) ~ r for r > 1e-12 (grid spacing >> eps).
    # See scripts/run_regularization_sensitivity.py for eps sensitivity analysis.
    r_safe = jnp.sqrt(r**2 + 1e-24)

    # Numerically stable log-cosh difference
    a = sigma * (r_safe - R)
    b = sigma * (r_safe + R)
    log_ratio = _stable_logcosh(a) - _stable_logcosh(b)

    sinh_R_sigma = jnp.sinh(R * sigma)
    cosh_R_sigma = jnp.cosh(R * sigma)

    numerator = 2.0 * r_safe * sigma * sinh_R_sigma + cosh_R_sigma * log_ratio
    # denominator uses sinh(x/2)*cosh(x/2) = sinh(x)/2 simplification
    denominator = 2.0 * r_safe * sigma * sinh_R_sigma

    return numerator / jnp.maximum(denominator, 1e-30)


def _rodal_G(
    r: Float[Array, "..."], R: float, sigma: float
) -> Float[Array, "..."]:
    """Lab-frame irrotational angular profile G(r) = 1 - g_paper(r).

    G(0) = 1, G(inf) = 0.  Matches Alcubierre far-field convention.
    """
    return 1.0 - _rodal_g_paper(r, R, sigma)


# ---------------------------------------------------------------------------
# RodalMetric (ADM decomposition)
# ---------------------------------------------------------------------------


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
        rho_perp = jnp.sqrt(y**2 + z**2)
        rho_perp_safe = jnp.sqrt(rho_perp**2 + 1e-24)

        cos_theta = dx / r_s
        sin_theta = rho_perp / r_s

        # Lab-frame shift components in spherical tetrad
        # F(r) = f_Alc(r): F(0)=1, F(inf)=0  (radial, same as Alcubierre)
        # G(r) = 1 - g_paper(r): G(0)=1, G(inf)=0  (angular, irrotational)
        F_val = _alcubierre_shape(r_s, self.R, self.sigma)
        G_val = _rodal_G(r_s, self.R, self.sigma)

        beta_r_hat = -self.v_s * F_val * cos_theta
        beta_theta_hat = self.v_s * G_val * sin_theta

        # Spherical to Cartesian (theta measured from +x direction of travel)
        beta_x = beta_r_hat * cos_theta - beta_theta_hat * sin_theta
        radial_plus_angular = beta_r_hat * sin_theta + beta_theta_hat * cos_theta
        cos_phi = y / rho_perp_safe
        sin_phi = z / rho_perp_safe
        beta_y = radial_plus_angular * cos_phi
        beta_z = radial_plus_angular * sin_phi

        # Origin safety: at r_s=0, F=1, G=1, so shift = (-v_s, 0, 0)
        # The spherical-to-Cartesian conversion is well-defined at center
        # when approaching along x-axis, but numerical issues arise when
        # rho_perp -> 0. Use analytical limit.
        beta_x = jnp.where(r_s_raw < 1e-12, -self.v_s, beta_x)
        beta_y = jnp.where(r_s_raw < 1e-12, 0.0, beta_y)
        beta_z = jnp.where(r_s_raw < 1e-12, 0.0, beta_z)

        return jnp.array([beta_x, beta_y, beta_z])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation.

        The symbolic form uses f_Alc(r) for the radial shift (same as
        Alcubierre) since the angular correction g(r) is complex to
        represent symbolically and is handled numerically.
        """
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)

        dx = x - v_s * t
        r_s = sp.sqrt(dx**2 + y**2 + z**2)

        # Alcubierre shape function (lab-frame radial profile)
        f_alc = (
            sp.tanh(sigma_val * (r_s + R_val))
            - sp.tanh(sigma_val * (r_s - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        # Simplified x-only shift for symbolic form
        beta_x_sym = -v_s * f_alc

        g = sp.Matrix([
            [-(1 - beta_x_sym**2), beta_x_sym, 0, 0],
            [beta_x_sym, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Rodal"


# ---------------------------------------------------------------------------
# Ground truth for validation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
    "hawking_ellis_type": 1,
}
