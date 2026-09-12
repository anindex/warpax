"""Natario zero-expansion warp drive metric.

The trace of the extrinsic curvature K = div(beta) = 0 everywhere by
construction (arXiv:gr-qc/0110086): no volume change of spatial slices.

ADM components:
    alpha = 1 (unit lapse)
    gamma_ij = delta_ij (flat spatial metric)
    beta^i: divergence-free Cartesian shift

The laboratory coordinates x = xi + v_s t use
    beta_lab = -X(x - v_s t, y, z) - v_s e_x,
where Natario writes ds^2 = -dt^2 + |d xi - X(xi) dt|^2. Thus
    beta^x = v_s * (2*n(r) + r*n'(r)*sin^2(theta) - 1)
    beta^y = -v_s * n'(r) * (x - v_s*t)*y/r
    beta^z = -v_s * n'(r) * (x - v_s*t)*z/r.

The analytic profile n = (1 - f_Alc)/2 has n(0)=0 and n(infinity)=1/2;
it approaches the constant interior/exterior profiles smoothly. The center
worldline (t, v_s*t, 0, 0) has unit timelike tangent, and beta tends to zero
at spatial infinity. The Eulerian density is nonpositive; a negative value
violates WEC. NEC requires its separate null-contraction check.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric
from ._common import alcubierre_shape


def _natario_n(
    r_s: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
) -> Float[Array, "..."]:
    """Natario shape function n(r_s).

    n(r_s) = (1/2) * (1 - f_Alc(r_s))

    n(0) = 0, n(inf) = 1/2.
    """
    return 0.5 * (1.0 - alcubierre_shape(r_s, R, sigma))


def _natario_dn_dr(
    r_s: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
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
    r_s: Float[Array, "..."], R: float | Float[Array, ""], sigma: float | Float[Array, ""]
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

    d2f_dr2 = (
        sigma**2 * (-2.0 * tanh_a * sech2_a + 2.0 * tanh_b * sech2_b) / (2.0 * jnp.tanh(sigma * R))
    )
    return -0.5 * d2f_dr2


def _natario_n_and_nq(q, R, sigma):
    """Evaluate n(q) and dn/dq, including their analytic center limits.

    q=r^2. The even tanh profile is evaluated by its Taylor series when
    sigma^2*q < 1e-6; the first omitted term is O((sigma^2*q)^5), below
    binary64 rounding there. This preserves the center derivatives instead
    of replacing a finite ball by a constant shift.
    """
    u = sigma**2 * q
    near_center = u < 1e-6
    r = jnp.sqrt(jnp.where(near_center, 1.0 / sigma**2, q))
    n = _natario_n(r, R, sigma)
    nq = _natario_dn_dr(r, R, sigma) / (2 * r)
    a = jnp.tanh(sigma * R) ** 2
    c = 0.5 / jnp.cosh(sigma * R) ** 2
    c2 = a - 2.0 / 3
    c3 = a**2 - 4 * a / 3 + 17.0 / 45
    c4 = a**3 - 2 * a**2 + 6 * a / 5 - 62.0 / 315
    n_series = c * u * (1 + u * (c2 + u * (c3 + u * c4)))
    nq_series = c * sigma**2 * (1 + u * (2 * c2 + u * (3 * c3 + 4 * u * c4)))
    return jnp.where(near_center, n_series, n), jnp.where(near_center, nq_series, nq)


class NatarioMetric(ADMMetric):
    """Natario zero-expansion warp drive metric via ADM 3+1 decomposition.

    Uses laboratory coordinates, with beta=-v_s e_x at the moving center
    and beta tending to zero at infinity. The shift is divergence-free.

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
        r_sq = dx**2 + y**2 + z**2
        n_val, nq_val = _natario_n_and_nq(r_sq, self.R, self.sigma)

        # beta_lab=-X(x-v_s*t)-v_s e_x, with dn/dr=2*r*dn/dq.
        beta_x = self.v_s * (2 * n_val + 2 * nq_val * (y**2 + z**2) - 1)
        beta_y = -2 * self.v_s * nq_val * dx * y
        beta_z = -2 * self.v_s * nq_val * dx * z

        return jnp.array([beta_x, beta_y, beta_z])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Alcubierre-convention shape function f(r_s) underlying the Natario n(r)."""
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_s = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        return alcubierre_shape(r_s, self.R, self.sigma)

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

        f_alc_r = (sp.tanh(sigma_val * (r + R_val)) - sp.tanh(sigma_val * (r - R_val))) / (
            2 * sp.tanh(sigma_val * R_val)
        )
        n_r = sp.Rational(1, 2) * (1 - f_alc_r)
        dn_dr_r = sp.diff(n_r, r)

        n_val = n_r.subs(r, r_s_expr)
        dn_dr_val = dn_dr_r.subs(r, r_s_expr)

        sin2_theta = (y**2 + z**2) / r_s_expr**2
        beta = sp.Matrix(
            [
                v_s * (2 * n_val + r_s_expr * dn_dr_val * sin2_theta - 1),
                -v_s * dn_dr_val * dx_sym * y / r_s_expr,
                -v_s * dn_dr_val * dx_sym * z / r_s_expr,
            ]
        )
        g = sp.eye(4)
        g[0, 0] = -1 + beta.dot(beta)
        g[0, 1:] = beta.T
        g[1:, 0] = beta
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Natario"


def natario_eulerian_energy_density(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    v_s: float = 0.1,
    R: float = 100.0,
    sigma: float = 0.03,
    t: float = 0.0,
) -> Float[Array, "..."]:
    """Analytical Eulerian energy density for Natario metric.

    rho = -(v_s^2 / kappa) * [3*(dn/dr)^2*cos^2(theta)
           + (dn/dr + r/2*d2n/dr2)^2*sin^2(theta)]

    where kappa = 8*pi (in geometric units; Natario 2002 Eq. for T_{mu nu} u^mu u^nu).

    Uses co-moving radius ``dx = x - v_s t`` consistent with the metric shift.

    This is nonpositive and violates WEC wherever the squared bracket is positive.

    Parameters
    ----------
    x, y, z : array-like
        Spatial coordinates (bubble center at origin in lab frame at t=0).
    v_s : float
        Bubble velocity.
    R : float
        Bubble radius.
    sigma : float
        Wall thickness parameter.
    t : float
        Time coordinate for moving bubble center.

    Returns
    -------
    rho : array-like
        Eulerian energy density (strictly <= 0).
    """
    dx = x - v_s * t
    r_sq = dx**2 + y**2 + z**2
    nq, nqq = jax.jvp(
        lambda q: _natario_n_and_nq(q, R, sigma)[1],
        (r_sq,),
        (jnp.ones_like(r_sq),),
    )
    # Rewrite the spherical formula in q=r^2 so the center has no 0/0 limit.
    radial = 12 * nq**2 * dx**2
    angular = (3 * nq + 2 * r_sq * nqq) ** 2 * (y**2 + z**2)
    return -(v_s**2 / (8 * jnp.pi)) * (radial + angular)


GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
}
