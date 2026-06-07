"""Garattini-Zatrimaylov de Sitter warp bubble.

Garattini & Zatrimaylov, "Positive-Energy Warp Drive in a de Sitter Universe"
(arXiv:2502.13153, 2025). A warp bubble on a de Sitter background whose Eulerian
energy density can be non-negative and which satisfies *averaged* energy
conditions (ANEC/AWEC) when the bubble speed matches the de Sitter expansion
rate, even though the pointwise NEC/WEC are violated at the wall.

ADM form (flat de Sitter slicing):

    alpha   = 1                       (unit lapse)
    beta^x  = -v_s * f(r_s)           (Alcubierre-type x-shift, irrotational)
    gamma_ij = e^{2 H t} delta_ij     (isotropic de Sitter expansion)

The de Sitter slicing adds isotropic spatial expansion (nonzero ``theta``) on
top of the Alcubierre-type bubble wall (which carries the usual Alcubierre wall
shear and vorticity). At ``H = 0`` the metric reduces exactly to Alcubierre.

Unlike the irrotational Rodal angular profile, the shift and spatial factor here
are elementary, so :meth:`symbolic` is a *faithful* closed form (usable for
symbolic cross-checks), not a structural placeholder.

Notes
-----
The spacetime is NOT asymptotically flat: ``gamma_ij`` grows as ``e^{2 H t}``.
Certification is performed on a reference slice ``t = t0`` (default ``t0 = 0``,
where ``gamma = delta``); the H-dependence still enters through the time
derivative of the metric in the curvature chain. ANEC along a complete geodesic
should be gated on ``geodesic_complete`` because dS geodesics can leave the
integration box.
"""
from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric
from ._common import alcubierre_shape


class GarattiniMetric(ADMMetric):
    """Garattini-Zatrimaylov de Sitter warp bubble (ADM 3+1).

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity (matched to ``H * R`` for the averaged-condition
        sweet spot; see :meth:`matched`).
    R : float
        Warp bubble radius.
    sigma : float
        Wall thickness parameter (inverse thickness).
    H : float
        de Sitter Hubble (expansion) rate.
    t0 : float
        Reference slice for the spatial expansion factor ``e^{2 H t}``. The
        certification grid is built at ``t = t0``; ``t0 = 0`` gives ``gamma =
        delta`` on the slice while retaining the H-dependence in the curvature.
    """

    v_s: float = 0.1
    R: float = 1.0
    sigma: float = 8.0
    H: float = 0.1
    t0: float = 0.0

    @classmethod
    def matched(cls, R: float = 1.0, sigma: float = 8.0, H: float = 0.1) -> "GarattiniMetric":
        """Speed-matched construction ``v_s = H * R`` (averaged-condition regime)."""
        return cls(v_s=H * R, R=R, sigma=sigma, H=H, t0=0.0)

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_safe = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        f_val = alcubierre_shape(r_safe, self.R, self.sigma)
        beta_x = -self.v_s * f_val
        return jnp.array([beta_x, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        t = coords[0]
        scale = jnp.exp(2.0 * self.H * t)
        return scale * jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_safe = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        return alcubierre_shape(r_safe, self.R, self.sigma)

    # __call__ inherited from ADMMetric (adm_to_full_metric).

    def symbolic(self) -> SymbolicMetric:
        """Faithful SymPy form (elementary shift + exponential spatial factor)."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)
        H = sp.Symbol("H", positive=True)

        dx = x - v_s * t
        r_s = sp.sqrt(dx**2 + y**2 + z**2)
        f_alc = (
            sp.tanh(sigma_val * (r_s + R_val))
            - sp.tanh(sigma_val * (r_s - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        beta_up_x = -v_s * f_alc          # beta^x
        scale = sp.exp(2 * H * t)         # gamma_ij = scale * delta_ij
        beta_low_x = scale * beta_up_x    # beta_x = gamma_xx beta^x
        beta_sq = beta_low_x * beta_up_x  # beta_i beta^i

        g = sp.Matrix([
            [-(1 - beta_sq), beta_low_x, 0, 0],
            [beta_low_x, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, scale],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Garattini"


def garattini_default(v_s: float = 0.1) -> GarattiniMetric:
    """Default Garattini bubble (R=1, sigma=8, H=0.1)."""
    return GarattiniMetric(v_s=v_s, R=1.0, sigma=8.0, H=0.1, t0=0.0)


GROUND_TRUTH = {
    "stress_energy_zero": False,
    # Pointwise conditions fail at the wall (it is still a warp bubble); the
    # Garattini-Zatrimaylov claim is about AVERAGED conditions at v_s = H*R.
    "energy_conditions": {"WEC": False, "NEC": False, "DEC": False, "SEC": False},
    "averaged": {"ANEC_satisfied_at_matched_speed": True},
    # Alcubierre-type wall -> mixed Type I / Type IV (like Alcubierre), with
    # added de Sitter expansion; the distinguishing claim is the AVERAGED
    # condition at v_s = H*R, not a pointwise type.
    "hawking_ellis_type": None,
    "note": (
        "de Sitter background; averaged conditions satisfied when v_s = H*R "
        "(Garattini-Zatrimaylov 2025). Not asymptotically flat; certify on the "
        "t=t0 slice. Reduces exactly to Alcubierre at H=0."
    ),
}
