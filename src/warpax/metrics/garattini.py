"""Garattini-Zatrimaylov de Sitter warp bubble.

Garattini & Zatrimaylov, "Positive-Energy Warp Drive in a de Sitter Universe"
(arXiv:2502.13153, 2025). A warp bubble on a de Sitter background whose Eulerian
energy density can be non-negative and which satisfies *averaged* energy
conditions (ANEC/AWEC) when the bubble speed matches the de Sitter expansion
rate, even though the pointwise NEC/WEC are violated at the wall.

ADM form, exactly as published (their Sec. 2; shift-only class):

    alpha    = 1                                  (unit lapse)
    gamma_ij = delta_ij                           (spatial slices are FLAT)
    N^i      = -(1 - f(r_s)) x^i / L  -  f(r_s) v^i,      L = 1/H

The de Sitter expansion is carried by the BACKGROUND SHIFT -x^i/L, not by a
scale factor: this is de Sitter in Painleve-Gullstrand (flat-slicing) form. The
bubble switches that background flow off inside the wall and replaces it with
-v^i, which is what the interpolation by f does.

An earlier implementation here used ``gamma_ij = e^{2Ht} delta_ij`` with an
Alcubierre shift ``beta^x = -v_s f``. That is a different spacetime: it keeps the
Hubble flow everywhere instead of switching it off inside the bubble, its slices
are not flat, and its shift is NOT irrotational -- so it reported wall vorticity
and Type-IV structure that the published construction does not have.

Irrotationality is the point of the paper, and it holds under the matching condition
``v = r_0 / L`` (the bubble moves with the Hubble flow at its own position). Then

    N = -x/L + f (x - x_s)/L,

a sum of two radial gradients -- about the origin and about the bubble centre --
hence curl N = 0 identically. By the momentum constraint the Eulerian momentum
density then vanishes and the stress-energy is Hawking-Ellis Type I everywhere.

This class realises the matched family and ONLY that family. ``shift`` takes
``x_s(t) = (v_s/H) e^{Ht}`` and ``v = H x_s``, so ``v = r_0 / L`` is an identity of the
parametrisation rather than a condition on it, so ``curl beta`` vanishes
*identically* at every ``v_s``, ``R``, ``sigma`` and ``H`` -- an algebraic fact
about the shift, not a numerical result. What the autodiff pipeline returns is
that fact through float64: the realised values are rounding noise at 1e-18 to
1e-22, not the literal 0.0 an earlier version of this docstring claimed. No
irrotationality tolerance appears anywhere downstream because the vanishing is
structural, not because the measurement is exact.

Two consequences worth stating plainly, because an earlier version of this docstring
got both wrong. First, ``v_s`` fixes the bubble's position at ``t = 0`` as well as its
speed there: ``r_0 = v_s / H`` and ``|N(x_s)| = v_s``. Second, the un-matched case is
NOT implemented, so no claim about it is tested here; the previous text quoted a
measurement (``|curl beta| ~ 0.29``) from a configuration this class cannot produce.
``matched()`` is the inverse parametrisation -- specify ``r_0`` and get ``v_s = H r_0``
-- not a different spacetime.

At ``H = 0`` with the matching relaxed this reduces to Alcubierre.

Unlike the irrotational Rodal angular profile, the shift and spatial factor here
are elementary, so :meth:`symbolic` is a *faithful* closed form (usable for
symbolic cross-checks), not a structural placeholder.

Notes
-----
The slices are flat by construction -- ``spatial_metric`` returns the identity at
every ``t`` -- because this is de Sitter in Painleve-Gullstrand form, where the
expansion is carried entirely by the background shift ``-x^i/L``. An earlier version
of this note described ``gamma_ij`` as growing like ``e^{2 H t}``; that was the
superseded exponential-slicing implementation, not this one. The H-dependence enters
through the shift and through its time derivative in the curvature chain. ANEC along a complete geodesic
should be conditional on ``geodesic_complete`` because dS geodesics can leave the
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
        Retained for backward compatibility with the superseded exponential-slicing
        implementation, which built its certification grid on a reference slice. It is
        read by nothing: the slices here are flat at every ``t``, so
        ``max|g(t0=0) - g(t0=17)| = 0`` exactly. Do not add a use for it without
        deciding what it should mean.
    """

    v_s: float = 0.1
    R: float = 1.0
    sigma: float = 8.0
    H: float = 0.1
    t0: float = 0.0

    @classmethod
    def matched(cls, R: float = 1.0, sigma: float = 8.0, H: float = 0.1,
                r0: float | None = None) -> "GarattiniMetric":
        """The paper's matching condition ``v = r_0 / L = H r_0``.

        ``r_0`` is the bubble's position at ``t = 0``; the shift is irrotational
        only on the slice where the bubble sits at ``r_0 = v_s / H``. Defaulting
        ``r_0 = R`` reproduces the previous ``v_s = H R`` convention.
        """
        r0 = R if r0 is None else r0
        return cls(v_s=H * r0, R=R, sigma=sigma, H=H, t0=0.0)

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        # The bubble co-moves with the Hubble flow: r(t) = r_0 e^{Ht}, so its
        # velocity is v(t) = H r(t). A constant-velocity centre x_s = v_s t
        # breaks the matching at every t != 0 and the shift is then NOT
        # irrotational -- measured |curl beta| ~ 0.29 at a generic wall point.
        H = jnp.asarray(self.H)
        big = jnp.abs(H) > 1e-30
        H_safe = jnp.where(big, H, 1.0)
        # Matched bubble: r(t) = r_0 e^{Ht} with r_0 = v_s / H and v = H r(t).
        # At H = 0 the de Sitter flow is absent and the construction degenerates
        # to a constant-velocity Alcubierre bubble, which is the limit the tests
        # pin; branch rather than divide by H.
        x_s = jnp.where(big, (self.v_s / H_safe) * jnp.exp(H * t), self.v_s * t)
        v_x = jnp.where(big, H * x_s, self.v_s)
        dx = x - x_s
        r_safe = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        f_val = alcubierre_shape(r_safe, self.R, self.sigma)
        pos = jnp.array([x, y, z])
        vel = jnp.array([v_x, 0.0, 0.0])
        return -(1.0 - f_val) * pos * H - f_val * vel

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        H = jnp.asarray(self.H)
        big = jnp.abs(H) > 1e-30
        H_safe = jnp.where(big, H, 1.0)
        x_s = jnp.where(big, (self.v_s / H_safe) * jnp.exp(H * t), self.v_s * t)
        dx = x - x_s
        r_safe = jnp.sqrt(dx**2 + y**2 + z**2 + 1e-60)
        return alcubierre_shape(r_safe, self.R, self.sigma)

    # __call__ inherited from ADMMetric (adm_to_full_metric).

    def symbolic(self) -> SymbolicMetric:
        """Faithful SymPy form: flat slices, interpolated Hubble/bubble shift."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        R_val = sp.Symbol("R", positive=True)
        sigma_val = sp.Symbol("sigma", positive=True)
        H = sp.Symbol("H", positive=True)

        x_s = (v_s / H) * sp.exp(H * t)
        dx = x - x_s
        r_s = sp.sqrt(dx**2 + y**2 + z**2)
        f_alc = (
            sp.tanh(sigma_val * (r_s + R_val))
            - sp.tanh(sigma_val * (r_s - R_val))
        ) / (2 * sp.tanh(sigma_val * R_val))

        # N^i = -(1 - f) H x^i - f v^i with v = H x_s; gamma = delta so
        # beta_i = beta^i.
        b = [-(1 - f_alc) * H * c - f_alc * (H * x_s if c is x else 0)
             for c in (x, y, z)]
        beta_sq = sum(c * c for c in b)

        g = sp.Matrix([
            [-(1 - beta_sq), b[0], b[1], b[2]],
            [b[0], 1, 0, 0],
            [b[1], 0, 1, 0],
            [b[2], 0, 0, 1],
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
    # Under the paper's matching condition the shift is irrotational, so the
    # Eulerian momentum density vanishes and the wall is Hawking-Ellis Type I.
    "hawking_ellis_type": 1,
    "note": (
        "de Sitter background; averaged conditions satisfied when v_s = H*R "
        "(Garattini-Zatrimaylov 2025). Not asymptotically flat; certify on the "
        "t=t0 slice. Reduces exactly to Alcubierre at H=0."
    ),
}
