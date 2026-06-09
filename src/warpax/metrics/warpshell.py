"""WarpShell physical warp drive metric.

Bobrick-Martire / Fell-Heisenberg constant-velocity construction
(arXiv:2102.06824, arXiv:2405.02709) realised via the WarpFactory
"Bobrick-Martire Modified Time" simplification (arXiv:2404.03095 §3.3).
The shell is the only metric in the suite with non-unit lapse and
non-flat spatial metric simultaneously.

Shell structure: flat interior (``r < R_1``), Schwarzschild-like
shell (``R_1 < r < R_2``), flat exterior (``r > R_2``). The shell
spatial metric uses ``gamma_rr = 1 / (1 - r_s_param / r)`` projected
to Cartesians as ``gamma_ij = delta_ij + (gamma_rr - 1) x_i x_j / r^2``;
the shift is ``beta^x = -S_warp(r) v_s``. Transitions use Hermite
``smoothstep`` (``transition_order=1`` C1 cubic, ``=2`` C2 quintic;
default keeps the Riemann tensor continuous).
"""

from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric
from ..geometry.transitions import smoothstep
from ..numerics import R_EPS


def _safe_radial_norm(
    x_rel: Float[Array, ""],
    y: Float[Array, ""],
    z: Float[Array, ""],
) -> Float[Array, ""]:
    """Co-moving radial coordinate ``sqrt(x_rel^2 + y^2 + z^2 + R_EPS)``.

    Floors the radicand by :data:`numerics.R_EPS` so the gradient of
    ``sqrt`` stays finite at the origin (autodiff-safe) without changing
    the numerical value at any practical scale.
    """
    return jnp.sqrt(x_rel**2 + y**2 + z**2 + R_EPS)


def _hermite_smoothstep(t, order=2):
    """Delegate to ``geometry.transitions.smoothstep`` (order=1 C1, 2 C2)."""
    return smoothstep(t, order=order)


def _warpshell_transition(
    r: Float[Array, "..."],
    R_inner: float,
    R_outer: float,
    R_b: float,
    order: int = 2,
) -> Float[Array, "..."]:
    """Smooth transition function for WarpShell regions.

    Returns exactly 1 for r <= R_inner, smoothly transitions to exactly 0
    for r >= R_outer, using Hermite smoothstep interpolation. The
    derivative is zero at both endpoints, ensuring at least C1 contact
    (C2 with order=2).

    Parameters
    ----------
    r : array
        Radial distance from bubble center.
    R_inner : float
        Inner boundary of transition region.
    R_outer : float
        Outer boundary of transition region.
    R_b : float
        Unused (kept for API compatibility).
    order : int
        Continuity order: 1 for C1 cubic, 2 for C2 quintic (default).
    """
    t = jnp.clip((r - R_inner) / (R_outer - R_inner), 0.0, 1.0)
    return 1.0 - _hermite_smoothstep(t, order=order)


def _shell_indicator(
    r: Float[Array, "..."],
    R_inner: float,
    R_outer: float,
    smooth_width: float,
    order: int = 2,
) -> Float[Array, "..."]:
    """Smooth indicator for the shell region.

    Nonzero only between (R_inner - smooth_width) and
    (R_outer + smooth_width), using two Hermite smoothstep ramps.
    The function is exactly 1 for R_inner <= r <= R_outer and
    exactly 0 outside the blending zones. All transitions have zero
    derivative at endpoints (C1 or C2 contact depending on order).

    Parameters
    ----------
    r : array
        Radial distance.
    R_inner : float
        Inner shell boundary.
    R_outer : float
        Outer shell boundary.
    smooth_width : float
        Width of the smooth blending zone at each boundary.
    order : int
        Continuity order: 1 for C1 cubic, 2 for C2 quintic (default).
    """
    sw = jnp.maximum(smooth_width, 1e-12)
    ramp_in = _hermite_smoothstep((r - (R_inner - sw)) / sw, order=order)
    ramp_out = _hermite_smoothstep(((R_outer + sw) - r) / sw, order=order)
    return ramp_in * ramp_out


class WarpShellMetric(ADMMetric):
    """WarpShell warp drive metric via ADM 3+1 decomposition.

    Features non-unit lapse and non-flat spatial metric in the shell
    region.

    All parameters are dynamic fields (no recompilation on change).

    Parameters
    ----------
    v_s : float
        Warp bubble velocity (subluminal).
    R_1 : float
        Inner shell radius.
    R_2 : float
        Outer shell radius.
    R_b : float
        Legacy buffer zone width (kept for API compatibility; used only
        by ``_warpshell_transition`` in the shift computation).
    r_s_param : float
        Schwarzschild radius of shell (2GM/c^2, geometric units).
    smooth_width : float | None
        Width of smooth blending zone at each shell boundary for the
        lapse and spatial metric indicator. If None, defaults to
        0.12 * (R_2 - R_1). This parameter enables sensitivity
        ablation studies.
    transition_order : int
        Smoothstep continuity order: 1 for C1 cubic (legacy),
        2 for C2 quintic (default). C2 guarantees continuous Riemann
        tensor at shell boundaries.
    """

    v_s: float = 0.02
    R_1: float = 10.0
    R_2: float = 20.0
    R_b: float = 1.0
    r_s_param: float = 5.0
    smooth_width: float | None = None
    transition_order: int = 2

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)

        # Clamp ratio so 1 - r_s/r > 0 under stress-test r_s_param >= R_1.
        ratio = self.r_s_param / r
        ratio_safe = jnp.minimum(ratio, 1.0 - 1e-12)
        alpha_shell = jnp.sqrt(1.0 - ratio_safe)

        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        alpha = 1.0 - S_shell * (1.0 - alpha_shell)
        alpha = jnp.maximum(alpha, 1e-12)
        return alpha

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)

        S_warp = _warpshell_transition(r, self.R_1, self.R_2, self.R_b, order=self.transition_order)
        return jnp.array([-S_warp * self.v_s, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r_sq = x_rel**2 + y**2 + z**2
        r = jnp.sqrt(r_sq + 1e-60)

        ratio = self.r_s_param / r
        ratio_safe = jnp.minimum(ratio, 1.0 - 1e-12)
        gamma_rr_sph = 1.0 / (1.0 - ratio_safe)

        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)
        gamma_rr_eff = 1.0 + S_shell * (gamma_rr_sph - 1.0)

        x_vec = jnp.array([x_rel, y, z])
        n_hat = x_vec / r
        radial_proj = jnp.outer(n_hat, n_hat)
        gamma = jnp.eye(3) + (gamma_rr_eff - 1.0) * radial_proj
        # Radial projector is singular at r=0; force flat metric there.
        gamma = jnp.where(r_sq < 1e-20, jnp.eye(3), gamma)
        return gamma

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Warp transition function S_warp(r): 1 inside, 0 outside."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)
        return _warpshell_transition(
            r, self.R_1, self.R_2, self.R_b, order=self.transition_order
        )

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation.

        Due to the piecewise transition functions, this provides a
        simplified symbolic form showing the shell-region metric
        structure.
        """
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        r_s = sp.Symbol("r_s", positive=True)
        R_1 = sp.Symbol("R_1", positive=True)
        R_2 = sp.Symbol("R_2", positive=True)

        x_rel = x - v_s * t
        r = sp.sqrt(x_rel**2 + y**2 + z**2)

        S = sp.Function("S_shell")(r, R_1, R_2)
        S_warp = sp.Function("S_warp")(r, R_1, R_2)

        alpha = 1 - S * (1 - sp.sqrt(1 - r_s / r))
        beta_x = -S_warp * v_s
        gamma_rr = 1 + S * (1 / (1 - r_s / r) - 1)

        g = sp.Matrix([
            [-(alpha**2 - beta_x**2), beta_x, 0, 0],
            [beta_x, gamma_rr, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "WarpShell"


# Alias: WarpShellStressTest carries clamps for r_s_param >= R_1 sweeps;
# WarpShellMetric is kept for backward compatibility.
WarpShellStressTest = WarpShellMetric


class WarpShellPhysical(ADMMetric):
    """Physical-regime WarpShell where the Schwarzschild radius is buried
    inside the flat interior (``r_s_param < R_1``).

    In this regime the shell coordinate ``r`` is always larger than
    ``r_s_param``, so the Schwarzschild factor ``1 - r_s / r`` is bounded
    away from zero and the lapse is a clean ``sqrt(1 - r_s / r)`` without
    needing the ``minimum(ratio, 1.0 - 1e-12)`` clamp or the
    ``maximum(alpha, 1e-12)`` lapse floor that :class:`WarpShellMetric`
    carries. Use this class when modeling a physical thin-shell
    configuration; use :class:`WarpShellStressTest` when you want the
    clamps for robustness under unphysical parameter sweeps.

    Construction validates ``r_s_param < R_1`` and raises ``ValueError``
    otherwise so sensitivity studies can't accidentally silently switch
    back to the clamped behavior.

    Parameters are the same as :class:`WarpShellMetric`.
    """

    v_s: float = 0.02
    R_1: float = 10.0
    R_2: float = 20.0
    R_b: float = 1.0
    r_s_param: float = 5.0
    smooth_width: float | None = None
    transition_order: int = 2

    def __check_init__(self) -> None:
        if not (self.r_s_param < self.R_1):
            raise ValueError(
                "WarpShellPhysical requires r_s_param < R_1 (physical regime: "
                "the Schwarzschild radius must sit strictly inside the flat "
                f"interior).  Got r_s_param={self.r_s_param}, R_1={self.R_1}. "
                "Use WarpShellStressTest if you want the clamped stress-test "
                "variant."
            )

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)

        # Floor r -> R_1 inside the Schwarzschild factor: prevents 0*inf NaN
        # at interior points; no-op in the shell since r >= R_1 > r_s_param.
        r_shell = jnp.maximum(r, self.R_1)
        alpha_shell = jnp.sqrt(1.0 - self.r_s_param / r_shell)

        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        alpha = 1.0 - S_shell * (1.0 - alpha_shell)
        return jnp.maximum(alpha, 1e-12)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)
        S_warp = _warpshell_transition(
            r, self.R_1, self.R_2, self.R_b, order=self.transition_order
        )
        return jnp.array([-S_warp * self.v_s, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r_sq = x_rel**2 + y**2 + z**2
        r = jnp.sqrt(r_sq + 1e-60)

        # Same r -> R_1 floor as the lapse: no-op in the shell.
        r_shell = jnp.maximum(r, self.R_1)
        gamma_rr_sph = 1.0 / (1.0 - self.r_s_param / r_shell)

        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        gamma_rr_eff = 1.0 + S_shell * (gamma_rr_sph - 1.0)

        x_vec = jnp.array([x_rel, y, z])
        n_hat = x_vec / r
        radial_proj = jnp.outer(n_hat, n_hat)
        gamma = jnp.eye(3) + (gamma_rr_eff - 1.0) * radial_proj

        # Origin flat patch: radial projector is singular at r=0.
        gamma = jnp.where(r_sq < 1e-20, jnp.eye(3), gamma)
        return gamma

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)
        return _warpshell_transition(
            r, self.R_1, self.R_2, self.R_b, order=self.transition_order
        )

    def symbolic(self) -> SymbolicMetric:
        """Same symbolic form as ``WarpShellMetric``; the floors are
        numeric-only."""
        return WarpShellMetric(
            v_s=self.v_s, R_1=self.R_1, R_2=self.R_2, R_b=self.R_b,
            r_s_param=self.r_s_param, smooth_width=self.smooth_width,
            transition_order=self.transition_order,
        ).symbolic()

    def name(self) -> str:
        return "WarpShellPhysical"


GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {
        "WEC": True,
        "NEC": True,
        "DEC": False,
        "SEC": True,
    },
    "note": (
        "Interior satisfies WEC/NEC/SEC for subluminal v_s; boundary DEC "
        "violation at the shell transition (margin ~1e-3 to 1e-4)"
    ),
}
