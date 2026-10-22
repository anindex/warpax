"""WarpShell physical warp drive metric.

Bobrick-Martire / Fell-Heisenberg construction
(arXiv:2102.06824, arXiv:2104.06488) realized via the WarpFactory
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

The exterior is exactly flat while the shell carries the areal mass
``r_s_param / 2``, so the outer blend must shed that mass again: there
``m'(r) = 4 pi r^2 rho < 0``, the energy conditions fail and the ADM mass is
zero. This toy is not the Fuchs et al. construction, which keeps a
Schwarzschild exterior.
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


def _warp_shape_at(metric, coords: Float[Array, "4"]) -> Float[Array, ""]:
    """``S_warp(r)`` in the bubble frame, shared by both shell metrics."""
    t, x, y, z = coords
    r = _safe_radial_norm(x - metric.v_s * t, y, z)
    return _warpshell_transition(
        r, metric.R_1, metric.R_2, metric.R_b, order=metric.transition_order
    )


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
        Smoothstep continuity order: 1 for C1 cubic,
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
        """Lapse blended from 1 to ``sqrt(1 - r_s/r)`` across the shell.

        The ratio ``r_s/r`` is clamped below ``1 - 1e-12`` so ``1 - r_s/r > 0``
        under stress-test ``r_s_param >= R_1``.
        """
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)

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
        return jnp.array([-self.shape_function_value(coords) * self.v_s, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric ``delta_ij + (gamma_rr - 1) n_i n_j``.

        The radial projector is singular at r = 0, so the metric is forced flat
        where ``r^2 < 1e-20``.
        """
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
        gamma = jnp.where(r_sq < 1e-20, jnp.eye(3), gamma)
        return gamma

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Warp transition function S_warp(r): 1 inside, 0 outside."""
        return _warp_shape_at(self, coords)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation.

        Due to the piecewise transition functions, this provides a
        simplified symbolic form showing the shell-region metric
        structure. ``beta_x`` is the contravariant ``beta^x``, so
        ``g_{0x} = gamma_xx beta^x`` and
        ``g_{00} = -(alpha^2 - gamma_xx (beta^x)^2)``; without the
        ``gamma_xx`` factor this is a different spacetime from the numeric
        ``__call__`` wherever ``gamma_rr != 1``.
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

        beta_x_low = gamma_rr * beta_x
        g = sp.Matrix(
            [
                [-(alpha**2 - gamma_rr * beta_x**2), beta_x_low, 0, 0],
                [beta_x_low, gamma_rr, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "WarpShell"


class WarpShellPhysical(ADMMetric):
    """WarpShell with the Schwarzschild radius buried inside the flat interior.

    Requires ``r_s_param < R_1 - smooth_width`` so ``1 - r_s_param / r`` stays
    positive wherever the shell indicator is nonzero. The Schwarzschild factors
    are evaluated at ``max(r, R_1 - smooth_width)``; the indicator and its first
    two derivatives vanish there, so the lapse and spatial metric are C2 across
    the inner blend and no lapse floor or ratio clamp is needed. Use
    :class:`WarpShellMetric` for stress sweeps with ``r_s_param >= R_1``.

    Parameters are the same as :class:`WarpShellMetric`.
    """

    v_s: float = 0.02
    R_1: float = 10.0
    R_2: float = 20.0
    R_b: float = 1.0
    r_s_param: float = 5.0
    smooth_width: float | None = None
    transition_order: int = 2

    def _blend_width(self) -> float:
        return self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)

    def __check_init__(self) -> None:
        r_in = self.R_1 - self._blend_width()
        if not (self.r_s_param < r_in):
            raise ValueError(
                "WarpShellPhysical requires r_s_param < R_1 - smooth_width so the "
                "Schwarzschild factor 1 - r_s_param / r stays positive on the whole "
                f"inner blend. Got r_s_param={self.r_s_param}, R_1={self.R_1}, "
                f"smooth_width={self._blend_width()}. Use WarpShellMetric for the "
                "clamped stress-test variant."
            )

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = _safe_radial_norm(x_rel, y, z)

        sw = self._blend_width()
        r_shell = jnp.maximum(r, self.R_1 - sw)
        alpha_shell = jnp.sqrt(1.0 - self.r_s_param / r_shell)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        alpha = 1.0 - S_shell * (1.0 - alpha_shell)
        return jnp.maximum(alpha, 1e-12)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        return jnp.array([-self.shape_function_value(coords) * self.v_s, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric ``delta_ij + (gamma_rr - 1) n_i n_j``.

        The radial projector is singular at r = 0, so the metric is forced flat
        where ``r^2 < 1e-20``.
        """
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r_sq = x_rel**2 + y**2 + z**2
        r = jnp.sqrt(r_sq + 1e-60)

        sw = self._blend_width()
        r_shell = jnp.maximum(r, self.R_1 - sw)
        gamma_rr_sph = 1.0 / (1.0 - self.r_s_param / r_shell)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        gamma_rr_eff = 1.0 + S_shell * (gamma_rr_sph - 1.0)

        x_vec = jnp.array([x_rel, y, z])
        n_hat = x_vec / r
        radial_proj = jnp.outer(n_hat, n_hat)
        gamma = jnp.eye(3) + (gamma_rr_eff - 1.0) * radial_proj

        gamma = jnp.where(r_sq < 1e-20, jnp.eye(3), gamma)
        return gamma

    @jaxtyped(typechecker=beartype)
    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Warp transition function S_warp(r): 1 inside, 0 outside."""
        return _warp_shape_at(self, coords)

    def symbolic(self) -> SymbolicMetric:
        """Same symbolic form as ``WarpShellMetric``; the floors are
        numeric-only."""
        return WarpShellMetric(
            v_s=self.v_s,
            R_1=self.R_1,
            R_2=self.R_2,
            R_b=self.R_b,
            r_s_param=self.r_s_param,
            smooth_width=self.smooth_width,
            transition_order=self.transition_order,
        ).symbolic()

    def name(self) -> str:
        return "WarpShellPhysical"


GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {
        "WEC": False,
        "NEC": False,
        "DEC": False,
        "SEC": False,
    },
    "note": (
        "The flat exterior forces the shell mass r_s_param / 2 to vanish across the "
        "outer blend, so rho < 0 there (rho(20.6) = -7.2e-4 and min(rho + p_i)(20.9) "
        "= -1.94e-2 at the defaults) and every pointwise condition fails; the ADM "
        "mass is zero. The interior and the shell proper satisfy WEC/NEC/SEC for "
        "subluminal v_s, with the DEC failing at both transitions."
    ),
}
