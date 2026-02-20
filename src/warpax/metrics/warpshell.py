"""WarpShell warp drive metric.

Implements the WarpShell / Bobrick-Martire / Fell-Heisenberg
constant-velocity physical warp drive (arXiv:2102.06824,
arXiv:2405.02709) using the WarpFactory "Bobrick-Martire Modified
Time" simplification (arXiv:2404.03095, Section 3.3).

This is the most physically complex warp drive metric in the suite
because it has BOTH non-unit lapse AND non-flat spatial metric in the
shell region.

Shell structure:
    r < R_1          : Flat interior (Minkowski with shift)
    R_1 < r < R_2    : Curved shell (Schwarzschild-like)
    r > R_2          : Flat exterior (Minkowski, no shift)

ADM components:
    alpha != 1  (non-unit lapse in shell: Schwarzschild time dilation)
    beta^x = -S_warp(r) * v_s  (uniform shift inside, zero outside)
    gamma_ij != delta_ij  (Schwarzschild radial stretching in shell)

The spatial metric in the shell uses a Schwarzschild-like radial
component gamma_rr = 1/(1 - r_s_param/r), converted to Cartesian
coordinates via the standard spherical-to-Cartesian projection:
    gamma_ij = delta_ij + (gamma_rr - 1) * (x_i * x_j / r^2)

Transition functions use C2-smooth quintic Hermite smoothstep
(6t^5 - 15t^4 + 10t^3) by default. The ``transition_order``
parameter selects the smoothness class:
    transition_order=1: C1 cubic (3t^2 - 2t^3) legacy
    transition_order=2: C2 quintic (6t^5 - 15t^4 + 10t^3) default

C2 smoothness guarantees continuous second derivatives of the metric
at shell boundaries, which means the Riemann tensor (computed via
two applications of jax.jacfwd) is continuous across transition seams.
The ``smooth_width`` constructor parameter independently controls the
transition zone width for the shell indicator.

JAX-based implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import ADMMetric, SymbolicMetric
from ..geometry.transitions import smoothstep


# ---------------------------------------------------------------------------
# Transition function helpers (pure JAX, C1/C2 Hermite smoothstep)
#
# Both C1 cubic and C2 quintic smoothstep are supported via the ``order``
# parameter. The actual polynomials live in geometry/transitions.py;
# these helpers apply them to WarpShell's specific transition geometry.
# ---------------------------------------------------------------------------


def _hermite_smoothstep(t, order=2):
    """Smoothstep: h(0)=0, h(1)=1, h'(0)=h'(1)=0.

    Delegates to the shared smoothstep function in geometry/transitions.py.

    Parameters
    ----------
    t : array
        Input values.
    order : int
        Continuity order: 1 for C1 cubic, 2 for C2 quintic (default).
    """
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


# ---------------------------------------------------------------------------
# WarpShellMetric (ADM decomposition)
# ---------------------------------------------------------------------------


class WarpShellMetric(ADMMetric):
    """WarpShell warp drive metric via ADM 3+1 decomposition.

    Features non-unit lapse and non-flat spatial metric in the shell
    region, making it the most physically complex warp metric.

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

        # Radial distance from bubble center (moving at v_s along x)
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2)
        r_safe = jnp.sqrt(r**2 + 1e-24)

        # Schwarzschild lapse in shell region
        # alpha_shell = sqrt(1 - r_s / r)
        ratio = self.r_s_param / r_safe
        # Clamp ratio to avoid negative under sqrt (r_s_param < r in shell)
        ratio_safe = jnp.minimum(ratio, 1.0 - 1e-12)
        alpha_shell = jnp.sqrt(1.0 - ratio_safe)

        # Shell indicator: nonzero only in [R_1, R_2] with smooth blending
        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        # Smooth interpolation: alpha = 1 outside shell, alpha_shell inside
        # No jnp.where hard clamps needed the Hermite smoothstep in
        # _shell_indicator reaches exact 0 outside the blending zone.
        alpha = 1.0 - S_shell * (1.0 - alpha_shell)

        # Safety: lapse must be positive
        alpha = jnp.maximum(alpha, 1e-12)

        return alpha

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords

        # Radial distance from bubble center
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2)

        # Warp transition: 1 inside shell, 0 outside
        S_warp = _warpshell_transition(r, self.R_1, self.R_2, self.R_b, order=self.transition_order)

        # Uniform shift inside, zero outside
        beta_x = -S_warp * self.v_s

        return jnp.array([beta_x, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        t, x, y, z = coords

        # Radial distance from bubble center
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2)
        r_safe = jnp.sqrt(r**2 + 1e-24)

        # Schwarzschild radial component in shell
        ratio = self.r_s_param / r_safe
        ratio_safe = jnp.minimum(ratio, 1.0 - 1e-12)
        gamma_rr_sph = 1.0 / (1.0 - ratio_safe)

        # Shell indicator with smooth blending
        sw = self.smooth_width if self.smooth_width is not None else 0.12 * (self.R_2 - self.R_1)
        S_shell = _shell_indicator(r, self.R_1, self.R_2, sw, order=self.transition_order)

        # Effective radial component: 1 outside shell, gamma_rr_sph in shell
        gamma_rr_eff = 1.0 + S_shell * (gamma_rr_sph - 1.0)

        # Convert spherical radial stretching to Cartesian:
        # gamma_ij = delta_ij + (gamma_rr_eff - 1) * (x_i * x_j / r^2)
        # This stretches only the radial direction.
        x_vec = jnp.array([x_rel, y, z])
        r_sq = r_safe**2

        # Outer product x_i x_j / r^2
        n_hat = x_vec / r_safe
        radial_proj = jnp.outer(n_hat, n_hat)

        gamma = jnp.eye(3) + (gamma_rr_eff - 1.0) * radial_proj

        # For r near zero (interior), force flat metric
        gamma = jnp.where(r < 1e-10, jnp.eye(3), gamma)

        return gamma

    # __call__ is inherited from ADMMetric (uses adm_to_full_metric)

    def symbolic(self) -> SymbolicMetric:
        """Return SymPy symbolic form for inspection and cross-validation.

        Due to the piecewise transition functions, this provides a
        simplified symbolic form showing the shell-region metric
        structure.
        """
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        r_s = sp.Symbol("r_s", positive=True)  # Schwarzschild radius param
        R_1 = sp.Symbol("R_1", positive=True)
        R_2 = sp.Symbol("R_2", positive=True)

        # Radial distance from bubble center
        x_rel = x - v_s * t
        r = sp.sqrt(x_rel**2 + y**2 + z**2)

        # Symbolic indicator for shell region (simplified)
        S = sp.Function("S_shell")(r, R_1, R_2)
        S_warp = sp.Function("S_warp")(r, R_1, R_2)

        # Lapse
        alpha = 1 - S * (1 - sp.sqrt(1 - r_s / r))

        # Shift
        beta_x = -S_warp * v_s

        # Spatial metric (simplified: radial direction only)
        gamma_rr = 1 + S * (1 / (1 - r_s / r) - 1)

        # Full metric (using ADM reconstruction)
        # g_00 = -(alpha^2 - beta_x^2)
        # g_0x = beta_x (with gamma_xx ~ gamma_rr along x)
        g = sp.Matrix([
            [-(alpha**2 - beta_x**2), beta_x, 0, 0],
            [beta_x, gamma_rr, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "WarpShell"


# ---------------------------------------------------------------------------
# Ground truth for validation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "stress_energy_zero": False,
    "energy_conditions": {
        "WEC": True,
        "NEC": True,
        "DEC": True,
        "SEC": True,
    },
    "note": (
        "Claimed to satisfy all energy conditions for subluminal "
        "velocities"
    ),
}
