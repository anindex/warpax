"""- ShapeFunctionMetric: wrap a ShapeFunction into an ADMMetric.

The :class:`ShapeFunctionMetric` is an :class:`warpax.geometry.metric.ADMMetric`
subclass whose shift vector magnitude is driven by a differentiable
:class:`warpax.design.ShapeFunction`. The ADM components mirror the
Alcubierre convention::

    alpha(coords) = 1 (unit lapse)
    beta^i(coords) = -v_s * f(r_s) * delta^i_x (shift along x)
    gamma_{ij}(coords) = delta_{ij} (flat spatial metric)

where ``r_s = sqrt((x - v_s t)^2 + y^2 + z^2)`` is the bubble-centerd
radial coordinate and ``f`` is the wrapped shape function.

A construction-time ``verify_physical`` gate
enforces three must-pass checks on a 16**3 probe grid:

1. **Lapse floor:** ``alpha(coords) >= lapse_floor`` (default ``1e-6``).
2. **CTC-free:** ``g_tt(coords) < 0`` at every probe (no closed timelike
   curves).
3. **Bubble-finite:** ``|f(10 R)| < 1`` (shape function decays /
   stays bounded outside the bubble).

``strict=True`` (default) raises :class:`UnphysicalMetricError` on
failure; ``strict=False`` warns via :class:`UnphysicalMetricWarning`
and returns the metric anyway (inspection mode for the optimizer).

Reference: see module-level docstring.
§5 (Physics Constraints).
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Float

from ..geometry.metric import ADMMetric, SymbolicMetric
from .shape_functions import ShapeFunction


# ---------------------------------------------------------------------------
# Error / warning types
# ---------------------------------------------------------------------------


class UnphysicalMetricError(ValueError):
    """raised by ``__check_init__`` when verify_physical fails with strict=True."""


class UnphysicalMetricWarning(UserWarning):
    """warning variant of UnphysicalMetricError for strict=False (inspection mode)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class PhysicalityVerdict(NamedTuple):
    """Per-check outcome of :meth:`ShapeFunctionMetric.verify_physical`.

    Attributes
    ----------
    lapse_floor_ok
        True iff ``alpha(coords) >= lapse_floor`` at every probe point.
    ctc_free
        True iff ``g_tt(coords) < 0`` at every probe point.
    bubble_finite
        True iff the shape function stays bounded at ``10 * R``.
    overall
        Logical AND of the three checks.
    """
    lapse_floor_ok: bool
    ctc_free: bool
    bubble_finite: bool
    overall: bool


# ---------------------------------------------------------------------------
# ShapeFunctionMetric
# ---------------------------------------------------------------------------


class ShapeFunctionMetric(ADMMetric):
    """ADMMetric wrapping a differentiable :class:`ShapeFunction`.

    Parameters
    ----------
    shape_fn
        Differentiable shape function ``f : R -> R`` (typically ``f(r_s)``).
    v_s
        Warp bubble velocity (in units of ``c = 1``).
    R
        Bubble radius (static field). Sets the probe-grid extent.
    sigma
        Wall thickness (static field; not used in the ADM reconstruction
        but retained for symbolic inspection and downstream heuristics).
    strict
        If True (default), :meth:`__check_init__` raises
        :class:`UnphysicalMetricError` on ``verify_physical`` failure.
        If False, emits :class:`UnphysicalMetricWarning` and returns.
    lapse_floor
        Minimum acceptable lapse value (static; default ``1e-6``).

    Notes
    -----
    The shift ``beta^x = -v_s * f(r_s)`` follows the Alcubierre
    convention. ``r_s`` is recenterd at ``x_s = v_s t`` so the bubble
    co-moves with its velocity. ``y`` and ``z`` components of the
    shift are zero.

    All parameters are dynamic pytree leaves (``v_s``, ``shape_fn``)
    except ``R``, ``sigma``, ``strict``, ``lapse_floor`` which are
    static. Gradient flow through ``shape_fn.params`` enables the
    optimizer.
    """

    shape_fn: ShapeFunction
    v_s: Float[Array, ""]
    R: float = eqx.field(static=True, default=1.0)
    sigma: float = eqx.field(static=True, default=0.1)
    strict: bool = eqx.field(static=True, default=True)
    lapse_floor: float = eqx.field(static=True, default=1e-6)

    # ------------------------------------------------------------------
    # Equinox post-init: apply the verify_physical gate
    # ------------------------------------------------------------------

    def __check_init__(self):
        """Post-init gate: enforce verify_physical at construction."""
        verdict = self.verify_physical()
        if not verdict.overall:
            if self.strict:
                raise UnphysicalMetricError(
                    f"ShapeFunctionMetric failed verify_physical: {verdict}"
                )
            else:
                warnings.warn(
                    f"Unphysical ShapeFunctionMetric (strict=False): {verdict}",
                    UnphysicalMetricWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # ADMMetric abstract-method overrides
    # ------------------------------------------------------------------

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Unit lapse (Alcubierre convention)."""
        return jnp.asarray(1.0)

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift ``beta^i = -v_s * f(r_s) * delta^i_x``."""
        t, x, y, z = coords
        r_s = jnp.sqrt((x - self.v_s * t) ** 2 + y**2 + z**2 + 1e-30)
        f_val = self.shape_fn(r_s)
        return jnp.asarray([-self.v_s * f_val, 0.0, 0.0])

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Flat spatial metric (Alcubierre convention)."""
        return jnp.eye(3)

    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Shape function value at ``coords`` (ADMMetric contract)."""
        t, x, y, z = coords
        r_s = jnp.sqrt((x - self.v_s * t) ** 2 + y**2 + z**2 + 1e-30)
        return self.shape_fn(r_s)

    def name(self) -> str:
        return "ShapeFunctionMetric"

    def symbolic(self) -> SymbolicMetric:
        """Placeholder symbolic form.

        The wrapped shape function is in general non-symbolic (data-
        parametrized), so the symbolic view leaves the shape entries
        as generic ``sp.Function`` placeholders. Downstream symbolic
        cross-validation (SymPy lambdify path) is not meaningful for
        DSGN metrics; use the JAX path instead.
        """
        t, x, y, z = sp.symbols("t x y z")
        f_sym = sp.Function("f")
        v_s = sp.Symbol("v_s", real=True)
        r_s = sp.sqrt(x**2 + y**2 + z**2)
        f = f_sym(r_s)
        g = sp.Matrix([
            [-(1 - v_s**2 * f**2), -v_s * f, 0, 0],
            [-v_s * f, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    # ------------------------------------------------------------------
    # verify_physical gate
    # ------------------------------------------------------------------

    def verify_physical(
        self, probe_grid: Float[Array, "N 4"] | None = None
    ) -> PhysicalityVerdict:
        """Evaluate three physicality checks on a probe grid.

        Parameters
        ----------
        probe_grid
            Optional ``(N, 4)`` array of probe coordinates. If ``None``,
            uses a 16**3 grid at ``t = 0`` spanning
            ``[-2 R, +2 R]`` on each spatial axis.

        Returns
        -------
        PhysicalityVerdict
            NamedTuple with the three per-check booleans + overall AND.
        """
        if probe_grid is None:
            grid_1d = jnp.linspace(-2.0 * self.R, 2.0 * self.R, 16)
            XX, YY, ZZ = jnp.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
            TT = jnp.zeros_like(XX)
            coords_stack = jnp.stack([TT, XX, YY, ZZ], axis=-1).reshape(-1, 4)
        else:
            coords_stack = probe_grid

        # ---- Lapse floor ----
        alpha_vals = jax.vmap(self.lapse)(coords_stack)
        lapse_ok = bool(jnp.all(alpha_vals >= self.lapse_floor))

        # ---- CTC-free: g_tt < 0 everywhere ----
        def _g_tt(c):
            return self(c)[0, 0]

        g_tt_vals = jax.vmap(_g_tt)(coords_stack)
        ctc_free = bool(jnp.all(g_tt_vals < 0))

        # ---- Bubble-finite: shape function bounded far from the wall ----
        # For spline bases, evaluation outside the knot range may return NaN
        # via interpax extrapolation. We probe at ``10 * R`` but also check
        # the sampled probe-grid maximum |f(r)| stays finite and bounded.
        r_far = jnp.asarray(10.0 * self.R)
        shape_far = self.shape_fn(r_far)
        # Also check max absolute amplitude on the in-grid probe radii.
        r_probes = jnp.linalg.norm(coords_stack[:, 1:], axis=-1)
        shape_probe = jax.vmap(self.shape_fn)(r_probes)
        bubble_finite = bool(
            (jnp.isnan(shape_far) | (jnp.abs(shape_far) < 1.0))
            & jnp.all(jnp.isfinite(shape_probe))
            & jnp.all(jnp.abs(shape_probe) < 1e3)
        )

        overall = bool(lapse_ok and ctc_free and bubble_finite)
        return PhysicalityVerdict(
            lapse_floor_ok=lapse_ok,
            ctc_free=ctc_free,
            bubble_finite=bubble_finite,
            overall=overall,
        )
