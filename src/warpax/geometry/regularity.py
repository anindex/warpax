"""Metric regularity diagnostics (C^k continuity checks).

Evaluates whether a spacetime metric and its first k derivatives are
continuous across specified boundaries or along radial sweeps.

Physical motivation (Barzegar-Buchert-Vigneron 2026, arXiv:2602.16495):
classical energy-condition evaluation requires the Einstein tensor to be
well-defined, which demands at least C^2 smoothness of the metric. Below
C^2, the Riemann tensor (computed via two derivatives of g) has delta-function
contributions that invalidate pointwise EC evaluation. Thick-shell metrics
must be C^2; thin-shell metrics require distributional/junction treatment.

Method:
    1. Sweep metric components along a radial line through the bubble center.
    2. Compute first and second derivatives of each component via JAX autodiff.
    3. At each level, detect discontinuities by bounding the finite-difference
       approximation to the next derivative.
    4. Report the maximum jump at each continuity level as a diagnostic.

A metric passes C^k if the (k+1)-th derivative proxy (finite differences
of the k-th derivative) is bounded. Large jumps indicate a seam where the
k-th derivative is discontinuous.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class RegularityDiagnostic(NamedTuple):
    """C^k regularity diagnostic for a single metric component sweep.

    Attributes
    ----------
    r_values : Float[Array, "N"]
        Radial values used in the sweep.
    values : Float[Array, "N"]
        Metric component values g_{ab}(r).
    first_deriv : Float[Array, "N"]
        First derivative dg/dr via autodiff.
    second_deriv : Float[Array, "N"]
        Second derivative d^2g/dr^2 via autodiff.
    c0_max_jump : float
        Maximum jump in the function values (should be ~0 for C^0).
    c1_max_jump : float
        Maximum jump in first derivative (proxy for C^1 failure).
    c2_max_jump : float
        Maximum jump in second derivative (proxy for C^2 failure).
    """

    r_values: Float[Array, "N"]
    values: Float[Array, "N"]
    first_deriv: Float[Array, "N"]
    second_deriv: Float[Array, "N"]
    c0_max_jump: float
    c1_max_jump: float
    c2_max_jump: float


class RegularityReport(NamedTuple):
    """Full regularity report for a metric across shell boundaries.

    Attributes
    ----------
    is_c0 : bool
        True if the metric is C^0 (continuous) across boundaries.
    is_c1 : bool
        True if the metric is C^1 (continuous first derivative).
    is_c2 : bool
        True if the metric is C^2 (continuous second derivative).
    components : dict[str, RegularityDiagnostic]
        Per-component diagnostics for g_{tt}, g_{tx}, g_{xx}, g_{yy}.
    """

    is_c0: bool
    is_c1: bool
    is_c2: bool
    components: dict[str, RegularityDiagnostic]


def metric_c2_diagnostic(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    r_values: Float[Array, "N"],
    component: tuple[int, int] = (0, 0),
    axis: str = "x",
) -> RegularityDiagnostic:
    """Sweep a single metric component along a radial line and diagnose C^k.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
        Spacetime metric function.
    r_values : array of shape (N,)
        Radial coordinate values to sweep.
    component : (i, j) tuple
        Metric tensor component index (0-indexed). Default: g_{tt}.
    axis : 'x', 'y', or 'z'
        Spatial axis along which to sweep.

    Returns
    -------
    RegularityDiagnostic
        Diagnostic with values, derivatives, and jump magnitudes.
    """
    a, b = component
    axis_idx = {"x": 1, "y": 2, "z": 3}[axis]

    def g_component(r: Float[Array, ""]) -> Float[Array, ""]:
        coords = jnp.zeros(4)
        coords = coords.at[axis_idx].set(r)
        return metric_fn(coords)[a, b]

    dg_dr = jax.grad(g_component)
    d2g_dr2 = jax.grad(dg_dr)

    values = jax.vmap(g_component)(r_values)
    first_deriv = jax.vmap(dg_dr)(r_values)
    second_deriv = jax.vmap(d2g_dr2)(r_values)

    dr = r_values[1] - r_values[0]

    # C0: jumps in the function itself
    diffs_0 = jnp.diff(values)
    c0_max_jump = float(jnp.max(jnp.abs(diffs_0)) / dr)

    # C1: jumps in first derivative (second differences of values)
    diffs_1 = jnp.diff(first_deriv)
    c1_max_jump = float(jnp.max(jnp.abs(diffs_1)) / dr)

    # C2: jumps in second derivative (third differences of values)
    diffs_2 = jnp.diff(second_deriv)
    c2_max_jump = float(jnp.max(jnp.abs(diffs_2)) / dr)

    return RegularityDiagnostic(
        r_values=r_values,
        values=values,
        first_deriv=first_deriv,
        second_deriv=second_deriv,
        c0_max_jump=c0_max_jump,
        c1_max_jump=c1_max_jump,
        c2_max_jump=c2_max_jump,
    )


def regularity_report(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    r_min: float = 5.0,
    r_max: float = 25.0,
    n_points: int = 200,
    axis: str = "x",
    c0_tol: float = 100.0,
    c1_tol: float = 500.0,
    c2_tol: float = 500.0,
) -> RegularityReport:
    """Full regularity report for a metric across shell boundaries.

    Sweeps the diagonal and off-diagonal metric components along a
    radial line and evaluates C^0/C^1/C^2 continuity based on
    derivative jump thresholds.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
        Spacetime metric function.
    r_min, r_max : float
        Radial range for the sweep (should span shell boundaries).
    n_points : int
        Number of sweep points.
    axis : 'x', 'y', or 'z'
        Spatial axis for the radial sweep.
    c0_tol, c1_tol, c2_tol : float
        Maximum acceptable jump per dr for C^0, C^1, C^2 classification.
        The defaults are calibrated for warp shell metrics with R_1~10,
        R_2~20 and r_s_param~5.

    Returns
    -------
    RegularityReport
        Report with per-component diagnostics and C^k pass/fail flags.
    """
    r_values = jnp.linspace(r_min, r_max, n_points)

    # Key metric components to check
    component_specs = {
        "g_tt": (0, 0),
        "g_tx": (0, 1),
        "g_xx": (1, 1),
        "g_yy": (2, 2),
    }

    components: dict[str, RegularityDiagnostic] = {}
    all_c0 = True
    all_c1 = True
    all_c2 = True

    for name, (i, j) in component_specs.items():
        diag = metric_c2_diagnostic(metric_fn, r_values, (i, j), axis)
        components[name] = diag

        if diag.c0_max_jump > c0_tol:
            all_c0 = False
        if diag.c1_max_jump > c1_tol:
            all_c1 = False
        if diag.c2_max_jump > c2_tol:
            all_c2 = False

    return RegularityReport(
        is_c0=all_c0,
        is_c1=all_c0 and all_c1,
        is_c2=all_c0 and all_c1 and all_c2,
        components=components,
    )
