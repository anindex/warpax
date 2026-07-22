"""Continuous extremum polishing via nested local grid refinement.

The stress-energy tensor here is autodiff-exact pointwise (``jax.jacfwd``), so a
grid-sampled extremum (the minimum NEC/DEC margin, or the maximum Type-IV
imaginary eigenvalue ``max|Im lambda|``) is only a *sampling* estimate. It
undershoots the true continuous extremum of a smooth field by an
``O(Delta x^2)`` grid-alignment gap that does NOT admit a clean Richardson order
(the gap oscillates with grid alignment; aliasing). Richardson-extrapolating a
grid-sampled min/max is therefore unsound.

This module removes the gap instead of extrapolating it. Starting from the
coarse-grid ``argmin``/``argmax`` seed, it evaluates the exact tensor on a small
local grid, recentres on the local extremum, shrinks the window, and repeats.
Because every evaluation is exact, the limit is the continuum extremum of the
seeded basin to the requested tolerance, a value independent of the ladder
resolution, and (when the caller seeds from the most extreme sample and clamps to
it) guaranteed at least as extreme as every grid sample. It is a high-accuracy
local refinement, not a global-optimality proof; grid spacing enters only as a
search control, never as a discretization error.

The field extractor is caller-supplied (e.g. wrapping
``certify_grid_frame_free``) so this stays generic over the diagnostic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.types import GridSpec

__all__ = ["refine_extremum"]


def refine_extremum(
    metric,
    seed_xyz,
    field_fn: Callable,
    *,
    mode: str = "min",
    half_width: float = 0.15,
    n: int = 9,
    levels: int = 7,
    shrink: float = 0.34,
    tol: float = 1e-9,
    batch_size: int = 512,
    t: float = 0.0,
) -> dict:
    """Polish a grid-seeded spatial extremum to its exact continuous value.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive metric (its curvature chain is autodiff-exact).
    seed_xyz : (3,) array-like
        Coarse-grid ``argmin``/``argmax`` location (the basin seed).
    field_fn : callable
        Maps a ``CurvatureResult`` (over a local grid) to a real ``(n, n, n)``
        scalar field. Fill invalid cells with ``+inf`` (min mode) or ``-inf``
        (max mode) so they are never selected.
    mode : {"min", "max"}
        Whether the target is a minimum or a maximum of the field.
    half_width : float
        Initial local half-window per axis (should exceed the coarse spacing so
        the true extremum lies inside the first window).
    n : int
        Local grid points per axis.
    levels, shrink, tol : int, float, float
        Zoom depth, per-level window shrink factor, and early-stop tolerance on
        the value change between levels.
    batch_size, t : int, float
        Curvature-evaluation batch size and slice time.

    Returns
    -------
    dict
        ``value`` (exact extremum), ``coord`` ((3,) location), ``levels_used``,
        ``last_delta`` (final value change), and ``history``.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    center = np.asarray(seed_xyz, dtype=float).copy()
    hw = float(half_width)
    best_val = np.inf if mode == "min" else -np.inf
    best_xyz = center.copy()
    history = []
    last_delta = np.inf
    for lvl in range(levels):
        bounds = [(float(center[i] - hw), float(center[i] + hw)) for i in range(3)]
        grid = GridSpec(bounds=bounds, shape=(n, n, n))
        curv = evaluate_curvature_grid(
            metric, grid, batch_size=batch_size, t=t, compute_invariants=False
        )
        field = np.asarray(field_fn(curv)).reshape(n, n, n)
        flat = field.ravel()
        k = int(np.argmin(flat)) if mode == "min" else int(np.argmax(flat))
        val = float(flat[k])
        i0, i1, i2 = np.unravel_index(k, field.shape)
        axes = [np.linspace(bounds[a][0], bounds[a][1], n) for a in range(3)]
        xyz = np.array([axes[0][i0], axes[1][i1], axes[2][i2]])
        improved = (val < best_val) if mode == "min" else (val > best_val)
        if improved or lvl == 0:
            last_delta = abs(val - best_val) if np.isfinite(best_val) else np.inf
            best_val, best_xyz = val, xyz
        history.append({"level": lvl, "value": val, "coord": xyz.tolist(), "hw": hw})
        center = best_xyz.copy()
        hw *= shrink
        if np.isfinite(last_delta) and last_delta < tol and lvl >= 2:
            break
    # on-wall diagnostic: shape function value at the extremum (band [0.05,0.95])
    try:
        import jax.numpy as jnp
        f_here = float(metric.shape_function_value(
            jnp.array([t, best_xyz[0], best_xyz[1], best_xyz[2]])))
    except (NotImplementedError, Exception):  # noqa: BLE001
        f_here = None
    return {
        "value": best_val,
        "coord": best_xyz.tolist(),
        "levels_used": len(history),
        "last_delta": float(last_delta) if np.isfinite(last_delta) else None,
        "shape_at_extremum": f_here,
        "history": history,
    }


def _demo():
    """Self-check: polishing raises the sampled max|Im lambda| and lands on the
    Alcubierre wall (shape function strictly interior to [0, 1])."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from warpax.benchmarks import AlcubierreMetric
    from warpax.grids import wall_clustered
    from warpax.energy_conditions import certify_grid_frame_free
    from warpax.geometry import evaluate_curvature_grid as _ecg

    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3.0, 3.0)] * 3, (60, 60, 60), a=2.0)
    curv = _ecg(metric, grid, batch_size=4096)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    imag = np.max(np.abs(np.asarray(ff.eigenvalues_imag)), axis=-1)
    k = int(np.argmax(imag.ravel()))
    axes = [np.asarray(grid.axes[a]) for a in range(3)]
    i0, i1, i2 = np.unravel_index(k, imag.shape)
    seed = [axes[0][i0], axes[1][i1], axes[2][i2]]
    coarse = float(imag.ravel()[k])

    def field_max_imag(c):
        f = certify_grid_frame_free(c.stress_energy, c.metric, c.metric_inv)
        return np.max(np.abs(np.asarray(f.eigenvalues_imag)), axis=-1)

    res = refine_extremum(metric, seed, field_max_imag, mode="max",
                          half_width=0.15, n=9, levels=7)
    print(f"coarse max|Im| = {coarse:.6f}  seed={seed}")
    print(f"polished       = {res['value']:.6f}  coord={res['coord']}")
    print(f"shape@extremum = {res['shape_at_extremum']}  levels={res['levels_used']}")
    assert res["value"] >= coarse - 1e-9, "polished max must not fall below sampled max"
    assert res["shape_at_extremum"] is not None and 0.02 < res["shape_at_extremum"] < 0.995, \
        "Type-IV imaginary-eigenvalue peak must sit on the wall"
    print("OK: continuous extremum polishing certified.")


if __name__ == "__main__":
    _demo()
