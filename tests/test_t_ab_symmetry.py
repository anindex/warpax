"""Contract test: T_ab symmetry.

Pins the invariant `max|T_ab - T_ab^T| / max|T_ab| < 1e-10` at every
grid point on every warp metric. Includes a direct pin at WarpShell
v_s=0.5 idx=8, a known worst-case point for reduction-order noise.

This is a separate file from the existing TestStressEnergySymmetry suite
(in test_curvature_chain.py, 10x10x10 interior slabs @ 1e-13 tolerance)
because scope is the full 50x50x50 WarpShell grid where XLA
reduction-order noise accumulates over more grid points.

Note: the pipeline T_ab (= G / 8π as emitted by `stress_energy_tensor`)
is symmetric to machine precision on the full WarpShell v_s=0.5 50^3
grid (max rel asymmetry ~1.34e-11). The post-compute symmetrization
`T = 0.5 * (T + T.T)` is defensive hardening that bounds reduction-order
drift at machine precision.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.geometry import GridSpec, compute_curvature_chain, evaluate_curvature_grid
from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)


REL_TOL = 1e-10


def _max_rel_asymmetry(T_grid: np.ndarray) -> float:
    """Compute max_i |T_i - T_i.T| / max|T_i| across a grid of (N, 4, 4) tensors.

    NaN-safe: ignores any (4, 4) slice with non-finite entries (e.g., WarpShell
    evaluated at r=0 may produce 1/0 singularities outside the shell geometry).
    """
    finite_mask = np.all(np.isfinite(T_grid), axis=(-1, -2))
    if finite_mask.sum() == 0:
        raise ValueError("all T_ab slices are non-finite; grid likely degenerate")
    T_finite = T_grid[finite_mask]
    asym = np.max(np.abs(T_finite - np.swapaxes(T_finite, -1, -2)), axis=(-1, -2))
    scale = np.max(np.abs(T_finite), axis=(-1, -2))
    # Avoid div-by-zero at Minkowski-like points
    rel = asym / np.maximum(scale, np.finfo(T_finite.dtype).tiny)
    return float(np.max(rel))


class TestTabSymmetryFullGrid:
    """`max|T_ab - T_ab^T| / max|T_ab| < 1e-10` on full production grids."""

    @pytest.mark.slow
    def test_warpshell_idx_8_post_fix(self):
        """Direct pin at the WarpShell idx=8 worst-case point.

        """
        metric = WarpShellMetric(v_s=0.5)  # defaults R_1=10, R_2=20, r_s_param=5
        grid = GridSpec(
            bounds=((-12.0, 12.0), (-6.0, 6.0), (-6.0, 6.0)),
            shape=(50, 50, 50),
        )
        result = evaluate_curvature_grid(metric, grid, batch_size=256)
        T_grid = np.asarray(result.stress_energy).reshape(-1, 4, 4)
        # Specific pin at idx=8
        T_8 = T_grid[8]
        asym_8 = np.max(np.abs(T_8 - T_8.T))
        scale_8 = np.max(np.abs(T_8))
        rel_8 = asym_8 / max(scale_8, np.finfo(T_8.dtype).tiny)
        assert rel_8 < REL_TOL, (
            f"WarpShell idx=8 T_ab asymmetry regressed post-fix; "
            f"expected < {REL_TOL:.0e}, got {rel_8:.2e}. "
            f"Post-fix contract: pipeline emits symmetric T_ab at machine precision."
        )
        # And pin the whole grid (NaN-safe: WarpShell has geometric singularities outside the shell)
        rel_max = _max_rel_asymmetry(T_grid)
        assert rel_max < REL_TOL, (
            f"WarpShell v_s=0.5 full grid has a T_ab-asymmetry worst-point at "
            f"rel = {rel_max:.2e} > tol {REL_TOL:.0e}."
        )

    @pytest.mark.parametrize(
        "name,metric_factory,bounds",
        [
            ("alcubierre", lambda: AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0), (-2.0, 2.0)),
            ("rodal", lambda: RodalMetric(v_s=0.5, R=1.0, sigma=0.1), (-2.0, 2.0)),
            ("natario", lambda: NatarioMetric(v_s=0.1, R=100.0, sigma=0.03), (-150.0, 150.0)),
            ("lentz", lambda: LentzMetric(v_s=0.1, R=100.0, sigma=8.0), (-150.0, 150.0)),
            ("vdb", lambda: VanDenBroeckMetric(v_s=0.5), (-2.0, 2.0)),
        ],
    )
    def test_t_ab_symmetric_all_metrics(self, name, metric_factory, bounds):
        """contract on 5 warp metrics at production-scale grids.

        WarpShell tested separately (see test_warpshell_idx_8_post_fix; has its own
        grid bounds). Tolerance 1e-10 (looser than TestStressEnergySymmetry's 1e-13
        because full grids run up more reduction-order noise than 10x10x10 slabs).
        """
        metric = metric_factory()
        grid = GridSpec(
            bounds=(bounds, bounds, bounds),
            shape=(20, 20, 20),
        )
        result = evaluate_curvature_grid(metric, grid, batch_size=256)
        T_grid = np.asarray(result.stress_energy).reshape(-1, 4, 4)
        rel_max = _max_rel_asymmetry(T_grid)
        assert rel_max < REL_TOL, (
            f"Metric {name} T_ab asymmetry rel={rel_max:.2e} > tol {REL_TOL:.0e}. "
            f"Post-compute symmetrization should bound this at machine precision."
        )
