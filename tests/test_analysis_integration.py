"""End-to-end integration tests for the analysis pipeline.

Tests the full pipeline (curvature grid -> EC comparison -> results) on
real warp metrics at small grid scales (8^3 or 10^3) to verify correctness
without excessive runtime.

These tests catch real physics bugs that unit tests with synthetic data
would miss: wrong sign conventions, comparison logic errors, incorrect
optimizer behavior, etc.
"""
from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric, SchwarzschildMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import (
    ComparisonResult,
    compare_eulerian_vs_robust,
    compute_convergence_quantity,
    compute_kinematic_scalars_grid,
    richardson_extrapolation,
)
from warpax.energy_conditions.verifier import verify_grid
from warpax.metrics import NatarioMetric


# ---------------------------------------------------------------------------
# Core pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_alcubierre_comparison_pipeline():
    """Full Eulerian vs robust pipeline on Alcubierre at 10^3 grid.

    Validates:
    - ComparisonResult has all expected fields
    - Margins dicts have 4 condition keys
    - Missed arrays are boolean with correct shape
    - Robust margins <= Eulerian margins + epsilon (key invariant)
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(10, 10, 10))

    # Step 1: Curvature grid
    curv = evaluate_curvature_grid(metric, grid, batch_size=100)
    assert curv.stress_energy.shape == (10, 10, 10, 4, 4)
    assert curv.metric.shape == (10, 10, 10, 4, 4)
    assert curv.metric_inv.shape == (10, 10, 10, 4, 4)

    # Step 2: Comparison
    result = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid.shape,
        n_starts=4,
        batch_size=50,
    )

    # Validate ComparisonResult structure
    assert isinstance(result, ComparisonResult)
    conditions = {"nec", "wec", "sec", "dec"}
    assert set(result.eulerian_margins.keys()) == conditions
    assert set(result.robust_margins.keys()) == conditions
    assert set(result.missed.keys()) == conditions
    assert set(result.severity.keys()) == conditions
    assert set(result.pct_missed.keys()) == conditions
    assert set(result.pct_violated_robust.keys()) == conditions

    # Validate shapes
    for cond in conditions:
        assert result.eulerian_margins[cond].shape == (10, 10, 10)
        assert result.robust_margins[cond].shape == (10, 10, 10)
        assert result.missed[cond].shape == (10, 10, 10)

    # Key physics invariant: robust margins <= Eulerian margins + epsilon
    # The optimizer searches over ALL observers, so it can only find
    # worse (more negative) margins than any specific observer.
    #
    # For WEC/SEC/DEC: Eulerian IS one specific timelike observer in the
    # search space, so robust <= Eulerian should hold tightly.
    #
    # For NEC: Eulerian check uses 6 discrete null directions while the
    # optimizer uses a continuous parameterization of the null cone.
    # The optimizer may find slightly more negative margins at points
    # where the 6-direction check happened to return ~ 0.
    # This is physically correct (the optimizer found a worse null vector).
    #
    # Use per-condition tolerances: tighter for WEC/SEC/DEC, relaxed for NEC.
    tol = {"nec": 1e-4, "wec": 1e-6, "sec": 1e-6, "dec": 1e-6}
    for cond in conditions:
        eul = np.asarray(result.eulerian_margins[cond])
        rob = np.asarray(result.robust_margins[cond])
        # At each point, robust should be <= Eulerian + tolerance
        valid = np.isfinite(eul) & np.isfinite(rob)
        if np.any(valid):
            assert np.all(rob[valid] <= eul[valid] + tol[cond]), (
                f"{cond}: robust margins exceed Eulerian by more than tolerance. "
                f"Max excess: {np.max(rob[valid] - eul[valid]):.2e}"
            )

    # Percentages should be in [0, 100]
    for cond in conditions:
        assert 0.0 <= result.pct_missed[cond] <= 100.0
        assert 0.0 <= result.pct_violated_robust[cond] <= 100.0


# ---------------------------------------------------------------------------
# Worst-observer caching test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_worst_observer_data_cached():
    """Verify worst_params and worst_observers are present in ECGridResult.

    Validates:
    - worst_params shape is (*grid_shape, 3) for (zeta, theta, phi)
    - worst_observers shape is (*grid_shape, 4) for 4-velocity
    - worst_params dtype is float64
    - worst_params can be serialized to .npz and reloaded
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(8, 8, 8))

    # Compute curvature grid
    curv = evaluate_curvature_grid(metric, grid, batch_size=64)

    # Run verify_grid to get ECGridResult with worst-observer data
    ec_grid = verify_grid(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        n_starts=4,
        batch_size=32,
        compute_eulerian=False,
    )

    # Check worst_params shape: (*grid_shape, 3) = (8, 8, 8, 3)
    assert ec_grid.worst_params.shape == (8, 8, 8, 3), (
        f"worst_params shape: {ec_grid.worst_params.shape}, expected (8, 8, 8, 3)"
    )

    # Check worst_observers shape: (*grid_shape, 4) = (8, 8, 8, 4)
    assert ec_grid.worst_observers.shape == (8, 8, 8, 4), (
        f"worst_observers shape: {ec_grid.worst_observers.shape}, expected (8, 8, 8, 4)"
    )

    # Check dtype is float64
    assert ec_grid.worst_params.dtype == jnp.float64, (
        f"worst_params dtype: {ec_grid.worst_params.dtype}, expected float64"
    )

    # Verify .npz round-trip
    wp_np = np.asarray(ec_grid.worst_params)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez(tmp_path, worst_params=wp_np)
        loaded = np.load(tmp_path)
        assert loaded["worst_params"].shape == (8, 8, 8, 3)
        np.testing.assert_array_equal(loaded["worst_params"], wp_np)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Schwarzschild baseline test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_schwarzschild_no_missed_violations():
    """Schwarzschild vacuum: 0% missed violations.

    Schwarzschild is vacuum (T_ab = 0), so all margins should be ~0 or NaN
    near singularity. The comparison logic should never produce false positives.
    We place the grid far from the singularity (r in [5, 15]).
    """
    metric = SchwarzschildMetric(M=1.0)
    grid = GridSpec(bounds=[(5, 15)] * 3, shape=(8, 8, 8))

    curv = evaluate_curvature_grid(metric, grid, batch_size=64)

    result = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid.shape,
        n_starts=4,
        batch_size=32,
    )

    # For vacuum spacetime: no missed violations
    for cond in ("nec", "wec"):
        pct = result.pct_missed[cond]
        assert pct == 0.0, (
            f"Schwarzschild {cond.upper()} pct_missed = {pct:.4f}%, expected 0.0%"
        )


# ---------------------------------------------------------------------------
# Kinematic scalar physics tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_natario_zero_expansion():
    """Natario metric has zero expansion by construction (div(beta) = 0).

    Also verifies vorticity is zero for Eulerian observers.
    """
    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(8, 8, 8))

    theta, sigma_sq, omega_sq = compute_kinematic_scalars_grid(
        metric, grid, batch_size=64,
    )

    # Natario: theta = 0 everywhere (zero-expansion by construction)
    theta_np = np.asarray(theta)
    assert np.allclose(theta_np, 0.0, atol=1e-6), (
        f"Natario theta not zero: min={np.min(theta_np):.2e}, max={np.max(theta_np):.2e}"
    )

    # Vorticity: always zero for Eulerian observers (Frobenius theorem)
    omega_np = np.asarray(omega_sq)
    assert np.allclose(omega_np, 0.0, atol=1e-8), (
        f"Natario omega_sq not zero: max={np.max(np.abs(omega_np)):.2e}"
    )


@pytest.mark.slow
def test_alcubierre_nonzero_expansion():
    """Alcubierre metric has nonzero expansion at the bubble wall.

    Also validates omega_sq = 0 and sigma_sq >= 0 everywhere.
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(8, 8, 8))

    theta, sigma_sq, omega_sq = compute_kinematic_scalars_grid(
        metric, grid, batch_size=64,
    )

    # Alcubierre: theta NOT all zero (has expansion at bubble wall)
    theta_np = np.asarray(theta)
    assert not np.allclose(theta_np, 0.0, atol=1e-6), (
        "Alcubierre theta unexpectedly zero everywhere"
    )

    # Vorticity: always zero for Eulerian observers
    omega_np = np.asarray(omega_sq)
    assert np.allclose(omega_np, 0.0, atol=1e-8), (
        f"Alcubierre omega_sq not zero: max={np.max(np.abs(omega_np)):.2e}"
    )

    # Shear-squared: positive semidefinite everywhere
    sigma_np = np.asarray(sigma_sq)
    assert np.all(sigma_np >= -1e-10), (
        f"Alcubierre sigma_sq has negative values: min={np.min(sigma_np):.2e}"
    )


# ---------------------------------------------------------------------------
# Richardson extrapolation test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_richardson_with_real_data():
    """Richardson extrapolation on Alcubierre Eulerian NEC at 3 resolutions.

    Uses small grids (8^3, 12^3, 16^3) for speed. Validates:
    - Observed order > 0 (convergence, not divergence)
    - Error estimate is bounded (extrapolation is a refinement)
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    resolutions = [8, 12, 16]
    values = []

    for N in resolutions:
        grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(N, N, N))
        curv = evaluate_curvature_grid(metric, grid, batch_size=64)

        # Compute Eulerian NEC margin via comparison
        result = compare_eulerian_vs_robust(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            grid.shape,
            n_starts=2,
            batch_size=32,
        )

        nec_margin = np.asarray(result.eulerian_margins["nec"])
        val = compute_convergence_quantity(nec_margin, "min_margin")
        values.append(val)

    # Richardson extrapolation
    rich = richardson_extrapolation(values, resolutions)

    # Convergence: observed order should be > 0
    assert rich["observed_order"] > 0, (
        f"Richardson order = {rich['observed_order']:.2f}, expected > 0"
    )

    # Error estimate should be finite
    assert np.isfinite(rich["error_estimate"]), (
        f"Richardson error estimate not finite: {rich['error_estimate']}"
    )


# ---------------------------------------------------------------------------
# Dtype test
# ---------------------------------------------------------------------------


def test_analysis_float64():
    """All analysis output arrays should be float64 dtype."""
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=[(-3, 3)] * 3, shape=(5, 5, 5))

    curv = evaluate_curvature_grid(metric, grid, batch_size=25)

    result = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid.shape,
        n_starts=2,
        batch_size=25,
    )

    for cond in ("nec", "wec", "sec", "dec"):
        eul = result.eulerian_margins[cond]
        rob = result.robust_margins[cond]
        assert eul.dtype == jnp.float64, (
            f"Eulerian {cond} dtype: {eul.dtype}, expected float64"
        )
        assert rob.dtype == jnp.float64, (
            f"Robust {cond} dtype: {rob.dtype}, expected float64"
        )
