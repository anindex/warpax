"""Unit tests for the analysis library module.

Tests three submodules:
- comparison: ComparisonResult construction, missed-flag logic, table builder
- convergence: Richardson extrapolation with known analytical convergence
- kinematic_scalars: Minkowski (zero), Schwarzschild (zero), Alcubierre (nonzero)
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.analysis import (
    ComparisonResult,
    build_comparison_table,
    compute_kinematic_scalars,
    compute_kinematic_scalars_grid,
    richardson_extrapolation,
)
from warpax.analysis.convergence import compute_convergence_quantity
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.geometry.types import GridSpec


# ===========================================================================
# Comparison tests
# ===========================================================================


class TestComparisonResult:
    """Tests for ComparisonResult construction and field access."""

    def test_construction_and_field_access(self):
        """ComparisonResult can be constructed with all fields."""
        eul = {"nec": jnp.array([0.1, -0.2]), "wec": jnp.array([0.3, 0.0])}
        rob = {"nec": jnp.array([-0.1, -0.3]), "wec": jnp.array([0.1, -0.1])}
        missed = {"nec": jnp.array([True, False]), "wec": jnp.array([False, False])}
        severity = {"nec": jnp.array([0.2, 0.0]), "wec": jnp.array([0.0, 0.0])}
        pct_m = {"nec": 50.0, "wec": 0.0}
        pct_v = {"nec": 100.0, "wec": 50.0}

        result = ComparisonResult(
            eulerian_margins=eul,
            robust_margins=rob,
            missed=missed,
            severity=severity,
            pct_missed=pct_m,
            pct_violated_robust=pct_v,
            conditional_miss_rate={"nec": 50.0, "wec": 0.0},
            classification_stats={
                "n_type_i": 2, "n_type_ii": 0, "n_type_iii": 0,
                "n_type_iv": 0, "max_imag_eigenvalue": 0.0,
            },
            opt_margins={"nec": rob["nec"], "wec": rob["wec"]},
            he_types=jnp.array([1.0, 1.0]),
        )

        assert result.pct_missed["nec"] == 50.0
        assert result.pct_violated_robust["wec"] == 50.0
        assert result.eulerian_margins["nec"].shape == (2,)

    def test_missed_flag_logic(self):
        """Missed flag: (eul >= 0) & (rob < -1e-10) produces correct mask."""
        # Synthetic margins
        eul_nec = jnp.array([0.1, -0.2, 0.0, 0.5, 0.3])
        rob_nec = jnp.array([-0.5, -0.3, -0.5, 0.1, -1e-11])

        missed = (eul_nec >= 0.0) & (rob_nec < -1e-10)

        # Point 0: eul=0.1>=0, rob=-0.5<-1e-10 -> MISSED (True)
        # Point 1: eul=-0.2<0 -> NOT missed (False)
        # Point 2: eul=0.0>=0, rob=-0.5<-1e-10 -> MISSED (True)
        # Point 3: eul=0.5>=0, rob=0.1>=0 -> NOT missed (False)
        # Point 4: eul=0.3>=0, rob=-1e-11 > -1e-10 -> NOT missed (False)
        expected = jnp.array([True, False, True, False, False])

        np.testing.assert_array_equal(np.asarray(missed), np.asarray(expected))

    def test_severity_ratio(self):
        """Severity ratio is eul_margin - rob_margin at missed points, 0 otherwise."""
        eul = jnp.array([0.1, -0.2, 0.0])
        rob = jnp.array([-0.5, -0.3, -0.5])
        missed = (eul >= 0.0) & (rob < -1e-10)
        severity = jnp.where(missed, eul - rob, 0.0)

        # Point 0: missed, severity = 0.1 - (-0.5) = 0.6
        # Point 1: not missed, severity = 0.0
        # Point 2: missed, severity = 0.0 - (-0.5) = 0.5
        np.testing.assert_allclose(np.asarray(severity), [0.6, 0.0, 0.5], atol=1e-14)

    def test_build_comparison_table(self, tmp_path):
        """build_comparison_table loads .npz files and produces correct JSON."""
        results_dir = str(tmp_path)

        # Create synthetic .npz files
        for v_s in [0.1, 0.5]:
            eul_nec = np.array([0.1, -0.2, 0.0, 0.5])
            rob_nec = np.array([-0.5, -0.3, -0.5, 0.1])
            eul_wec = np.array([0.2, 0.1, -0.1, 0.3])
            rob_wec = np.array([-0.1, -0.2, -0.3, 0.2])
            eul_sec = np.array([0.3, 0.2, 0.1, 0.4])
            rob_sec = np.array([0.1, 0.0, -0.1, 0.3])
            eul_dec = np.array([0.4, 0.3, 0.2, 0.5])
            rob_dec = np.array([0.2, 0.1, 0.0, 0.4])

            np.savez(
                os.path.join(results_dir, f"testmetric_vs{v_s}.npz"),
                nec_eulerian=eul_nec,
                nec_robust=rob_nec,
                wec_eulerian=eul_wec,
                wec_robust=rob_wec,
                sec_eulerian=eul_sec,
                sec_robust=rob_sec,
                dec_eulerian=eul_dec,
                dec_robust=rob_dec,
            )

        rows = build_comparison_table(results_dir, ["testmetric"], [0.1, 0.5])

        assert len(rows) == 2
        assert rows[0]["metric"] == "testmetric"
        assert rows[0]["v_s"] == 0.1

        # Check NEC fields
        assert "nec_eulerian_min" in rows[0]
        assert "nec_robust_min" in rows[0]
        assert "nec_pct_violated_robust" in rows[0]
        assert "nec_pct_missed" in rows[0]

        # Verify values for NEC
        assert rows[0]["nec_eulerian_min"] == pytest.approx(-0.2)
        assert rows[0]["nec_robust_min"] == pytest.approx(-0.5)

        # Verify JSON file written
        json_path = os.path.join(results_dir, "comparison_table.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2


# ===========================================================================
# Convergence tests
# ===========================================================================


class TestRichardsonExtrapolation:
    """Tests for Richardson extrapolation with known analytical cases."""

    def test_quadratic_convergence(self):
        """Recover known Q_exact from Q(h) = Q_exact + C*h^2."""
        Q_exact = 3.14159
        C = 2.0

        # Q(h) = Q_exact + C * h^2, where h = 1/N
        grid_sizes = [25, 50, 100]
        values = [Q_exact + C * (1.0 / N) ** 2 for N in grid_sizes]

        result = richardson_extrapolation(values, grid_sizes, expected_order=2)

        assert result["observed_order"] == pytest.approx(2.0, abs=0.05)
        assert result["extrapolated_value"] == pytest.approx(Q_exact, rel=0.01)
        assert result["converged"] is True

    def test_first_order_convergence(self):
        """Recover p=1 from Q(h) = Q_exact + C*h."""
        Q_exact = 1.0
        C = 5.0

        grid_sizes = [25, 50, 100]
        values = [Q_exact + C * (1.0 / N) for N in grid_sizes]

        result = richardson_extrapolation(values, grid_sizes, expected_order=1)

        assert result["observed_order"] == pytest.approx(1.0, abs=0.05)
        assert result["extrapolated_value"] == pytest.approx(Q_exact, rel=0.01)
        assert result["converged"] is True

    def test_converged_flag_false(self):
        """converged=False when observed order far from expected."""
        Q_exact = 2.0
        C = 1.0

        # First-order convergence Q(h) = Q_exact + C*h
        grid_sizes = [25, 50, 100]
        values = [Q_exact + C * (1.0 / N) for N in grid_sizes]

        # Expected order 4 but actual is 1 |1 - 4| = 3 > 1.0
        result = richardson_extrapolation(values, grid_sizes, expected_order=4)

        assert result["converged"] is False

    def test_too_few_resolutions_raises(self):
        """Raise ValueError with fewer than 3 resolutions."""
        with pytest.raises(ValueError, match="at least 3"):
            richardson_extrapolation([1.0, 2.0], [25, 50])


class TestConvergenceQuantity:
    """Tests for compute_convergence_quantity."""

    def test_min_margin(self):
        """min_margin returns nanmin of margins."""
        margins = np.array([0.1, -0.5, 0.3, np.nan, -0.2])
        val = compute_convergence_quantity(margins, "min_margin")
        assert val == pytest.approx(-0.5)

    def test_l2_violation(self):
        """l2_violation returns L2 norm of negative margins."""
        margins = np.array([0.1, -0.3, 0.5, -0.4])
        # Violated: -0.3, -0.4
        expected = float(np.sqrt(0.3**2 + 0.4**2))
        val = compute_convergence_quantity(margins, "l2_violation")
        assert val == pytest.approx(expected, rel=1e-10)

    def test_integrated_violation(self):
        """integrated_violation returns sum of |margin| * cell_volume."""
        margins = np.array([0.1, -0.3, 0.5, -0.4])
        cell_volume = 0.5
        expected = (0.3 + 0.4) * 0.5
        val = compute_convergence_quantity(margins, "integrated_violation", cell_volume=cell_volume)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_no_violations(self):
        """l2_violation and integrated_violation return 0 when no violations."""
        margins = np.array([0.1, 0.2, 0.3])
        assert compute_convergence_quantity(margins, "l2_violation") == 0.0
        assert compute_convergence_quantity(margins, "integrated_violation") == 0.0

    def test_unknown_quantity_raises(self):
        """Raise ValueError for unknown quantity name."""
        with pytest.raises(ValueError, match="Unknown convergence quantity"):
            compute_convergence_quantity(np.array([1.0]), "invalid")


# ===========================================================================
# Kinematic scalar tests
# ===========================================================================


class TestKinematicScalars:
    """Tests for kinematic scalar computation."""

    def test_minkowski_zero(self):
        """Minkowski metric: theta=0, sigma_sq=0, omega_sq=0."""
        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])

        theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric, coords)

        assert theta.dtype == jnp.float64
        assert sigma_sq.dtype == jnp.float64
        np.testing.assert_allclose(float(theta), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(sigma_sq), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(omega_sq), 0.0, atol=1e-15)

    def test_schwarzschild_static_zero(self):
        """Schwarzschild (static): K_ij=0 at t=0, so theta=0, sigma_sq=0."""
        metric = SchwarzschildMetric(M=1.0)
        # Far from origin to avoid coordinate issues
        coords = jnp.array([0.0, 10.0, 0.0, 0.0])

        theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric, coords)

        assert theta.dtype == jnp.float64
        assert sigma_sq.dtype == jnp.float64
        np.testing.assert_allclose(float(theta), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(sigma_sq), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(omega_sq), 0.0, atol=1e-15)

    def test_alcubierre_nonzero_expansion(self):
        """Alcubierre: theta is nonzero at the bubble wall.

        For the Alcubierre metric, theta = -v_s * df/dx at the bubble wall.
        At a point near the bubble edge, the expansion should be nonzero.
        """
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        # Point near the bubble wall (r_s ~ R = 1.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])

        theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric, coords)

        assert theta.dtype == jnp.float64
        assert sigma_sq.dtype == jnp.float64

        # Expansion should be nonzero at the bubble wall
        assert abs(float(theta)) > 1e-3, (
            f"Expected nonzero expansion at bubble wall, got theta={float(theta)}"
        )

        # omega_sq = 0 always (Eulerian observers)
        np.testing.assert_allclose(float(omega_sq), 0.0, atol=1e-15)

    def test_grid_minkowski(self):
        """Grid version on Minkowski: all theta~0, sigma_sq~0, omega_sq=0."""
        metric = MinkowskiMetric()
        grid = GridSpec(
            bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
            shape=(5, 5, 5),
        )

        theta_grid, sigma_sq_grid, omega_sq_grid = compute_kinematic_scalars_grid(
            metric, grid, t=0.0
        )

        assert theta_grid.shape == (5, 5, 5)
        assert sigma_sq_grid.shape == (5, 5, 5)
        assert omega_sq_grid.shape == (5, 5, 5)

        assert theta_grid.dtype == jnp.float64

        np.testing.assert_allclose(
            np.asarray(theta_grid), 0.0, atol=1e-12,
            err_msg="Minkowski expansion should be zero everywhere",
        )
        np.testing.assert_allclose(
            np.asarray(sigma_sq_grid), 0.0, atol=1e-12,
            err_msg="Minkowski shear should be zero everywhere",
        )
        np.testing.assert_allclose(
            np.asarray(omega_sq_grid), 0.0, atol=1e-15,
            err_msg="Eulerian vorticity must be zero",
        )
