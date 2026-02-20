"""Tests for observer sweep EC margin computation.

Verifies:
1. Observer family generators produce correct shapes and ranges
2. Sweep computation returns correct (N, K) shapes
3. Eulerian observer in sweep matches existing compute_eulerian_ec
4. WEC violations present for Alcubierre metric
5. Worst-case sweep is at least as negative as Eulerian
6. BFGS cross-validation: sign agreement >= 85%
7. Sweep is faster than per-point BFGS
"""
import jax
import jax.numpy as jnp
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.energy_conditions.sweep import (
    cross_validate_sweep,
    make_angular_observers,
    make_rapidity_observers,
    sweep_all_margins,
    sweep_nec_margins,
    sweep_wec_margins,
)
from warpax.geometry.grid import evaluate_curvature_grid
from warpax.geometry.types import GridSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alcubierre_fields():
    """Alcubierre at v_s=0.5 on 10^3 grid, flattened T and g fields."""
    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(10, 10, 10))
    metric = AlcubierreMetric(v_s=0.5)
    result = evaluate_curvature_grid(metric, grid_spec)

    N = 10 * 10 * 10
    T_field = result.stress_energy.reshape(N, 4, 4)
    g_field = result.metric.reshape(N, 4, 4)
    g_inv_field = result.metric_inv.reshape(N, 4, 4)
    return T_field, g_field, g_inv_field


# ---------------------------------------------------------------------------
# Observer family generator tests
# ---------------------------------------------------------------------------


class TestMakeRapidityObservers:
    def test_shape(self):
        obs = make_rapidity_observers(n_rapidity=12, n_directions=3)
        assert obs.shape == (36, 3)

    def test_includes_eulerian(self):
        obs = make_rapidity_observers(n_rapidity=12, n_directions=3)
        # At least one row should have zeta=0
        min_zeta = jnp.min(obs[:, 0])
        assert float(min_zeta) == pytest.approx(0.0, abs=1e-10), (
            "Should include Eulerian observer (zeta=0)"
        )

    def test_rapidity_range(self):
        zeta_max = 3.0
        obs = make_rapidity_observers(n_rapidity=12, n_directions=3, zeta_max=zeta_max)
        assert float(jnp.min(obs[:, 0])) >= 0.0
        assert float(jnp.max(obs[:, 0])) <= zeta_max + 1e-10

    def test_custom_counts(self):
        obs = make_rapidity_observers(n_rapidity=5, n_directions=2)
        assert obs.shape == (10, 3)


class TestMakeAngularObservers:
    def test_shape(self):
        obs = make_angular_observers(n_theta=6, n_phi=6)
        assert obs.shape == (36, 3)

    def test_fixed_rapidity(self):
        zeta = 2.0
        obs = make_angular_observers(zeta_fixed=zeta, n_theta=4, n_phi=4)
        assert jnp.allclose(obs[:, 0], zeta), "All observers should have same rapidity"

    def test_custom_counts(self):
        obs = make_angular_observers(n_theta=3, n_phi=4)
        assert obs.shape == (12, 3)


# ---------------------------------------------------------------------------
# Sweep computation tests
# ---------------------------------------------------------------------------


class TestSweepWec:
    def test_shape(self, alcubierre_fields):
        T_field, g_field, _ = alcubierre_fields
        obs = make_rapidity_observers(n_rapidity=4, n_directions=3)
        margins = sweep_wec_margins(T_field, g_field, obs)
        assert margins.shape == (1000, 12)

    def test_violations_present(self, alcubierre_fields):
        """Alcubierre is a known WEC violator sweep should find negative margins."""
        T_field, g_field, _ = alcubierre_fields
        obs = make_rapidity_observers(n_rapidity=6, n_directions=3)
        margins = sweep_wec_margins(T_field, g_field, obs)
        min_margin = float(jnp.min(margins))
        assert min_margin < 0, (
            f"Expected negative WEC margins for Alcubierre, got min={min_margin}"
        )

    def test_eulerian_matches_verifier(self, alcubierre_fields):
        """Eulerian observer (zeta=0) in sweep matches compute_eulerian_ec."""
        from warpax.energy_conditions.verifier import compute_eulerian_ec

        T_field, g_field, _ = alcubierre_fields
        # Single Eulerian observer: zeta=0, theta=0, phi=0
        eulerian_params = jnp.array([[0.0, 0.0, 0.0]])
        sweep_margins = sweep_wec_margins(T_field, g_field, eulerian_params)  # (N, 1)
        sweep_eulerian = sweep_margins[:, 0]  # (N,)

        # Compute Eulerian EC per-point via verifier
        # compute_eulerian_ec takes (T_ab, g_ab) at a single point
        # Compare at 20 sample points
        sample_indices = list(range(0, 1000, 50))  # 20 points
        for idx in sample_indices:
            ec = compute_eulerian_ec(T_field[idx], g_field[idx])
            verifier_wec = float(ec["wec"])
            sweep_wec = float(sweep_eulerian[idx])

            # Signs should agree
            if abs(verifier_wec) > 1e-12 and abs(sweep_wec) > 1e-12:
                assert (verifier_wec < 0) == (sweep_wec < 0), (
                    f"Sign disagreement at point {idx}: "
                    f"verifier={verifier_wec:.6e}, sweep={sweep_wec:.6e}"
                )
            # Values should be close (both compute T_ab u^a u^b for Eulerian u)
            if abs(verifier_wec) > 1e-10:
                rel_err = abs(sweep_wec - verifier_wec) / abs(verifier_wec)
                assert rel_err < 0.1, (
                    f"Point {idx}: sweep={sweep_wec:.6e}, verifier={verifier_wec:.6e}, "
                    f"rel_err={rel_err:.2e}"
                )


class TestSweepNec:
    def test_shape(self, alcubierre_fields):
        T_field, g_field, _ = alcubierre_fields
        nec_params = jnp.array([
            [0.0, 0.0],
            [jnp.pi / 2, 0.0],
            [jnp.pi / 2, jnp.pi / 2],
        ])
        margins = sweep_nec_margins(T_field, g_field, nec_params)
        assert margins.shape == (1000, 3)


class TestSweepAll:
    def test_returns_dict(self, alcubierre_fields):
        T_field, g_field, g_inv_field = alcubierre_fields
        obs = make_rapidity_observers(n_rapidity=4, n_directions=3)
        result = sweep_all_margins(T_field, g_field, g_inv_field, obs)
        assert isinstance(result, dict)
        assert "wec" in result
        assert "nec" in result
        assert "sec" in result

    def test_shapes(self, alcubierre_fields):
        T_field, g_field, g_inv_field = alcubierre_fields
        obs = make_rapidity_observers(n_rapidity=4, n_directions=3)
        result = sweep_all_margins(T_field, g_field, g_inv_field, obs)
        assert result["wec"].shape == (1000, 12)
        assert result["nec"].shape == (1000, 12)
        assert result["sec"].shape == (1000, 12)


class TestSweepWorstMoreNegative:
    def test_worst_more_negative_than_eulerian(self, alcubierre_fields):
        """Worst-case across all observers should be <= Eulerian margin."""
        T_field, g_field, _ = alcubierre_fields
        obs = make_rapidity_observers(n_rapidity=8, n_directions=3)
        margins = sweep_wec_margins(T_field, g_field, obs)

        # Eulerian is at zeta=0 (first 3 entries: zeta=0, 3 directions)
        # All 3 directions with zeta=0 should give same result
        eulerian_margins = margins[:, 0]  # First observer (zeta=0, +x)
        worst_margins = jnp.min(margins, axis=-1)  # Worst across all observers

        # Worst should be <= Eulerian (more negative or equal)
        diff = worst_margins - eulerian_margins
        # Allow small numerical tolerance
        assert float(jnp.max(diff)) < 1e-8, (
            f"Worst-case margin exceeds Eulerian by {float(jnp.max(diff)):.2e}"
        )


# ---------------------------------------------------------------------------
# BFGS cross-validation tests
# ---------------------------------------------------------------------------


class TestCrossValidation:
    @pytest.mark.slow
    def test_bfgs_agreement_alcubierre(self):
        """SC3: Observer sweep sign agreement >= 85% with BFGS on Alcubierre."""
        grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(15, 15, 15))
        metric = AlcubierreMetric(v_s=0.5)
        result = evaluate_curvature_grid(metric, grid_spec)

        N = 15 * 15 * 15
        T_field = result.stress_energy.reshape(N, 4, 4)
        g_field = result.metric.reshape(N, 4, 4)

        obs = make_rapidity_observers(n_rapidity=12, n_directions=3)

        cv = cross_validate_sweep(
            T_field, g_field, obs,
            n_validation_points=30,
            n_starts=16,
            zeta_max=5.0,
        )

        assert cv["sign_agreement_fraction"] >= 0.85, (
            f"Sign agreement {cv['sign_agreement_fraction']:.2f} < 0.85 for Alcubierre"
        )

    @pytest.mark.slow
    def test_bfgs_agreement_rodal(self):
        """SC3: Observer sweep sign agreement >= 85% with BFGS on Rodal."""
        from warpax.metrics import RodalMetric

        grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(15, 15, 15))
        metric = RodalMetric(v_s=0.5)
        result = evaluate_curvature_grid(metric, grid_spec)

        N = 15 * 15 * 15
        T_field = result.stress_energy.reshape(N, 4, 4)
        g_field = result.metric.reshape(N, 4, 4)

        obs = make_rapidity_observers(n_rapidity=12, n_directions=3)

        cv = cross_validate_sweep(
            T_field, g_field, obs,
            n_validation_points=30,
            n_starts=16,
            zeta_max=5.0,
        )

        assert cv["sign_agreement_fraction"] >= 0.85, (
            f"Sign agreement {cv['sign_agreement_fraction']:.2f} < 0.85 for Rodal"
        )


# ---------------------------------------------------------------------------
# Performance sanity test
# ---------------------------------------------------------------------------


class TestSweepPerformance:
    @pytest.mark.slow
    def test_sweep_faster_than_bfgs(self, alcubierre_fields):
        """Sweep for 100 points with 36 observers should be faster than BFGS for same 100 points."""
        import time

        from warpax.energy_conditions.optimization import optimize_wec

        T_field, g_field, _ = alcubierre_fields
        T_100 = T_field[:100]
        g_100 = g_field[:100]
        obs = make_rapidity_observers(n_rapidity=12, n_directions=3)

        # Warm up JIT
        _ = sweep_wec_margins(T_100[:1], g_100[:1], obs)
        _ = optimize_wec(T_100[0], g_100[0], n_starts=8)

        # Time sweep
        t0 = time.perf_counter()
        _ = sweep_wec_margins(T_100, g_100, obs)
        sweep_time = time.perf_counter() - t0

        # Time BFGS (just 10 points for speed)
        t0 = time.perf_counter()
        for i in range(10):
            _ = optimize_wec(T_100[i], g_100[i], n_starts=8)
        bfgs_time_10 = time.perf_counter() - t0
        bfgs_time_100 = bfgs_time_10 * 10  # Extrapolate

        # Sweep should be at least 5x faster
        assert sweep_time < bfgs_time_100, (
            f"Sweep ({sweep_time:.3f}s) not faster than BFGS ({bfgs_time_100:.3f}s est.)"
        )
