"""ADM decomposition and constraint residual tests.

Verifies 3+1 ADM split (lapse, shift, spatial metric, extrinsic curvature)
and Hamiltonian/momentum constraint evaluations against known analytical
solutions (Minkowski, Schwarzschild, WarpShell).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 3+1 ADM decomposition
# ---------------------------------------------------------------------------

class TestADMSplit:
    """Verify (alpha, beta, gamma, K) extraction from full spacetime metrics."""

    def test_minkowski_decomposition(self):
        """Minkowski: alpha=1, beta=0, gamma=I, K=0."""
        from warpax.geometry import adm_split
        from warpax.benchmarks import MinkowskiMetric

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.allclose(adm.lapse, 1.0, atol=1e-14)
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.spatial_metric, jnp.eye(3), atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-10)

    def test_schwarzschild_decomposition(self):
        """Schwarzschild isotropic: alpha = (1-M/2r)/(1+M/2r), beta=0, K=0."""
        from warpax.geometry import adm_split
        from warpax.benchmarks import SchwarzschildMetric

        metric = SchwarzschildMetric(M=1.0)
        r_iso = 10.0
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        adm = adm_split(metric, coords)

        ratio = 1.0 / (2.0 * r_iso)
        expected_lapse = (1.0 - ratio) / (1.0 + ratio)

        assert jnp.allclose(adm.lapse, expected_lapse, rtol=1e-12), \
            f"Lapse: got {adm.lapse}, expected {expected_lapse}"
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-8), \
            f"Static K must vanish, got max|K|={jnp.max(jnp.abs(adm.extrinsic_curvature))}"

    def test_warpshell_nonzero_shift(self):
        """WarpShell interior has beta^x = -v_s."""
        from warpax.geometry import adm_split
        from warpax.metrics import WarpShellPhysical

        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.abs(adm.shift_upper[0] - (-0.02)) < 1e-6, \
            f"Interior shift should be -v_s=-0.02, got {adm.shift_upper[0]}"
        assert jnp.allclose(adm.shift_upper[1:], 0.0, atol=1e-10)

    def test_warpshell_shell_region(self):
        """WarpShell transition region produces finite K_{ij}."""
        from warpax.geometry import adm_split
        from warpax.metrics import WarpShellPhysical

        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        K_max = jnp.max(jnp.abs(adm.extrinsic_curvature))
        assert K_max < 100, f"K should be finite, got max|K|={K_max}"


# ---------------------------------------------------------------------------
# Constraint residuals
# ---------------------------------------------------------------------------

class TestConstraints:
    """Verify Hamiltonian and momentum constraint evaluations."""

    def test_minkowski_hamiltonian(self):
        """Minkowski vacuum: H = 0."""
        from warpax.constraints import hamiltonian_constraint

        gamma = jnp.eye(3)
        K = jnp.zeros((3, 3))
        H = hamiltonian_constraint(gamma, K, jnp.array(0.0), R=jnp.array(0.0))
        assert jnp.abs(H) < 1e-14, f"H should be 0 for Minkowski, got {H}"

    def test_pure_K_trace(self):
        """Flat space with K=delta_{ij}: H = K^2 - K_{ij}K^{ij} = 9-3 = 6."""
        from warpax.constraints import hamiltonian_constraint

        gamma = jnp.eye(3)
        K = jnp.eye(3)
        H = hamiltonian_constraint(gamma, K, jnp.array(0.0), R=jnp.array(0.0))
        assert jnp.allclose(H, 6.0, atol=1e-12), f"Expected H=6, got {H}"

    def test_normalized_residuals_minkowski(self):
        """Minkowski: eps_H ~0, eps_M ~0."""
        from warpax.constraints import normalized_residuals
        from warpax.benchmarks import MinkowskiMetric

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        result = normalized_residuals(metric, coords)

        assert result["epsilon_H"] < 1e-10, f"eps_H={result['epsilon_H']}"

    def test_normalized_residuals_schwarzschild(self):
        """Schwarzschild vacuum: eps_H ~0."""
        from warpax.constraints import normalized_residuals
        from warpax.benchmarks import SchwarzschildMetric

        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 10.0, 0.0, 0.0])
        result = normalized_residuals(metric, coords)

        assert result["epsilon_H"] < 1e-6, f"eps_H={result['epsilon_H']}"

    def test_momentum_constraint_minkowski(self):
        """Minkowski: M_i = 0."""
        from warpax.constraints import momentum_constraint

        gamma = jnp.eye(3)
        K = jnp.zeros((3, 3))
        M_i = momentum_constraint(gamma, K, jnp.zeros(3))
        assert jnp.allclose(M_i, 0.0, atol=1e-14)
