"""Physics verification tests for ADM decomposition, constraints, TOV,
ADM mass, Fuchs metric, transport diagnostics, junction conditions,
and source-consistency checks.

Each test verifies a specific analytical property against known
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


# ---------------------------------------------------------------------------
# Anisotropic TOV
# ---------------------------------------------------------------------------

class TestTOV:
    """Verify the Bowers-Liang anisotropic TOV equilibrium residual."""

    def test_constant_pressure_no_gravity(self):
        """Constant isotropic pressure with Phi'=0, residual = 0."""
        from warpax.tov import tov_residual

        r = jnp.array(5.0)
        rho = lambda r: jnp.array(1.0)
        p_r = lambda r: jnp.array(0.5)
        p_t = lambda r: jnp.array(0.5)
        res = tov_residual(r, rho, p_r, p_t, jnp.array(0.0))
        assert jnp.abs(res) < 1e-10, f"Expected 0, got {res}"

    def test_anisotropic_term(self):
        """The 2(p_t - p_r)/r anisotropy contribution."""
        from warpax.tov import tov_residual

        r = jnp.array(5.0)
        rho = lambda r: jnp.array(1.0)
        p_r = lambda r: jnp.array(0.5)
        p_t = lambda r: jnp.array(1.0)
        res = tov_residual(r, rho, p_r, p_t, jnp.array(0.0))
        # dp_r/dr = 0, Phi'=0, so residual = -2(0.5)/5 = -0.2
        assert jnp.allclose(res, -0.2, atol=1e-8), f"Expected -0.2, got {res}"

    def test_autodiff_pressure_gradient(self):
        """dp_r/dr via JAX autodiff matches analytical 2r."""
        from warpax.tov import tov_residual

        r = jnp.array(5.0)
        rho = lambda r: jnp.array(0.0)
        p_r = lambda r: r**2
        p_t = lambda r: r**2
        res = tov_residual(r, rho, p_r, p_t, jnp.array(0.0))
        assert jnp.allclose(res, 10.0, atol=1e-8), f"Expected 10.0, got {res}"


# ---------------------------------------------------------------------------
# ADM mass surface integral
# ---------------------------------------------------------------------------

class TestADMMass:
    """Verify the ADM mass integral and asymptotic falloff."""

    def test_minkowski_mass(self):
        """Minkowski: M_ADM = 0."""
        from warpax.adm import adm_mass
        from warpax.benchmarks import MinkowskiMetric

        M = adm_mass(MinkowskiMetric(), r_surface=50.0, n_theta=8, n_phi=16)
        assert jnp.abs(M) < 1e-10, f"Minkowski M_ADM should be 0, got {M}"

    def test_schwarzschild_mass(self):
        """Schwarzschild M=1: ADM mass ~1.0."""
        from warpax.adm import adm_mass
        from warpax.benchmarks import SchwarzschildMetric

        M = adm_mass(SchwarzschildMetric(M=1.0), r_surface=100.0, n_theta=16, n_phi=32)
        assert jnp.abs(M - 1.0) < 0.05, f"Expected ~1.0, got {M}"

    def test_falloff_minkowski(self):
        """Minkowski: all diagonal components are exactly flat."""
        from warpax.adm import falloff_check
        from warpax.benchmarks import MinkowskiMetric

        result = falloff_check(MinkowskiMetric(), r_test=100.0)
        for comp, passed in result.items():
            assert passed, f"Falloff failed for {comp}"

    def test_falloff_schwarzschild(self):
        """Schwarzschild: 1/r falloff for spatial components."""
        from warpax.adm import falloff_check
        from warpax.benchmarks import SchwarzschildMetric

        result = falloff_check(SchwarzschildMetric(M=1.0), r_test=200.0)
        for comp, passed in result.items():
            assert passed, f"Falloff failed for {comp}"


# ---------------------------------------------------------------------------
# Fuchs metric
# ---------------------------------------------------------------------------

class TestFuchsMetric:
    """Verify the Fuchs et al. CQG 2024 metric wrapper."""

    def test_fuchs_creates(self):
        """FuchsMetric instantiates and returns a (4,4) tensor."""
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        assert metric.name() == "Fuchs-CQG2024"
        assert metric(jnp.array([0.0, 5.0, 0.0, 0.0])).shape == (4, 4)

    def test_fuchs_interior_is_shifted_minkowski(self):
        """Deep interior: flat spatial metric with shift beta_x = -v_s."""
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        g = metric(jnp.array([0.0, 1.0, 0.0, 0.0]))

        assert jnp.abs(g[0, 0] - (-(1.0 - 0.02**2))) < 1e-4
        assert jnp.abs(g[0, 1] - (-0.02)) < 1e-3

    def test_fuchs_exterior_is_flat(self):
        """Far exterior converges to eta_{munu}."""
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        g = metric(jnp.array([0.0, 50.0, 0.0, 0.0]))
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, eta, atol=1e-6), \
            f"Exterior deviation = {jnp.max(jnp.abs(g - eta))}"


# ---------------------------------------------------------------------------
# Transport diagnostics
# ---------------------------------------------------------------------------

class TestTransport:
    """Verify geodesic-based transport observables."""

    def test_geodesic_deviation_minkowski(self):
        """Minkowski: zero tidal forces, A_geo = 0."""
        from warpax.transport import geodesic_deviation_diagnostic
        from warpax.benchmarks import MinkowskiMetric

        A = geodesic_deviation_diagnostic(
            MinkowskiMetric(), jnp.array([0.0, 5.0, 0.0, 0.0])
        )
        assert jnp.abs(A) < 1e-10, f"A_geo should be 0 for Minkowski, got {A}"

    def test_geodesic_deviation_schwarzschild(self):
        """Schwarzschild: tidal force ~ M/r^3 > 0."""
        from warpax.transport import geodesic_deviation_diagnostic
        from warpax.benchmarks import SchwarzschildMetric

        A = geodesic_deviation_diagnostic(
            SchwarzschildMetric(M=1.0), jnp.array([0.0, 10.0, 0.0, 0.0])
        )
        assert A > 1e-6, f"Schwarzschild should have nonzero tidal force, got {A}"


# ---------------------------------------------------------------------------
# Junction conditions
# ---------------------------------------------------------------------------

class TestJunction:
    """Verify Israel junction surface stress-energy."""

    def test_vacuum_surface_stress(self):
        """Smooth metric, [K] ~0, so S ~0."""
        from warpax.junction import surface_stress_energy
        from warpax.benchmarks import MinkowskiMetric

        metric = MinkowskiMetric()
        boundary_fn = lambda c: c[1] - 5.0

        inside = jnp.array([0.0, 4.99, 0.0, 0.0])
        outside = jnp.array([0.0, 5.01, 0.0, 0.0])

        S = surface_stress_energy(metric, boundary_fn, inside, outside)
        assert S.shape == (4, 4)
        assert jnp.max(jnp.abs(S)) < 1e-8, f"max|S|={jnp.max(jnp.abs(S))}"


# ---------------------------------------------------------------------------
# Source consistency
# ---------------------------------------------------------------------------

class TestSourceConsistency:
    """Verify DeltaT = T_input - G/(8pi) residual."""

    def test_minkowski_vacuum_consistency(self):
        """Minkowski with T=0: DeltaT = 0."""
        from warpax.constraints import stress_energy_residual
        from warpax.benchmarks import MinkowskiMetric

        result = stress_energy_residual(
            MinkowskiMetric(), jnp.array([0.0, 5.0, 0.0, 0.0])
        )
        assert result["max_residual"] < 1e-12, f"DeltaT={result['max_residual']}"

    def test_schwarzschild_vacuum_consistency(self):
        """Schwarzschild with T=0: DeltaT ~0 (vacuum solution)."""
        from warpax.constraints import stress_energy_residual
        from warpax.benchmarks import SchwarzschildMetric

        result = stress_energy_residual(
            SchwarzschildMetric(M=1.0), jnp.array([0.0, 10.0, 0.0, 0.0])
        )
        assert result["max_residual"] < 1e-8, f"DeltaT={result['max_residual']}"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptions:
    """Verify domain exception inheritance."""

    def test_exception_inheritance(self):
        from warpax.exceptions import (
            WarpAXError,
            ConstraintViolationError,
            TOVInconsistencyError,
            JunctionDiscontinuityError,
            AsymptoticFalloffError,
            TransportUndefinedError,
        )
        for exc_cls in [
            ConstraintViolationError,
            TOVInconsistencyError,
            JunctionDiscontinuityError,
            AsymptoticFalloffError,
            TransportUndefinedError,
        ]:
            assert issubclass(exc_cls, WarpAXError)
