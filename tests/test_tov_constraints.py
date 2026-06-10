"""TOV equilibrium residuals and ADM-aux constraint diagnostics."""

from warpax.tov import tov_residual
import jax
import jax.numpy as jnp



jax.config.update("jax_enable_x64", True)


def test_tov_residual_isotropic_perfect_fluid():
    """Isotropic fluid (p_r = p_t) with zero derivative should have zero residual."""
    r = jnp.array(1.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5)
    Phi_prime = jnp.array(0.0)
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    assert jnp.isclose(result, 0.0, atol=1e-14)


def test_tov_residual_anisotropic():
    """Anisotropic fluid with balancing terms should have zero residual."""
    r = jnp.array(2.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5 + 0.1)  # p_t > p_r
    # Choose Phi_prime to balance: Phi' = 2(p_t - p_r) / (r * (rho + p_r))
    Phi_prime = jnp.array(2.0 * 0.1 / (2.0 * (1.0 + 0.5)))
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    assert jnp.isclose(result, 0.0, atol=1e-6)


def test_tov_residual_with_gravity():
    """Fluid with non-zero Phi_prime should show non-zero residual if not in equilibrium."""
    r = jnp.array(1.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5)
    Phi_prime = jnp.array(1.0)
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    # For isotropic fluid with constant p_r, dp_r/dr = 0, so residual = (rho + p_r) * Phi'
    expected = (1.0 + 0.5) * 1.0
    assert jnp.isclose(result, expected, atol=1e-6)


def test_tov_residual_pressure_gradient_equilibrium():
    """Non-constant p_r: dp_r/dr balanced by gravity gives zero residual.

    The constant-p_r tests above never exercise the jax.grad(p_r) term;
    this one does. At r=2: p_r = 0.3, dp_r/dr = -0.1, so equilibrium
    needs Phi' = 0.1 / (rho + p_r) = 0.1 / 1.3.
    """
    r = jnp.array(2.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: 0.5 - 0.1 * rr
    p_t = lambda rr: 0.5 - 0.1 * rr  # isotropic, anisotropy term drops
    Phi_prime = jnp.array(0.1 / (1.0 + 0.3))
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    assert jnp.isclose(result, 0.0, atol=1e-14)


def test_tov_residual_pressure_gradient_only():
    """Non-constant p_r with Phi' = 0 and p_t = p_r: residual is exactly dp_r/dr."""
    r = jnp.array(2.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: 0.5 - 0.1 * rr
    p_t = lambda rr: 0.5 - 0.1 * rr
    result = tov_residual(r, rho, p_r, p_t, jnp.array(0.0))
    assert jnp.isclose(result, -0.1, atol=1e-14)


def test_tov_residual_from_metric_vacuum_zero():
    """Minkowski + vacuum profiles: Phi' = 0 and all terms vanish exactly."""
    from warpax.benchmarks import MinkowskiMetric
    from warpax.tov.residuals import tov_residual_from_metric

    zero = lambda rr: jnp.array(0.0)
    result = tov_residual_from_metric(MinkowskiMetric(), jnp.array(2.0), zero, zero, zero)
    assert jnp.isclose(result, 0.0, atol=1e-14)


jax.config.update("jax_enable_x64", True)

from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.constraints import hamiltonian_constraint, momentum_constraint, normalized_residuals


def test_hamiltonian_constraint_minkowski():
    """Minkowski spacetime has zero Hamiltonian constraint."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    energy_density = jnp.array(0.0, dtype=jnp.float64)
    result = hamiltonian_constraint(gamma, K, energy_density, R=jnp.array(0.0))
    assert jnp.isclose(result, 0.0, atol=1e-14)


def test_hamiltonian_constraint_pure_trace_k():
    """Flat metric with K = lambda I gives H = 6*lambda^2."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.eye(3, dtype=jnp.float64)  # lambda = 1
    result = hamiltonian_constraint(gamma, K, jnp.array(0.0, dtype=jnp.float64), R=jnp.array(0.0))
    assert jnp.isclose(result, 6.0, atol=1e-14)


def test_hamiltonian_constraint_pure_energy():
    """Flat metric with E gives H = -16piE."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    result = hamiltonian_constraint(gamma, K, jnp.array(1.0, dtype=jnp.float64), R=jnp.array(0.0))
    assert jnp.isclose(result, -16.0 * jnp.pi, atol=1e-14)


def test_momentum_constraint_minkowski():
    """Minkowski spacetime has zero momentum constraint."""
    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    momentum_density = jnp.zeros(3, dtype=jnp.float64)
    result = momentum_constraint(
        gamma, K, momentum_density, metric_fn=metric, coords=coords,
    )
    assert jnp.allclose(result, 0.0, atol=1e-14)


def test_momentum_constraint_flat_with_momentum():
    """Flat metric with non-zero momentum density gives M_i = -8piS_i."""
    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    S_i = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    result = momentum_constraint(
        gamma, K, S_i, metric_fn=metric, coords=coords,
    )
    expected = -8.0 * jnp.pi * S_i
    assert jnp.allclose(result, expected, atol=1e-14)


def test_momentum_constraint_raises_without_metric():
    """Calling momentum_constraint without metric_fn/coords raises."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    S_i = jnp.zeros(3, dtype=jnp.float64)
    import pytest
    with pytest.raises(ValueError, match="metric_fn"):
        momentum_constraint(gamma, K, S_i)


def test_hamiltonian_constraint_raises_without_R_or_metric():
    """Calling hamiltonian_constraint without R or metric raises."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    energy_density = jnp.array(0.0, dtype=jnp.float64)
    import pytest
    with pytest.raises(ValueError, match="R"):
        hamiltonian_constraint(gamma, K, energy_density)


def test_normalized_residuals_minkowski():
    """Normalized residuals for Minkowski are exactly zero."""
    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    res = normalized_residuals(metric, coords)
    assert jnp.isclose(res["epsilon_H"], 0.0, atol=1e-14)
    assert jnp.isclose(res["epsilon_M"], 0.0, atol=1e-14)


def test_normalized_residuals_schwarzschild():
    """Schwarzschild vacuum satisfies constraints to high precision."""
    metric = SchwarzschildMetric()
    coords = jnp.array([0.0, 10.0, 0.0, 0.0], dtype=jnp.float64)
    res = normalized_residuals(metric, coords)
    assert res["epsilon_H"] < 1e-6, f"eps_H={res['epsilon_H']}"
    assert res["epsilon_M"] < 1e-6, f"eps_M={res['epsilon_M']}"
