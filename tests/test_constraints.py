"""Tests for ADM constraint residuals."""

import jax
import jax.numpy as jnp

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
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    momentum_density = jnp.zeros(3, dtype=jnp.float64)
    result = momentum_constraint(gamma, K, momentum_density)
    assert jnp.allclose(result, 0.0, atol=1e-14)


def test_momentum_constraint_flat_with_momentum():
    """Flat metric with non-zero momentum density gives M_i = -8piS_i."""
    gamma = jnp.eye(3, dtype=jnp.float64)
    K = jnp.zeros((3, 3), dtype=jnp.float64)
    S_i = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    result = momentum_constraint(gamma, K, S_i)
    expected = -8.0 * jnp.pi * S_i
    assert jnp.allclose(result, expected, atol=1e-14)


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
