"""Tests for Israel junction surface stress-energy."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.junction import surface_stress_energy

jax.config.update("jax_enable_x64", True)


def test_surface_stress_energy_vacuum():
    """Vacuum region should have zero surface stress-energy."""
    metric = MinkowskiMetric()
    boundary_fn = lambda coords: coords[1] - 5.0  # x = 5 boundary
    inside = jnp.array([0.0, 4.99, 0.0, 0.0], dtype=jnp.float64)
    outside = jnp.array([0.0, 5.01, 0.0, 0.0], dtype=jnp.float64)
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert jnp.allclose(S_ab, 0.0, atol=1e-8)


def test_surface_stress_energy_shape():
    """Surface stress-energy should return a 4x4 matrix (full tensor)."""
    metric = MinkowskiMetric()
    boundary_fn = lambda coords: coords[1] - 5.0
    inside = jnp.array([0.0, 4.99, 0.0, 0.0], dtype=jnp.float64)
    outside = jnp.array([0.0, 5.01, 0.0, 0.0], dtype=jnp.float64)
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert S_ab.shape == (4, 4)


def test_surface_stress_energy_symmetry():
    """S_{ab} should be symmetric for smooth metrics."""
    metric = MinkowskiMetric()
    boundary_fn = lambda coords: coords[1] - 5.0
    inside = jnp.array([0.0, 4.99, 0.0, 0.0], dtype=jnp.float64)
    outside = jnp.array([0.0, 5.01, 0.0, 0.0], dtype=jnp.float64)
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert jnp.allclose(S_ab, S_ab.T, atol=1e-14)
