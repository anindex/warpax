"""Tests for ADM mass and falloff computations."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from warpax.adm import adm_mass, falloff_check

jax.config.update("jax_enable_x64", True)


def test_adm_mass_schwarzschild():
    """ADM mass of Schwarzschild should equal its parameter M."""
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    metric = SchwarzschildMetric()
    M = adm_mass(metric, r_surface=100.0, n_theta=16, n_phi=32)
    assert jnp.abs(M - 1.0) < 0.05, f"Expected ~1.0, got {M}"


def test_adm_mass_minkowski():
    """ADM mass of Minkowski should be zero."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    M = adm_mass(metric, r_surface=100.0, n_theta=8, n_phi=16)
    assert jnp.abs(M) < 1e-10, f"Expected ~0, got {M}"


def test_falloff_check_minkowski():
    """Minkowski has exact flat falloff."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    result = falloff_check(metric, r_test=100.0, expected_order=1)
    assert result["g_tt"] is True
    assert result["g_xx"] is True


def test_falloff_check_schwarzschild():
    """Schwarzschild should show O(1/r) falloff."""
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    metric = SchwarzschildMetric()
    result = falloff_check(metric, r_test=200.0, expected_order=1)
    assert result["g_tt"] is True
    assert result["g_xx"] is True
