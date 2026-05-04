"""Integration test: full admissibility check on known metrics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from warpax.adm import adm_mass, falloff_check
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.constraints import normalized_residuals

jax.config.update("jax_enable_x64", True)


def test_admissibility_audit_minkowski():
    """Minkowski should pass all admissibility criteria."""
    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

    # Constraints
    res = normalized_residuals(metric, coords)
    assert res["epsilon_H"] < 1e-10, f"eps_H={res['epsilon_H']}"
    assert res["epsilon_M"] < 1e-10, f"eps_M={res['epsilon_M']}"

    # ADM mass
    M = adm_mass(metric, r_surface=100.0, n_theta=8, n_phi=16)
    assert jnp.abs(M) < 1e-8, f"M_ADM={M}"

    # Falloff
    falloff = falloff_check(metric, r_test=100.0)
    assert all(falloff.values())


def test_admissibility_audit_schwarzschild():
    """Schwarzschild should pass constraint and mass checks."""
    metric = SchwarzschildMetric()
    coords = jnp.array([0.0, 10.0, 0.0, 0.0], dtype=jnp.float64)

    # Constraints (vacuum)
    res = normalized_residuals(metric, coords)
    assert res["epsilon_H"] < 1e-6, f"eps_H={res['epsilon_H']}"
    assert res["epsilon_M"] < 1e-6, f"eps_M={res['epsilon_M']}"

    # ADM mass should be ~1.0
    M = adm_mass(metric, r_surface=100.0, n_theta=16, n_phi=32)
    assert jnp.abs(M - 1.0) < 0.05, f"M_ADM={M}"
