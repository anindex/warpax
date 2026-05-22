"""Tests for the is_vacuum flag on ClassificationResult.

The classifier's near-vacuum early-out (max|Re lambda| < tol) is folded into
the Type-I branch, so near-vacuum grid points are classified Type-I and the
full-domain Type-I fraction is dominated by vacuum. The classifier still
returns he_type=1 for these points but also sets is_vacuum=1, which lets
wall-restricted statistics exclude them. These tests pin that behavior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from warpax.energy_conditions import classify_hawking_ellis


def _g_minkowski():
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def test_exact_vacuum_is_tagged_vacuum():
    """T = 0 must produce is_vacuum=1 (and he_type=1 by convention)."""
    T = jnp.zeros((4, 4))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0, (
        f"Vacuum point should be classified Type-I (convention); got he_type={float(res.he_type)}"
    )
    assert float(res.is_vacuum) == 1.0, (
        f"Vacuum point should have is_vacuum=1; got {float(res.is_vacuum)}"
    )


def test_near_vacuum_is_tagged_vacuum():
    """T with all eigenvalues below tol must be tagged vacuum."""
    # Eigenvalues at scale ~1e-12 < default tol=1e-10
    T = jnp.diag(jnp.array([-1e-12, 1e-12, 1e-12, 1e-12]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 1.0, (
        f"Near-vacuum (1e-12) should have is_vacuum=1; got {float(res.is_vacuum)}"
    )


def test_genuine_type_i_perfect_fluid_is_not_tagged_vacuum():
    """A perfect fluid with rho=p=1 is genuine Type-I, not vacuum."""
    T = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 0.0, (
        f"Perfect fluid should have is_vacuum=0; got {float(res.is_vacuum)}"
    )


def test_genuine_type_i_anisotropic_is_not_tagged_vacuum():
    """Anisotropic pressures: rho=1, p_x=2, p_y=0.5, p_z=0.5."""
    T = jnp.diag(jnp.array([-1.0, 2.0, 0.5, 0.5]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 0.0


def test_vacuum_tag_just_below_tolerance():
    """Eigenvalues at exactly tol should NOT be tagged vacuum
    (strict less-than in the check)."""
    # max|Re lambda| = tol = 1e-10 exactly: should NOT be vacuum
    T = jnp.diag(jnp.array([-1e-10, 1e-10, 1e-10, 1e-10]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.is_vacuum) == 0.0, (
        "Eigenvalues exactly at tol should not trigger near_vacuum (uses <, not <=)"
    )


def test_grid_vacuum_count_matches_n_vacuum():
    """ECGridResult.n_vacuum reports the grid-aggregate count consistent
    with per-point is_vacuum tags."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec, evaluate_curvature_grid
    from warpax.energy_conditions.verifier import verify_grid

    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=0.5)
    # Small grid for test speed
    grid = GridSpec(bounds=[(-5, 5)] * 3, shape=(20, 20, 20))
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ec = verify_grid(
        curv.stress_energy, curv.metric, curv.metric_inv,
        n_starts=2, batch_size=64, compute_eulerian=False,
    )
    assert ec.n_vacuum is not None, "n_vacuum should be populated on ECGridResult"
    # On a (±5)^3 box for R=1, sigma=8 Alcubierre, the wall occupies
    # ~0.35% of the volume, so the vast majority of points should be
    # near-vacuum (T ~ 0 outside the wall).
    total = 20 ** 3
    vacuum_frac = ec.n_vacuum / total
    assert vacuum_frac > 0.5, (
        f"Expected > 50% vacuum for Alcubierre on (±5)^3; got {vacuum_frac:.3f}"
    )
    # And n_vacuum should be <= n_type_i (since vacuum points are tagged Type-I)
    assert ec.n_vacuum <= ec.n_type_i, (
        f"n_vacuum={ec.n_vacuum} exceeds n_type_i={ec.n_type_i}; vacuum "
        "points should be a subset of Type-I."
    )
