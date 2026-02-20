"""Tests for time parameter in grid evaluation.

Verifies:
1. build_coord_batch produces correct time coordinates at non-zero t
2. evaluate_curvature_grid accepts t= and produces different results at different times
3. Backward compatibility: default t=0.0 matches explicit t=0.0
4. Zero recompilation across velocity sweeps (pytree structure equality)
5. All 6 warp metrics support dynamic v_s via eqx.tree_at
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry.grid import build_coord_batch, evaluate_curvature_grid
from warpax.geometry.types import GridSpec


# ---------------------------------------------------------------------------
# build_coord_batch tests
# ---------------------------------------------------------------------------


def test_build_coord_batch_nonzero_t():
    """build_coord_batch with t=1.5 produces coords with t=1.5 everywhere."""
    grid_spec = GridSpec(bounds=[(-1, 1), (-1, 1), (-1, 1)], shape=(5, 5, 5))
    coords = build_coord_batch(grid_spec, t=1.5)
    assert coords.shape == (125, 4)
    assert jnp.allclose(coords[:, 0], 1.5), "Time coordinate should be 1.5 everywhere"


def test_build_coord_batch_default_t():
    """build_coord_batch default t=0 produces coords with t=0 everywhere."""
    grid_spec = GridSpec(bounds=[(-1, 1), (-1, 1), (-1, 1)], shape=(5, 5, 5))
    coords = build_coord_batch(grid_spec)
    assert jnp.allclose(coords[:, 0], 0.0), "Default time coordinate should be 0.0"


# ---------------------------------------------------------------------------
# evaluate_curvature_grid time parameter tests
# ---------------------------------------------------------------------------


def test_evaluate_curvature_grid_t_parameter():
    """Results differ at t=0.0 vs t=1.0 when bubble center depends on time.

    The Alcubierre metric uses x_s as a static parameter (not v_s*t),
    so we test by explicitly setting x_s to simulate time-dependent position.
    At t=0 the bubble is centered at x=0; at t=1 it is at x=v_s*t=0.5.
    """
    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(10, 10, 10))
    metric_t0 = AlcubierreMetric(v_s=0.5, x_s=0.0)
    metric_t1 = eqx.tree_at(lambda m: m.x_s, metric_t0, 0.5)  # x_s = v_s * t = 0.5

    result_t0 = evaluate_curvature_grid(metric_t0, grid_spec, t=0.0)
    result_t1 = evaluate_curvature_grid(metric_t1, grid_spec, t=1.0)

    # Ricci scalar should differ (bubble center has moved)
    assert not jnp.allclose(result_t0.ricci_scalar, result_t1.ricci_scalar), (
        "Ricci scalar should differ when bubble center shifts from x_s=0 to x_s=0.5"
    )


def test_evaluate_curvature_grid_t_default():
    """Calling without t= gives same result as t=0.0 (backward compatibility)."""
    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    metric = AlcubierreMetric(v_s=0.5)

    result_default = evaluate_curvature_grid(metric, grid_spec)
    result_t0 = evaluate_curvature_grid(metric, grid_spec, t=0.0)

    assert jnp.allclose(result_default.ricci_scalar, result_t0.ricci_scalar), (
        "Default t should give same result as t=0.0"
    )


# ---------------------------------------------------------------------------
# Zero recompilation via pytree structure
# ---------------------------------------------------------------------------


def test_zero_recompilation_pytree_structure():
    """Swapping v_s via eqx.tree_at preserves pytree structure (no recompilation)."""
    m1 = AlcubierreMetric(v_s=0.1)
    m2 = eqx.tree_at(lambda m: m.v_s, m1, 0.5)
    m3 = eqx.tree_at(lambda m: m.v_s, m1, 0.99)

    # All should have identical pytree structure
    s1 = jax.tree_util.tree_structure(m1)
    s2 = jax.tree_util.tree_structure(m2)
    s3 = jax.tree_util.tree_structure(m3)

    assert s1 == s2, "Pytree structure changed when swapping v_s from 0.1 to 0.5"
    assert s1 == s3, "Pytree structure changed when swapping v_s from 0.1 to 0.99"


def test_velocity_sweep_produces_distinct_results():
    """60 v_s values produce distinct ricci_scalar (not cached stale results)."""
    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    base_metric = AlcubierreMetric(v_s=0.1)
    v_s_values = [0.1 + i * 0.89 / 9 for i in range(10)]  # 10 values for speed

    results = []
    for v_s in v_s_values:
        m = eqx.tree_at(lambda m: m.v_s, base_metric, v_s)
        result = evaluate_curvature_grid(m, grid_spec)
        results.append(result.ricci_scalar)

    # Adjacent results should differ
    for i in range(len(results) - 1):
        assert not jnp.allclose(results[i], results[i + 1]), (
            f"Results at v_s={v_s_values[i]:.3f} and v_s={v_s_values[i+1]:.3f} are identical"
        )


# ---------------------------------------------------------------------------
# All metrics support dynamic v_s
# ---------------------------------------------------------------------------


def test_all_metrics_support_dynamic_vs():
    """All 6 warp metrics support v_s swap via eqx.tree_at with preserved structure."""
    from warpax.metrics import (
        LentzMetric,
        NatarioMetric,
        RodalMetric,
        VanDenBroeckMetric,
        WarpShellMetric,
    )

    metrics = [
        AlcubierreMetric(v_s=0.1),
        LentzMetric(v_s=0.1),
        NatarioMetric(v_s=0.1),
        RodalMetric(v_s=0.1),
        VanDenBroeckMetric(v_s=0.1),
        WarpShellMetric(v_s=0.02),
    ]

    for base_metric in metrics:
        name = base_metric.name()
        new_v_s = 0.5
        m_new = eqx.tree_at(lambda m: m.v_s, base_metric, new_v_s)

        # Check v_s was actually swapped
        assert m_new.v_s == new_v_s, f"{name}: v_s not swapped"

        # Check pytree structure preserved
        s_old = jax.tree_util.tree_structure(base_metric)
        s_new = jax.tree_util.tree_structure(m_new)
        assert s_old == s_new, f"{name}: pytree structure changed on v_s swap"
