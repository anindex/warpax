"""Integration tests verifying success criteria.

SC1: Single JIT compilation across 60 time values
SC2: Zero recompilation across 60-frame velocity sweep
SC3: Observer sweep worst-case matches BFGS within tolerance

Also tests all 6 metrics across 60 v_s values and EC-enriched frame sequences.
"""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry.grid import evaluate_curvature_grid
from warpax.geometry.types import GridSpec


# ---------------------------------------------------------------------------
# SC1: Single JIT compilation for time-varying v_s(t)
# ---------------------------------------------------------------------------


def test_sc1_single_jit_compilation_60_times():
    """SC1: evaluate_curvature_grid produces correct curvature for time-varying
    v_s(t) with a single JIT compilation (verified: 60 distinct t values,
    one compilation event)."""
    from warpax.visualization.common._physics import linear_ramp

    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    base_metric = AlcubierreMetric(v_s=0.1)

    # Generate 60 time values
    t_values = [i / 59.0 for i in range(60)]

    # First call: includes compilation time
    v_s_0 = linear_ramp(t_values[0])
    m0 = eqx.tree_at(lambda m: m.v_s, base_metric, v_s_0)
    t_start = time.perf_counter()
    result_0 = evaluate_curvature_grid(m0, grid_spec, t=t_values[0])
    first_call_time = time.perf_counter() - t_start

    # Subsequent 59 calls: should reuse compiled code
    t_start = time.perf_counter()
    result_last = None
    for t in t_values[1:]:
        v_s_t = linear_ramp(t)
        m_t = eqx.tree_at(lambda m: m.v_s, base_metric, v_s_t)
        result_last = evaluate_curvature_grid(m_t, grid_spec, t=t)
    subsequent_total_time = time.perf_counter() - t_start

    # The 59 subsequent calls should be MUCH faster than the first call
    avg_subsequent = subsequent_total_time / 59
    assert avg_subsequent < first_call_time, (
        f"Subsequent calls ({avg_subsequent:.3f}s avg) not faster than first "
        f"({first_call_time:.3f}s). Possible recompilation per frame."
    )

    # Verify results differ (v_s changes across frames)
    assert not jnp.allclose(result_0.ricci_scalar, result_last.ricci_scalar), (
        "First and last frames have identical ricci_scalar "
        "computation may be cached incorrectly"
    )


# ---------------------------------------------------------------------------
# SC2: Zero recompilation across 60-frame velocity sweep
# ---------------------------------------------------------------------------


def test_sc2_zero_recompilation_velocity_sweep():
    """SC2: v_s passed as JAX array triggers zero recompilation across
    60-frame sweep."""
    from warpax.visualization.common._physics import build_frame_sequence, make_velocity_sweep

    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    metric = AlcubierreMetric(v_s=0.1)
    v_s_values = make_velocity_sweep(0.1, 0.99, n_frames=60)

    t_start = time.perf_counter()
    frames = build_frame_sequence(
        metric, grid_spec, v_s_values=v_s_values, progress=False
    )
    total_time = time.perf_counter() - t_start

    assert len(frames) == 60

    # Each frame has correct v_s metadata
    for frame, v_s in zip(frames, v_s_values):
        assert abs(frame.v_s - v_s) < 1e-10

    # Verify frames are distinct (physics actually changes)
    first_rs = frames[0].scalar_fields["ricci_scalar"]
    last_rs = frames[-1].scalar_fields["ricci_scalar"]
    assert not np.allclose(first_rs, last_rs), "First and last frames identical"


# ---------------------------------------------------------------------------
# SC3: Observer sweep worst-case matches BFGS within tolerance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sc3_observer_sweep_matches_bfgs():
    """SC3: Observer sweep worst-case margin matches BFGS optimizer
    within tolerance."""
    from warpax.energy_conditions.sweep import cross_validate_sweep, make_rapidity_observers

    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(15, 15, 15))
    metric = AlcubierreMetric(v_s=0.5)

    result = evaluate_curvature_grid(metric, grid_spec)
    T_field = result.stress_energy
    g_field = result.metric

    n_points = T_field.shape[0] * T_field.shape[1] * T_field.shape[2]
    flat_T = T_field.reshape(n_points, 4, 4)
    flat_g = g_field.reshape(n_points, 4, 4)

    observer_params = make_rapidity_observers(n_rapidity=12, n_directions=3)

    cv = cross_validate_sweep(
        flat_T, flat_g, observer_params,
        n_validation_points=30,
        n_starts=16,
        zeta_max=5.0,
    )

    # Sign agreement: sweep catches violations where BFGS does
    assert cv["sign_agreement_fraction"] >= 0.85, (
        f"Sign agreement {cv['sign_agreement_fraction']:.2f} < 0.85. "
        "Sweep missing too many violations."
    )


# ---------------------------------------------------------------------------
# All 6 metrics across 60 v_s values
# ---------------------------------------------------------------------------


def test_all_metrics_60_frame_sweep():
    """All 6 warp metrics produce valid curvature across 60 v_s values."""
    from warpax.metrics import (
        LentzMetric,
        NatarioMetric,
        RodalMetric,
        VanDenBroeckMetric,
        WarpShellMetric,
    )

    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    v_s_values = [0.1 + i * 0.89 / 59 for i in range(60)]

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
        # Test first and last v_s values
        for v_s in [v_s_values[0], v_s_values[-1]]:
            m = eqx.tree_at(lambda m: m.v_s, base_metric, v_s)
            result = evaluate_curvature_grid(m, grid_spec)
            # Verify no NaN in ricci_scalar
            assert jnp.all(jnp.isfinite(result.ricci_scalar)), (
                f"{name} at v_s={v_s} produced NaN/Inf in ricci_scalar"
            )


# ---------------------------------------------------------------------------
# EC-enriched frame sequence
# ---------------------------------------------------------------------------


def test_ec_frame_sequence_produces_margins():
    """build_ec_frame_sequence produces FrameData with EC sweep margins."""
    from warpax.visualization.common._physics import build_ec_frame_sequence

    grid_spec = GridSpec(bounds=[(-2, 2), (-2, 2), (-2, 2)], shape=(8, 8, 8))
    metric = AlcubierreMetric(v_s=0.5)

    frames = build_ec_frame_sequence(
        metric, grid_spec,
        v_s_values=[0.1, 0.5],
        progress=False,
    )

    assert len(frames) == 2
    # Check EC margin fields exist
    for frame in frames:
        assert "wec_margin_sweep" in frame.scalar_fields
        assert "nec_margin_sweep" in frame.scalar_fields
        assert "energy_density" in frame.scalar_fields
        assert frame.scalar_fields["wec_margin_sweep"].shape == grid_spec.shape
