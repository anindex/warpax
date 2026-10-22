"""Tests for :func:`warpax.certify`."""

from __future__ import annotations

import sys

import beartype
import jax.numpy as jnp
import numpy as np
import pytest

import warpax.analysis.invariant_verification as iv
from warpax.analysis.invariant_verification import single_frame_miss
from warpax.benchmarks import AlcubierreMetric
from warpax.certify import certify
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free, type_fractions
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import proper_volume_weights, wall_clustered
from warpax.metrics import VanDenBroeckMetric

SHAPE = (8, 8, 8)
BOUNDS = [(-3, 3)] * 3


def test_frame_free_certification_runs_once(monkeypatch):
    """certify computes the frame-free result once and hands it to single_frame_miss."""
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("lmi_where"))
        return certify_grid_frame_free(*args, **kwargs)

    monkeypatch.setattr(sys.modules["warpax.certify"], "certify_grid_frame_free", counted)
    monkeypatch.setattr(iv, "certify_grid_frame_free", counted)

    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    res = certify(metric, shape=SHAPE)
    assert len(calls) == 1
    assert res.eulerian_available

    grid = wall_clustered(metric, BOUNDS, SHAPE, a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv
    coords = build_coord_batch(grid, t=0.0)
    mask = np.asarray(shape_function_mask(metric, coords, SHAPE, f_low=0.1, f_high=0.9)).ravel()
    w = np.asarray(proper_volume_weights(grid.volume_weights_array, g)).ravel()
    ref = single_frame_miss(T, g, gi, mask=mask, volume_weights=w, frame_free=res.frame_free)
    assert res.single_frame_miss == ref
    assert len(calls) == 1


def test_uniform_grid_fractions_carry_the_slice_jacobian():
    """On a uniform grid the coordinate cell volume cancels in a fraction but
    sqrt(det gamma) does not; Van den Broeck has gamma_ij = B^2 delta_ij."""
    metric = VanDenBroeckMetric(v_s=0.5, R=1.0, sigma=2.0, R_tilde=0.5, alpha_vdb=0.5, sigma_B=2.0)
    res = certify(metric, shape=SHAPE, clustered=False)

    grid = GridSpec(bounds=BOUNDS, shape=SHAPE)
    g = evaluate_curvature_grid(metric, grid, batch_size=256).metric
    coords = build_coord_batch(grid, t=0.0)
    mask = np.asarray(shape_function_mask(metric, coords, SHAPE, f_low=0.1, f_high=0.9)).ravel()
    w = np.sqrt(np.linalg.det(np.asarray(g)[..., 1:, 1:])).ravel() * mask
    he = np.asarray(res.frame_free.he_types).ravel()
    assert w.sum() > 0
    assert not np.allclose(w[mask], w[mask][0])
    for key, t in (("i", 1.0), ("ii", 2.0), ("iii", 3.0), ("iv", 4.0)):
        assert res.type_fractions[f"frac_type_{key}"] == pytest.approx(
            float(np.sum(w * (he == t)) / w.sum()), rel=1e-12, abs=0.0
        )


def test_numpy_masks_and_uniform_weights_pass_beartype():
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=BOUNDS, shape=SHAPE)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    mask = np.ones(SHAPE, dtype=bool)
    fr = beartype.beartype(type_fractions)(ff, mask=mask)
    assert fr["n_selected"] == mask.size
    w = beartype.beartype(proper_volume_weights)(None, curv.metric)
    assert w.shape == SHAPE
    assert jnp.all(w == 1.0)
    miss = beartype.beartype(single_frame_miss)(
        curv.stress_energy, curv.metric, curv.metric_inv, mask=mask, volume_weights=np.asarray(w)
    )
    assert miss["n_selected"] == mask.size
