"""Tests for the Contribution 2 invariant all-observer verification."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from warpax.analysis.invariant_verification import (
    integrated_exotic_content,
    peak_proper_energy_deficit,
    reduction_factors,
    single_frame_miss,
)
import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import RodalMetric


def _curv(metric, N=24):
    grid = wall_clustered(metric, [(-3, 3)] * 3, (N, N, N), a=1.2)
    c = evaluate_curvature_grid(metric, grid, batch_size=256)
    return c.stress_energy, c.metric, c.metric_inv, grid.volume_weights_array


def _wall(metric, N=24):
    grid = wall_clustered(metric, [(-3, 3)] * 3, (N, N, N), a=1.2)
    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, (N, N, N), f_low=0.1, f_high=0.9)
    return np.asarray(jnp.reshape(mask, (-1,))).astype(bool)


def test_rodal_single_frame_miss_is_substantial():
    """Rodal (100% Type I): within the wall, the Eulerian frame misses a large
    fraction of the all-observer DEC/WEC violations -- the verification of its
    positive-energy claim. NEC is missed far less (Eulerian probes null directly)."""
    metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
    T, g, gi, vw = _curv(metric)
    mask = _wall(metric)
    miss = single_frame_miss(T, g, gi, mask=mask,
                             volume_weights=np.asarray(jnp.reshape(vw, (-1,))))
    # DEC/WEC: most violations are off-Eulerian (boosted observers); NEC less so.
    assert 0.4 < miss["dec"]["miss_rate"] < 0.95
    assert 0.4 < miss["wec"]["miss_rate"] < 0.95
    assert miss["dec"]["miss_rate"] > miss["nec"]["miss_rate"]
    assert miss["dec"]["n_violated"] > 0


def test_exotic_content_signs_and_manual_match():
    """E_- <= 0 <= E_+, and the invariant E_- equals a direct integral of the
    negative Type-I proper energy density."""
    metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
    T, g, gi, vw = _curv(metric)
    exotic = integrated_exotic_content(T, g, gi, vw)
    # Strict signs: an all-NaN rho would give 0 here, so these catch a dead
    # certifier. Golden pins make this the Rodal exotic-content regression
    # anchor (rel=1e-3 tolerates cross-platform eigenvalue jitter).
    assert exotic["E_minus_inv"] < 0.0
    assert exotic["E_plus_inv"] > 0.0
    assert exotic["E_minus_eul"] < 0.0
    assert exotic["E_minus_inv"] == pytest.approx(-0.03419251994163071, rel=1e-3)
    assert exotic["E_plus_inv"] == pytest.approx(0.04663450787803846, rel=1e-3)

    # Manual cross-check against the frame-free rho (Rodal is ~100% Type I).
    from warpax.energy_conditions.frame_free import certify_grid_frame_free
    ff = certify_grid_frame_free(T, g, gi)
    rho = np.asarray(ff.rho).ravel()
    w = np.asarray(vw).ravel()
    rho0 = np.where(np.isfinite(rho), rho, 0.0)
    manual = float(np.sum(w * np.minimum(rho0, 0.0)))
    assert manual == pytest.approx(exotic["E_minus_inv"], rel=1e-6, abs=1e-9)


def test_reduction_factor_self_is_one():
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    T, g, gi, _ = _curv(metric)
    peaks = {"Alcubierre": peak_proper_energy_deficit(T, g, gi)}
    # The peaks are the real content (the self-ratio is x/x for any finite
    # nonzero x). Measured: eul=0.036140687, inv=0.0011412952.
    assert peaks["Alcubierre"]["peak_deficit_eul"] > 0.0
    assert peaks["Alcubierre"]["peak_deficit_inv"] > 0.0
    rf = reduction_factors(peaks, baseline="Alcubierre")
    assert rf["Alcubierre"]["vs_Alcubierre_eul"] == 1.0
    assert rf["Alcubierre"]["vs_Alcubierre_inv"] == 1.0


def test_reduction_factors_logic():
    """Grid-free check of the ratio bookkeeping and degenerate-peak handling."""
    peaks = {
        "A": {"peak_deficit_inv": 2.0, "peak_deficit_eul": 4.0},
        "B": {"peak_deficit_inv": 1.0, "peak_deficit_eul": 2.0},
        "C": {"peak_deficit_inv": 0.0, "peak_deficit_eul": float("nan")},
    }
    rf = reduction_factors(peaks, baseline="A")
    assert rf["B"] == {"vs_A_inv": 2.0, "vs_A_eul": 2.0}
    # zero or NaN peaks have no meaningful ratio
    assert rf["C"] == {"vs_A_inv": None, "vs_A_eul": None}
