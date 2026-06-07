"""Tests for frame-independent, all-velocity EC certification (Contribution 3 engine)."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.frame_free import (
    certify_grid_frame_free,
    certify_point_frame_free,
    type_fractions,
)
from warpax.energy_conditions.verifier import verify_grid
from warpax.geometry import evaluate_curvature_grid
from warpax.grids import wall_clustered
from warpax.metrics import RodalMetric

MINKOWSKI = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def test_minkowski_vacuum_is_type_i():
    """Vacuum (T=0) in Minkowski classifies Type I with ~zero margins."""
    T = jnp.zeros((4, 4))
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    for k in ("nec", "wec", "sec", "dec"):
        assert abs(float(out[k])) < 1e-8


def test_perfect_fluid_margins_exact():
    """Type-I perfect fluid: margins equal the eigenvalue inequalities."""
    rho, p = 1.0, 0.5
    T = jnp.diag(jnp.array([rho, p, p, p]))  # T_{ab}, energy density T_tt=rho
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    assert float(out["rho"]) == pytest.approx(rho, abs=1e-9)
    assert float(out["nec"]) == pytest.approx(rho + p, abs=1e-9)  # 1.5
    assert float(out["wec"]) == pytest.approx(min(rho, rho + p), abs=1e-9)  # 1.0
    assert float(out["dec"]) == pytest.approx(rho - abs(p), abs=1e-9)  # 0.5
    assert float(out["sec"]) == pytest.approx(min(rho + p, rho + 3 * p), abs=1e-9)


def test_negative_pressure_violates():
    """rho=1, p=-2 violates NEC/WEC/DEC; margins are negative."""
    rho, p = 1.0, -2.0
    T = jnp.diag(jnp.array([rho, p, p, p]))
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    assert float(out["nec"]) < 0
    assert float(out["wec"]) < 0
    assert float(out["dec"]) < 0


def test_type_iv_yields_nan_margins():
    """A complex-eigenvalue T^a_b is Type IV with no invariant rest-frame margin."""
    # T_{ab} with a (t,x) block giving |T_tx| > |T_tt+T_xx|/2 -> complex pair.
    T = jnp.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    out = certify_point_frame_free(T, MINKOWSKI, solver="standard")
    assert int(out["he_type"]) == 4
    assert np.isnan(float(out["nec"]))
    assert np.isnan(float(out["wec"]))
    assert np.isnan(float(out["dec"]))


def test_grid_matches_verify_grid_typeI_margins():
    """On Rodal (100% Type I, v_s=0.5) the frame-free Type-I margins match the
    validated verify_grid eigenvalue branch."""
    metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (24, 24, 24), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)

    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    ec = verify_grid(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=1)

    he = np.asarray(ff.he_types).ravel()
    typeI = he == 1.0
    assert typeI.mean() > 0.99  # Rodal is ~100% Type I

    nec_ff = np.asarray(ff.nec_margins).ravel()[typeI]
    nec_vg = np.asarray(ec.nec_margins).ravel()[typeI]
    # NEC/WEC/SEC are algebraic-exact in both paths -> should match closely.
    np.testing.assert_allclose(nec_ff, nec_vg, atol=1e-8, rtol=1e-6)


def test_runs_superluminal_no_nan_types():
    """The frame-free engine runs at v_s=1.5 (Eulerian normal undefined) and
    returns finite Hawking-Ellis types with Type-IV present at the wall."""
    metric = AlcubierreMetric(v_s=1.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (24, 24, 24), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    he = np.asarray(ff.he_types)
    assert np.all(np.isfinite(he))
    assert ff.n_type_iv > 0  # Alcubierre wall is Type IV, even superluminally
    fr = type_fractions(ff)
    assert abs(sum(fr[f"frac_type_{k}"] for k in ("i", "ii", "iii", "iv")) - 1.0) < 1e-6
