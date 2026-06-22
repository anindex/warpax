"""Tests for the vorticity -> Type-IV analytic mechanism (f = kappa * omega)."""
from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from warpax.analysis.shift_kinematics import compute_shift_kinematics
from warpax.analysis.vorticity_type_analytic import (
    excess_over_pure_rotation,
    fit_kappa,
    imaginary_part_estimate,
    typeIV_threshold,
)
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.metric import ADMMetric, SymbolicMetric
from warpax.metrics import NatarioMetric


class _RotationShift(ADMMetric):
    """Pure-rotation shift: zero expansion/shear, tunable vorticity."""

    c: float = 0.1
    w: float = 1.0

    def lapse(self, coords):
        return jnp.array(1.0)

    def shift(self, coords):
        t, x, y, z = coords
        env = jnp.exp(-(x * x + y * y + z * z) / (2.0 * self.w * self.w))
        return jnp.array([-self.c * y * env, self.c * x * env, 0.0])

    def spatial_metric(self, coords):
        return jnp.eye(3)

    def shape_function_value(self, coords):
        t, x, y, z = coords
        return jnp.exp(-(x * x + y * y + z * z) / (2.0 * self.w * self.w))

    def symbolic(self):
        t, x, y, z = sp.symbols("t x y z")
        g = sp.eye(4)
        g[0, 0] = -1
        return SymbolicMetric([t, x, y, z], g)

    def name(self):
        return "RotationShift"


class TestEstimator:
    def test_estimator_sanity(self):
        # f = kappa*omega: zero at zero vorticity, monotone in omega
        vals = [imaginary_part_estimate(w, kappa=0.06) for w in (0.0, 0.1, 0.2, 0.4)]
        assert vals[0] == 0.0
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_fit_recovers_kappa(self):
        omega = np.array([0.05, 0.1, 0.2, 0.4])
        imag = 0.0597 * omega
        fit = fit_kappa(omega, imag)
        assert np.isclose(fit["kappa"], 0.0597, rtol=1e-6)
        assert fit["r_squared"] > 0.999999

    def test_threshold_positive(self):
        assert typeIV_threshold(kappa=0.06) > 0.0
        assert typeIV_threshold(kappa=0.0) == float("inf")


class TestMechanism:
    """The physics: pure vorticity flips Type I -> Type IV, linearly in omega."""

    def _omega_imag_type(self, c):
        pt = jnp.array([0.0, 0.5, 0.5, 0.0])
        m = _RotationShift(c=c, w=1.0)
        _, _, omega_sq = compute_shift_kinematics(m, pt)
        cur = compute_curvature_chain(m, pt)
        cls = classify_hawking_ellis(cur.metric_inv @ cur.stress_energy, cur.metric)
        return (float(np.sqrt(max(float(omega_sq), 0.0))),
                float(jnp.max(jnp.abs(cls.eigenvalues_imag))),
                int(cls.he_type))

    def test_irrotational_is_type_i(self):
        omega, imag, he = self._omega_imag_type(0.0)
        assert omega == 0.0
        assert imag < 1e-10
        assert he == 1

    def test_rotational_is_type_iv(self):
        omega, imag, he = self._omega_imag_type(0.2)
        assert omega > 0.0
        assert imag > 1e-6
        assert he == 4

    def test_imag_is_linear_in_vorticity(self):
        omegas, imags = [], []
        for c in (0.05, 0.1, 0.2, 0.3, 0.4):
            om, im, _ = self._omega_imag_type(c)
            omegas.append(om)
            imags.append(im)
        fit = fit_kappa(np.array(omegas), np.array(imags))
        # f = kappa * omega holds to ~machine precision for pure rotation.
        assert fit["r_squared"] > 0.9999
        assert fit["kappa"] > 0.0


class TestExcessRatio:
    def test_exact_value(self):
        assert excess_over_pure_rotation(0.12, 0.5, 0.06) == pytest.approx(4.0)

    def test_irrotational_returns_none(self):
        assert excess_over_pure_rotation(0.1, 0.0, 0.06) is None
        assert excess_over_pure_rotation(0.1, 5e-14, 0.06) is None

    def test_full_metric_exceeds_pure_rotation_slope(self):
        """Physics sentinel: the high-shear Natario wall sits well above the
        pure-rotation prediction (measured excess ~32x; 2x is a loose floor)."""
        omegas, imags = [], []
        for c in (0.1, 0.2, 0.4):
            pt = jnp.array([0.0, 0.5, 0.5, 0.0])
            m = _RotationShift(c=c, w=1.0)
            _, _, omega_sq = compute_shift_kinematics(m, pt)
            cur = compute_curvature_chain(m, pt)
            cls = classify_hawking_ellis(
                cur.metric_inv @ cur.stress_energy, cur.metric)
            omegas.append(float(np.sqrt(max(float(omega_sq), 0.0))))
            imags.append(float(jnp.max(jnp.abs(cls.eigenvalues_imag))))
        kappa = fit_kappa(np.array(omegas), np.array(imags))["kappa"]

        pt = jnp.array([0.0, 1.0, 0.3, 0.0])  # wall sample
        nat = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
        _, _, omega_sq = compute_shift_kinematics(nat, pt)
        cur = compute_curvature_chain(nat, pt)
        cls = classify_hawking_ellis(cur.metric_inv @ cur.stress_energy, cur.metric)
        omega = float(np.sqrt(max(float(omega_sq), 0.0)))
        imag = float(jnp.max(jnp.abs(cls.eigenvalues_imag)))
        assert int(cls.he_type) == 4
        assert imag > 2.0 * kappa * omega

    def test_cached_cross_metric_excess(self):
        path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "vorticity_type_analytic.json")
        if not os.path.exists(path):
            pytest.skip("results/vorticity_type_analytic.json not present")
        cross = json.load(open(path))["cross_metric"]
        assert cross["Rodal"]["imag_ratio"] is None
        vortical = ["Van den Broeck", "Alcubierre", "Natário"]
        for name in vortical:
            assert cross[name]["imag_ratio"] > 1.0
            assert cross[name]["sigma"] > 0.0
        # The excess over kappa*omega orders with the shear-to-vorticity ratio.
        ratios = [cross[n]["imag_ratio"] for n in vortical]
        shears = [cross[n]["shear_to_vorticity"] for n in vortical]
        assert ratios == sorted(ratios)
        assert shears == sorted(shears)


def test_fit_kappa_uncentered_r2_with_scatter():
    """For a through-origin fit, R^2 must use the uncentered total sum of
    squares (baseline 0), not the mean. On proportional-plus-noise data the
    uncentered R^2 (~0.999) exceeds the mean-centered value (~0.996); this
    pins the correct no-intercept convention and would fail with the old
    mean-centered formula."""
    omega = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    imag = 0.06 * omega + 5e-4 * np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    fit = fit_kappa(omega, imag)
    assert fit["r_squared"] > 0.999
