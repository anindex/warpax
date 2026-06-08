"""Tests for the vorticity -> Type-IV analytic mechanism (f = kappa * omega)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import sympy as sp

from warpax.analysis.shift_kinematics import compute_shift_kinematics
from warpax.analysis.vorticity_type_analytic import (
    fit_kappa,
    imaginary_part_estimate,
    typeIV_threshold,
)
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.metric import ADMMetric, SymbolicMetric


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
    def test_zero_vorticity_zero_imag(self):
        assert imaginary_part_estimate(0.0, kappa=0.06) == 0.0

    def test_monotone_in_omega(self):
        k = 0.06
        vals = [imaginary_part_estimate(w, k) for w in (0.0, 0.1, 0.2, 0.4)]
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
