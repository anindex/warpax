"""Tests for the Garattini-Zatrimaylov de Sitter warp metric."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.metric import SymbolicMetric, sympy_metric_to_jax
from warpax.metrics import GarattiniMetric, garattini_default


class TestGarattiniBasic:
    def test_metric_finite_at_wall(self):
        m = garattini_default()
        g = m(jnp.array([0.0, 1.0, 0.3, 0.0]))
        assert g.shape == (4, 4)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_lorentzian_signature(self):
        m = garattini_default()
        g = m(jnp.array([0.0, 0.5, 0.2, 0.1]))
        assert float(jnp.linalg.det(g)) < 0.0

    def test_curvature_finite(self):
        m = garattini_default()
        c = compute_curvature_chain(m, jnp.array([0.0, 1.0, 0.3, 0.0]))
        assert bool(jnp.all(jnp.isfinite(c.stress_energy)))

    def test_float64(self):
        m = garattini_default()
        g = m(jnp.array([0.0, 0.5, 0.0, 0.0]))
        assert g.dtype == jnp.float64

    def test_jit(self):
        m = garattini_default()
        pt = jnp.array([0.0, 0.7, 0.1, 0.0])
        assert bool(jnp.allclose(m(pt), jax.jit(m.__call__)(pt), atol=1e-15))


class TestGarattiniFaithfulSymbolic:
    """Unlike Rodal, the Garattini symbolic form is a faithful closed form."""

    def test_symbolic_matches_numeric(self):
        m = garattini_default()
        sm = m.symbolic()
        subs = {
            sp.Symbol("v_s", positive=True): m.v_s,
            sp.Symbol("R", positive=True): m.R,
            sp.Symbol("sigma", positive=True): m.sigma,
            sp.Symbol("H", positive=True): m.H,
        }
        fn = sympy_metric_to_jax(SymbolicMetric(sm.coords, sm.g.subs(subs)))
        for pt in (
            jnp.array([0.0, 0.7, 0.2, 0.1]),
            jnp.array([0.0, 1.3, -0.4, 0.2]),
        ):
            assert float(jnp.max(jnp.abs(m(pt) - fn(pt)))) < 1e-12


class TestGarattiniLimits:
    def test_H_zero_reduces_to_alcubierre(self):
        g0 = GarattiniMetric(v_s=0.5, R=1.0, sigma=8.0, H=0.0)
        alc = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        pt = jnp.array([0.0, 1.0, 0.3, 0.0])
        c0 = compute_curvature_chain(g0, pt)
        ca = compute_curvature_chain(alc, pt)
        assert float(jnp.max(jnp.abs(c0.stress_energy - ca.stress_energy))) < 1e-10

    def test_matched_speed(self):
        m = GarattiniMetric.matched(R=2.0, H=0.05)
        assert np.isclose(m.v_s, 0.05 * 2.0)
