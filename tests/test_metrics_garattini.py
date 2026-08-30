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


class TestPublishedConstruction:
    """The metric must be the one in arXiv:2502.13153, not an Alcubierre look-alike.

    The published bubble is shift-only on FLAT slices, beta^i = -(1-f) x^i / L - f v^i
    with L = 1/H, and it co-moves with the Hubble flow, r(t) = r_0 e^{Ht}. Under that
    matching the last two terms cancel and the shift becomes a sum of two radial
    gradients, hence irrotational, which is the property the paper's positive-energy
    claim rests on. An earlier implementation used gamma_ij = e^{2Ht} delta_ij with an
    Alcubierre shift on a constant-velocity centre; that shift has |curl beta| ~ 0.29
    at a generic wall point and reported Type-IV wall structure the real construction
    does not have.
    """

    def test_spatial_slices_are_flat(self):
        m = GarattiniMetric.matched(R=1.0, sigma=8.0, H=0.1)
        for t in (0.0, 1.7):
            gam = m.spatial_metric(jnp.array([t, 0.9, 0.2, 0.1]))
            assert float(jnp.max(jnp.abs(gam - jnp.eye(3)))) == 0.0

    def test_shift_is_irrotational_under_matching(self):
        import jax

        m = GarattiniMetric.matched(R=1.0, sigma=8.0, H=0.1)
        for t in (0.0, 1.0):
            jac = jax.jacfwd(lambda p: m.shift(jnp.array([t, p[0], p[1], p[2]])))
            for pt in ([1.5, 0.3, 0.2], [1.0, 0.0, 0.0], [1.7, -0.4, 0.6]):
                j = np.asarray(jac(jnp.array(pt)))
                curl = np.array([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])
                assert float(np.linalg.norm(curl)) < 1e-12, f"t={t} x={pt}"

    def test_wall_is_type_i_with_vanishing_momentum(self):
        from warpax.energy_conditions.classification import classify_hawking_ellis

        m = GarattiniMetric.matched(R=1.0, sigma=8.0, H=0.1)
        # A point on the bubble wall: the centre sits at r_0 = v_s / H = 1 at t = 0.
        pt = jnp.array([0.0, 2.0, 0.0, 0.0])
        c = compute_curvature_chain(m, pt)
        T, gi, g = c.stress_energy, c.metric_inv, c.metric
        n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
        n_up = gi @ n_low
        n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
        j = np.array([float(sum(n_up[a] * T[a, i + 1] for a in range(4))) for i in range(3)])
        assert float(np.linalg.norm(j)) < 1e-12, "irrotational shift carries no momentum"
        assert int(classify_hawking_ellis(gi @ T, g).he_type) == 1
