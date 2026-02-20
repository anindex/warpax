"""Tests for observer parameterization (pure JAX).

Validates:
- Orthonormal tetrad construction for Minkowski, Schwarzschild, Alcubierre metrics
- Rapidity-parameterized timelike vectors satisfy g_{ab} u^a u^b = -1
- Null vectors satisfy g_{ab} k^a k^b = 0
- Sigmoid reparameterization roundtrip
- JIT and vmap compatibility
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.energy_conditions.observer import (
    bounded_param,
    compute_orthonormal_tetrad,
    null_from_angles,
    timelike_from_rapidity,
    unbounded_param,
)

# Ensure float64 is active
assert jnp.array(1.0).dtype == jnp.float64, "Float64 not enabled"

# Flat Minkowski metric: eta = diag(-1, 1, 1, 1)
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _orthonormality_check(tetrad: jnp.ndarray, g_ab: jnp.ndarray) -> jnp.ndarray:
    """Compute g_{ab} e_I^a e_J^b and return the deviation from eta_{IJ}."""
    # tetrad[I, a], g_ab[a, b] -> product[I, J] = g_{ab} e_I^a e_J^b
    product = jnp.einsum("Ia,ab,Jb->IJ", tetrad, g_ab, tetrad)
    return product - ETA


# =========================================================================
# Tetrad tests
# =========================================================================


class TestTetradMinkowski:
    """Tetrad construction for Minkowski (flat) metric."""

    def test_minkowski_tetrad_is_identity(self):
        """For flat metric diag(-1,1,1,1), tetrad should be identity."""
        g = ETA
        tetrad = compute_orthonormal_tetrad(g)
        np.testing.assert_allclose(tetrad, jnp.eye(4), atol=1e-14)

    def test_minkowski_orthonormality(self):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Minkowski."""
        g = ETA
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-14)


class TestTetradSchwarzschild:
    """Tetrad construction for Schwarzschild metric at r=3M."""

    @pytest.fixture
    def schwarzschild_metric_at_3M(self):
        """Schwarzschild metric evaluated at r_iso=3M (x=3, y=0, z=0)."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        return metric(coords)

    def test_schwarzschild_orthonormality(self, schwarzschild_metric_at_3M):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Schwarzschild at r=3M."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)

    def test_schwarzschild_e0_timelike(self, schwarzschild_metric_at_3M):
        """e_0 should be timelike: g_{ab} e_0^a e_0^b = -1."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        norm = jnp.einsum("a,ab,b->", tetrad[0], g, tetrad[0])
        np.testing.assert_allclose(norm, -1.0, atol=1e-12)

    def test_schwarzschild_spatial_spacelike(self, schwarzschild_metric_at_3M):
        """e_1, e_2, e_3 should be spacelike: g_{ab} e_i^a e_i^b = +1."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        for i in range(1, 4):
            norm = jnp.einsum("a,ab,b->", tetrad[i], g, tetrad[i])
            np.testing.assert_allclose(norm, 1.0, atol=1e-12)


class TestTetradAlcubierre:
    """Tetrad construction for Alcubierre metric inside the bubble."""

    @pytest.fixture
    def alcubierre_metric_inside(self):
        """Alcubierre metric at a point inside the bubble (near center)."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        # Inside the bubble: near center
        coords = jnp.array([0.0, 0.1, 0.1, 0.0])
        return metric(coords)

    def test_alcubierre_orthonormality(self, alcubierre_metric_inside):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Alcubierre."""
        g = alcubierre_metric_inside
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)


class TestTetradJITVmap:
    """JIT and vmap compatibility for tetrad construction."""

    def test_jit_compilation(self):
        """jax.jit(compute_orthonormal_tetrad) runs without error."""
        g = ETA
        jit_fn = jax.jit(compute_orthonormal_tetrad)
        tetrad = jit_fn(g)
        np.testing.assert_allclose(tetrad, jnp.eye(4), atol=1e-14)

    def test_vmap_batch(self):
        """jax.vmap(compute_orthonormal_tetrad) over batch of metrics."""
        metrics = jnp.stack([
            ETA,
            SchwarzschildMetric(M=1.0)(jnp.array([0.0, 3.0, 0.0, 0.0])),
            ETA,
        ])
        vmap_fn = jax.vmap(compute_orthonormal_tetrad)
        tetrads = vmap_fn(metrics)
        assert tetrads.shape == (3, 4, 4)
        # Each should be orthonormal
        for i in range(3):
            deviation = _orthonormality_check(tetrads[i], metrics[i])
            np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)


# =========================================================================
# Timelike vector tests
# =========================================================================


class TestTimelikeFromRapidity:
    """Timelike vector construction from rapidity parameters."""

    @pytest.fixture
    def flat_tetrad(self):
        """Tetrad for flat Minkowski metric."""
        return compute_orthonormal_tetrad(ETA)

    def test_eulerian_observer_zeta_zero(self, flat_tetrad):
        """zeta=0: u^a = e_0^a (Eulerian observer)."""
        u = timelike_from_rapidity(
            jnp.float64(0.0), jnp.float64(0.5), jnp.float64(0.3), flat_tetrad
        )
        np.testing.assert_allclose(u, flat_tetrad[0], atol=1e-14)

    def test_boosted_observer_unit_norm(self, flat_tetrad):
        """zeta=1.0, various (theta, phi): g_{ab} u^a u^b = -1."""
        zeta = jnp.float64(1.0)
        for theta_val in [0.3, 0.7, 1.2, 2.5]:
            for phi_val in [0.0, 1.0, 3.14, 5.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", u, ETA, u)
                np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_high_rapidity_unit_norm(self, flat_tetrad):
        """zeta=5.0 (Lorentz factor ~74): still unit norm."""
        zeta = jnp.float64(5.0)
        theta = jnp.float64(0.5)
        phi = jnp.float64(0.3)
        u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
        norm = jnp.einsum("a,ab,b->", u, ETA, u)
        np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_angular_coverage(self, flat_tetrad):
        """Sweep theta/phi: all produce unit timelike vectors."""
        thetas = [0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4, jnp.pi]
        phis = [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
        zeta = jnp.float64(1.5)
        for theta_val in thetas:
            for phi_val in phis:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", u, ETA, u)
                np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_curved_spacetime_unit_norm(self):
        """Boosted observer in Schwarzschild: g_{ab} u^a u^b = -1."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        g = metric(coords)
        tetrad = compute_orthonormal_tetrad(g)
        u = timelike_from_rapidity(
            jnp.float64(2.0), jnp.float64(0.7), jnp.float64(1.2), tetrad
        )
        norm = jnp.einsum("a,ab,b->", u, g, u)
        np.testing.assert_allclose(norm, -1.0, atol=1e-12)


# =========================================================================
# Null vector tests
# =========================================================================


class TestNullFromAngles:
    """Null vector construction from angular parameters."""

    @pytest.fixture
    def flat_tetrad(self):
        """Tetrad for flat Minkowski metric."""
        return compute_orthonormal_tetrad(ETA)

    def test_null_norm_flat(self, flat_tetrad):
        """For various (theta, phi), g_{ab} k^a k^b = 0 in flat spacetime."""
        for theta_val in [0.0, 0.5, 1.0, 1.5, jnp.pi]:
            for phi_val in [0.0, 0.5, 1.0, 2.0, 3.14, 5.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                k = null_from_angles(theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", k, ETA, k)
                np.testing.assert_allclose(norm, 0.0, atol=1e-14)

    def test_null_in_schwarzschild(self):
        """Null vector in Schwarzschild: g_{ab} k^a k^b = 0."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        g = metric(coords)
        tetrad = compute_orthonormal_tetrad(g)
        for theta_val in [0.3, 1.0, 2.5]:
            for phi_val in [0.0, 1.5, 4.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                k = null_from_angles(theta, phi, tetrad)
                norm = jnp.einsum("a,ab,b->", k, g, k)
                np.testing.assert_allclose(norm, 0.0, atol=1e-12)


# =========================================================================
# Sigmoid parameterization tests
# =========================================================================


class TestBoundedParam:
    """Sigmoid reparameterization tests."""

    def test_bounded_range(self):
        """bounded_param(raw, 0, 5) always in (0, 5) for various raw values."""
        raw_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        for raw in raw_values:
            result = bounded_param(raw, jnp.float64(0.0), jnp.float64(5.0))
            assert float(result) > 0.0, f"bounded_param({float(raw)}) = {float(result)} <= 0"
            assert float(result) < 5.0, f"bounded_param({float(raw)}) = {float(result)} >= 5"

    def test_roundtrip(self):
        """unbounded_param(bounded_param(raw, lo, hi), lo, hi) = raw."""
        raw_values = jnp.array([-3.0, -1.0, 0.0, 0.5, 2.0])
        lo = jnp.float64(0.0)
        hi = jnp.float64(5.0)
        for raw in raw_values:
            bounded = bounded_param(raw, lo, hi)
            recovered = unbounded_param(bounded, lo, hi)
            np.testing.assert_allclose(float(recovered), float(raw), atol=1e-12)

    def test_gradient_at_zero(self):
        """Gradient of bounded_param at raw=0 is maximal: sigmoid'(0) * range = 0.25 * 5 = 1.25."""
        grad_fn = jax.grad(lambda r: bounded_param(r, jnp.float64(0.0), jnp.float64(5.0)))
        grad_at_zero = grad_fn(jnp.float64(0.0))
        # sigmoid'(0) = 0.25, range = 5, so gradient = 1.25
        np.testing.assert_allclose(float(grad_at_zero), 1.25, atol=1e-14)

    def test_dtype_float64(self):
        """Output should be float64."""
        result = bounded_param(jnp.float64(0.0), jnp.float64(0.0), jnp.float64(5.0))
        assert result.dtype == jnp.float64
