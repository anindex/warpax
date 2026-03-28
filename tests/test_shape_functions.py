"""Tests for shape_function_value across all metrics.

Verifies the unified shape function API introduced by - Warp metrics return ~1.0 at bubble center, ~0.0 in far field
- Benchmark metrics return exactly 0.0 everywhere
- All implementations are JIT-compatible and return float64 scalars
- Natario uses Alcubierre convention (NOT inverted n(r))
- Lentz uses L1 (Manhattan) distance geometry
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)

# ---------------------------------------------------------------------------
# Shared test coordinates
# ---------------------------------------------------------------------------

ORIGIN = jnp.array([0.0, 0.0, 0.0, 0.0])
FAR_FIELD = jnp.array([0.0, 100.0, 0.0, 0.0])

WARP_METRICS = [
    RodalMetric(),
    NatarioMetric(),
    LentzMetric(),
    VanDenBroeckMetric(),
    WarpShellMetric(),
    AlcubierreMetric(),
]

ALL_METRICS = WARP_METRICS + [MinkowskiMetric(), SchwarzschildMetric()]

WARP_METRIC_IDS = [m.name() for m in WARP_METRICS]
ALL_METRIC_IDS = [m.name() for m in ALL_METRICS]


# ---------------------------------------------------------------------------
# Origin value tests
# ---------------------------------------------------------------------------


class TestShapeFunctionOrigin:
    """Warp metrics should return ~1.0 at the bubble center."""

    @pytest.mark.parametrize("metric", WARP_METRICS, ids=WARP_METRIC_IDS)
    def test_origin_value(self, metric):
        f = metric.shape_function_value(ORIGIN)
        npt.assert_allclose(float(f), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Far-field value tests
# ---------------------------------------------------------------------------


class TestShapeFunctionFarField:
    """Warp metrics should return ~0.0 far from the bubble."""

    @pytest.mark.parametrize("metric", WARP_METRICS, ids=WARP_METRIC_IDS)
    def test_far_field_value(self, metric):
        # Use a coordinate well outside any bubble radius
        far = jnp.array([0.0, 1000.0, 0.0, 0.0])
        f = metric.shape_function_value(far)
        npt.assert_allclose(float(f), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Benchmark zero tests
# ---------------------------------------------------------------------------


class TestBenchmarkZero:
    """Non-warp metrics return identically 0.0."""

    def test_minkowski_zero(self):
        m = MinkowskiMetric()
        f = m.shape_function_value(ORIGIN)
        assert float(f) == 0.0

    def test_schwarzschild_zero(self):
        m = SchwarzschildMetric()
        f = m.shape_function_value(ORIGIN)
        assert float(f) == 0.0

    def test_minkowski_zero_arbitrary(self):
        """Minkowski returns 0.0 at arbitrary coordinates."""
        m = MinkowskiMetric()
        coords = jnp.array([5.0, 10.0, -3.0, 7.0])
        f = m.shape_function_value(coords)
        assert float(f) == 0.0

    def test_schwarzschild_zero_arbitrary(self):
        """Schwarzschild returns 0.0 at arbitrary coordinates."""
        m = SchwarzschildMetric()
        coords = jnp.array([5.0, 10.0, -3.0, 7.0])
        f = m.shape_function_value(coords)
        assert float(f) == 0.0


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------


class TestShapeFunctionDtype:
    """All shape functions must return float64."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_float64(self, metric):
        f = metric.shape_function_value(ORIGIN)
        assert f.dtype == jnp.float64, f"{metric.name()} dtype: {f.dtype}"


# ---------------------------------------------------------------------------
# Scalar shape tests
# ---------------------------------------------------------------------------


class TestShapeFunctionShape:
    """All shape functions must return a scalar (shape == )."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_scalar_shape(self, metric):
        f = metric.shape_function_value(ORIGIN)
        assert f.shape == (), f"{metric.name()} shape: {f.shape}"


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestShapeFunctionJIT:
    """JIT-compiled shape functions must match eager evaluation."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_jit_compat(self, metric):
        coords = jnp.array([0.0, 1.0, 2.0, 0.5])
        f_eager = metric.shape_function_value(coords)
        f_jit = jax.jit(metric.shape_function_value)(coords)
        npt.assert_allclose(float(f_jit), float(f_eager), atol=1e-15)


# ---------------------------------------------------------------------------
# Wall region tests
# ---------------------------------------------------------------------------


class TestShapeFunctionWallRegion:
    """Shape function in the wall transition region produces intermediate values."""

    def test_rodal_wall_region(self):
        """Rodal at r ~ R should give 0.1 < f < 0.9."""
        m = RodalMetric()  # R=100.0, sigma=0.03
        # Place point at approximately r = R = 100
        coords = jnp.array([0.0, 100.0, 0.0, 0.0])
        f = float(m.shape_function_value(coords))
        assert 0.1 < f < 0.9, f"Expected wall region value, got {f}"


# ---------------------------------------------------------------------------
# Natario convention correctness
# ---------------------------------------------------------------------------


class TestNatarioConvention:
    """Natario must use Alcubierre convention (1 inside, 0 outside), not n(r)."""

    def test_natario_not_inverted(self):
        """Origin value > 0.5 confirms Alcubierre convention, not n(r)."""
        m = NatarioMetric()
        f = float(m.shape_function_value(ORIGIN))
        assert f > 0.5, (
            f"Natario origin value {f} <= 0.5, "
            "suggesting inverted n(r) convention instead of Alcubierre"
        )

    def test_natario_matches_alcubierre_convention(self):
        """Natario shape function should be close to 1.0 at origin."""
        m = NatarioMetric()
        f = float(m.shape_function_value(ORIGIN))
        npt.assert_allclose(f, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Lentz L1 geometry tests
# ---------------------------------------------------------------------------


class TestLentzL1Geometry:
    """Lentz shape function uses L1 (Manhattan) distance, not Euclidean."""

    def test_l1_axis_symmetry(self):
        """Points at equal L1 distance along different axes give same value.

        For axis-aligned points: (R/2, 0, 0) and (0, R/2, 0) have the
        same L1 distance d = R/2. They should produce identical shape
        function values.
        """
        m = LentzMetric()  # R=100.0
        half_R = m.R / 2.0
        coords_x = jnp.array([0.0, half_R, 0.0, 0.0])
        coords_y = jnp.array([0.0, 0.0, half_R, 0.0])
        f_x = float(m.shape_function_value(coords_x))
        f_y = float(m.shape_function_value(coords_y))
        npt.assert_allclose(f_x, f_y, atol=1e-12)

    def test_l1_diagonal_different(self):
        """Diagonal point has larger L1 distance than axis-aligned point.

        Point (R/2, R/2, 0) has L1 distance R, while (R/2, 0, 0) has
        L1 distance R/2. The diagonal point should have a smaller shape
        function value (closer to exterior).
        """
        m = LentzMetric()  # R=100.0
        half_R = m.R / 2.0
        coords_axis = jnp.array([0.0, half_R, 0.0, 0.0])
        coords_diag = jnp.array([0.0, half_R, half_R, 0.0])
        f_axis = float(m.shape_function_value(coords_axis))
        f_diag = float(m.shape_function_value(coords_diag))
        assert f_axis > f_diag, (
            f"L1 geometry violated: axis f={f_axis} should be > diagonal f={f_diag}"
        )
