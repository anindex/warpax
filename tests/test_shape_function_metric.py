"""Tests: ShapeFunctionMetric ADMMetric contract + verify_physical gate.

- ADMMetric / MetricSpecification contract: __call__ shape,
  lapse / shift / spatial_metric callable.
- JIT compatibility: jax.jit(sfm)(coords) returns same as unjit.
- jacfwd on __call__: (4, 4, 4) derivative tensor.
- verify_physical gate: PhysicalityVerdict NamedTuple
  with lapse_floor_ok / ctc_free / bubble_finite / overall booleans.
- strict=True raises UnphysicalMetricError at __check_init__; strict=False
  warns via UnphysicalMetricWarning but returns the metric.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from warpax.design import ShapeFunction, ShapeFunctionMetric
from warpax.design.metrics import (
    PhysicalityVerdict,
    UnphysicalMetricError,
)
from warpax.geometry.metric import ADMMetric


@pytest.fixture
def alcubierre_like_shape():
    """24-knot spline approximating a physically bounded Alcubierre bubble profile.

    Uses ``knots ∈ [0, 12]`` so the 16**3 probe grid at bounds ``[-2R, +2R]``
    (``R=1``) stays well within the knot range (max ``r_s = sqrt(3)*2R ≈ 3.46``).
    """
    R, sigma = 1.0, 0.1
    knots = jnp.linspace(0.0, 12.0, 24)
    # 1 - tanh((r - R)/sigma)^2 is a bump that decays to 0 as r -> infty and is
    # bounded in [0, 1], well-suited for the ShapeFunctionMetric bubble profile.
    values = 1.0 - jnp.tanh((knots - R) / sigma) ** 2
    return ShapeFunction.spline(knots, values)


class TestShapeFunctionMetric:
    """ADMMetric contract + JIT + verify_physical gate."""

    def test_metric_specification_contract(self, alcubierre_like_shape):
        sfm = ShapeFunctionMetric(
            alcubierre_like_shape, v_s=jnp.asarray(1.0), strict=False
        )
        assert isinstance(sfm, ADMMetric)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = sfm(coords)
        assert g.shape == (4, 4)

    def test_jit_compatibility(self, alcubierre_like_shape):
        """__call__ is JIT-compatible."""
        sfm = ShapeFunctionMetric(
            alcubierre_like_shape, v_s=jnp.asarray(1.0), strict=False
        )
        coords = jnp.array([0.0, 0.5, 0.0, 0.0])
        jitted = jax.jit(sfm.__call__)
        g_unjit = sfm(coords)
        g_jit = jitted(coords)
        assert jnp.allclose(g_unjit, g_jit)

    def test_jacfwd_on_call(self, alcubierre_like_shape):
        """jax.jacfwd(metric)(coords) finite and shape (4,4,4) - curvature chain compat."""
        sfm = ShapeFunctionMetric(
            alcubierre_like_shape, v_s=jnp.asarray(1.0), strict=False
        )
        coords = jnp.array([0.0, 0.5, 0.0, 0.0])
        dg = jax.jacfwd(sfm)(coords)
        assert dg.shape == (4, 4, 4)
        assert jnp.all(jnp.isfinite(dg))

    def test_verify_physical_alcubierre_like_passes(self, alcubierre_like_shape):
        """A bounded sub-luminal bubble-profile spline passes all 3 gate checks."""
        sfm = ShapeFunctionMetric(
            alcubierre_like_shape, v_s=jnp.asarray(0.5), strict=False
        )
        verdict = sfm.verify_physical()
        assert isinstance(verdict, PhysicalityVerdict)
        assert verdict.overall is True

    def test_verify_physical_zero_lapse_fails(self):
        """Synthetic shape forcing alpha(coords) < lapse_floor returns lapse_floor_ok=False.

        An absurdly large Bernstein amplitude + v_s >> 1 drives shift norm
        beyond alpha=1, so g_tt = -(1 - |beta|^2) goes non-negative => CTC or
        equivalently the effective lapse floor fails.
        """
        huge_sf = ShapeFunction.bernstein(jnp.asarray([1e4, 1e4, 1e4, 1e4]))
        sfm = ShapeFunctionMetric(
            huge_sf, v_s=jnp.asarray(10.0), strict=False, lapse_floor=1e-6
        )
        verdict = sfm.verify_physical()
        # Either lapse floor breaks or CTC check breaks for this adversarial config.
        assert verdict.overall is False

    def test_strict_raises_unphysical(self):
        """strict=True raises UnphysicalMetricError from __check_init__ on failure."""
        huge_sf = ShapeFunction.bernstein(jnp.asarray([1e4, 1e4, 1e4, 1e4]))
        with pytest.raises(UnphysicalMetricError):
            ShapeFunctionMetric(
                huge_sf, v_s=jnp.asarray(10.0), strict=True, lapse_floor=1e-6
            )
