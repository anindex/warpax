"""DSGN constraints tests - bubble-size / velocity / boundedness.

Per the design specification:
- Each constraint returns a signed margin (positive => satisfied).
- Registry (`CONSTRAINT_REGISTRY`) enables string-dispatch from the
  optimizer.
- `jax.grad` flows through the `.margin` field of each result.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from warpax.design import (
    CONSTRAINT_REGISTRY,
    ConstraintResult,
    ShapeFunction,
    boundedness_constraint,
    bubble_size_constraint,
    velocity_constraint,
)


class TestDesignConstraints:
    """constraints: 6 tests per behavior spec."""

    def test_bubble_size_within_bounds(self):
        """Decaying GMM at r=max_radius yields margin > 0, satisfied=True."""
        sf = ShapeFunction.gmm(
            means=jnp.asarray([0.5]),
            widths=jnp.asarray([0.1]),
            amps=jnp.asarray([1.0]),
        )
        res = bubble_size_constraint(sf, max_radius=10.0)
        assert isinstance(res, ConstraintResult)
        assert res.name == "bubble_size"
        assert res.satisfied is True
        assert float(res.margin) > 0.0

    def test_bubble_size_exceeds_bounds(self):
        """Non-decaying Bernstein at r=max_radius yields margin < 0, satisfied=False."""
        # Constant Bernstein => f(r) ~ 1 at the r_max bound
        sf = ShapeFunction.bernstein(jnp.ones(4), r_max=10.0)
        res = bubble_size_constraint(sf, max_radius=10.0)
        assert res.satisfied is False
        assert float(res.margin) < 0.0

    def test_velocity_constraint_at_boundary(self):
        """v_s=10, max_v=10 => margin ~ 0; v_s=5, max_v=10 => margin=5."""
        res_boundary = velocity_constraint(jnp.asarray(10.0), max_v=10.0)
        assert abs(float(res_boundary.margin)) < 1e-10
        res_safe = velocity_constraint(jnp.asarray(5.0), max_v=10.0)
        assert float(res_safe.margin) == 5.0
        assert res_safe.satisfied is True

    def test_boundedness_constraint(self):
        """GMM with amps=2 (> amp_max=1) yields satisfied=False."""
        sf = ShapeFunction.gmm(
            means=jnp.asarray([0.5]),
            widths=jnp.asarray([0.1]),
            amps=jnp.asarray([2.0]),
        )
        res = boundedness_constraint(sf, amp_max=1.0)
        assert res.satisfied is False
        assert float(res.margin) < 0.0

    def test_registry_lookup(self):
        """CONSTRAINT_REGISTRY['bubble_size'] is callable and matches direct call."""
        assert "bubble_size" in CONSTRAINT_REGISTRY
        assert "velocity" in CONSTRAINT_REGISTRY
        assert "boundedness" in CONSTRAINT_REGISTRY
        sf = ShapeFunction.gmm(
            means=jnp.asarray([0.5]),
            widths=jnp.asarray([0.1]),
            amps=jnp.asarray([1.0]),
        )
        res_direct = bubble_size_constraint(sf, 10.0)
        res_registry = CONSTRAINT_REGISTRY["bubble_size"](sf, 10.0)
        assert float(res_direct.margin) == float(res_registry.margin)

    def test_jax_grad_through_constraint(self):
        """jax.grad(lambda v: velocity_constraint(v, ...).margin) returns finite gradient."""
        g = jax.grad(lambda v: velocity_constraint(v, 10.0).margin)(
            jnp.asarray(5.0)
        )
        assert jnp.isfinite(g)
