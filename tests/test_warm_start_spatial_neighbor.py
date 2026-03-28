"""TestSpatialNeighborWarmStart - warm-start kwarg contract tests.

Scope of this test suite:
These tests cover the API surface (kwargs + validation).
The spatial-neighbor pool composition helper (which
would swap 1-of-16 starts with the neighbor grid point's previous
worst-observer) requires grid-aware state plumbing through
``_solve_multistart_3d``. These tests therefore pin the CURRENT contract:

- default ``warm_start='cold'`` is bit-exact to the v0.1.x path
- ``warm_start='spatial_neighbor'`` is an accepted string value (no
  ValueError) and currently dispatches identically to cold
- invalid values raise ValueError with verbatim message
- ``neighbor_fraction`` range validation (0 < f <= 1)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from warpax.energy_conditions import (
    optimize_dec,
    optimize_nec,
    optimize_sec,
    optimize_wec,
)


class TestSpatialNeighborWarmStart:
    """warm_start kwarg contract tests (v1.1+ API surface)."""

    def _bench_inputs(self):
        """Standard Minkowski-like inputs that exercise all 4 optimize_*."""
        g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        key = jax.random.PRNGKey(42)
        return T, g, key

    def test_default_cold_preserves_v10_bit_exact(self):
        """Default + explicit warm_start='cold' are bit-identical to v1.0."""
        T, g, key = self._bench_inputs()

        for fn in [optimize_wec, optimize_sec, optimize_dec]:
            r_default = fn(T, g, key=key)
            r_explicit = fn(T, g, warm_start="cold", key=key)
            assert jnp.array_equal(r_default.margin, r_explicit.margin), (
                f"{fn.__name__} drifted under warm_start='cold' explicit call"
            )

        # NEC (2D) - check margin bit-exactness
        r_default_nec = optimize_nec(T, g, key=key)
        r_explicit_nec = optimize_nec(T, g, warm_start="cold", key=key)
        assert jnp.array_equal(r_default_nec.margin, r_explicit_nec.margin)

    def test_spatial_neighbor_accepted(self):
        """warm_start='spatial_neighbor' is accepted and returns a valid OptimizationResult.

        Currently, spatial_neighbor dispatches identically
        to cold (the pool-composition helper is in a separate module). The test pins
        that the string value is accepted and the call returns normally.
        """
        T, g, key = self._bench_inputs()
        r = optimize_wec(T, g, warm_start="spatial_neighbor", key=key)
        assert r.margin is not None
        assert r.worst_observer.shape == (4,)

    def test_invalid_warm_start_raises(self):
        """warm_start='bad' raises ValueError with verbatim message."""
        T, g, _ = self._bench_inputs()
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_wec(T, g, warm_start="bad")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_nec(T, g, warm_start="cold_start")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_sec(T, g, warm_start="warm")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_dec(T, g, warm_start="")

    def test_neighbor_fraction_validates(self):
        """neighbor_fraction must satisfy 0 < f <= 1; other values raise."""
        T, g, _ = self._bench_inputs()
        for bad in [-0.5, 0.0, 1.5, 2.0]:
            with pytest.raises(ValueError, match="neighbor_fraction must satisfy"):
                optimize_wec(T, g, neighbor_fraction=bad)

    def test_default_neighbor_fraction_accepted(self):
        """Default neighbor_fraction=1/16 is explicitly accepted."""
        T, g, key = self._bench_inputs()
        r = optimize_wec(T, g, neighbor_fraction=1.0 / 16.0, key=key)
        assert r.margin is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
