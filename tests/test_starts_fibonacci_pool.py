"""TestStartsFibonacciPool - starts kwarg contract tests.

API surface and validation tests.
Fibonacci lattice + BFGS-top-k pool composition helper is implemented
in a separate module.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions import (
    optimize_dec,
    optimize_nec,
    optimize_sec,
    optimize_wec,
)


class TestStartsFibonacciPool:
    """starts kwarg contract tests (v1.1+ API surface)."""

    def _bench_inputs(self):
        g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        return T, g, jax.random.PRNGKey(42)

    def test_default_axis_gaussian_preserves_v10(self):
        """Default + explicit starts='axis+gaussian' are bit-identical to v1.0."""
        T, g, key = self._bench_inputs()
        for fn in [optimize_wec, optimize_sec, optimize_dec]:
            r_default = fn(T, g, key=key)
            r_explicit = fn(T, g, starts="axis+gaussian", key=key)
            assert jnp.array_equal(r_default.margin, r_explicit.margin), (
                f"{fn.__name__} drifted under starts='axis+gaussian'"
            )
        r_default_nec = optimize_nec(T, g, key=key)
        r_explicit_nec = optimize_nec(T, g, starts="axis+gaussian", key=key)
        assert jnp.array_equal(r_default_nec.margin, r_explicit_nec.margin)

    def test_fibonacci_pool_accepted(self):
        """starts='fibonacci+bfgs_top_k' is accepted; returns valid result.

        Currently dispatches identically to
        'axis+gaussian' (pool helper in separate module).
        """
        T, g, key = self._bench_inputs()
        r = optimize_wec(T, g, key=key, starts="fibonacci+bfgs_top_k")
        assert r.margin is not None
        assert r.worst_observer.shape == (4,)

    def test_invalid_starts_raises(self):
        """starts='bad' raises ValueError with verbatim message."""
        T, g, _ = self._bench_inputs()
        for bad in ["bad", "random", ""]:
            with pytest.raises(ValueError, match="starts must be one of"):
                optimize_wec(T, g, starts=bad)

    def test_composes_with_strategy_hard_bound(self):
        """Orthogonal kwargs: strategy + starts compose without conflict."""
        T, g, key = self._bench_inputs()
        r = optimize_wec(
            T, g, key=key,
            strategy="hard_bound",
            starts="fibonacci+bfgs_top_k",
        )
        assert r.margin is not None

    def test_golden_fixture_metadata_present(self):
        """c7_fibonacci_pool_v1_1_0.npy contains required header fields."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "golden"
            / "c7_fibonacci_pool_v1_1_0.npy"
        )
        assert fixture_path.exists(), f"golden fixture missing: {fixture_path}"

        d = np.load(fixture_path, allow_pickle=True).item()
        for k in (
            "warpax_version", "jaxlib_version", "jax_random_seed",
            "backend", "starts", "strategy",
        ):
            assert k in d, f"fixture missing required key: {k!r}"
        assert d["starts"] == "fibonacci+bfgs_top_k"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
