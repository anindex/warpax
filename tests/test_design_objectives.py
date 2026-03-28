"""Tests: design objectives - EC-margin integrals + averaged + quantum.

- Pointwise EC margin (NEC/WEC/SEC/DEC) via `ec_margin_objective`.
- Averaged objectives (ANEC / AWEC).
- Quantum objective (Ford-Roman).
- `OBJECTIVE_REGISTRY` string-dispatch for the optimizer.
"""
from __future__ import annotations

import jax.numpy as jnp

from warpax.benchmarks import MinkowskiMetric
from warpax.design import (
    OBJECTIVE_REGISTRY,
    averaged_objective,
    ec_margin_objective,
    quantum_objective,
)


class TestDesignObjectives:
    """Design objectives: 5 tests per behavior spec."""

    def test_nec_margin_on_minkowski(self):
        """NEC margin on Minkowski (vacuum) is near zero (or positive)."""
        m = MinkowskiMetric()
        margin = ec_margin_objective(m, objective="nec",
                                     grid_shape=(4, 4, 4),
                                     bounds=((-1, 1), (-1, 1), (-1, 1)))
        assert jnp.isfinite(margin)
        # Minkowski has T_ab = 0 => margin should be near 0 or positive
        assert float(margin) >= -1e-6

    def test_wec_margin_on_minkowski(self):
        """WEC margin on Minkowski is finite (regression-pinned)."""
        m = MinkowskiMetric()
        margin = ec_margin_objective(m, objective="wec",
                                     grid_shape=(4, 4, 4),
                                     bounds=((-1, 1), (-1, 1), (-1, 1)))
        assert jnp.isfinite(margin)

    def test_registry_lookup(self):
        """OBJECTIVE_REGISTRY has all 7 keys and dispatches to ec_margin_objective."""
        for key in ["nec", "wec", "sec", "dec", "anec", "awec", "ford_roman"]:
            assert key in OBJECTIVE_REGISTRY, f"Missing key {key}"
        # Check the NEC dispatcher is callable and returns finite
        m = MinkowskiMetric()
        margin_direct = ec_margin_objective(
            m, objective="nec", grid_shape=(4, 4, 4),
            bounds=((-1, 1), (-1, 1), (-1, 1))
        )
        margin_registry = OBJECTIVE_REGISTRY["nec"](
            m, grid_shape=(4, 4, 4),
            bounds=((-1, 1), (-1, 1), (-1, 1))
        )
        assert float(margin_direct) == float(margin_registry)

    def test_anec_composition(self):
        """averaged_objective(..., kind='anec') agrees with warpax.averaged.anec."""
        from warpax.averaged import anec

        m = MinkowskiMetric()
        # Straight null geodesic as a callable: gamma(lambda) = (lambda, lambda, 0, 0)
        def gamma(lam):
            return jnp.array([lam, lam, 0.0, 0.0])

        direct = anec(m, gamma, affine_bounds=(0.0, 1.0), n_samples=16).line_integral
        composed = averaged_objective(m, gamma, kind="anec",
                                       affine_bounds=(0.0, 1.0), n_samples=16)
        assert float(direct) == float(composed)

    def test_ford_roman_composition(self):
        """quantum_objective(...) agrees with warpax.quantum.ford_roman."""
        from warpax.quantum import ford_roman

        m = MinkowskiMetric()
        # Static timelike worldline: (tau, 0, 0, 0)
        def worldline(tau):
            return jnp.array([tau, 0.0, 0.0, 0.0])

        direct = ford_roman(m, worldline, tau0=1.0).margin
        composed = quantum_objective(m, worldline, tau0=1.0)
        assert float(direct) == float(composed)
