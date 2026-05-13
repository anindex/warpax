"""Tests for Fuchs shell EC pipeline, constraints, and source consistency.

Verifies the warpax energy-condition pipeline, ADM constraint residuals,
and source-consistency diagnostics on the Fuchs et al. (CQG 2024,
arXiv:2405.02709) constant-velocity warp shell.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestFuchsECPipeline:
    """Energy condition pipeline on the Fuchs shell."""

    def test_interior_type_i(self):
        """Interior (r=1) classifies as Hawking-Ellis Type I."""
        from warpax.energy_conditions import classify_mixed_tensor
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        cls = classify_mixed_tensor(
            curv.stress_energy, curv.metric, curv.metric_inv,
        )
        assert int(cls.he_type) == 1

    def test_shell_classification_and_margins(self):
        """Shell (r=15): classify, verify finite margins, compare Eulerian.

        At v_s=0.02 the shell eigenvalues are ~1e-7 (near-degenerate).
        jnp.linalg.eig can produce spurious imaginary parts -> Type IV.
        The BFGS path still returns finite margins regardless.
        """
        from warpax.energy_conditions import (
            classify_mixed_tensor,
            compute_eulerian_ec,
            verify_point,
        )
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

        cls = classify_mixed_tensor(T, g, gi)
        assert int(cls.he_type) in (1, 4)
        assert jnp.all(jnp.isfinite(cls.eigenvalues))

        ec = verify_point(T, g, gi, n_starts=16)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert jnp.isfinite(m), f"{name} not finite"

        eul = compute_eulerian_ec(T, g, gi)
        for k in ["nec", "wec", "sec", "dec"]:
            assert jnp.isfinite(eul[k]), f"Eulerian {k} not finite"

    def test_interior_ec_non_negative(self):
        """Interior (r=1, near-vacuum) has non-negative EC margins."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        ec = verify_point(
            curv.stress_energy, curv.metric, curv.metric_inv, n_starts=16,
        )
        assert float(ec.nec_margin) >= -1e-8
        assert float(ec.wec_margin) >= -1e-8
        assert float(ec.sec_margin) >= -1e-8
        assert float(ec.dec_margin) >= -1e-8


class TestFuchsConstraints:
    """ADM constraint residuals on Fuchs initial data."""

    def test_interior_constraints_vanish(self):
        """Interior (r=1): flat Minkowski, constraints vanish."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        assert float(res["epsilon_H"]) < 1e-8
        assert float(res["epsilon_M"]) < 1e-8

    def test_shell_constraints_finite(self):
        """Shell (r=15): constraints are finite (no NaN)."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])

    def test_exterior_constraints_vanish(self):
        """Exterior (r=30): flat Minkowski, constraints vanish."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 30.0, 0.0, 0.0]),
        )
        assert float(res["epsilon_H"]) < 1e-8
        assert float(res["epsilon_M"]) < 1e-8


class TestFuchsSourceConsistency:
    """Input-vs-derived stress-energy comparison."""

    def test_input_stress_energy_construction(self):
        """T_input is symmetric (4,4) float64; nonzero at shell, zero outside."""
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()

        T_shell = fuchs_input_stress_energy(
            metric, jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert T_shell.shape == (4, 4)
        assert T_shell.dtype == jnp.float64
        assert jnp.allclose(T_shell, T_shell.T, atol=1e-15)
        assert jnp.max(jnp.abs(T_shell)) > 1e-15

        T_ext = fuchs_input_stress_energy(
            metric, jnp.array([0.0, 30.0, 0.0, 0.0]),
        )
        assert jnp.max(jnp.abs(T_ext)) < 1e-10

    def test_exterior_source_consistency(self):
        """Exterior (r=30): both T_input and G/8pi are ~0, delta_T vanishes."""
        from warpax.constraints import stress_energy_residual
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()
        coords = jnp.array([0.0, 30.0, 0.0, 0.0])
        T_input = fuchs_input_stress_energy(metric, coords)

        sc = stress_energy_residual(metric, coords, T_input=T_input)
        assert float(sc["max_residual"]) < 1e-8

    def test_shell_source_consistency_finite(self):
        """Shell (r=15): source consistency residual is finite.

        A nonzero residual is expected: the isotropic pressure
        approximation differs from the exact Einstein solution.
        """
        from warpax.constraints import stress_energy_residual
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        T_input = fuchs_input_stress_energy(metric, coords)

        sc = stress_energy_residual(metric, coords, T_input=T_input)
        assert jnp.isfinite(sc["max_residual"])
        assert jnp.isfinite(sc["relative_residual"])
