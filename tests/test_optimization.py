"""Tests for the source-first optimization framework.

Covers Bernstein basis, multi-objective loss, EC constraints, and optimizer.
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestBernsteinBasis:
    """Bernstein basis mathematical properties."""

    def test_partition_of_unity(self):
        """Sum of B_{n,k}(t) = 1 for all t in [0,1] (Bernstein property)."""
        from warpax.optimization import bernstein_basis

        for n in [3, 5, 8]:
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                basis_sum = jnp.sum(bernstein_basis(n, jnp.asarray(t_val)))
                assert abs(float(basis_sum) - 1.0) < 1e-12

    def test_endpoint_values(self):
        """B_{n,0}(0) = 1 and B_{n,n}(1) = 1."""
        from warpax.optimization import bernstein_basis

        for n in [3, 5, 8]:
            assert abs(float(bernstein_basis(n, jnp.asarray(0.0))[0]) - 1.0) < 1e-12
            assert abs(float(bernstein_basis(n, jnp.asarray(1.0))[-1]) - 1.0) < 1e-12

    def test_eval_consistency(self):
        """bernstein_eval matches manual basis * coeffs sum."""
        from warpax.optimization import bernstein_basis, bernstein_eval

        coeffs = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
        t = jnp.asarray(0.3)
        assert abs(float(jnp.sum(coeffs * bernstein_basis(4, t)) - bernstein_eval(coeffs, t))) < 1e-14

    def test_jax_differentiable(self):
        """jax.grad flows through bernstein_eval."""
        from warpax.optimization import bernstein_eval

        coeffs = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
        grad = jax.grad(lambda c: bernstein_eval(c, jnp.asarray(0.5)))(coeffs)
        assert jnp.all(jnp.isfinite(grad))


class TestEndpointClamping:
    """Compact support via endpoint clamping."""

    def test_clamp_structure(self):
        """Clamped output has c_0 = c_n = 0, interior unchanged."""
        from warpax.optimization import clamp_endpoints

        theta = jnp.array([1.0, 2.0, 3.0])
        coeffs = clamp_endpoints(theta)
        assert coeffs.shape == (5,)
        assert float(coeffs[0]) == 0.0
        assert float(coeffs[-1]) == 0.0
        assert jnp.allclose(coeffs[1:-1], theta)

    def test_softplus_positivity(self):
        """positive=True ensures all coefficients >= 0."""
        from warpax.optimization import clamp_endpoints

        coeffs = clamp_endpoints(jnp.array([-5.0, 0.0, 5.0]), positive=True)
        assert jnp.all(coeffs >= 0.0)

    def test_compact_support_property(self):
        """Bernstein polynomial vanishes at t=0 and t=1 with clamped endpoints."""
        from warpax.optimization import clamp_endpoints, bernstein_eval

        coeffs = clamp_endpoints(jnp.array([0.5, 1.0, 0.5]))
        assert abs(float(bernstein_eval(coeffs, jnp.asarray(0.0)))) < 1e-14
        assert abs(float(bernstein_eval(coeffs, jnp.asarray(1.0)))) < 1e-14
        assert float(bernstein_eval(coeffs, jnp.asarray(0.5))) > 0


class TestPackUnpack:
    """Parameter vector serialization."""

    def test_round_trip_shape(self):
        """pack -> unpack preserves structure and endpoint clamping."""
        from warpax.optimization import pack_theta, unpack_theta

        packed = pack_theta(
            jnp.array([0.1, 0.5, 0.3, 0.2]),
            jnp.array([0.2, 0.8, 0.5, 0.1]),
            0.1, 1e-4,
        )
        assert packed.shape == (10,)

        coeffs = unpack_theta(packed, n_density=4, n_velocity=4)
        assert coeffs.density_coeffs.shape == (6,)
        assert coeffs.velocity_coeffs.shape == (6,)
        assert float(coeffs.density_coeffs[0]) == 0.0
        assert float(coeffs.density_coeffs[-1]) == 0.0

    def test_default_theta_valid(self):
        """default_theta produces non-negative density coefficients."""
        from warpax.optimization import default_theta, unpack_theta

        theta = default_theta(n_density=4, n_velocity=4)
        coeffs = unpack_theta(theta, n_density=4, n_velocity=4)
        assert jnp.all(coeffs.density_coeffs >= 0)


class TestCoeffsToProfiles:
    """Profile factory integration."""

    def test_sshell_valid_profiles(self):
        """S-shell profiles have positive mass and correct radii."""
        from warpax.optimization import default_theta, unpack_theta, coeffs_to_profiles_sshell

        coeffs = unpack_theta(default_theta(), n_density=4, n_velocity=4)
        profiles = coeffs_to_profiles_sshell(coeffs)
        assert profiles.total_mass > 0
        assert profiles.R_1 == 10.0
        assert profiles.R_2 == 20.0

    def test_tshell_valid_profiles(self):
        """T-shell profiles have positive mass and correct radii."""
        from warpax.optimization import default_theta, unpack_theta, coeffs_to_profiles_tshell

        coeffs = unpack_theta(default_theta(), n_density=4, n_velocity=4)
        profiles = coeffs_to_profiles_tshell(coeffs)
        assert profiles.total_mass > 0
        assert profiles.R_1 == 10.0


class TestLoss:
    """Multi-objective loss evaluation."""

    def test_default_sshell_finite(self):
        """Default S-shell produces finite loss with all components finite."""
        from warpax.optimization import default_theta, evaluate_loss

        loss, components = evaluate_loss(
            default_theta(), ansatz="sshell", n_probes=5, n_grid=256, n_ec_starts=2,
        )
        assert jnp.isfinite(loss)
        assert jnp.isfinite(components.constraint)
        assert jnp.isfinite(components.ec_penalty)
        assert jnp.isfinite(components.tidal)
        assert jnp.isfinite(components.mass)
        assert components.mass > 0

    def test_weight_sensitivity(self):
        """Increasing w_ec increases total loss when EC penalty > 0."""
        from warpax.optimization import default_theta, evaluate_loss, LossWeights

        theta = default_theta()
        l1, c1 = evaluate_loss(
            theta, ansatz="sshell", n_probes=3, n_grid=256, n_ec_starts=2,
            weights=LossWeights(w_ec=1.0),
        )
        l2, _ = evaluate_loss(
            theta, ansatz="sshell", n_probes=3, n_grid=256, n_ec_starts=2,
            weights=LossWeights(w_ec=100.0),
        )
        if c1.ec_penalty > 1e-10:
            assert float(l2) > float(l1)


class TestECConstraints:
    """EC soft penalty and hard feasibility."""

    def test_penalty_finite_and_nonnegative(self):
        """EC penalty is finite and >= 0 (softplus^2 is non-negative)."""
        from warpax.optimization import ec_penalty
        from warpax.metrics import sshell_default

        penalty = ec_penalty(sshell_default(), jnp.linspace(9.0, 21.0, 5), n_starts=2)
        assert jnp.isfinite(penalty)
        assert float(penalty) >= 0.0

    def test_feasibility_structure(self):
        """ECFeasibilityResult has correct shape and condition keys."""
        from warpax.optimization import ec_feasibility_check
        from warpax.metrics import sshell_default

        result = ec_feasibility_check(sshell_default(), jnp.linspace(9.0, 21.0, 5), n_starts=4)
        assert isinstance(result.feasible, bool)
        for cond in ("nec", "wec", "dec"):
            assert cond in result.margins
            assert result.margins[cond].shape == (5,)

    def test_distinct_prng_keys_per_probe_point(self, monkeypatch):
        """Regression (PRNG key reuse): ``_ec_margins_at_point`` was called
        with no key, so every probe point fell back to ``PRNGKey(42)`` and
        used identical random multistarts. Each probe point must now
        receive a distinct ``fold_in`` key.
        """
        import warpax.optimization.ec_constraints as ecc

        captured: list = []

        def fake_margins(T, g, conditions, n_starts, key=None):
            captured.append(key)
            return {c: jnp.asarray(1.0) for c in conditions}

        def fake_probe_T_g(metric, r_probes):
            n = r_probes.shape[0]
            eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
            return (
                jnp.zeros((n, 4, 4)),
                jnp.broadcast_to(eta, (n, 4, 4)),
            )

        monkeypatch.setattr(ecc, "_ec_margins_at_point", fake_margins)
        monkeypatch.setattr(ecc, "_probe_T_g", fake_probe_T_g)

        r_probes = jnp.array([10.0, 15.0])

        ecc.ec_penalty(object(), r_probes)
        assert len(captured) == 2
        assert captured[0] is not None and captured[1] is not None
        assert not jnp.array_equal(captured[0], captured[1]), (
            "ec_penalty consumed identical PRNG keys at two probe points"
        )

        captured.clear()
        ecc.ec_feasibility_check(object(), r_probes)
        assert len(captured) == 2
        assert not jnp.array_equal(captured[0], captured[1]), (
            "ec_feasibility_check consumed identical PRNG keys at two probe points"
        )


class TestOptimizerIntegration:
    """End-to-end optimizer smoke test."""

    @pytest.mark.slow
    def test_optimize_shell_completes(self):
        """optimize_shell with maxiter=3 completes and returns finite loss."""
        from warpax.optimization import optimize_shell

        result = optimize_shell(
            ansatz="sshell",
            n_density=3, n_velocity=3,
            n_grid=256, n_probes=3, n_ec_starts=2,
            maxiter=3, certify_ec=False,
        )
        assert result.theta_opt is not None
        assert jnp.isfinite(result.loss_final)
