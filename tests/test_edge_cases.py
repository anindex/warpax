"""Edge-case tests for the Hawking-Ellis classifier, NaN propagation, optimiser
convergence, and determinant-guard boundary behavior.

Four categories:

1. Classifier near-degenerate inputs: near-vacuum, exactly-zero, Type I/IV
   boundary (below/above ``imag_rtol``), near-degenerate eigenvalues, and
   large-scale eigenvalues with tiny imaginary parts.

2. NaN propagation at sharp walls: ``compute_curvature_chain`` at wall-center
   points with high ``sigma`` for Alcubierre (sigma=32, 64) and Rodal
   (sigma=0.3); classifier NaN sanitisation.

3. Optimiser convergence near walls: ``verify_point`` at wall-center
   coordinates where curvature is maximal.

4. Determinant-guard boundary: ``determinant_guard_mask`` against healthy,
   singular, and threshold-crossing metric fields, plus a superluminal g_00
   sign-flip diagnostic.

Tolerances: 1e-12 for synthetic tensor tests, 1e-6 for optimiser margins.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.filtering import determinant_guard_mask
from warpax.energy_conditions.verifier import verify_point
from warpax.geometry import compute_curvature_chain
from warpax.metrics import RodalMetric

# Minkowski metric (common to all tests)
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Category 1: Classifier near-degenerate inputs
# ---------------------------------------------------------------------------


class TestClassifierNearDegenerateInputs:
    """Probe the classifier at boundaries where misclassification is most likely.

    Covers near-vacuum, exactly-zero, Type I/IV transition at the relative
    imaginary-part threshold (imag_rtol = 3e-3 by default), near-degenerate
    real eigenvalues, and large-scale eigenvalues with tiny imaginary parts.
    """

    def test_near_vacuum_classifies_as_type_i(self):
        """Near-vacuum tensors must classify as Type I (not Type IV noise)."""
        T_mixed = jnp.diag(jnp.array([-1e-12, 1e-13, 1e-13, 1e-13]))
        result = classify_hawking_ellis(T_mixed, ETA)
        assert int(result.he_type) == 1

    def test_exactly_zero_tensor_classifies_as_type_i(self):
        """Vacuum is Type I by construction."""
        T_mixed = jnp.zeros((4, 4))
        result = classify_hawking_ellis(T_mixed, ETA)
        assert int(result.he_type) == 1

    def test_type_i_iv_boundary_below_threshold(self):
        """Complex conjugate eigenvalue pair with |Im|/|Re| = 0.002 < 3e-3.

        Real Jordan block [[lam, -eps], [eps, lam]] has eigenvalues lam +/- i*eps.
        With lam=1.0 and eps=0.002, the relative imaginary ratio is 0.002,
        below the default imag_rtol=0.003, so the classifier must treat the
        spectrum as effectively real and return Type I.
        """
        lam = 1.0
        eps = 0.002 * lam
        block = jnp.array([[lam, -eps], [eps, lam]])
        T = (
            jnp.zeros((4, 4))
            .at[:2, :2]
            .set(block)
            .at[2, 2]
            .set(lam)
            .at[3, 3]
            .set(lam)
        )
        result = classify_hawking_ellis(T, ETA)
        assert int(result.he_type) == 1

    def test_type_i_iv_boundary_above_threshold(self):
        """Complex conjugate eigenvalue pair with |Im|/|Re| = 0.005 > 3e-3.

        Above the default imag_rtol=0.003, the classifier must return Type IV.
        """
        lam = 1.0
        eps = 0.005 * lam
        block = jnp.array([[lam, -eps], [eps, lam]])
        T = (
            jnp.zeros((4, 4))
            .at[:2, :2]
            .set(block)
            .at[2, 2]
            .set(lam)
            .at[3, 3]
            .set(lam)
        )
        result = classify_hawking_ellis(T, ETA)
        assert int(result.he_type) == 4

    def test_near_degenerate_eigenvalues(self):
        """Near-degenerate but real eigenvalues must classify as Type I."""
        T_mixed = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0 + 1e-14]))
        result = classify_hawking_ellis(T_mixed, ETA)
        assert int(result.he_type) == 1

    def test_large_scale_eigenvalues_with_tiny_imaginary(self):
        """Scale ~1e6 with |Im|/|Re| = 0.001 < 3e-3 must still classify as Type I.

        Tests the scale-dependent behavior from Research the
        classifier's unclamped relative criterion catches split degenerate
        pairs at large norm where the absolute criterion (tol * scale) alone
        would misclassify as Type IV.
        """
        lam = 1e6
        eps = 0.001 * lam
        block = jnp.array([[lam, -eps], [eps, lam]])
        T = (
            jnp.zeros((4, 4))
            .at[:2, :2]
            .set(block)
            .at[2, 2]
            .set(lam)
            .at[3, 3]
            .set(lam)
        )
        result = classify_hawking_ellis(T, ETA)
        assert int(result.he_type) == 1


# ---------------------------------------------------------------------------
# Category 2: NaN propagation at sharp walls
# ---------------------------------------------------------------------------


class TestNaNPropagationAtSharpWalls:
    """Autodiff chain at sharp-wall wall-center points must not produce NaN.

    Uses compute_curvature_chain at single points with high sigma (sharp walls)
    to probe for NaN propagation through Gamma -> R -> Ric -> G -> T. Also
    verifies the classifier's NaN sanitization.
    """

    def test_alcubierre_sharp_wall_no_nan(self):
        """Alcubierre(sigma=32) at wall center must produce finite curvature."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=32.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])
        curv = compute_curvature_chain(metric, coords)
        assert jnp.all(jnp.isfinite(curv.stress_energy))
        assert jnp.all(jnp.isfinite(curv.riemann))
        assert jnp.all(jnp.isfinite(curv.christoffel))

    def test_alcubierre_very_sharp_wall_no_nan(self):
        """Alcubierre(sigma=64) at wall center must still produce finite values."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=64.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])
        curv = compute_curvature_chain(metric, coords)
        assert jnp.all(jnp.isfinite(curv.stress_energy))
        assert jnp.all(jnp.isfinite(curv.riemann))

    def test_rodal_sharp_wall_no_nan(self):
        """Rodal(sigma=0.3) at wall center must produce finite curvature.

        sigma=0.3 is the sharpest sigma value in the Rodal DEC ablation sweep.
        """
        metric = RodalMetric(v_s=0.5, R=100.0, sigma=0.3)
        # Offset slightly from wall center on axis to avoid 0/0 forms
        coords = jnp.array([0.0, 100.0, 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)
        assert jnp.all(jnp.isfinite(curv.stress_energy))
        assert jnp.all(jnp.isfinite(curv.riemann))

    def test_classifier_with_nan_input_sanitizes(self):
        """Classifier replaces NaN entries with zero before eigendecomposition."""
        T = jnp.array(
            [
                [-1.0, float("nan"), 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Must not raise and must produce a finite he_type
        result = classify_hawking_ellis(T, ETA)
        he = int(result.he_type)
        assert he in (1, 2, 3, 4)
        # NaN sanitization replaces NaN -> 0, so eigenvalues must be finite
        assert jnp.all(jnp.isfinite(result.eigenvalues))


# ---------------------------------------------------------------------------
# Category 3: Optimizer convergence near walls
# ---------------------------------------------------------------------------


class TestOptimizerConvergenceNearWalls:
    """BFGS optimizer must produce finite margins at wall-center points."""

    def test_alcubierre_wall_center_converges(self):
        """verify_point at Alcubierre wall center must yield finite margins."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        coords = jnp.array([0.0, 1.0, 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)
        ec = verify_point(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            n_starts=8,
            zeta_max=5.0,
        )
        assert bool(jnp.isfinite(ec.nec_margin))
        assert bool(jnp.isfinite(ec.wec_margin))
        assert bool(jnp.isfinite(ec.sec_margin))
        assert bool(jnp.isfinite(ec.dec_margin))

    def test_rodal_wall_center_converges(self):
        """verify_point at Rodal wall center must yield finite margins."""
        metric = RodalMetric(v_s=0.5, R=100.0, sigma=0.03)
        coords = jnp.array([0.0, 100.0, 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)
        ec = verify_point(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            n_starts=8,
            zeta_max=5.0,
        )
        assert bool(jnp.isfinite(ec.nec_margin))
        assert bool(jnp.isfinite(ec.wec_margin))
        assert bool(jnp.isfinite(ec.sec_margin))
        assert bool(jnp.isfinite(ec.dec_margin))

    def test_sharp_wall_optimizer_finite_margins(self):
        """verify_point on a sharp Alcubierre wall (sigma=32) must yield finite margins."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=32.0)
        coords = jnp.array([0.0, 1.0, 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)
        ec = verify_point(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            n_starts=8,
            zeta_max=5.0,
        )
        assert bool(jnp.isfinite(ec.nec_margin))
        assert bool(jnp.isfinite(ec.wec_margin))
        assert bool(jnp.isfinite(ec.sec_margin))
        assert bool(jnp.isfinite(ec.dec_margin))


# ---------------------------------------------------------------------------
# Category 4: Determinant guard boundary
# ---------------------------------------------------------------------------


class TestDeterminantGuardBoundary:
    """determinant_guard_mask must correctly flag degenerate and superluminal metrics."""

    def test_healthy_metric_passes_guard(self):
        """A field of Minkowski metrics must fully pass the determinant guard."""
        grid_shape = (2, 2, 2)
        g_single = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        g_field = jnp.broadcast_to(g_single, (*grid_shape, 4, 4))
        # No warning should be emitted for a healthy field
        mask = determinant_guard_mask(g_field, threshold=1e-10)
        assert mask.shape == grid_shape
        assert bool(jnp.all(mask))

    def test_singular_metric_fails_guard(self):
        """A rank-deficient metric must fail the determinant guard."""
        grid_shape = (1, 1, 1)
        # Pure-rank-1 metric: det = 0
        g_single = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        g_field = g_single.reshape(1, 1, 1, 4, 4)
        with pytest.warns(UserWarning, match="grid points have"):
            mask = determinant_guard_mask(g_field, threshold=1e-10)
        assert mask.shape == grid_shape
        assert not bool(mask[0, 0, 0])

    def test_near_threshold_metric(self):
        """Metric with |det(g)| near the threshold must behave correctly on both sides.

        Below threshold (|det| = 1e-11 < 1e-10) -> guard False.
        Above threshold (|det| = 1e-9 > 1e-10) -> guard True.
        """
        # Build a diagonal metric with arbitrary target determinant.
        # det(diag(a, b, c, d)) = a*b*c*d. Use diag(-d0, 1, 1, 1) -> det = -d0.

        # Below threshold
        g_low = jnp.diag(jnp.array([-1e-11, 1.0, 1.0, 1.0]))
        g_field_low = g_low.reshape(1, 1, 1, 4, 4)
        with pytest.warns(UserWarning, match="grid points have"):
            mask_low = determinant_guard_mask(g_field_low, threshold=1e-10)
        assert not bool(mask_low[0, 0, 0])

        # Above threshold
        g_high = jnp.diag(jnp.array([-1e-9, 1.0, 1.0, 1.0]))
        g_field_high = g_high.reshape(1, 1, 1, 4, 4)
        mask_high = determinant_guard_mask(g_field_high, threshold=1e-10)
        assert bool(mask_high[0, 0, 0])

    def test_superluminal_g00_positive_detected(self):
        """Superluminal Alcubierre: det(g) = -1 but g_00 > 0 inside bubble.

        The superluminal failure mode is a g_00 sign flip, not a det(g)=0
        collapse. The determinant guard alone does NOT detect this -- we
        construct a metric with g_00 > 0 manually and verify that the
        pipeline handles it without crashing and the classifier returns a
        finite type.
        """
        # Build a diagonal metric with g_00 = +0.5 (sign-flipped) and |det(g)| = 0.5
        g_bad = jnp.diag(jnp.array([0.5, 1.0, 1.0, 1.0]))
        g_field = g_bad.reshape(1, 1, 1, 4, 4)
        # det(g) = 0.5 (healthy magnitude), so guard should pass
        mask = determinant_guard_mask(g_field, threshold=1e-10)
        assert bool(mask[0, 0, 0]), (
            "Determinant guard does NOT detect g_00 sign flip "
            "(it checks |det(g)|, not signature)."
        )

        # Classifier must not crash on such a tensor either
        T_mixed = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        result = classify_hawking_ellis(T_mixed, g_bad)
        assert int(result.he_type) in (1, 2, 3, 4)
        assert jnp.all(jnp.isfinite(result.eigenvalues))
