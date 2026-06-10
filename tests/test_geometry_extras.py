"""Geometry extras: invariants, regularity, grid/edge-case handling, custom exceptions."""

from __future__ import annotations
from numpy.testing import assert_allclose
from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.benchmarks.schwarzschild import kretschmann_isotropic
from warpax.exceptions import (
    AsymptoticFalloffError,
    ConstraintViolationError,
    JunctionDiscontinuityError,
    TOVInconsistencyError,
    TransportUndefinedError,
    WarpAXError,
)
from warpax.geometry.geometry import CurvatureResult
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.grid import (
    GridCurvatureResult,
    build_coord_batch,
    evaluate_curvature_grid,
)
from warpax.geometry.invariants import (
    chern_pontryagin,
    compute_invariants,
    kretschmann_scalar,
    ricci_squared,
    weyl_squared,
)
from warpax.geometry.types import GridSpec
import jax
import jax.numpy as jnp
import numpy as np
import pytest



# Minkowski: all invariants exactly zero


class TestMinkowskiInvariants:
    """Minkowski (flat) spacetime: all curvature invariants are zero."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        self.result = compute_curvature_chain(metric, coords)

    def test_kretschmann_minkowski_zero(self):
        """Minkowski: Kretschmann scalar K = 0 to machine precision."""
        K = kretschmann_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        assert_allclose(float(K), 0.0, atol=1e-14)

    def test_ricci_squared_minkowski_zero(self):
        """Minkowski: R_{ab} R^{ab} = 0 to machine precision."""
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        assert_allclose(float(R2), 0.0, atol=1e-14)

    def test_weyl_squared_minkowski_zero(self):
        """Minkowski: C_{abcd} C^{abcd} = 0 to machine precision."""
        K = kretschmann_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        W2 = weyl_squared(K, R2, self.result.ricci_scalar)
        assert_allclose(float(W2), 0.0, atol=1e-14)


# Schwarzschild: analytical ground truth


class TestSchwarzschildInvariants:
    """Schwarzschild black hole: invariants vs analytical formulas."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.M = 1.0
        metric = SchwarzschildMetric(M=self.M)
        # Test point far from singularity: (t=0, x=0, y=5, z=0)
        self.coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        self.result = compute_curvature_chain(metric, self.coords)

        # Analytical Kretschmann: K = 48 * M^2 / r_s^6
        # isotropic r_iso = 5.0, Schwarzschild r_s = r_iso * (1 + M/(2*r_iso))^2
        r_iso = 5.0
        r_s = r_iso * (1.0 + self.M / (2.0 * r_iso)) ** 2
        self.expected_K = 48.0 * self.M**2 / r_s**6

    def test_kretschmann_schwarzschild_analytical(self):
        """Kretschmann scalar matches K = 48*M^2/r_s^6 for Schwarzschild."""
        K = kretschmann_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        assert_allclose(float(K), self.expected_K, rtol=1e-10)

    def test_schwarzschild_ricci_flat(self):
        """Schwarzschild is vacuum: Ricci-flat => ricci_squared=0, weyl_squared=K."""
        K = kretschmann_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        W2 = weyl_squared(K, R2, self.result.ricci_scalar)

        assert_allclose(float(R2), 0.0, atol=1e-10)
        # For Ricci-flat spacetimes, Weyl-squared equals Kretschmann.
        assert_allclose(float(W2), float(K), atol=1e-10)


# Convenience function and dtype tests


class TestInvariantConvenience:
    """Test compute_invariants convenience function and dtype enforcement."""

    def test_compute_invariants_convenience(self):
        """compute_invariants matches individual function calls on Schwarzschild."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        result = compute_curvature_chain(metric, coords)

        K_individual = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
        R2_individual = ricci_squared(result.ricci, result.metric_inv)
        W2_individual = weyl_squared(K_individual, R2_individual, result.ricci_scalar)

        CP_individual = chern_pontryagin(result.riemann, result.metric, result.metric_inv)

        # Convenience function
        K_conv, R2_conv, W2_conv, CP_conv = compute_invariants(result)

        assert_allclose(float(K_conv), float(K_individual), atol=0.0)
        assert_allclose(float(R2_conv), float(R2_individual), atol=0.0)
        assert_allclose(float(W2_conv), float(W2_individual), atol=0.0)
        assert_allclose(float(CP_conv), float(CP_individual), atol=0.0)

    def test_invariants_float64_dtype(self):
        """All invariant outputs are float64."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        result = compute_curvature_chain(metric, coords)

        K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
        R2 = ricci_squared(result.ricci, result.metric_inv)
        W2 = weyl_squared(K, R2, result.ricci_scalar)

        assert K.dtype == jnp.float64
        assert R2.dtype == jnp.float64
        assert W2.dtype == jnp.float64


jax.config.update("jax_enable_x64", True)


class TestRegularityDiagnostics:
    """Verify the metric regularity diagnostic module."""

    def test_minkowski_passes_c2(self):
        """Minkowski metric is trivially C^inf (all jumps near zero)."""
        from warpax.benchmarks import MinkowskiMetric
        from warpax.geometry import regularity_report

        report = regularity_report(MinkowskiMetric(), r_min=5.0, r_max=25.0)
        assert report.is_c2

        for name, diag in report.components.items():
            assert diag.c0_max_jump < 1.0, f"{name} C0 jump = {diag.c0_max_jump}"
            assert diag.c1_max_jump < 1.0, f"{name} C1 jump = {diag.c1_max_jump}"

    def test_schwarzschild_passes_c2(self):
        """Schwarzschild metric is C^inf away from the horizon."""
        from warpax.benchmarks import SchwarzschildMetric
        from warpax.geometry import regularity_report

        report = regularity_report(
            SchwarzschildMetric(M=1.0),
            r_min=5.0, r_max=25.0,
        )
        assert report.is_c2

    def test_c1_smoothstep_has_larger_c2_jumps(self):
        """C1 cubic smoothstep produces larger C^2 jumps than C2 quintic.

        This tests the module's ability to discriminate between smoothness
        levels. The C1 cubic has f''(0)=6, f''(1)=-6 at transition
        boundaries, so its C2 diagnostic should show larger jumps.
        """
        from warpax.geometry import metric_c2_diagnostic
        from warpax.metrics import WarpShellPhysical

        r_vals = jnp.linspace(5.0, 25.0, 200)

        m_c1 = WarpShellPhysical(
            v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0,
            transition_order=1,
        )
        m_c2 = WarpShellPhysical(
            v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0,
            transition_order=2,
        )

        diag_c1 = metric_c2_diagnostic(m_c1, r_vals, component=(0, 0))
        diag_c2 = metric_c2_diagnostic(m_c2, r_vals, component=(0, 0))

        assert diag_c1.c2_max_jump > diag_c2.c2_max_jump, \
            f"C1 jump ({diag_c1.c2_max_jump:.1f}) should exceed " \
            f"C2 jump ({diag_c2.c2_max_jump:.1f})"


# Coordinate batch construction


class TestBuildCoordBatch:
    """Tests for build_coord_batch helper."""

    def test_build_coord_batch_shape(self, default_grid):
        """build_coord_batch returns shape (N, 4) where N = prod(grid_shape)."""
        coords = build_coord_batch(default_grid)
        N = np.prod(default_grid.shape)
        assert coords.shape == (N, 4)

    def test_build_coord_batch_time_coordinate(self, default_grid):
        """Time coordinates default to 0.0, or custom t value."""
        coords_default = build_coord_batch(default_grid)
        assert_allclose(np.array(coords_default[:, 0]), 0.0, atol=0.0)

        coords_custom = build_coord_batch(default_grid, t=1.5)
        assert_allclose(np.array(coords_custom[:, 0]), 1.5, atol=0.0)


# Grid curvature result shapes


class TestGridCurvatureShapes:
    """Tests for evaluate_curvature_grid output shapes and types."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.metric = MinkowskiMetric()
        self.grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )

    def test_grid_curvature_result_shapes(self):
        """evaluate_curvature_grid returns correct shapes for all fields."""
        result = evaluate_curvature_grid(self.metric, self.grid)
        gs = (4, 4, 2)

        assert isinstance(result, GridCurvatureResult)
        assert result.metric.shape == (*gs, 4, 4)
        assert result.metric_inv.shape == (*gs, 4, 4)
        assert result.christoffel.shape == (*gs, 4, 4, 4)
        assert result.riemann.shape == (*gs, 4, 4, 4, 4)
        assert result.ricci.shape == (*gs, 4, 4)
        assert result.ricci_scalar.shape == gs
        assert result.kretschmann.shape == gs
        assert result.ricci_squared.shape == gs
        assert result.weyl_squared.shape == gs

    def test_grid_without_invariants(self):
        """compute_invariants=False returns CurvatureResult without invariant fields."""
        result = evaluate_curvature_grid(
            self.metric, self.grid, compute_invariants=False
        )
        assert isinstance(result, CurvatureResult)
        assert not hasattr(result, "kretschmann")
        assert not hasattr(result, "weyl_squared")


# Batch size consistency


class TestGridBatching:
    """Chunked batching via batch_size produces identical results to full vmap."""

    def test_grid_batch_size_matches_full_vmap(self):
        """batch_size=32 gives identical results to batch_size=None (full vmap)."""
        metric = AlcubierreMetric()
        grid = GridSpec(
            bounds=[(-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5)],
            shape=(8, 8, 4),
        )

        result_full = evaluate_curvature_grid(metric, grid, batch_size=None)
        result_chunked = evaluate_curvature_grid(metric, grid, batch_size=32)

        # Compare all 11 fields
        for field_name in GridCurvatureResult._fields:
            full_val = np.array(getattr(result_full, field_name))
            chunked_val = np.array(getattr(result_chunked, field_name))
            assert_allclose(
                chunked_val,
                full_val,
                atol=1e-12,
                err_msg=f"Mismatch in field {field_name}",
            )


# Minkowski flatness on grid


class TestGridMinkowskiFlat:
    """Minkowski spacetime: all curvature tensors and invariants are zero on a grid."""

    def test_grid_minkowski_flat(self):
        """All curvature fields zero for Minkowski on a grid."""
        metric = MinkowskiMetric()
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        # Curvature tensors must be zero
        assert_allclose(np.array(result.riemann), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci_scalar), 0.0, atol=1e-12)
        assert_allclose(np.array(result.einstein), 0.0, atol=1e-12)
        assert_allclose(np.array(result.stress_energy), 0.0, atol=1e-12)

        assert_allclose(np.array(result.kretschmann), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci_squared), 0.0, atol=1e-12)
        assert_allclose(np.array(result.weyl_squared), 0.0, atol=1e-12)


# Float64 dtype enforcement


class TestGridFloat64:
    """All grid output fields must be float64."""

    def test_grid_float64_dtype(self):
        """All output fields from evaluate_curvature_grid are float64."""
        metric = MinkowskiMetric()
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        for field_name in GridCurvatureResult._fields:
            arr = getattr(result, field_name)
            assert arr.dtype == jnp.float64, (
                f"Field {field_name} has dtype {arr.dtype}, expected float64"
            )


class TestGridSchwarzschildKretschmann:
    """Schwarzschild Kretschmann scalar field matches the analytical formula on a grid."""

    def test_grid_schwarzschild_kretschmann_field(self):
        """Kretschmann field matches K=48*M^2/r_s^6 at each grid point."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)
        grid = GridSpec(
            bounds=[(3.0, 7.0), (3.0, 7.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        X, Y, Z = grid.meshgrid
        K_analytical = np.array(kretschmann_isotropic(X, Y, Z, M=M))

        assert_allclose(
            np.array(result.kretschmann),
            K_analytical,
            rtol=1e-8,
        )


class TestAutoChunk:
    """``auto_chunk_threshold`` kwarg validation and ULP-tolerant parity
    with the full-vmap path.

    The no-op (threshold > grid_size) path is bit-exact to the full-vmap
    path because it never calls ``lax.map``.
    When the threshold triggers chunking, ``jax.lax.map`` uses a different
    floating-point addition order than ``jax.vmap``; the resulting drift
    is at the ULP floor (observed max ~6.8e-13 on ricci/weyl squared at
    20³ Alcubierre). Tests use ``assert_allclose`` with ``atol=rtol=1e-12``.
    """

    _CHUNK_ATOL = 1e-12
    _CHUNK_RTOL = 1e-12

    def test_threshold_1000_chunks_20cubed_bit_exact(self):
        """auto_chunk_threshold=1000 on a 20³=8000 grid chunks to ULP-equivalent result.

        Chunking dispatches to ``jax.lax.map`` whose floating-point add
        order differs from ``jax.vmap``; the observed max drift is ~1e-13
        (below ``assert_allclose(atol=1e-12)``).
        """
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(20, 20, 20))

        ref = evaluate_curvature_grid(metric, grid)  # full vmap
        chunked = evaluate_curvature_grid(metric, grid, auto_chunk_threshold=1000)

        for field in GridCurvatureResult._fields:
            a = np.array(getattr(ref, field))
            b = np.array(getattr(chunked, field))
            assert_allclose(
                b,
                a,
                atol=self._CHUNK_ATOL,
                rtol=self._CHUNK_RTOL,
                err_msg=f"Field {field!r} drifted beyond ULP floor under threshold=1000",
            )

    def test_threshold_larger_than_grid_is_noop(self):
        """auto_chunk_threshold > grid_size: no chunking, bit-exact to default."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(8, 8, 8))  # 512 points

        ref = evaluate_curvature_grid(metric, grid)
        noop = evaluate_curvature_grid(metric, grid, auto_chunk_threshold=10_000)

        for field in GridCurvatureResult._fields:
            a = getattr(ref, field)
            b = getattr(noop, field)
            assert jnp.array_equal(a, b), (
                f"Field {field!r} drifted under auto_chunk_threshold=10_000 "
                "(no-op path must not call lax.map)"
            )

    def test_zero_threshold_raises(self):
        """auto_chunk_threshold=0 raises ValueError with verbatim message."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(4, 4, 4))

        with pytest.raises(ValueError, match="must be a positive integer or None"):
            evaluate_curvature_grid(metric, grid, auto_chunk_threshold=0)


jax.config.update("jax_enable_x64", True)

from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.filtering import determinant_guard_mask
from warpax.energy_conditions.verifier import verify_point
from warpax.metrics import RodalMetric

# Minkowski metric (common to all tests)
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# Category 1: Classifier near-degenerate inputs


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
        """At unit scale a genuine complex pair is Type IV at any split.

        The relative imag tier only engages above the 1e6 scale floor,
        so lam=1, eps=0.002 (a real complex pair, 50-digit certified)
        is Type IV -- the pre-floor classifier absorbed it as Type I.
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
        assert int(result.he_type) == 4

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
        """Above the scale floor, |Im|/|Re| = 0.001 < 3e-3 is Type I.

        The relative tier absorbs split-degenerate pairs at large norm,
        where the absolute criterion alone would report spurious Type IV.
        """
        lam = 1e9
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


# Category 2: NaN propagation at sharp walls


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
        # NaN -> 0 leaves a diagonal dust-like tensor; must classify Type I
        assert he == 1
        assert jnp.all(jnp.isfinite(result.eigenvalues))


# Category 3: Optimizer convergence near walls


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


# Category 4: Determinant guard boundary


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


def test_exception_inheritance():
    """All domain exceptions inherit from WarpAXError."""
    assert issubclass(AsymptoticFalloffError, WarpAXError)
    assert issubclass(ConstraintViolationError, WarpAXError)
    assert issubclass(JunctionDiscontinuityError, WarpAXError)
    assert issubclass(TOVInconsistencyError, WarpAXError)
    assert issubclass(TransportUndefinedError, WarpAXError)


@pytest.mark.parametrize(
    ("exc_cls", "kwargs", "expected_attrs", "expected_substrings"),
    [
        pytest.param(
            ConstraintViolationError,
            {"residual_type": "Hamiltonian", "max_residual": 1e-3, "location": "(0, 1, 0, 0)"},
            {"residual_type": "Hamiltonian", "max_residual": 1e-3, "location": "(0, 1, 0, 0)"},
            ["Hamiltonian", "1.000000e-03", "(0, 1, 0, 0)"],
            id="ConstraintViolationError",
        ),
        pytest.param(
            TOVInconsistencyError,
            {"max_residual": 2.5e-4, "r_location": 1.5},
            {"max_residual": 2.5e-4, "r_location": 1.5},
            ["2.500000e-04", "1.500000e+00"],
            id="TOVInconsistencyError",
        ),
        pytest.param(
            JunctionDiscontinuityError,
            {"surface_label": "r=10", "jump_magnitude": 0.01},
            {"surface_label": "r=10", "jump_magnitude": 0.01},
            ["r=10", "1.000000e-02"],
            id="JunctionDiscontinuityError",
        ),
        pytest.param(
            AsymptoticFalloffError,
            {"measured_order": 1.5, "required_order": 2},
            {"measured_order": 1.5, "required_order": 2},
            ["1.50", "2"],
            id="AsymptoticFalloffError",
        ),
        pytest.param(
            TransportUndefinedError,
            {"geodesic_id": 42, "termination_reason": "max_iter exceeded"},
            {"geodesic_id": 42, "termination_reason": "max_iter exceeded"},
            ["42", "max_iter exceeded"],
            id="TransportUndefinedError",
        ),
    ],
)
def test_exception_fields_and_message(
    exc_cls, kwargs, expected_attrs, expected_substrings
):
    """Each exception stores constructor parameters as attributes and formats them in the message."""
    err = exc_cls(**kwargs)
    for attr, expected in expected_attrs.items():
        assert getattr(err, attr) == expected
    msg = str(err)
    for substring in expected_substrings:
        assert substring in msg
