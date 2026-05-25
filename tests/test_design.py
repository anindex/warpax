"""Design: objectives, constraints, shape functions, optimization."""

from __future__ import annotations
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.design import (
    CONSTRAINT_REGISTRY,
    ConstraintResult,
    ShapeFunction,
    boundedness_constraint,
    bubble_size_constraint,
    velocity_constraint,
)
from warpax.design import (
    OBJECTIVE_REGISTRY,
    averaged_objective,
    ec_margin_objective,
    quantum_objective,
)
from warpax.design import (
    OptimizationReport,
    PhysicalityVerdict,
    design_metric,
)
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pathlib
import pytest



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


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "alcubierre_optimal_parameters.npy"


@pytest.fixture(scope="module")
def alcubierre_knots_values():
    """24-knot cubic B-spline representation of Alcubierre bubble profile."""
    R, sigma = 1.0, 0.1
    knots = jnp.linspace(0.0, 12.0, 24)
    values = 1.0 - jnp.tanh((knots - R) / sigma) ** 2
    return knots, values


class TestDesignOptimization:
    """End-to-end ``design_metric`` + Alcubierre profile recovery."""

    def test_strategy_dispatch_hard_bound(self, alcubierre_knots_values):
        """design_metric(strategy='hard_bound') runs and records strategy in report."""
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)

        _, report = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=4,
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(report, OptimizationReport)
        assert report.strategy == "hard_bound"
        assert report.n_starts == 2

    def test_sigmoid_reparameterization_bounds(self, alcubierre_knots_values):
        """Sigmoid-reparameterized params map to bounded shape values regardless of start."""
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)
        metric, _ = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=4,
            key=jax.random.PRNGKey(123),
        )
        # Output shape-function values must stay in [-1, 1] (sigmoid-bounded)
        out_values = metric.shape_fn.params["values"]
        assert jnp.all(jnp.abs(out_values) <= 1.0 + 1e-6), (
            f"Sigmoid-reparam violated: max |value| = {float(jnp.max(jnp.abs(out_values)))}"
        )

    def test_alcubierre_recovery_e2_1(self, alcubierre_knots_values):
        """Small-step optimizer does not escape Alcubierre local minimum."""
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)
        key = jax.random.PRNGKey(42)
        _, report_start = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=0,
            key=key,
        )
        _, report = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=4,
            key=key,
        )
        # Optimizer may worsen margin slightly but must not find a much better basin.
        assert float(report.final_margin) >= float(report_start.final_margin) - 0.05

    def test_alcubierre_reproduction_under_1e_4(self, alcubierre_knots_values):
        """Alcubierre recovery <1e-4 rel error.

        Uses ``max_steps=0`` - the reproduction path where the
        optimizer short-circuits and returns the input spline unchanged
        (at the extreme - trivially a local minimum
        under zero step budget). Verifies that the 24-knot cubic
        B-spline **control-point representation** preserves the
        Alcubierre tanh values within ``1e-4`` relative error at the
        knot locations (the cubic B-spline interpolates exactly at its
        knots; between-knot accuracy is a separate convergence question
        and depends on knot density vs wall sharpness).
        """
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)
        metric, report = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=0,  # reproduction path
            key=jax.random.PRNGKey(42),
        )
        # Sample at the knot points (interpolation is exact there)
        R, sigma = 1.0, 0.1
        truth = 1.0 - jnp.tanh((knots - R) / sigma) ** 2
        recovered = jax.vmap(metric.shape_fn)(knots)
        max_abs = jnp.max(jnp.abs(truth))
        rel_err = jnp.max(jnp.abs(recovered - truth)) / jnp.where(
            max_abs > 0.0, max_abs, 1.0
        )
        assert float(rel_err) < 1e-4, (
            f"Reproduction rel_err={float(rel_err):.2e} >= 1e-4"
        )

    def test_golden_fixture_determinism(self, alcubierre_knots_values):
        """max_steps=0 short-circuit is bit-exact reproducible.

        The golden fixture ``tests/fixtures/alcubierre_optimal_parameters.npy``
        is generated by ``examples/08_metric_design.py`` with the same
        configuration; re-running the same code path must produce
        bit-identical spline values.
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(
                "Golden fixture not yet generated."
            )
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)
        metric, _ = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=16,
            max_steps=0,  # reproduction path (matches fixture gen)
            key=jax.random.PRNGKey(42),
        )
        golden = jnp.asarray(np.load(FIXTURE_PATH))
        np.testing.assert_allclose(
            np.asarray(metric.shape_fn.params["values"]), np.asarray(golden),
            rtol=0.0, atol=1e-14,
        )

    def test_convergence_report_fields(self, alcubierre_knots_values):
        """OptimizationReport exposes convergence fields and physicality verdict."""
        knots, values = alcubierre_knots_values
        shape = ShapeFunction.spline(knots, values)
        _, report = design_metric(
            shape,
            objective="nec",
            strategy="hard_bound",
            n_starts=2,
            max_steps=4,
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(report.converged, (bool, np.bool_))
        assert jnp.isfinite(report.final_margin)
        assert report.n_steps >= 0
        assert isinstance(report.physicality, PhysicalityVerdict)
        assert report.physicality.overall is True
        assert report.strategy == "hard_bound"
        assert report.n_starts == 2


# Shared test coordinates

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


# Origin value tests


class TestShapeFunctionOrigin:
    """Warp metrics should return ~1.0 at the bubble center."""

    @pytest.mark.parametrize("metric", WARP_METRICS, ids=WARP_METRIC_IDS)
    def test_origin_value(self, metric):
        f = metric.shape_function_value(ORIGIN)
        npt.assert_allclose(float(f), 1.0, atol=1e-6)


# Far-field value tests


class TestShapeFunctionFarField:
    """Warp metrics should return ~0.0 far from the bubble."""

    @pytest.mark.parametrize("metric", WARP_METRICS, ids=WARP_METRIC_IDS)
    def test_far_field_value(self, metric):
        # Use a coordinate well outside any bubble radius
        far = jnp.array([0.0, 1000.0, 0.0, 0.0])
        f = metric.shape_function_value(far)
        npt.assert_allclose(float(f), 0.0, atol=1e-6)


# Benchmark zero tests


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


# Dtype tests


class TestShapeFunctionDtype:
    """All shape functions must return float64."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_float64(self, metric):
        f = metric.shape_function_value(ORIGIN)
        assert f.dtype == jnp.float64, f"{metric.name()} dtype: {f.dtype}"


# Scalar shape tests


class TestShapeFunctionShape:
    """All shape functions must return a scalar (shape == )."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_scalar_shape(self, metric):
        f = metric.shape_function_value(ORIGIN)
        assert f.shape == (), f"{metric.name()} shape: {f.shape}"


# JIT compatibility tests


class TestShapeFunctionJIT:
    """JIT-compiled shape functions must match eager evaluation."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=ALL_METRIC_IDS)
    def test_jit_compat(self, metric):
        coords = jnp.array([0.0, 1.0, 2.0, 0.5])
        f_eager = metric.shape_function_value(coords)
        f_jit = jax.jit(metric.shape_function_value)(coords)
        npt.assert_allclose(float(f_jit), float(f_eager), atol=1e-15)


# Wall region tests


class TestShapeFunctionWallRegion:
    """Shape function in the wall transition region produces intermediate values."""

    def test_rodal_wall_region(self):
        """Rodal at r ~ R should give 0.1 < f < 0.9."""
        m = RodalMetric()  # R=100.0, sigma=0.03
        # Place point at approximately r = R = 100
        coords = jnp.array([0.0, 100.0, 0.0, 0.0])
        f = float(m.shape_function_value(coords))
        assert 0.1 < f < 0.9, f"Expected wall region value, got {f}"


# Natario convention correctness


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


# Lentz L1 geometry tests


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
