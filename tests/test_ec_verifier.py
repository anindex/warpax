"""Integration tests for the full EC verification pipeline.

Validates the two-tier verifier orchestrator end-to-end:
- Dust in Minkowski (all conditions satisfied)
- Known WEC violation detection with correct worst observer
- Eulerian vs observer-robust comparison (core paper argument)
- Grid-level Alcubierre verification with NEC/WEC violations
- Summary statistics correctness
- ANEC integrand and stub
- Float64 dtype enforcement
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.energy_conditions import (
    ECGridResult,
    ECPointResult,
    ECSummary,
    verify_grid,
    verify_point,
)
from warpax.energy_conditions.verifier import (
    anec_integrand,
    anec_integral,
    compute_eulerian_ec,
)
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.grid import evaluate_curvature_grid
from warpax.geometry.types import GridSpec

# Standard Minkowski metric
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# 1. Dust in Minkowski (no violation)
# ---------------------------------------------------------------------------


class TestDustVerifyPoint:
    """T_{ab} for dust (rho=1, p=0). All ECs satisfied."""

    T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))

    def test_all_conditions_satisfied(self):
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert float(r.wec_margin) >= 0, f"WEC margin: {r.wec_margin}"
        assert float(r.nec_margin) >= 0, f"NEC margin: {r.nec_margin}"
        assert float(r.sec_margin) >= 0, f"SEC margin: {r.sec_margin}"
        assert float(r.dec_margin) >= 0, f"DEC margin: {r.dec_margin}"

    def test_he_type_is_one(self):
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert int(r.he_type) == 1

    def test_rho_and_pressures(self):
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert float(r.rho) == pytest.approx(1.0, abs=1e-10)
        assert jnp.allclose(r.pressures, jnp.zeros(3), atol=1e-10)

    def test_result_is_namedtuple(self):
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert isinstance(r, ECPointResult)
        # All fields accessible
        _ = r.he_type
        _ = r.eigenvalues
        _ = r.rho
        _ = r.pressures
        _ = r.nec_margin
        _ = r.wec_margin
        _ = r.sec_margin
        _ = r.dec_margin
        _ = r.worst_observer
        _ = r.worst_params

    def test_wec_margin_value(self):
        """For dust: WEC margin = rho = 1.0 (eigenvalue method)."""
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert float(r.wec_margin) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Known WEC violation
# ---------------------------------------------------------------------------


class TestWECViolationVerifyPoint:
    """Negative energy density: rho = -0.5."""

    T_bad = jnp.diag(jnp.array([-0.5, 0.0, 0.0, 0.0]))

    def test_wec_violated(self):
        r = verify_point(self.T_bad, ETA, n_starts=4)
        assert float(r.wec_margin) < 0

    def test_worst_observer_returned(self):
        r = verify_point(self.T_bad, ETA, n_starts=4)
        assert r.worst_observer.shape == (4,)
        assert r.worst_params.shape == (3,)


# ---------------------------------------------------------------------------
# 3. Eulerian vs observer-robust comparison (CORE PAPER ARGUMENT)
# ---------------------------------------------------------------------------


class TestEulerianVsObserverRobust:
    """Construct T_{ab} where Eulerian observer sees no WEC violation
    but a boosted observer does.

    For an off-diagonal T_{ab} in Minkowski:
    T = [[rho, q, 0, 0],
         [q,   p, 0, 0],
         [0,   0, p, 0],
         [0,   0, 0, p]]

    Eulerian: T_{ab} u^a u^b = T_{00} = rho (positive).
    Boosted in x: T_{ab} u^a u^b = rho*cosh^2 + 2q*cosh*sinh + p*sinh^2
    The minimum can be negative when q is large enough.

    With rho=0.5, q=-0.4, p=0.3:
    f(z) = 0.5*cosh^2(z) - 0.8*cosh(z)*sinh(z) + 0.3*sinh^2(z)
    f'(z) = (0.5+0.3)*sinh(2z)/2 - 0.8*cosh(2z) = 0.8*sinh(2z)/2 - 0.8*cosh(2z)
    Minimum when: tanh(2z) = 2, which has no solution, but the function does
    decrease below rho for large enough z when rho + p < 2|q|.
    Here rho+p = 0.8, 2|q| = 0.8, so borderline. Use larger q.
    """

    # rho=0.5, q=-0.6, p=0.3: rho+p=0.8, 2|q|=1.2 > 0.8 -> violation exists
    rho = 0.5
    q = -0.6
    p = 0.3
    T_offdiag = jnp.array([
        [rho, q, 0.0, 0.0],
        [q, p, 0.0, 0.0],
        [0.0, 0.0, p, 0.0],
        [0.0, 0.0, 0.0, p],
    ])

    def test_eulerian_wec_satisfied(self):
        """Eulerian observer sees WEC margin = rho = 0.5 >= 0."""
        e = compute_eulerian_ec(self.T_offdiag, ETA)
        assert float(e["wec"]) >= 0, f"Eulerian WEC = {e['wec']}, expected >= 0"

    def test_observer_robust_wec_violated(self):
        """Boosted observer finds WEC violation (margin < 0)."""
        r = verify_point(self.T_offdiag, ETA, n_starts=16, zeta_max=5.0)
        assert float(r.wec_margin) < 0, (
            f"Observer-robust WEC margin = {r.wec_margin}, expected < 0"
        )

    def test_observer_robust_finds_worse_than_eulerian(self):
        """The observer-robust margin is strictly lower than the Eulerian margin."""
        e = compute_eulerian_ec(self.T_offdiag, ETA)
        r = verify_point(self.T_offdiag, ETA, n_starts=16, zeta_max=5.0)
        assert float(r.wec_margin) < float(e["wec"]), (
            f"robust={r.wec_margin} should be < eulerian={e['wec']}"
        )


# ---------------------------------------------------------------------------
# 4. Small Alcubierre grid (5x5x5)
# ---------------------------------------------------------------------------


class TestAlcubierreGrid:
    """End-to-end grid verification on Alcubierre metric.

    The Alcubierre warp drive is known to violate NEC and WEC
    in the bubble wall region (Alcubierre 1994).
    """

    @pytest.fixture(scope="class")
    def alcubierre_grid_data(self):
        """Compute curvature grid for Alcubierre metric (5x5x5)."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        grid = GridSpec(
            bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
            shape=(5, 5, 5),
        )
        result = evaluate_curvature_grid(metric, grid, compute_invariants=False)
        return result, grid

    def test_grid_ec_runs(self, alcubierre_grid_data):
        """verify_grid completes without error on Alcubierre data."""
        result, grid = alcubierre_grid_data
        T_field = result.stress_energy   # (5, 5, 5, 4, 4)
        g_field = result.metric          # (5, 5, 5, 4, 4)
        g_inv_field = result.metric_inv  # (5, 5, 5, 4, 4)

        ec = verify_grid(T_field, g_field, g_inv_field, n_starts=4)
        assert isinstance(ec, ECGridResult)
        assert ec.he_types.shape == (5, 5, 5)
        assert ec.wec_margins.shape == (5, 5, 5)

    def test_some_type_i_points(self, alcubierre_grid_data):
        """Some interior points should classify as Type I."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy, result.metric, result.metric_inv,
            n_starts=4,
        )
        n_type_i = jnp.sum(ec.he_types == 1.0)
        assert int(n_type_i) > 0, "Expected some Type I points in Alcubierre grid"

    def test_nec_violation_detected(self, alcubierre_grid_data):
        """NEC violations in the bubble wall region (Alcubierre 1994)."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy, result.metric, result.metric_inv,
            n_starts=4,
        )
        # Use nanmin: center point (r_s=0) may produce NaN from coordinate singularity
        assert float(jnp.nanmin(ec.nec_margins)) < 0, (
            f"Expected NEC violation but min margin = {jnp.nanmin(ec.nec_margins)}"
        )

    def test_wec_violation_detected(self, alcubierre_grid_data):
        """WEC violations in the bubble wall region (Alcubierre 1994)."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy, result.metric, result.metric_inv,
            n_starts=4,
        )
        # Use nanmin: center point (r_s=0) may produce NaN from coordinate singularity
        assert float(jnp.nanmin(ec.wec_margins)) < 0, (
            f"Expected WEC violation but min margin = {jnp.nanmin(ec.wec_margins)}"
        )

    def test_summary_fraction_violated(self, alcubierre_grid_data):
        """Summary statistics report fraction_violated > 0 for WEC/NEC."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy, result.metric, result.metric_inv,
            n_starts=4,
        )
        assert float(ec.nec_summary.fraction_violated) > 0
        assert float(ec.wec_summary.fraction_violated) > 0


# ---------------------------------------------------------------------------
# 5. Eulerian comparison on grid
# ---------------------------------------------------------------------------


class TestEulerianGridComparison:
    """Run verify_grid with compute_eulerian=True on Alcubierre data."""

    @pytest.fixture(scope="class")
    def alcubierre_grid_data(self):
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        grid = GridSpec(
            bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
            shape=(5, 5, 5),
        )
        result = evaluate_curvature_grid(metric, grid, compute_invariants=False)
        return result, grid

    def test_eulerian_grid_runs(self, alcubierre_grid_data):
        """verify_grid with compute_eulerian=True completes."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy, result.metric, result.metric_inv,
            n_starts=4,
            compute_eulerian=True,
        )
        assert isinstance(ec, ECGridResult)


# ---------------------------------------------------------------------------
# 6. Summary statistics correctness
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Verify summary statistics are finite and reasonable."""

    def test_summary_types(self):
        """Summary fields are ECSummary NamedTuples."""
        T_field = jnp.zeros((3, 3, 3, 4, 4))
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        # Use a simple diagonal T for quick test
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert isinstance(ec.nec_summary, ECSummary)
        assert isinstance(ec.wec_summary, ECSummary)
        assert isinstance(ec.sec_summary, ECSummary)
        assert isinstance(ec.dec_summary, ECSummary)

    def test_dust_no_violations(self):
        """For all-dust grid, fraction_violated should be 0."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert float(ec.wec_summary.fraction_violated) == 0.0
        assert float(ec.nec_summary.fraction_violated) == 0.0

    def test_summary_finite(self):
        """All summary fields are finite."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        for summary in [ec.nec_summary, ec.wec_summary, ec.sec_summary, ec.dec_summary]:
            assert jnp.isfinite(summary.fraction_violated)
            assert jnp.isfinite(summary.max_violation)
            assert jnp.isfinite(summary.min_margin)


# ---------------------------------------------------------------------------
# 7. Worst observer stored correctly
# ---------------------------------------------------------------------------


class TestWorstObserver:
    """For a violated point, worst_observer is a unit timelike vector."""

    T_bad = jnp.diag(jnp.array([-0.5, 0.0, 0.0, 0.0]))

    def test_worst_observer_is_timelike(self):
        """g_{ab} u^a u^b = -1 for worst observer."""
        r = verify_point(self.T_bad, ETA, n_starts=8)
        norm_sq = float(jnp.einsum("a,ab,b->", r.worst_observer, ETA, r.worst_observer))
        assert norm_sq == pytest.approx(-1.0, abs=1e-4), (
            f"g_ab u^a u^b = {norm_sq}, expected -1"
        )

    def test_worst_params_shape(self):
        """worst_params = (zeta, theta, phi)."""
        r = verify_point(self.T_bad, ETA, n_starts=8)
        assert r.worst_params.shape == (3,)

    def test_worst_params_ranges(self):
        """Parameters in expected ranges: zeta >= 0, 0 <= theta <= pi, 0 <= phi <= 2pi."""
        r = verify_point(self.T_bad, ETA, n_starts=8)
        zeta, theta, phi = float(r.worst_params[0]), float(r.worst_params[1]), float(r.worst_params[2])
        assert zeta >= -1e-6, f"zeta={zeta} should be >= 0"
        assert -1e-6 <= theta <= jnp.pi + 1e-6, f"theta={theta} out of range"
        assert -1e-6 <= phi <= 2 * jnp.pi + 1e-6, f"phi={phi} out of range"


# ---------------------------------------------------------------------------
# 8. ANEC integrand
# ---------------------------------------------------------------------------


class TestANECIntegrand:
    """Pointwise ANEC integrand returns correct scalar."""

    def test_dust_null_vector(self):
        """For dust T = diag(1,0,0,0) and null k=(1,1,0,0): T_{ab} k^a k^b = 1."""
        T = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        val = anec_integrand(T, k)
        assert float(val) == pytest.approx(1.0, abs=1e-12)

    def test_sign_matches(self):
        """For negative-energy T: T_{ab} k^a k^b < 0."""
        T = jnp.diag(jnp.array([-0.5, -0.5, 0.0, 0.0]))
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        val = anec_integrand(T, k)
        expected = float(jnp.einsum("a,ab,b->", k, T, k))
        assert float(val) == pytest.approx(expected, abs=1e-12)

    def test_scalar_output(self):
        """Output is a scalar."""
        T = jnp.diag(jnp.array([1.0, 0.1, 0.1, 0.1]))
        k = jnp.array([1.0, 0.0, 0.0, 1.0])
        val = anec_integrand(T, k)
        assert val.shape == ()


# ---------------------------------------------------------------------------
# 9. ANEC line-integral stub
# ---------------------------------------------------------------------------


class TestANECIntegralStub:
    """anec_integral raises NotImplementedError."""

    def test_raises(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            anec_integral(None, None)


# ---------------------------------------------------------------------------
# 10. Float64 dtype
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9b. ECGridResult new fields
# ---------------------------------------------------------------------------


class TestECGridResultNewFields:
    """Verify ECGridResult contains optimizer margins and classification stats."""

    def test_opt_margins_present(self):
        """ECGridResult has nec/wec/sec/dec_opt_margins fields."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert ec.nec_opt_margins is not None
        assert ec.wec_opt_margins is not None
        assert ec.sec_opt_margins is not None
        assert ec.dec_opt_margins is not None
        assert ec.nec_opt_margins.shape == (3, 3, 3)

    def test_classification_stats_present(self):
        """ECGridResult has n_type_i..n_type_iv and max_imag_eigenvalue."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert ec.n_type_i is not None
        assert ec.n_type_ii is not None
        assert ec.n_type_iii is not None
        assert ec.n_type_iv is not None
        assert ec.max_imag_eigenvalue is not None

    def test_dust_all_type_i(self):
        """For all-dust grid, all points should be Type I."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert ec.n_type_i == 27
        assert ec.n_type_ii == 0
        assert ec.n_type_iii == 0
        assert ec.n_type_iv == 0


# ---------------------------------------------------------------------------
# 10b. DEC future-directedness
# ---------------------------------------------------------------------------


class TestDECFutureDirectedness:
    """Verify DEC checks future-directedness of energy flux, not just causality.

    Construct a synthetic tensor where the flux j^a = -T^a_b u^b is causal
    (timelike) but PAST-directed. DEC should be violated.

    In Minkowski with T^a_b = diag(-rho, p1, p2, p3), the Eulerian flux is
    j^a = (rho, 0, 0, 0), which is future-directed when rho > 0.
    To make j past-directed, we need j^0 < 0. For the Eulerian observer
    u = (1, 0, 0, 0): j^a = -T^a_b u^b = -T^a_0 = (rho, -T^1_0, ...).

    Strategy: use T_ab with off-diagonal components that make j^0 < 0
    for some boosted observer while keeping the flux causal.
    Simpler: T_ab = diag(-(-rho), p, p, p) with rho < 0 makes the Eulerian
    flux j = (rho, 0, 0, 0) past-directed (j^0 = rho < 0) while timelike
    (j^a j_a = -rho^2 < 0).
    But this also violates WEC (rho < 0), so DEC would catch it anyway.

    Better: construct T^a_b non-diagonal so j is past-directed for one
    observer but causal. Use mixed tensor directly.
    """

    def test_past_directed_flux_detected(self):
        """DEC catches past-directed but causal flux.

        Construct T_ab in Minkowski so that:
        - WEC satisfied (rho > 0 for all observers in range)
        - Flux is timelike (causal) but PAST-directed for the Eulerian observer

        T_ab with large T_{01} component creates past-directed flux for
        boosted observers.

        Actually, the simplest approach: build T_ab where the Eulerian
        j = -T^a_b n^b has j.n > 0 (past-directed in our convention).
        T_ab = diag(-rho, p, p, p) with rho > 0 always gives future-directed
        flux. We need off-diagonal terms.

        For Eulerian n^a = (1,0,0,0) in Minkowski:
        j^a = -eta^{ac} T_{cb} n^b = -eta^{ac} T_{c0}
        j^0 = -eta^{00} T_{00} = T_{00} (since eta^{00}=-1, T_{00}=rho)
        j^1 = -eta^{11} T_{10} = -T_{10}

        For j to be past-directed: j.n < 0 is future, j.n > 0 is past.
        j_a n^a = g_{ab} j^b n^a = eta_{0b} j^b = -j^0
        So -j^0 > 0 means future-directed, -j^0 < 0 means past-directed.
        j^0 = T_{00} = rho, so -rho < 0 means past-directed, i.e. rho > 0
        gives FUTURE-directed flux for Eulerian observer.

        To get past-directed flux, need j^0 < 0, i.e. T_{00} < 0 (negative
        energy density). But that also violates WEC.

        The only way to have past-directed causal flux WITHOUT WEC violation
        is with a boosted observer where the flux direction flips. This is
        actually only possible for non-Type-I tensors.

        For this test: use a Type I tensor where rho < 0, |p_i| < |rho|
        (DEC eigenvalue check: rho >= |p_i| fails), BUT also verify the
        optimizer catches the past-directed flux aspect.

        Simpler approach: just verify that the DEC optimizer objective now
        includes both causality and future-directedness by checking on a
        known tensor.
        """
        from warpax.energy_conditions.optimization import _dec_objective
        from warpax.energy_conditions.observer import (
            compute_orthonormal_tetrad,
            timelike_from_boost_vector,
        )

        # Construct T_ab in Minkowski where Eulerian flux is past-directed:
        # T_ab = diag(rho, p, p, p) in covariant form, with rho = -1.0
        # This gives j^0 = -1 (past-directed, since -j^0 = 1 > 0... wait)
        #
        # Let me recalculate carefully in Minkowski:
        # T_{ab} with T_{00} = -1, T_{11}=T_{22}=T_{33}=0.5 (past-directed rho)
        # g^{ab} = eta^{ab} = diag(-1,1,1,1)
        # T^a_b = g^{ac} T_{cb}
        # T^0_0 = g^{00} T_{00} = (-1)(-1) = 1
        # Actually no. T_{00} = -1 means in covariant form. Let me think about
        # what rho means. rho = T_{ab} n^a n^b = T_{00} * 1 * 1 = T_{00}.
        # So T_{00} = -1 means rho = -1 < 0 -> WEC violated.
        # j^0 = -T^0_b n^b = -T^0_0 = -(g^{00} T_{00}) = -(-1)(-1) = -1
        # So j^0 = -1 < 0, j_0 = g_{00} j^0 = (-1)(-1) = 1
        # j.n = j_a n^a = j_0 * 1 = 1 > 0 -> past-directed
        # future_margin = -j.n = -1 < 0 -> correctly detects violation
        #
        # Also flux_causality: j = (-1, 0, 0, 0)
        # g_{ab} j^a j^b = -1 < 0 -> timelike -> flux_causality = 1 > 0
        # So the past-directed check is the binding constraint here.

        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        g_ab = ETA
        tetrad = compute_orthonormal_tetrad(g_ab)
        g_inv = jnp.linalg.inv(g_ab)
        T_mixed = g_inv @ T_ab
        zeta_max = jnp.float64(5.0)

        # Eulerian observer (w = 0)
        w = jnp.zeros(3)
        args = (T_mixed, g_ab, tetrad, zeta_max)
        obj = _dec_objective(w, args)

        # The objective should be negative (violation detected)
        assert float(obj) < 0, (
            f"DEC objective at Eulerian = {obj}, expected < 0 for past-directed flux"
        )

    def test_dec_future_vs_causality(self):
        """Verify the DEC objective returns min(causality, future-directedness)."""
        from warpax.energy_conditions.optimization import _dec_objective
        from warpax.energy_conditions.observer import compute_orthonormal_tetrad

        # Perfect fluid dust: rho=1, p=0. Flux is future-directed and causal.
        T_ab = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        g_ab = ETA
        tetrad = compute_orthonormal_tetrad(g_ab)
        g_inv = jnp.linalg.inv(g_ab)
        T_mixed = g_inv @ T_ab
        zeta_max = jnp.float64(5.0)

        w = jnp.zeros(3)
        args = (T_mixed, g_ab, tetrad, zeta_max)
        obj = _dec_objective(w, args)

        # For dust at Eulerian: j = (1, 0, 0, 0)
        # causality = -g_{ab} j^a j^b = -(-1) = 1
        # future = -(j_a n^a) = -(g_{0b} j^b * n^0) = -(-1*1*1) = 1
        # min(1, 1) = 1
        assert float(obj) > 0, f"DEC objective for dust = {obj}, expected > 0"

    def test_eulerian_dec_catches_past_directed(self):
        """Eulerian DEC also detects past-directed flux."""
        # Same past-directed tensor as above
        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        e = compute_eulerian_ec(T_ab, ETA)
        # DEC margin should be negative (past-directed flux + WEC violation)
        assert float(e["dec"]) < 0, (
            f"Eulerian DEC = {e['dec']}, expected < 0 for past-directed flux"
        )

    def test_verify_point_dec_catches_past_directed(self):
        """Full verify_point pipeline detects DEC violation from past-directed flux."""
        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        r = verify_point(T_ab, ETA, n_starts=8)
        assert float(r.dec_margin) < 0, (
            f"DEC margin = {r.dec_margin}, expected < 0"
        )


# ---------------------------------------------------------------------------
# 10. Float64 dtype
# ---------------------------------------------------------------------------


class TestFloat64Dtype:
    """All output margins and observer vectors are float64."""

    T_test = jnp.diag(jnp.array([1.0, 0.1, 0.1, 0.1]))

    def test_verify_point_dtypes(self):
        r = verify_point(self.T_test, ETA, n_starts=4)
        assert r.nec_margin.dtype == jnp.float64
        assert r.wec_margin.dtype == jnp.float64
        assert r.sec_margin.dtype == jnp.float64
        assert r.dec_margin.dtype == jnp.float64
        assert r.worst_observer.dtype == jnp.float64
        assert r.worst_params.dtype == jnp.float64
        assert r.eigenvalues.dtype == jnp.float64

    def test_eulerian_ec_dtypes(self):
        e = compute_eulerian_ec(self.T_test, ETA)
        for name in ("wec", "nec", "sec", "dec"):
            assert e[name].dtype == jnp.float64, f"{name} dtype: {e[name].dtype}"

    def test_anec_integrand_dtype(self):
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        val = anec_integrand(self.T_test, k)
        assert val.dtype == jnp.float64
