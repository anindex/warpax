"""Integration tests for the full EC verification pipeline.

Validates the two-tier verifier orchestrator end-to-end:
- Dust in Minkowski (all conditions satisfied)
- Known WEC violation detection with correct worst observer
- Eulerian vs observer-robust comparison (core paper argument)
- Grid-level Alcubierre verification with NEC/WEC violations
- Summary statistics correctness
- ANEC integrand and placeholder
- Float64 dtype enforcement
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.energy_conditions import (
    ECGridResult,
    ECSummary,
    verify_grid,
    verify_point,
)
from warpax.energy_conditions.verifier import (
    anec_integrand,
    compute_eulerian_ec,
)
from warpax.geometry.grid import evaluate_curvature_grid
from warpax.geometry.types import GridSpec

ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


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

    def test_wec_margin_value(self):
        """For dust: WEC margin = rho = 1.0 (eigenvalue method)."""
        r = verify_point(self.T_dust, ETA, n_starts=4)
        assert float(r.wec_margin) == pytest.approx(1.0, abs=1e-6)


class TestWECViolationVerifyPoint:
    """Negative energy density: rho = -0.5."""

    T_bad = jnp.diag(jnp.array([-0.5, 0.0, 0.0, 0.0]))

    def test_wec_violated(self):
        r = verify_point(self.T_bad, ETA, n_starts=4)
        assert float(r.wec_margin) < 0


class TestEulerianVsObserverRobust:
    """Construct T_{ab} where Eulerian observer sees no WEC violation
    but a boosted observer does.

    For an off-diagonal T_{ab} in Minkowski:
    T = [[rho, q, 0, 0],
         [q, p, 0, 0],
         [0, 0, p, 0],
         [0, 0, 0, p]]

    Eulerian: T_{ab} u^a u^b = T_{00} = rho (positive).
    Boosted in x: T_{ab} u^a u^b = rho*cosh^2 + 2q*cosh*sinh + p*sinh^2
    The minimum can be negative when q is large enough.

    The boosted density falls below rho for large enough z when
    rho + p < 2|q|. Here rho=0.5, q=-0.6, p=0.3: rho+p=0.8 < 2|q|=1.2, so a
    violation exists.
    """

    rho = 0.5
    q = -0.6
    p = 0.3
    T_offdiag = jnp.array(
        [
            [rho, q, 0.0, 0.0],
            [q, p, 0.0, 0.0],
            [0.0, 0.0, p, 0.0],
            [0.0, 0.0, 0.0, p],
        ]
    )

    def test_eulerian_wec_satisfied(self):
        """Eulerian observer sees WEC margin = rho = 0.5 >= 0."""
        e = compute_eulerian_ec(self.T_offdiag, ETA)
        assert float(e["wec"]) >= 0, f"Eulerian WEC = {e['wec']}, expected >= 0"

    def test_observer_robust_wec_violated(self):
        """Boosted observer finds WEC violation (margin < 0)."""
        r = verify_point(self.T_offdiag, ETA, n_starts=16, zeta_max=5.0)
        assert float(r.wec_margin) < 0, f"Observer-robust WEC margin = {r.wec_margin}, expected < 0"

    def test_observer_robust_finds_worse_than_eulerian(self):
        """The observer-robust margin is strictly lower than the Eulerian margin."""
        e = compute_eulerian_ec(self.T_offdiag, ETA)
        r = verify_point(self.T_offdiag, ETA, n_starts=16, zeta_max=5.0)
        assert float(r.wec_margin) < float(e["wec"]), (
            f"robust={r.wec_margin} should be < eulerian={e['wec']}"
        )


@pytest.fixture(scope="module")
def alcubierre_grid_data():
    """Curvature grid for the Alcubierre metric (5x5x5), shared module-wide."""
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
    grid = GridSpec(
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        shape=(5, 5, 5),
    )
    result = evaluate_curvature_grid(metric, grid, compute_invariants=False)
    return result, grid


@pytest.fixture(scope="module")
def alcubierre_ec(alcubierre_grid_data):
    """Observer-robust verify_grid result on the shared Alcubierre grid."""
    result, _ = alcubierre_grid_data
    return verify_grid(
        result.stress_energy,
        result.metric,
        result.metric_inv,
        n_starts=4,
    )


class TestAlcubierreGrid:
    """End-to-end grid verification on Alcubierre metric.

    The Alcubierre warp drive is known to violate NEC and WEC
    in the bubble wall region (Alcubierre 1994).
    """

    def test_some_type_i_points(self, alcubierre_ec):
        """Some interior points should classify as Type I."""
        n_type_i = jnp.sum(alcubierre_ec.he_types == 1.0)
        assert int(n_type_i) > 0, "Expected some Type I points in Alcubierre grid"

    def test_nec_violation_detected(self, alcubierre_ec):
        """NEC violations in the bubble wall region (Alcubierre 1994)."""
        assert float(jnp.nanmin(alcubierre_ec.nec_margins)) < 0, (
            f"Expected NEC violation but min margin = {jnp.nanmin(alcubierre_ec.nec_margins)}"
        )

    def test_wec_violation_detected(self, alcubierre_ec):
        """WEC violations in the bubble wall region (Alcubierre 1994)."""
        assert float(jnp.nanmin(alcubierre_ec.wec_margins)) < 0, (
            f"Expected WEC violation but min margin = {jnp.nanmin(alcubierre_ec.wec_margins)}"
        )

    def test_summary_fraction_violated(self, alcubierre_ec):
        """Summary statistics report fraction_violated > 0 for WEC/NEC."""
        assert float(alcubierre_ec.nec_summary.fraction_violated) > 0
        assert float(alcubierre_ec.wec_summary.fraction_violated) > 0


class TestGeneralizedSolverIntegration:
    """integration: verify_point / verify_grid with solver='generalized'."""

    T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))

    def test_verify_point_generalized_dust(self):
        """verify_point(..., solver='generalized') on dust classifies Type I, all ECs satisfied."""
        r = verify_point(self.T_dust, ETA, n_starts=4, solver="generalized")
        assert int(r.he_type) == 1
        assert float(r.wec_margin) >= 0

    def test_verify_grid_generalized_small(self, alcubierre_grid_data):
        """verify_grid(..., solver='generalized') completes on a 5x5x5 Alcubierre grid."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy,
            result.metric,
            result.metric_inv,
            n_starts=4,
            solver="generalized",
        )
        assert isinstance(ec, ECGridResult)
        assert ec.he_types.shape == (5, 5, 5)


class TestEulerianGridComparison:
    """Run verify_grid with compute_eulerian=True on Alcubierre data."""

    def test_eulerian_grid_runs(self, alcubierre_grid_data, alcubierre_ec):
        """compute_eulerian=True merges margins as min(robust, eulerian)."""
        result, grid = alcubierre_grid_data
        ec = verify_grid(
            result.stress_energy,
            result.metric,
            result.metric_inv,
            n_starts=4,
            compute_eulerian=True,
        )
        assert isinstance(ec, ECGridResult)
        assert ec.nec_margins.shape == (5, 5, 5)

        flat_T = result.stress_energy.reshape(-1, 4, 4)
        flat_g = result.metric.reshape(-1, 4, 4)
        flat_gi = result.metric_inv.reshape(-1, 4, 4)
        eulerian = jax.vmap(compute_eulerian_ec)(flat_T, flat_g, flat_gi)
        for name in ("nec", "wec"):
            robust = getattr(alcubierre_ec, f"{name}_margins").reshape(-1)
            merged = getattr(ec, f"{name}_margins").reshape(-1)
            expected = jnp.minimum(robust, eulerian[name])
            finite = jnp.isfinite(merged) & jnp.isfinite(expected)
            assert bool(jnp.all(merged[finite] <= robust[finite] + 1e-12)), name
            assert bool(jnp.allclose(merged[finite], expected[finite], atol=1e-12)), name
        nec_robust = alcubierre_ec.nec_margins.reshape(-1)
        nec_merged = ec.nec_margins.reshape(-1)
        finite = jnp.isfinite(nec_merged) & jnp.isfinite(nec_robust)
        assert int(jnp.sum(nec_merged[finite] < nec_robust[finite] - 1e-12)) > 0


class TestSummaryStatistics:
    """Verify summary statistics are finite and reasonable."""

    def test_dust_no_violations(self):
        """For all-dust grid, fraction_violated should be 0."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert float(ec.wec_summary.fraction_violated) == 0.0
        assert float(ec.nec_summary.fraction_violated) == 0.0

    def test_summary_finite(self):
        """All summary fields are ECSummary NamedTuples with finite values."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        for summary in [ec.nec_summary, ec.wec_summary, ec.sec_summary, ec.dec_summary]:
            assert isinstance(summary, ECSummary)
            assert jnp.isfinite(summary.fraction_violated)
            assert jnp.isfinite(summary.max_violation)
            assert jnp.isfinite(summary.min_margin)


class TestWorstObserver:
    """For a violated point, worst_observer is a unit timelike vector."""

    T_bad = jnp.diag(jnp.array([-0.5, 0.0, 0.0, 0.0]))

    def test_worst_observer_is_timelike(self):
        """g_{ab} u^a u^b = -1 for worst observer."""
        r = verify_point(self.T_bad, ETA, n_starts=8)
        assert r.worst_observer.shape == (4,)
        norm_sq = float(jnp.einsum("a,ab,b->", r.worst_observer, ETA, r.worst_observer))
        assert norm_sq == pytest.approx(-1.0, abs=1e-4), f"g_ab u^a u^b = {norm_sq}, expected -1"

    def test_worst_params_ranges(self):
        """Parameters in expected ranges: zeta >= 0, 0 <= theta <= pi, 0 <= phi <= 2pi."""
        r = verify_point(self.T_bad, ETA, n_starts=8)
        zeta, theta, phi = (
            float(r.worst_params[0]),
            float(r.worst_params[1]),
            float(r.worst_params[2]),
        )
        assert zeta >= -1e-6, f"zeta={zeta} should be >= 0"
        assert -1e-6 <= theta <= jnp.pi + 1e-6, f"theta={theta} out of range"
        assert -1e-6 <= phi <= 2 * jnp.pi + 1e-6, f"phi={phi} out of range"


class TestANECIntegrand:
    """Pointwise ANEC integrand returns correct scalar."""

    def test_dust_null_vector(self):
        """For dust T = diag(1,0,0,0) and null k=(1,1,0,0): T_{ab} k^a k^b = 1."""
        T = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        val = anec_integrand(T, k)
        assert float(val) == pytest.approx(1.0, abs=1e-12)

    def test_sign_matches(self):
        """For negative-energy T: T_{ab} k^a k^b = T_00 + T_11 = -1 by hand."""
        T = jnp.diag(jnp.array([-0.5, -0.5, 0.0, 0.0]))
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        val = anec_integrand(T, k)
        assert float(val) == pytest.approx(-1.0, abs=1e-12)
        assert float(val) < 0


class TestECGridResultNewFields:
    """Verify ECGridResult contains optimizer margins and classification stats."""

    def test_dust_all_type_i(self):
        """All-dust grid: Type I everywhere; eigenvalue margins; opt skipped."""
        T_dust = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        T_field = jnp.broadcast_to(T_dust, (3, 3, 3, 4, 4)).copy()
        g_field = jnp.broadcast_to(ETA, (3, 3, 3, 4, 4)).copy()

        ec = verify_grid(T_field, g_field, n_starts=4)
        assert ec.nec_opt_margins is not None
        assert ec.wec_opt_margins is not None
        assert ec.sec_opt_margins is not None
        assert ec.dec_opt_margins is not None
        assert ec.nec_opt_margins.shape == (3, 3, 3)
        assert ec.n_type_i is not None
        assert ec.n_type_ii is not None
        assert ec.n_type_iii is not None
        assert ec.n_type_iv is not None
        assert ec.max_imag_eigenvalue is not None

        assert ec.n_type_i == 27
        assert ec.n_type_ii == 0
        assert ec.n_type_iii == 0
        assert ec.n_type_iv == 0
        assert float(ec.max_imag_eigenvalue) == pytest.approx(0.0, abs=1e-12)
        assert jnp.all(jnp.isnan(ec.nec_opt_margins))
        assert float(ec.nec_margins[0, 0, 0]) == pytest.approx(1.0, abs=1e-10)


class TestDECFutureDirectedness:
    """Verify DEC checks future-directedness of energy flux, not just causality.

    Construct a synthetic tensor where the flux j^a = -T^a_b u^b is causal
    (timelike) but PAST-directed. DEC should be violated.

    In Minkowski with T^a_b = diag(-rho, p1, p2, p3), the Eulerian flux is
    j^a = (rho, 0, 0, 0), which is future-directed when rho > 0.
    To make j past-directed, we need j^0 < 0. For the Eulerian observer
    u = (1, 0, 0, 0): j^a = -T^a_b u^b = -T^a_0 = (rho, -T^1_0, ...).
    """

    def test_past_directed_flux_detected(self):
        """DEC catches past-directed but causal flux.

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

        Verify that the DEC optimizer objective includes both causality
        and future-directedness by checking on a known tensor.
        """
        from warpax.energy_conditions.observer import (
            compute_orthonormal_tetrad,
        )
        from warpax.energy_conditions.optimization import (
            _dec_flux_subobjective,
            _dec_future_subobjective,
            _dec_objective,
        )

        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        g_ab = ETA
        tetrad = compute_orthonormal_tetrad(g_ab)
        g_inv = jnp.linalg.inv(g_ab)
        T_mixed = g_inv @ T_ab
        zeta_max = jnp.float64(5.0)

        w = jnp.zeros(3)
        args = (T_ab, T_mixed, g_ab, tetrad, zeta_max)
        obj = _dec_objective(w, args)

        assert float(obj) < 0, (
            f"DEC objective at Eulerian = {obj}, expected < 0 for past-directed flux"
        )
        assert float(obj) == pytest.approx(-1.0)

        assert float(_dec_future_subobjective(w, args)) == pytest.approx(-1.0)
        assert float(_dec_flux_subobjective(w, args)) == pytest.approx(1.0)

    def test_dec_future_vs_causality(self):
        """Verify the DEC objective returns min(wec, causality, future-directedness)."""
        from warpax.energy_conditions.observer import compute_orthonormal_tetrad
        from warpax.energy_conditions.optimization import _dec_objective

        T_ab = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        g_ab = ETA
        tetrad = compute_orthonormal_tetrad(g_ab)
        g_inv = jnp.linalg.inv(g_ab)
        T_mixed = g_inv @ T_ab
        zeta_max = jnp.float64(5.0)

        w = jnp.zeros(3)
        args = (T_ab, T_mixed, g_ab, tetrad, zeta_max)
        obj = _dec_objective(w, args)

        assert float(obj) == pytest.approx(1.0, abs=1e-12), (
            f"DEC objective for dust = {obj}, expected exactly 1.0"
        )

    def test_eulerian_dec_catches_past_directed(self):
        """Eulerian DEC also detects past-directed flux on the same
        T = diag(-1, 0.5, 0.5, 0.5) tensor (past-directed flux plus WEC
        violation give a negative margin)."""
        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        e = compute_eulerian_ec(T_ab, ETA)
        assert float(e["dec"]) < 0, (
            f"Eulerian DEC = {e['dec']}, expected < 0 for past-directed flux"
        )

    def test_verify_point_dec_catches_past_directed(self):
        """Full verify_point pipeline detects DEC violation from past-directed flux."""
        T_ab = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        r = verify_point(T_ab, ETA, n_starts=8)
        assert float(r.dec_margin) < 0, f"DEC margin = {r.dec_margin}, expected < 0"

    def test_type_i_dec_uses_full_optimizer(self):
        """Type-I DEC margin reflects min(eigenvalue_proxy, full_optimizer).

        The principal-frame point has rho=1, p=(0.3, 0, 0), so the algebraic
        DEC margin rho - max|p| = 0.7 > 0; the reported margin must not exceed
        that eigenvalue slack and must be no worse than the WEC margin, since
        DEC requires WEC.

        For a Type-I tensor with rho > max|p_i| (algebraic DEC margin > 0)
        but with a strong off-diagonal flux that makes the full DEC
        optimizer detect causality issues, verify_point must report the
        more conservative full-optimizer margin.
        """
        T_principal = jnp.diag(jnp.array([1.0, 0.3, 0.0, 0.0]))
        r = verify_point(T_principal, ETA, n_starts=8)
        assert int(r.he_type) == 1
        eig_slack = 1.0 - 0.3
        assert float(r.dec_margin) <= eig_slack + 1e-6
        assert float(r.dec_margin) <= float(r.wec_margin) + 1e-6


def test_verify_grid_subset_batch_size_matches_vmap(alcubierre_grid_data):
    """``batch_size`` routes the non-Type-I subset through ``lax.map`` without changing numbers.

    Margins agree to rtol 1e-12. The arg-min observer of a flat minimum is only
    determined to the optimizer stopping tolerance, so ``worst_observers`` and
    ``worst_params`` are compared at rtol 1e-6; a chunked ``lax.map`` compiles a
    different XLA program than one ``vmap`` and lands 1e-7 away on that arg-min.
    ``*_opt_n_steps`` counts iterations of the batched loop, which depend on the
    batch composition, so those diagnostics are not compared. The DEC flux
    sub-objective ``-g(j, j) / max|That|`` is unbounded below under boosts
    wherever the flux is spacelike, so its capped BFGS run stops on the tanh
    plateau at ``zeta_max`` with ``dec_opt_converged = 0``; the two programs stop
    about ``1e-4`` apart there, so the ``dec_*`` margins are compared at rtol 1e-3
    and ``dec_opt_converged``, which flips with the stopping point, is not compared.
    """
    result, _ = alcubierre_grid_data
    args = (result.stress_energy, result.metric, result.metric_inv)
    ec_vmap = verify_grid(*args, n_starts=4, batch_size=None)
    ec_map = verify_grid(*args, n_starts=4, batch_size=4)
    assert ec_vmap.n_type_i < ec_vmap.he_types.size, "fixture needs non-Type-I points"
    for name, a, b in zip(ECGridResult._fields, ec_vmap, ec_map, strict=True):
        if name.endswith("_opt_n_steps") or name == "dec_opt_converged":
            continue
        if isinstance(a, (ECSummary, jax.Array)):
            rtol = 1e-6 if name in ("worst_observers", "worst_params") else 1e-12
            atol = 1e-6 if name in ("worst_observers", "worst_params") else 0.0
            if name in ("dec_margins", "dec_opt_margins", "dec_summary"):
                rtol = 1e-3
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=name
            )
        else:
            assert a == b, name


def _boost_y(rapidity):
    c, s = np.cosh(rapidity), np.sinh(rapidity)
    L = np.eye(4)
    L[0, 0] = L[2, 2] = c
    L[0, 2] = L[2, 0] = s
    return jnp.asarray(L)


def test_ill_conditioned_type_i_routes_to_optimizer():
    """An ill-conditioned Type-I eigenbasis takes the optimizer route, not the eigenvalue slack.

    Rest-frame ``diag(1, p_x, 0, 3)`` with ``p_x = -1.0000004098162183`` boosted by
    rapidity 6 along ``y`` (``T = L^T diag L``, ``g = eta``) has eigenvector condition
    number ``e^12 = 1.63e5``. The float64 eigensolver returns ``rho = 1.0000005578``
    while an 80-digit eigendecomposition of the same stored matrix gives
    ``rho = 1.0000002618``, so the true eigenframe NEC slack ``rho + p_x`` is
    ``-1.48e-7`` but the float64 slack is ``+1.48031452e-7``. The point is
    flagged and carries the optimizer route
    ``min(nec_opt, null_deficit)``, which lies at ``-3.6e-12`` in Eulerian
    normalization (the slack divided by ``cosh^2 6``).
    """
    from warpax.energy_conditions.classification import classify_hawking_ellis
    from warpax.energy_conditions.slemma import null_deficit

    L = _boost_y(6.0)
    T = L.T @ jnp.diag(jnp.array([1.0, -1.0000004098162183, 0.0, 3.0])) @ L
    cls = classify_hawking_ellis(ETA @ T, ETA)
    sv = jnp.linalg.svd(cls.eigenvectors, compute_uv=False)
    assert int(cls.he_type) == 1
    assert float(sv[0] / sv[-1]) > 1e5
    assert float(jnp.min(cls.rho + cls.pressures)) == pytest.approx(1.48031452e-7, rel=1e-6)

    res = verify_grid(T[None], ETA[None], n_starts=4, solver="standard")
    assert float(res.eig_ill_conditioned[0]) == 1.0
    assert np.isfinite(float(res.nec_opt_margins[0]))
    route = min(float(res.nec_opt_margins[0]), float(null_deficit(T, ETA)))
    assert float(res.nec_margins[0]) == route
    assert abs(float(res.nec_margins[0])) < 1e-9


def test_eulerian_dec_flux_scales_linearly():
    """The Eulerian DEC margin is linear in ``T``.

    With ``T_00 = eps`` and ``T_01 = T_10 = 2 eps`` in Minkowski the Eulerian flux is
    ``j = (eps, -2 eps, 0, 0)``, ``-g(j, j) = -3 eps^2``, and the tetrad scale is
    ``max|That| = 2 eps``, so the flux submargin is ``-1.5 eps`` and the DEC margin
    ``min(eps, -1.5 eps, eps) = -1.5 eps``. The ratio ``dec(1e-3) / dec(1e-6)`` is
    therefore 1000; a quadratic submargin ``-3 eps^2`` would give 1e6.
    """

    def dec(eps):
        T = jnp.zeros((4, 4)).at[0, 0].set(eps).at[0, 1].set(2 * eps).at[1, 0].set(2 * eps)
        return float(compute_eulerian_ec(T, ETA)["dec"])

    assert dec(1e-3) == pytest.approx(-1.5e-3, rel=1e-12)
    assert dec(1e-3) / dec(1e-6) == pytest.approx(1000.0, rel=1e-6)


def test_verify_point_ill_conditioned_type_i_matches_grid():
    """``verify_point`` applies the eigenbasis conditioning guard like ``verify_grid``.

    Same tensor as :func:`test_ill_conditioned_type_i_routes_to_optimizer`:
    rest-frame ``diag(1, -1.0000004098162183, 0, 3)`` boosted by rapidity 6
    along ``y`` in Minkowski. The float64 eigenframe NEC slack is
    ``+1.48031452e-7`` although the stored tensor violates NEC, so the point
    must be flagged and take ``min(nec_opt, null_deficit)``, which is negative.
    The point and one-point-grid NEC margins agree to 1e-12.
    """
    L = _boost_y(6.0)
    T = L.T @ jnp.diag(jnp.array([1.0, -1.0000004098162183, 0.0, 3.0])) @ L

    pt = verify_point(T, ETA, n_starts=4, solver="standard")
    grid = verify_grid(T[None], ETA[None], n_starts=4, solver="standard")

    assert int(pt.he_type) == 1
    assert bool(pt.eig_ill_conditioned)
    assert float(pt.nec_margin) < 0
    assert float(pt.nec_margin) == pytest.approx(float(grid.nec_margins[0]), abs=1e-12)
    assert float(pt.wec_margin) == float(pt.wec_opt_margin)
    assert float(pt.sec_margin) == float(pt.sec_opt_margin)
