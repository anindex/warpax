"""Regression tests pinning confirmed findings, plus fidelity sentinels.

Each test names the confirmed finding it pins:

- near-vacuum modulus gate (classification.py / classification_mpmath.py):
  a purely imaginary eigenvalue spectrum is genuine Type IV, never vacuum.
- scale-aware violation gate (verifier._compute_summary, filtering):
  the violated/satisfied threshold carries a relative term ``rtol * max|lambda|``
  so float64 eigensolver noise at large ||T|| cannot mint violations.
- imag_rtol sentinel: paper-grid type fractions are insensitive to the
  split-degenerate imaginary tolerance, and the 50-digit gate catches the
  one adversarial input class the float64 tier absorbs.
- timelike tiebreak boundary: the scale-relative argmin bias selects the
  correct timelike eigenvector across 30 decades of ||T||.
- DEC necessary-only bound: the eigenvalue bound and the optimizer agree in
  sign on clear Type-I cases, and the DEC<=WEC merge invariant holds.
"""

from __future__ import annotations

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.classification import (
    classify_hawking_ellis,
    classify_with_solver,
)
from warpax.energy_conditions.classification_mpmath import (
    classify_hawking_ellis_mpmath,
    verify_classification_at_points,
)
from warpax.energy_conditions.eigenvalue_checks import check_dec, check_wec
from warpax.energy_conditions.optimization import optimize_dec
from warpax.energy_conditions.verifier import (
    _compute_summary,
    verify_grid,
    verify_point,
)
from warpax.geometry import GridSpec, evaluate_curvature_grid
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _boost_x(zeta: float) -> jnp.ndarray:
    """Mixed-index Lorentz boost along x with rapidity zeta."""
    c, s = jnp.cosh(zeta), jnp.sinh(zeta)
    B = jnp.eye(4)
    B = B.at[0, 0].set(c).at[0, 1].set(s).at[1, 0].set(s).at[1, 1].set(c)
    return B


class TestNearVacuumModulusGate:
    """Bug: ``near_vacuum`` tested only Re(lambda), so a pure momentum flux
    T_ab = q(dt (x) dx + dx (x) dt) with eigenvalues +/- iq was absorbed as
    vacuum Type I at any q."""

    def _pure_flux(self, q: float) -> jnp.ndarray:
        T = jnp.zeros((4, 4)).at[0, 1].set(q).at[1, 0].set(q)
        return T

    def test_pure_flux_imaginary_spectrum_is_type_iv(self):
        T = self._pure_flux(1.0)
        cls = classify_hawking_ellis(ETA @ T, ETA)
        assert int(cls.he_type) == 4
        assert float(cls.is_vacuum) == 0.0

    def test_pure_flux_small_but_above_tol_is_type_iv(self):
        T = self._pure_flux(1e-6)
        cls = classify_hawking_ellis(ETA @ T, ETA)
        assert int(cls.he_type) == 4

    def test_pure_flux_mpmath_agrees_type_iv(self):
        T = np.zeros((4, 4))
        T[0, 1] = T[1, 0] = 1.0
        g = np.diag([-1.0, 1.0, 1.0, 1.0])
        rep = classify_hawking_ellis_mpmath(np.linalg.inv(g) @ T, g)
        assert rep["he_type"] == 4
        assert not rep["near_vacuum"]

    def test_true_vacuum_still_vacuum(self):
        cls = classify_hawking_ellis(jnp.zeros((4, 4)), ETA)
        assert int(cls.he_type) == 1
        assert float(cls.is_vacuum) == 1.0

    def test_verify_point_pure_flux_reports_violation(self):
        T = self._pure_flux(1.0)
        res = verify_point(T, ETA, n_starts=4)
        assert int(res.he_type) == 4
        # Type IV: margins come from the optimizer, which sees the
        # unbounded WEC violation the old vacuum bypass hid.
        assert float(res.wec_margin) < 0.0


class TestScaleAwareViolationGate:
    """Bug: the fixed absolute atol=1e-10 violation gate flagged float64
    eigenvalue noise as 'violated' at ||T|| >~ 1e6 (54/200 synthetic
    marginal fluids at 1e11; ~24% of WarpShell NEC-'violated' points)."""

    def _marginal_margins(self, scale: float, n: int = 64) -> jnp.ndarray:
        """NEC margins of boosted, analytically marginal fluids."""
        key = jax.random.PRNGKey(0)
        zetas = jax.random.uniform(key, (n,), minval=0.5, maxval=2.5)
        T_rest = jnp.diag(jnp.array([-1.0, -1.0, 0.5, 0.5]) * scale)

        def margin(zeta):
            B = _boost_x(zeta)
            T_mixed = B @ T_rest @ jnp.linalg.inv(B)
            cls = classify_hawking_ellis(T_mixed, ETA)
            # NEC margin = min_i (rho + p_i); analytically exactly zero.
            return jnp.min(cls.rho + cls.pressures)

        return jax.vmap(margin)(zetas)

    @pytest.mark.parametrize("scale", [1.0, 1e11, 1e30])
    def test_marginal_fluids_not_flagged_at_any_scale(self, scale):
        margins = self._marginal_margins(scale)
        scales = jnp.full(margins.shape, scale)
        summary = _compute_summary(margins, scale=scales)
        assert float(summary.fraction_violated) == 0.0

    def test_legacy_absolute_gate_would_have_flagged(self):
        # Demonstrates the test is load-bearing: without the relative term
        # the same noise floor trips the gate at large ||T||.
        margins = self._marginal_margins(1e11)
        summary_abs = _compute_summary(margins)  # scale=None: pure absolute
        assert float(summary_abs.fraction_violated) > 0.0

    def test_genuine_violation_still_flagged_at_extreme_scale(self):
        scale = 1e30
        margins = jnp.array([-1e-3 * scale, 1e-3 * scale])
        scales = jnp.full((2,), scale)
        summary = _compute_summary(margins, scale=scales)
        assert float(summary.fraction_violated) == 0.5
        assert float(summary.max_violation) == pytest.approx(1e-3 * scale)

    def test_verify_grid_marginal_fluid_zero_fraction(self):
        scale = 1e11
        B = _boost_x(1.5)
        T_rest = jnp.diag(jnp.array([-1.0, -1.0, 0.5, 0.5]) * scale)
        T_mixed = B @ T_rest @ jnp.linalg.inv(B)
        T_ab = ETA @ T_mixed
        T_field = jnp.broadcast_to(T_ab, (2, 2, 1, 4, 4))
        g_field = jnp.broadcast_to(ETA, (2, 2, 1, 4, 4))
        res = verify_grid(T_field, g_field, n_starts=2)
        assert float(res.nec_summary.fraction_violated) == 0.0


class TestImagRtolSentinel:
    """Sentinel: the split-degenerate imaginary tolerance (imag_rtol=3e-3)
    must not move paper-grid type fractions, and the 50-digit gate must
    catch the adversarial small-Im/large-Re class it can absorb."""

    @pytest.fixture(scope="class")
    def wall_slab(self):
        metric = AlcubierreMetric(v_s=2.0)
        grid = GridSpec(bounds=((-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)), shape=(10, 6, 4))
        chain = evaluate_curvature_grid(metric, grid)
        t_mixed = (chain.metric_inv @ chain.stress_energy).reshape(-1, 4, 4)
        g_flat = chain.metric.reshape(-1, 4, 4)
        return t_mixed, g_flat

    def test_type_fractions_stable_under_imag_rtol_sweep(self, wall_slab):
        # With the scale floor on the relative tier, small-||T|| points are
        # governed by the absolute tier alone, so the type census must be
        # invariant under the imag_rtol sweep on this slab.
        t_mixed, g_flat = wall_slab
        counts = []
        for imag_rtol in (3e-4, 1e-3, 3e-3, 1e-2):
            cls = jax.vmap(
                lambda T, g: classify_with_solver(
                    T, g, None, solver="standard", imag_rtol=imag_rtol
                )
            )(t_mixed, g_flat)
            types = np.asarray(cls.he_type)
            counts.append(np.bincount(types.astype(int), minlength=5))
        for c in counts[1:]:
            np.testing.assert_array_equal(c, counts[0])

    def test_default_classifier_agrees_with_mpmath_on_slab(self, wall_slab):
        # The 50-digit gate is the authority; the default float64
        # classifier must agree with it everywhere on this slab,
        # including the weak Type-IV far-field tail (|Im| ~ 1e-8) that
        # the pre-fix relative tier absorbed as Type I.
        t_mixed, g_flat = wall_slab
        cls = jax.vmap(
            lambda T, g: classify_with_solver(T, g, None, solver="standard")
        )(t_mixed, g_flat)
        types = np.asarray(cls.he_type).astype(int)
        rep = verify_classification_at_points(
            np.asarray(t_mixed), np.asarray(g_flat), types
        )
        assert rep["n_flips"] == 0

    def test_mpmath_gate_catches_small_imag_large_real(self):
        # Eigenvalues s(1 +/- 1e-5j) at s=1e8: genuinely complex (Type IV),
        # but above the 1e6 scale floor the float64 relative tier
        # (3e-3 * max|Re|) absorbs the split. The 50-digit cross-check
        # (imag_rtol=0) is the authority and must report the flip.
        g = np.diag([-1.0, 1.0, 1.0, 1.0])
        s = 1e8
        M = np.zeros((4, 4))
        M[0, 0] = M[1, 1] = s
        M[0, 1], M[1, 0] = -1e-5 * s, 1e-5 * s
        M[2, 2], M[3, 3] = 2.0 * s, 3.0 * s
        f64 = classify_hawking_ellis(jnp.asarray(M), ETA)
        rep = verify_classification_at_points(
            M[None, ...], g[None, ...], np.array([int(f64.he_type)])
        )
        assert int(f64.he_type) != 4  # the relative tier absorbed the split
        assert rep["mpmath_he_types"][0] == 4
        assert rep["n_flips"] == 1

        # Below the floor the relative tier is off: float64 itself gets the
        # same spectrum (max|Re|=3) right, so the gate reports no flip.
        M_small = M / s
        f64_small = classify_hawking_ellis(jnp.asarray(M_small), ETA)
        rep_small = verify_classification_at_points(
            M_small[None, ...], g[None, ...], np.array([int(f64_small.he_type)])
        )
        assert int(f64_small.he_type) == 4
        assert rep_small["mpmath_he_types"][0] == 4
        assert rep_small["n_flips"] == 0


class TestTimelikeTiebreakBoundary:
    """Sentinel: the 1e-12 scale-relative timelike-index tiebreak selects
    the correct eigenvector (rho sign) across 30 decades of ||T||."""

    @pytest.mark.parametrize("scale", [1.0, 1e11, 1e30])
    def test_rho_sign_matches_mpmath_oracle(self, scale):
        T_mixed = jnp.diag(jnp.array([-0.7, 0.3, 0.5, 0.9]) * scale)
        B = _boost_x(0.8)
        T_b = B @ T_mixed @ jnp.linalg.inv(B)
        cls = classify_hawking_ellis(T_b, ETA)
        assert int(cls.he_type) == 1
        rep = classify_hawking_ellis_mpmath(
            np.asarray(T_b), np.diag([-1.0, 1.0, 1.0, 1.0])
        )
        assert rep["he_type"] == 1
        assert float(cls.rho) == pytest.approx(0.7 * scale, rel=1e-8)
        # mpmath oracle: rho = -(timelike eigenvalue); here the unique
        # negative eigenvalue of the boosted diag(-0.7, 0.3, 0.5, 0.9)*s.
        assert -min(rep["eigenvalues_real"]) == pytest.approx(
            0.7 * scale, rel=1e-8
        )


class TestDECNecessaryOnly:
    """Sentinel: check_dec is the eigenvalue (necessary) bound; the
    optimizer carries the flux-causality pillar. Their Boolean verdicts
    agree on clearly-signed Type-I cases, and DEC <= WEC after the merge."""

    @pytest.mark.parametrize(
        "rho,p",
        [
            (1.0, (0.5, 0.3, 0.1)),     # satisfied, margin +0.5
            (1.0, (1.5, 0.0, 0.0)),     # violated, margin -0.5
            (2.0, (-1.0, 0.5, 0.5)),    # satisfied, margin +1.0
        ],
    )
    def test_check_dec_sign_agrees_with_optimize_dec(self, rho, p):
        T_mixed = jnp.diag(jnp.array([-rho, *p]))
        T_ab = ETA @ T_mixed
        eig_margin = float(check_dec(jnp.array(rho), jnp.array(p)))
        opt = optimize_dec(T_ab, ETA, n_starts=4, key=jax.random.PRNGKey(1))
        opt_margin = float(opt.margin)
        # Sign agreement only: the three-term optimizer objective is not
        # magnitude-commensurable with the eigenvalue bound.
        assert (eig_margin < 0) == (opt_margin < -1e-8), (
            f"eigenvalue bound {eig_margin} vs optimizer {opt_margin}"
        )

    def test_dec_margin_merged_with_wec_invariant(self):
        # rho = -0.5 with p = 0.1: eigenvalue bounds are wec = -0.5 and
        # dec = rho - |p| = -0.6, so the DEC bound sits below WEC here.
        rho, p = -0.5, (0.1, 0.1, 0.1)
        assert float(check_wec(jnp.array(rho), jnp.array(p))) == pytest.approx(-0.5)
        assert float(check_dec(jnp.array(rho), jnp.array(p))) == pytest.approx(-0.6)

        T_mixed = jnp.diag(jnp.array([0.5, 0.1, 0.1, 0.1]))  # rho=-0.5
        res = verify_point(ETA @ T_mixed, ETA, n_starts=2)
        assert int(res.he_type) == 1
        # Type I publishes the eigenvalue WEC margin exactly.
        assert float(res.wec_margin) == pytest.approx(-0.5)
        assert float(res.dec_margin) <= float(res.wec_margin) + 1e-12

    def test_dec_wec_merge_on_optimizer_branch(self):
        # Non-Type-I point (pure flux, Type IV): all margins come from the
        # optimizer, so this exercises the dec = min(wec, dec) merge on the
        # branch the Type-I eigenvalue path never reaches.
        T = jnp.zeros((4, 4)).at[0, 1].set(1.0).at[1, 0].set(1.0)
        res = verify_point(T, ETA, n_starts=4)
        assert int(res.he_type) == 4
        assert float(res.dec_margin) <= float(res.wec_margin) + 1e-12

    def test_wec_margin_definition(self):
        rho, p = 1.0, (0.5, -0.2, 0.3)
        m = float(check_wec(jnp.array(rho), jnp.array(p)))
        assert m == pytest.approx(min(rho, *(rho + pi for pi in p)))
