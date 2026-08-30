"""Tests for frame-independent, all-velocity EC certification."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.frame_free import (
    certify_grid_frame_free,
    certify_point_frame_free,
    type_fractions,
)
from warpax.energy_conditions.verifier import verify_grid
from warpax.geometry import evaluate_curvature_grid
from warpax.grids import wall_clustered
from warpax.metrics import RodalMetric

MINKOWSKI = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def test_minkowski_vacuum_is_type_i():
    """Vacuum (T=0) in Minkowski classifies Type I with ~zero margins."""
    T = jnp.zeros((4, 4))
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    for k in ("nec", "wec", "sec", "dec"):
        assert abs(float(out[k])) < 1e-8


def test_perfect_fluid_margins_exact():
    """Type-I perfect fluid: margins equal the eigenvalue inequalities."""
    rho, p = 1.0, 0.5
    T = jnp.diag(jnp.array([rho, p, p, p]))  # T_{ab}, energy density T_tt=rho
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    assert float(out["rho"]) == pytest.approx(rho, abs=1e-9)
    assert float(out["nec"]) == pytest.approx(rho + p, abs=1e-9)  # 1.5
    assert float(out["wec"]) == pytest.approx(min(rho, rho + p), abs=1e-9)  # 1.0
    assert float(out["dec"]) == pytest.approx(rho - abs(p), abs=1e-9)  # 0.5
    assert float(out["sec"]) == pytest.approx(min(rho + p, rho + 3 * p), abs=1e-9)


def test_negative_pressure_violates():
    """rho=1, p=-2 violates NEC/WEC/DEC; margins are negative."""
    rho, p = 1.0, -2.0
    T = jnp.diag(jnp.array([rho, p, p, p]))
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    assert float(out["nec"]) < 0
    assert float(out["wec"]) < 0
    assert float(out["dec"]) < 0


def test_type_iv_certifies_nec_violation():
    """A complex-eigenvalue T^a_b is Type IV; the Eulerian null witness certifies
    NEC violation in closed form (cap-free), replacing the old NaN margins."""
    # T_{ab} with a (t,x) block giving |T_tx| > |T_tt+T_xx|/2 -> complex pair.
    # Eulerian decomposition: rho=1, |j|=2, S_par=1 -> Delta=(rho+S_par)^2-4|j|^2
    # = 4-16 = -12 < 0 (Type IV) and witness = rho+S_par-2|j| = 2-4 = -2.
    T = jnp.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    out = certify_point_frame_free(T, MINKOWSKI, solver="standard")
    assert int(out["he_type"]) == 4
    for cond in ("nec", "wec", "sec", "dec"):
        val = float(out[cond])
        assert np.isfinite(val), f"{cond} must be a finite certified margin, not NaN"
        assert val < 0, f"Type IV must certify {cond.upper()} violation"
    assert abs(float(out["nec"]) - (-2.0)) < 1e-9, "closed-form witness = rho+S_par-2|j|"


def test_conformal_type_iv_positive_witness_uses_lmi():
    """A Type-IV point whose momentum witness is >= 0 (conformal pair, not
    momentum-sourced) takes every margin from the LMI.

    It used to take ``-|Im lambda|`` instead. That sentinel was always negative, so
    every such point was reported violating whatever its stress-energy, true, by
    Martin-Moruno & Visser, but true by fiat, and at Type III (``imag = 0``) the
    reported number was a hard-coded ``-1e-30``. Deciding them on their own merits
    turns the unconditional-violation theorem into a test of the pipeline.
    """
    from warpax.energy_conditions.frame_free import _exact_margins

    he = jnp.float64(4.0)
    witness = jnp.float64(0.5)  # momentum plane is not the source
    nan = jnp.float64(jnp.nan)
    lmi = {
        "nec": jnp.float64(-0.7),
        "wec": jnp.float64(-0.8),
        "sec": jnp.float64(-0.9),
        "dec": jnp.float64(-1.1),
    }
    nec, wec, sec, dec = _exact_margins(he, nan, nan, nan, nan, witness, lmi)
    # The NEC slot carries the full null deficit, which is twice the LMI margin.
    assert float(nec) == pytest.approx(-1.4)
    assert float(wec) == pytest.approx(-0.8)
    assert float(sec) == pytest.approx(-0.9)
    assert float(dec) == pytest.approx(-1.1)


def test_nonI_nec_is_the_full_null_deficit_not_the_witness():
    """The NEC slot is one quantity at every type: min_{|s|=1} q(s).

    Mixing the momentum witness with ``lmi["nec"]`` would put three scales in one
    array, since the witness probes a single direction (an upper bound on the
    deficit) and ``lmi["nec"]`` is half of it. Non-Type-I always reports
    ``2*lmi["nec"]``,
    which is ``slemma.null_deficit`` and the same quantity as the Type-I
    ``min_i(rho + p_i)``.
    """
    from warpax.energy_conditions.frame_free import _exact_margins

    he = jnp.float64(4.0)
    nan = jnp.float64(jnp.nan)
    lmi = {k: jnp.float64(-0.5) for k in ("nec", "wec", "sec", "dec")}
    for witness in (jnp.float64(-2.0), jnp.float64(+0.5)):
        nec, _, _, _ = _exact_margins(he, nan, nan, nan, nan, witness, lmi)
        assert float(nec) == pytest.approx(-1.0)


def test_type_ii_nec_is_not_read_off_the_momentum_witness():
    """Regression: the Eulerian witness cannot decide the NEC at Type II.

    For the canonical block ``(mu, f, p_2, p_3) = (0, 1, -2, 0)`` the witness is
    exactly zero, which the old code reported as a satisfied NEC, while
    ``k = (1, 0, 1, 0)`` gives ``T_ab k^a k^b = -1``. The witness only probes the
    momentum plane; this violation lives in the transverse channel.
    """
    from warpax.energy_conditions.frame_free import eulerian_null_witness
    from warpax.energy_conditions.slemma import null_deficit

    T = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    witness = eulerian_null_witness(T, MINKOWSKI, jnp.linalg.inv(MINKOWSKI))
    assert float(witness) == pytest.approx(0.0, abs=1e-12)

    k = jnp.array([1.0, 0.0, 1.0, 0.0])
    assert float(k @ MINKOWSKI @ k) == pytest.approx(0.0, abs=1e-12)  # k is null
    assert float(k @ T @ k) == pytest.approx(-1.0)

    out = certify_point_frame_free(T, MINKOWSKI, solver="standard")
    assert int(out["he_type"]) == 2
    assert float(out["nec"]) < -1e-9, "Type-II NEC must come from the LMI"
    assert float(null_deficit(T, MINKOWSKI)) == pytest.approx(-4.0 / 3.0, abs=1e-6)


def test_grid_matches_verify_grid_typeI_margins():
    """On Rodal (100% Type I, v_s=0.5) the frame-free Type-I margins match the
    validated verify_grid eigenvalue branch."""
    metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (24, 24, 24), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)

    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    ec = verify_grid(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=1)

    he = np.asarray(ff.he_types).ravel()
    typeI = he == 1.0
    assert typeI.mean() > 0.99  # Rodal is ~100% Type I

    nec_ff = np.asarray(ff.nec_margins).ravel()[typeI]
    nec_vg = np.asarray(ec.nec_margins).ravel()[typeI]
    # NEC/WEC/SEC are algebraic-exact in both paths -> should match closely.
    np.testing.assert_allclose(nec_ff, nec_vg, atol=1e-8, rtol=1e-6)


def test_runs_superluminal_no_nan_types():
    """The frame-free engine runs at v_s=1.5 (Eulerian normal undefined) and
    returns finite Hawking-Ellis types with Type-IV present at the wall."""
    metric = AlcubierreMetric(v_s=1.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (24, 24, 24), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    he = np.asarray(ff.he_types)
    assert np.all(np.isfinite(he))
    assert ff.n_type_iv > 0  # Alcubierre wall is Type IV, even superluminally
    fr = type_fractions(ff)
    assert abs(sum(fr[f"frac_type_{k}"] for k in ("i", "ii", "iii", "iv")) - 1.0) < 1e-6


class TestTypeIIMarginsAreDecidedNotOverclaimed:
    """One null contraction decides the NEC, and nothing else, so use the LMI.

    The frame-free certifier used to hand the same Eulerian null witness to all
    four conditions at every non-Type-I point. At a Type-II point that silently
    certified WEC/SEC/DEC as satisfied whenever the null contraction happened to
    be non-negative, even with a manifestly negative Eulerian energy density.
    Returning NaN fixed the overclaim but left the point undecided. The S-lemma
    LMI decides it properly: this canonical block has null witness exactly zero
    and Eulerian energy density -1, and the WEC really is violated.
    """

    @staticmethod
    def _canonical_type_ii(mu=-2.0, f=1.0, p2=3.0, p3=3.0):
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T = jnp.array(
            [
                [mu + f, f, 0.0, 0.0],
                [f, -mu + f, 0.0, 0.0],
                [0.0, 0.0, p2, 0.0],
                [0.0, 0.0, 0.0, p3],
            ]
        )
        return T, eta

    def test_block_classifies_as_type_ii(self):
        T, eta = self._canonical_type_ii()
        res = certify_point_frame_free(T, eta, tol=1e-6)
        assert int(res["he_type"]) == 2

    def test_nec_is_the_null_contraction(self):
        T, eta = self._canonical_type_ii()
        res = certify_point_frame_free(T, eta, tol=1e-6)
        assert jnp.isfinite(res["nec"]), "NEC is decided at Type II and must be finite"

    def test_wec_sec_dec_are_decided_not_undefined(self):
        """All four are decided, and they do not all give the same answer.

        This block separates the conditions cleanly, which is the point: with
        rho = -1, b = (-1,0,0), S = 3*I and trace T = 10,

            q(w)        = -1 + 2 w_1 + 3|w|^2      -> -1 at w = 0, WEC VIOLATED
            q + T(1-|w|^2)/2 = 4 + 2 w_1 - 2|w|^2  ->  0 at w = (-1,0,0), SEC SATURATED

        so the WEC and DEC fail while the SEC holds with equality, which a single
        null contraction cannot express.
        """
        T, eta = self._canonical_type_ii()
        assert float(T[0, 0]) < 0.0
        res = certify_point_frame_free(T, eta, tol=1e-6)
        for cond in ("wec", "sec", "dec"):
            assert np.isfinite(float(res[cond])), f"{cond} must be decided at Type II"
        assert float(res["wec"]) < 0.0, "WEC is violated: rho_n = -1 at w = 0"
        assert float(res["dec"]) < 0.0, "DEC is violated because the WEC is"
        assert float(res["sec"]) == pytest.approx(0.0, abs=1e-8), "SEC is saturated"

    def test_the_null_witness_alone_would_have_missed_it(self):
        """The regression this guards: witness >= 0 while the WEC fails."""
        from warpax.energy_conditions.frame_free import eulerian_null_witness

        T, eta = self._canonical_type_ii()
        witness = float(eulerian_null_witness(T, eta, jnp.linalg.inv(eta)))
        assert witness >= 0.0, "this is the case where one null contraction says nothing"
        assert float(certify_point_frame_free(T, eta, tol=1e-6)["wec"]) < 0.0

    def test_explicit_violating_observer_exists(self):
        """A reported violation must come with an observer that sees it."""
        from warpax.energy_conditions.slemma import witness_observer

        T, eta = self._canonical_type_ii()
        w = np.asarray(witness_observer(T, eta))
        assert np.all(np.isfinite(w)) and np.linalg.norm(w) <= 1.0 + 1e-9
        u = np.concatenate([[1.0], w])
        assert float(u @ np.asarray(T) @ u) < 0.0


def test_lmi_is_evaluated_only_where_it_is_read():
    """The grid LMI must gather the non-Type-I points, and agree with the full sweep.

    ``_exact_margins`` takes the eigenvalue margins wherever ``he_type == 1``, so the
    LMI slot is read at a few per cent of a bubble-wall grid. It was nonetheless
    evaluated at every point: the guard read ``np.any(he != 1)``, which is true on any
    Type-IV dominated grid, while its comment and commit message both described it as
    firing only on a grid carrying a Type-II point. At a measured 0.95 ms per point
    against ~5 us for the classification it supports, that is what made the N=100
    velocity sweep unrunnable.

    Gathering must not perturb a single number.
    """
    from warpax.energy_conditions import slemma
    from warpax.energy_conditions.frame_free import certify_grid_frame_free

    metric = AlcubierreMetric(v_s=1.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (16, 16, 16), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=2048)

    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    he = np.asarray(ff.he_types).reshape(-1)
    assert np.any(he != 1), "fixture must carry non-Type-I points to be meaningful"
    assert np.any(he == 1), "fixture must carry Type-I points for gathering to matter"

    # Reference: the LMI at every point, then the same selection by hand.
    flat_T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    flat_g = jnp.reshape(curv.metric, (-1, 4, 4))
    full = jax.vmap(slemma.certify_point)(flat_T, flat_g)

    is_I = he == 1
    for cond, got in (("wec", ff.wec_margins), ("sec", ff.sec_margins), ("dec", ff.dec_margins)):
        got = np.asarray(got).reshape(-1)[~is_I]
        want = np.asarray(full[cond])[~is_I]
        np.testing.assert_array_equal(got, want, err_msg=cond)
