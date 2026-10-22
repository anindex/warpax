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


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_routed_nec_retains_the_lmi_inconclusive_band(monkeypatch, sign):
    from warpax.energy_conditions import frame_free
    from warpax.energy_conditions.slemma import noise_floor

    tensor = jnp.array(
        [[1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    floor = noise_floor(tensor, MINKOWSKI)
    monkeypatch.setattr(
        frame_free,
        "certify_point_lmi",
        lambda T, g: {c: sign * 0.75 * floor for c in ("nec", "wec", "sec", "dec")},
    )
    result = certify_grid_frame_free(tensor[None], MINKOWSKI[None])
    assert int(result.he_types[0]) == 4
    assert abs(float(result.nec_margins[0])) < float(result.nec_noise_floor[0])


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
    T = jnp.diag(jnp.array([rho, p, p, p]))
    out = certify_point_frame_free(T, MINKOWSKI)
    assert int(out["he_type"]) == 1
    assert float(out["rho"]) == pytest.approx(rho, abs=1e-9)
    assert float(out["nec"]) == pytest.approx(rho + p, abs=1e-9)
    assert float(out["wec"]) == pytest.approx(min(rho, rho + p), abs=1e-9)
    assert float(out["dec"]) == pytest.approx(rho - abs(p), abs=1e-9)
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
    NEC violation in closed form (cap-free). The (t,x) block has
    |T_tx| > |T_tt+T_xx|/2, a complex pair; Eulerian decomposition rho=1,
    |j|=2, S_par=1 gives Delta = (rho+S_par)^2 - 4|j|^2 = -12 < 0 (Type IV)
    and witness = rho + S_par - 2|j| = -2."""
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

    A ``-|Im lambda|`` sentinel is always negative, so every such point would be
    reported violating whatever its stress-energy, true by Martin-Moruno &
    Visser but true by fiat. Deciding them on their own merits turns the
    unconditional-violation theorem into a test of the pipeline. The NEC slot
    carries the full null deficit, which is twice the LMI margin.
    """
    from warpax.energy_conditions.frame_free import _exact_margins

    he = jnp.float64(4.0)
    witness = jnp.float64(0.5)
    nan = jnp.float64(jnp.nan)
    lmi = {
        "nec": jnp.float64(-0.7),
        "wec": jnp.float64(-0.8),
        "sec": jnp.float64(-0.9),
        "dec": jnp.float64(-1.1),
    }
    nec, wec, sec, dec = _exact_margins(he, nan, nan, nan, nan, witness, lmi)
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
    exactly zero, which must not be read as a satisfied NEC, because
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
    assert float(k @ MINKOWSKI @ k) == pytest.approx(0.0, abs=1e-12)
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
    assert typeI.mean() > 0.99

    nec_ff = np.asarray(ff.nec_margins).ravel()[typeI]
    nec_vg = np.asarray(ec.nec_margins).ravel()[typeI]
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
    assert ff.n_type_iv > 0
    fr = type_fractions(ff)
    assert abs(sum(fr[f"frac_type_{k}"] for k in ("i", "ii", "iii", "iv")) - 1.0) < 1e-6


class TestTypeIIMarginsAreDecidedNotOverclaimed:
    """One null contraction decides the NEC, and nothing else, so use the LMI.

    Handing the same Eulerian null witness to all four conditions at a
    non-Type-I point would silently certify WEC/SEC/DEC as satisfied whenever
    the null contraction happened to be non-negative, even with a manifestly
    negative Eulerian energy density, and returning NaN would leave the point
    undecided. The S-lemma LMI decides it properly: this canonical block has null witness exactly zero
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

    flat_T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    flat_g = jnp.reshape(curv.metric, (-1, 4, 4))
    full = jax.vmap(slemma.certify_point)(flat_T, flat_g)

    is_I = he == 1
    for cond, got in (("wec", ff.wec_margins), ("sec", ff.sec_margins), ("dec", ff.dec_margins)):
        got = np.asarray(got).reshape(-1)[~is_I]
        want = np.asarray(full[cond])[~is_I]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14, err_msg=cond)


def test_frame_free_kernels_match_eager():
    """The jitted grid kernels reproduce an eager, un-jitted evaluation of the same primitives."""
    from warpax.energy_conditions import slemma
    from warpax.energy_conditions.classification import (
        _standard_solver_unreliable_mask,
        classify_hawking_ellis,
    )
    from warpax.energy_conditions.eigenvalue_checks import check_all
    from warpax.energy_conditions.frame_free import (
        _exact_margins,
        eulerian_null_witness,
        ill_conditioned_eigenbasis,
    )

    metric = AlcubierreMetric(v_s=1.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3, 3)] * 3, (8, 8, 8), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=512)
    T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    g = jnp.reshape(curv.metric, (-1, 4, 4))
    gi = jnp.reshape(curv.metric_inv, (-1, 4, 4))
    tol = 1e-10

    with jax.disable_jit():
        Tm = jnp.einsum("nac,ncb->nab", gi, T)
        cls = jax.vmap(lambda a, b: classify_hawking_ellis(a, b, tol=tol))(Tm, g)
        bad = np.asarray(
            _standard_solver_unreliable_mask(cls.he_type, cls.eigenvalues, cls.eigenvalues_imag),
            dtype=bool,
        )
        if bad.any():
            idx = jnp.asarray(np.flatnonzero(bad))
            sub = jax.vmap(
                lambda a, b, c: classify_hawking_ellis(a, b, solver="generalized", T_ab=c, tol=tol)
            )(Tm[idx], g[idx], T[idx])
            cls = type(cls)(*(f.at[idx].set(s) for f, s in zip(cls, sub, strict=True)))
        nec_I, wec_I, sec_I, dec_I = jax.vmap(check_all)(cls.rho, cls.pressures)
        witness = jax.vmap(eulerian_null_witness)(T, g, gi)
        ill = jax.vmap(ill_conditioned_eigenbasis)(cls.eigenvectors)
        lmi_floor = jax.vmap(lambda a, b: slemma.noise_floor(a, b, condition="nec"))(T, g)
        eig_floor = slemma._NOISE_REL * jnp.max(jnp.abs(cls.eigenvalues), axis=-1)
        floor = jnp.where((cls.he_type == 1) & ~ill, eig_floor, 2 * lmi_floor)
        nonI = np.flatnonzero(np.asarray((cls.he_type != 1) | ill))
        assert nonI.size, "fixture must need the LMI somewhere"
        idx = jnp.asarray(nonI)
        sub = jax.vmap(slemma.certify_point)(T[idx], g[idx])
        lmi = {k: jnp.zeros_like(witness).at[idx].set(v) for k, v in sub.items()}
        nec, wec, sec, dec = jax.vmap(_exact_margins)(
            cls.he_type, nec_I, wec_I, sec_I, dec_I, witness, lmi, ill
        )

    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    expected = {
        "he_types": cls.he_type,
        "eigenvalues": cls.eigenvalues,
        "eigenvalues_imag": cls.eigenvalues_imag,
        "rho": cls.rho,
        "pressures": cls.pressures,
        "nec_noise_floor": floor,
        "nec_margins": nec,
        "wec_margins": wec,
        "sec_margins": sec,
        "dec_margins": dec,
        "is_vacuum": cls.is_vacuum,
        "lmi_substituted": ill.astype(nec.dtype),
    }
    for name, want in expected.items():
        got = np.asarray(getattr(ff, name)).reshape(np.shape(want))
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-12, atol=0, err_msg=name)
    assert ff.n_type_i == int(np.sum(np.asarray(cls.he_type) == 1))


def test_type_i_noise_floor_is_eigenframe_scale():
    """A well-conditioned Type-I point carries the eigenframe floor, not the LMI one.

    Rest frame ``rho = 1``, ``p = (-1 - 2e-10, 2, 3)`` boosted by rapidity 3 along
    ``y`` keeps the eigenframe NEC slack ``rho + p_1 = -1.987e-10`` (roundoff from
    the exact ``-2e-10``), but the Eulerian LMI floor grew to ``6.06e-10`` with the
    tetrad components and made the slack inconclusive. The routed floor is
    ``1e-12 * max|lambda| = 3e-12 < 1e-11`` and the slack lies below ``-floor``.
    """
    c, s = np.cosh(3.0), np.sinh(3.0)
    L = jnp.asarray(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    T = L.T @ jnp.diag(jnp.array([1.0, -1.0 - 2e-10, 2.0, 3.0])) @ L
    ff = certify_grid_frame_free(T[None], MINKOWSKI[None], solver="standard")
    assert int(ff.he_types[0]) == 1
    assert float(ff.lmi_substituted[0]) == 0.0
    floor = float(ff.nec_noise_floor[0])
    assert floor == pytest.approx(3e-12, rel=1e-9)
    assert floor < 1e-11
    assert float(ff.nec_margins[0]) == pytest.approx(-1.987e-10, rel=1e-3)
    assert float(ff.nec_margins[0]) < -floor
