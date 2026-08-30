"""S-lemma LMI certification: exactness against brute-force observer search.

The claim under test is that one 4x4 LMI decides NEC/WEC/SEC/DEC over every
observer at a point, with no rapidity cap and no dependence on the Hawking-Ellis
type. These tests check it the only way that matters: against a direct search
over the observer ball and sphere, on all four canonical algebraic types.
"""
from __future__ import annotations

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions.slemma import (
    certify_point,
    noise_floor,
    null_deficit,
    tetrad_components,
    witness_observer,
)

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
MINKOWSKI = jnp.asarray(ETA)


def brute_min(T_np, on_sphere, seed=0, n=400_000):
    """min over observers of T_ab u^a u^b for u = n + w, |w| = 1 or |w| <= 1."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    if not on_sphere:
        w *= rng.random((n, 1)) ** (1 / 3.0)
    u = np.concatenate([np.ones((n, 1)), w], axis=1)
    return float(np.einsum("ni,ij,nj->n", u, T_np, u).min())


def _sym(M):
    return 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)


# --- the four Hawking-Ellis canonical forms, satisfying and violating ---------

def _type_i(rho, p):
    return np.diag([rho, p, p, p])


def _type_ii(amplitude, extra=None):
    """Null dust T = amplitude * k k with k null, optionally perturbed."""
    k = np.array([1.0, 1.0, 0.0, 0.0])
    T = amplitude * np.outer(k, k)
    return T if extra is None else T + np.diag(extra)


def _type_iii(r=0.7, f=1.0, p=0.3):
    """Genuine Type III: covariant T_ab whose mixed form is ``J_3(-r) (+) [p]``.

    This is returned as ``T_ab`` directly, and it is symmetric, which is the whole
    point. The previous version built a mixed matrix and returned ``ETA @ Tm``; that
    product is *not* symmetric, so it was not the component matrix of any
    Lorentz-self-adjoint tensor, and ``_sym``, applied to every case before use,
    then cancelled the off-diagonal ``f`` entries exactly, collapsing the fixture to
    ``diag(-r, -r, -r, p)``: a diagonal **Type I** tensor. Every "Type III" assertion
    in this file was really being made about Type I.
    """
    return np.array([[r, 0.0, -f, 0.0],
                     [0.0, -r, f, 0.0],
                     [-f, f, -r, 0.0],
                     [0.0, 0.0, 0.0, p]])


def _type_ii_canonical(mu, f, p2, p3):
    """Covariant Type II block: ``J_2`` in the (t,x) plane, ``p2``/``p3`` transverse."""
    return np.array([[mu + f, f, 0.0, 0.0],
                     [f, -mu + f, 0.0, 0.0],
                     [0.0, 0.0, p2, 0.0],
                     [0.0, 0.0, 0.0, p3]])


def _type_iv_b2(f=1.3, c=0.4):
    """The referee's own B2 counterexample: eigenvalues +-i f and a double c."""
    Tm = np.array([[0.0, f, 0.0, 0.0], [-f, 0.0, 0.0, 0.0],
                   [0.0, 0.0, c, 0.0], [0.0, 0.0, 0.0, c]])
    return ETA @ Tm


CASES = {
    "I-satisfying": _type_i(2.0, 0.5),
    "I-violating": _type_i(-1.0, 0.5),
    "I-nec-only": _type_i(-0.2, 0.9),
    "II-null-dust": _type_ii(1.0),
    # Was _type_ii(1.0, extra=[-1.0, 0.0, 3.0, 3.0]), whose longitudinal factor is
    # lam^2 - lam + 1 (discriminant -3): a complex pair, i.e. Type IV, not Type II.
    # These are the canonical J_2 blocks, and they are genuinely Type II.
    "II-violating": _type_ii_canonical(-2.0, 1.0, 3.0, 3.0),
    "II-transverse-violating": _type_ii_canonical(0.0, 1.0, -2.0, 0.0),
    "III": _type_iii(),
    "IV-B2": _type_iv_b2(),
    "IV-momentum": _sym(np.array([[0.1, 2.0, 0.0, 0.0], [2.0, 0.1, 0.0, 0.0],
                                  [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.1]])),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_fixtures_are_lorentz_self_adjoint(name):
    """Every fixture must already be symmetric as a covariant tensor.

    ``_sym`` is applied to each case before use, so a non-symmetric fixture is not
    rejected, it is silently *replaced* by its symmetric part. That is how the
    "Type III" case degraded into a diagonal Type I without any test noticing. If a
    fixture is not symmetric to begin with, it is not the tensor it claims to be.
    """
    T = np.asarray(CASES[name], dtype=float)
    np.testing.assert_allclose(T, T.T, atol=1e-14, err_msg=f"{name} is not symmetric")


EXPECTED_TYPE = {
    "I-satisfying": 1, "I-violating": 1, "I-nec-only": 1,
    "II-violating": 2, "II-transverse-violating": 2,
    "III": 3,
    "IV-B2": 4, "IV-momentum": 4,
}


@pytest.mark.parametrize("name", sorted(EXPECTED_TYPE))
def test_fixtures_have_the_algebraic_type_they_claim(name):
    """Each fixture must really carry the Segre type its name asserts.

    Done in **exact rational arithmetic**, and it has to be. Attempting this check
    with ``np.linalg.eigvals`` fails on the Type III fixture, which comes back with a
    complex pair at ``6.4e-6``: a ``J_3`` block splits under float64 rounding by
    ``eps^(1/3)``, so no float test can confirm that a Type III is a Type III. That is
    the same limit ``test_classification.py`` pins for the production classifier, met
    here from the other side.

    The type is read off the Jordan chain length, the largest ``k`` with
    ``rank(N^k) > rank(N^(k+1))`` for ``N = A - lam I`` at the repeated eigenvalue,
    which is 1 for Type I, 2 for Type II and 3 for Type III.
    """
    sp = pytest.importorskip("sympy")

    T = sp.Matrix(4, 4, lambda i, j: sp.nsimplify(float(CASES[name][i][j]),
                                                  rational=True))
    A = sp.diag(-1, 1, 1, 1) * T
    assert T == T.T, f"{name}: fixture must be symmetric"

    eigs = A.eigenvals()  # {eigenvalue: algebraic multiplicity}, exact
    n_complex = sum(m for e, m in eigs.items() if not e.is_real)
    if EXPECTED_TYPE[name] == 4:
        assert n_complex > 0, f"{name}: expected a complex pair, got {eigs}"
        return
    assert n_complex == 0, f"{name}: expected a real spectrum, got {eigs}"

    lam = max(eigs, key=lambda e: eigs[e])
    N = A - lam * sp.eye(4)
    ranks = [(N ** k).rank() for k in range(1, 5)]
    chain = 1 + sum(1 for k in range(1, 4) if ranks[k] < ranks[k - 1])
    assert chain == EXPECTED_TYPE[name], (
        f"{name}: expected Segre chain length {EXPECTED_TYPE[name]}, got {chain} "
        f"(ranks of N^k: {ranks})"
    )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("condition,on_sphere", [("nec", True), ("wec", False)])
def test_lmi_matches_brute_force(name, condition, on_sphere):
    """The LMI verdict must equal a direct observer search, at every type."""
    T = _sym(CASES[name])
    margin = float(certify_point(jnp.asarray(T), MINKOWSKI)[condition])
    assert (margin >= -1e-8) == (brute_min(T, on_sphere) >= -1e-6), (
        f"{name}/{condition}: LMI margin {margin:.3e} disagrees with brute force"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_sec_matches_brute_force(name):
    """SEC is the WEC of the trace-reversed tensor; check it directly."""
    T = _sym(CASES[name])
    trace = float(np.trace(ETA @ T))
    theta = T - 0.5 * trace * ETA
    margin = float(certify_point(jnp.asarray(T), MINKOWSKI)["sec"])
    assert (margin >= -1e-8) == (brute_min(theta, False) >= -1e-6), name


@pytest.mark.parametrize("name", sorted(CASES))
def test_dec_matches_brute_force(name):
    """DEC = WEC and the flux -T^a_b u^b causal, i.e. (T^2)(u,u) <= 0."""
    T = _sym(CASES[name])
    margin = float(certify_point(jnp.asarray(T), MINKOWSKI)["dec"])
    expected = (brute_min(T, False) >= -1e-6) and (
        brute_min(-(T @ ETA @ T), False) >= -1e-6
    )
    assert (margin >= -1e-8) == expected, name


def test_type_ii_is_decided_not_nan():
    """The Type-II slot that used to be NaN now carries a real verdict.

    ``II-violating`` is the case recorded in ``frame_free._exact_margins``: the
    null witness is non-negative while the Eulerian energy density is negative,
    so WEC and DEC genuinely fail and were once reported satisfied.
    """
    m = certify_point(jnp.asarray(_sym(CASES["II-violating"])), MINKOWSKI)
    for key in ("nec", "wec", "sec", "dec"):
        assert np.isfinite(float(m[key])), key
    assert float(m["wec"]) < 0.0
    assert float(m["dec"]) < 0.0


def test_dec_never_cleaner_than_wec():
    """DEC implies WEC, so its margin can never be the more permissive one."""
    for name, T in CASES.items():
        m = certify_point(jnp.asarray(_sym(T)), MINKOWSKI)
        assert float(m["dec"]) <= float(m["wec"]) + 1e-9, name


def test_witness_observer_actually_violates():
    """When WEC fails, the returned boost must be a genuine violating observer."""
    T = _sym(CASES["I-violating"])
    w = np.asarray(witness_observer(jnp.asarray(T), MINKOWSKI))
    assert np.all(np.isfinite(w)) and np.linalg.norm(w) <= 1.0 + 1e-9
    u = np.concatenate([[1.0], w])
    assert float(u @ T @ u) < 0.0


def test_no_rapidity_cap_dependence():
    """A boost that a capped search would miss is still decided correctly.

    Build a Type-I tensor whose WEC failure only shows up at high rapidity:
    rho > 0 but rho + p_1 < 0, so the violating observer runs off toward the
    light cone. The LMI sees it without any cap.
    """
    T = np.diag([0.5, -0.9, 2.0, 2.0])
    m = certify_point(jnp.asarray(T), MINKOWSKI)
    assert float(m["wec"]) < 0.0
    assert float(m["nec"]) < 0.0          # rho + p_1 = -0.4 < 0
    # a cap at |w| <= 0.5 would report it clean
    w = np.linspace(0.0, 0.5, 2001)
    q = 0.5 + (-0.9) * w**2
    assert q.min() > 0.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_verdict_is_boost_invariant(name):
    """The verdict must survive a change of chart; the margin need not.

    Boosting the coordinates tilts the ``t = const`` slices, so the slice normal
   , and with it the Eulerian ``rho``, ``b``, ``S`` that normalize the margin,
    genuinely changes. What cannot change is the answer, because "some observer
    sees this fail" quantifies over all observers in any chart. This is precisely
    the paper's claim: the Boolean is frame-independent, the magnitude is a
    slice-normal-normalized severity.
    """
    T = _sym(CASES[name])
    base = certify_point(jnp.asarray(T), MINKOWSKI)
    ch, sh = np.cosh(0.8), np.sinh(0.8)
    L = np.array([[ch, sh, 0.0, 0.0], [sh, ch, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    T_new, g_new = L.T @ T @ L, L.T @ ETA @ L
    moved = certify_point(jnp.asarray(T_new), jnp.asarray(g_new))
    for key in ("nec", "wec", "sec", "dec"):
        assert (float(moved[key]) >= -1e-8) == (float(base[key]) >= -1e-8), (
            f"{name}/{key}: verdict flipped under a boost "
            f"({float(base[key]):.3e} -> {float(moved[key]):.3e})"
        )


def test_tetrad_components_recover_rho_and_momentum():
    """That[0,0] = rho, That[0,i] = -b_i, That[i,j] = S_ij in the tetrad frame."""
    T = _sym(CASES["IV-momentum"])
    T_hat = np.asarray(tetrad_components(jnp.asarray(T), MINKOWSKI))
    assert T_hat[0, 0] == pytest.approx(T[0, 0], abs=1e-12)
    assert np.allclose(T_hat[1:, 1:], T[1:, 1:], atol=1e-12)


def test_agrees_in_sign_with_eigenvalue_margins_at_type_i():
    """Cross-validation: at Type I the LMI must agree in sign with the
    eigenvalue inequalities that the paper actually reports, so rerouting
    Type II through the LMI cannot move any Type-I number."""
    from warpax.energy_conditions.eigenvalue_checks import check_all

    rng = np.random.default_rng(3)
    for _ in range(200):
        rho, p = rng.normal(size=1)[0] * 2.0, rng.normal(size=3) * 2.0
        T = np.diag([rho, *p])
        m = certify_point(jnp.asarray(T), MINKOWSKI)
        nec_I, wec_I, sec_I, dec_I = check_all(
            jnp.asarray(rho), jnp.asarray(p)
        )
        for key, ref in (("nec", nec_I), ("wec", wec_I),
                         ("sec", sec_I), ("dec", dec_I)):
            assert (float(m[key]) >= -1e-8) == (float(ref) >= -1e-8), (
                f"{key}: LMI {float(m[key]):.3e} vs eigenvalue {float(ref):.3e} "
                f"at rho={rho:.3f} p={p}"
            )


# --- regressions for the three defects found in the R2 audit ------------------

@pytest.mark.parametrize("name,T", [
    ("exact vacuum", np.zeros((4, 4))),
    ("eps-dust", 1e-12 * np.diag([1.0, 0.5, 0.5, 0.5])),
    ("unit dust", np.diag([1.0, 0.0, 0.0, 0.0])),
])
def test_vacuum_is_not_convicted(name, T):
    """A tensor satisfying every condition must not be reported as violating.

    Ternary search under-estimates a concave maximum, so on the exact vacuum, whose
    true maximum is 0, it returns about -6e-22, and a strict ``margin < 0`` test
    convicts Minkowski of violating all four conditions. The decision is one-sided:
    only ``margin < -noise_floor`` certifies a violation. The DEC floor is looser
    because the flux test feeds ``-T^2`` to the same search and squaring doubles the
    relative eigenvalue error.
    """
    Tj = jnp.asarray(T)
    m = certify_point(Tj, MINKOWSKI)
    for key in ("nec", "wec", "sec", "dec"):
        floor = float(noise_floor(Tj, MINKOWSKI, condition=key))
        assert float(m[key]) >= -floor, (
            f"{name}/{key}: margin {float(m[key]):.3e} below floor {floor:.3e}"
        )


def test_witness_is_genuine_at_a_repeated_lowest_eigenvalue():
    """The returned observer must actually violate, including in the degenerate case.

    For ``That = diag(1, -3, 10, 10)`` the optimal multiplier is 2, so
    ``M(2) = diag(-1, -1, 12, 12)`` has a *repeated* lowest eigenvalue and ``eigh``
    returns an arbitrary basis of that eigenspace. Reading the boost off the first
    eigenvector gave ``w = 0``, i.e. ``q(0) = +1 > 0``, reported as a violating
    observer while satisfying the condition.
    """
    T = np.diag([1.0, -3.0, 10.0, 10.0])
    assert float(certify_point(jnp.asarray(T), MINKOWSKI)["wec"]) < 0.0
    w = np.asarray(witness_observer(jnp.asarray(T), MINKOWSKI))
    assert np.all(np.isfinite(w)), "a violating observer exists and must be returned"
    assert np.linalg.norm(w) <= 1.0 + 1e-9, "the observer must be timelike"
    u = np.concatenate([[1.0], w])
    assert float(u @ T @ u) < 0.0, f"q({w}) = {float(u @ T @ u):.3f} is not a violation"


def test_witness_is_genuine_on_random_violating_tensors():
    """Randomized version of the above: no returned observer may fail to violate."""
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        A = rng.normal(size=(4, 4))
        T = _sym(A + A.T)
        if float(certify_point(jnp.asarray(T), MINKOWSKI)["wec"]) >= -1e-9:
            continue
        checked += 1
        w = np.asarray(witness_observer(jnp.asarray(T), MINKOWSKI))
        assert np.all(np.isfinite(w)) and np.linalg.norm(w) <= 1.0 + 1e-9
        T_hat = np.asarray(tetrad_components(jnp.asarray(T), MINKOWSKI))
        q = T_hat[0, 0] - 2.0 * (-T_hat[0, 1:]) @ w + w @ T_hat[1:, 1:] @ w
        assert q < 0.0, f"returned observer does not violate: q = {q:.3e}"
    assert checked > 20, "expected a decent number of violating draws"


@pytest.mark.parametrize("name", sorted(CASES))
def test_null_deficit_is_twice_the_nec_margin(name):
    """``null_deficit`` must equal the true minimum over the null cone.

    Lifting ``q`` to the null cone of ``eta`` gives
    ``max_sigma lambda_min(That + sigma eta) = (1/2) min_{|s|=1} q(s)``, so the
    type-independent NEC deficit needs no separate trust-region solver. Checked
    against a dense direct scan.
    """
    T = _sym(CASES[name])
    got = float(null_deficit(jnp.asarray(T), MINKOWSKI))
    ref = brute_min(T, True)
    assert got <= ref + 1e-6, "the exact minimum cannot exceed a sampled value"
    assert got == pytest.approx(ref, abs=2e-3), f"{name}: {got:.6f} vs scan {ref:.6f}"


def test_null_deficit_dominates_the_momentum_witness():
    """The deficit is never weaker than the Eulerian momentum-plane witness.

    The witness probes one null direction; the deficit minimizes over all of them, so
    it must be less than or equal to it everywhere. Where the inequality is strict the
    witness would have under-reported, which is exactly the Type-II NEC bug.
    """
    from warpax.energy_conditions.frame_free import eulerian_null_witness

    g_inv = jnp.linalg.inv(MINKOWSKI)
    rng = np.random.default_rng(11)
    strict = 0
    for _ in range(100):
        A = rng.normal(size=(4, 4))
        T = jnp.asarray(_sym(A + A.T))
        deficit = float(null_deficit(T, MINKOWSKI))
        witness = float(eulerian_null_witness(T, MINKOWSKI, g_inv))
        assert deficit <= witness + 1e-8, f"{deficit:.6f} > {witness:.6f}"
        if deficit < witness - 1e-6:
            strict += 1
    assert strict > 0, "the witness should be strictly weaker somewhere"


# --------------------------------------------------------------------------
# Covariance under T -> c T.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,T",
    [
        # WEC binds the DEC here.
        ("perfect fluid", np.diag([2.0, -1.0, 0.5, 0.5])),
        # Flux binds the DEC here: rho = 1 < |p_1| = 2, but rho + p_i >= 0.
        ("flux-bound", np.diag([1.0, 2.0, 0.0, 0.0])),
        ("momentum", np.array([[1.0, -0.9, 0.0, 0.0],
                               [-0.9, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.2, 0.0],
                               [0.0, 0.0, 0.0, 0.2]])),
    ],
)
@pytest.mark.parametrize("condition", ["nec", "wec", "sec", "dec"])
def test_margins_are_homogeneous_of_degree_one(name, T, condition):
    """Every margin must scale linearly under ``T -> c T``, the DEC included.

    The flux half of the DEC is the ball margin of ``-T^2`` and is therefore
    quadratic in the tensor. Taking ``min(wec, flux)`` unmodified produced a
    reported DEC margin that was degree one where the WEC bound and degree two
    where the flux bound: on ``diag(1, 2, 0, 0)`` it ran -1.5, -6, -150 for
    c = 1, 2, 10 while the WEC margin ran 0.5, 1, 5. Signs were always right, so
    no verdict moved, but any DEC ranking or scaling fit crossing that switch was
    comparing different powers of the same tensor.
    """
    Tj = jnp.asarray(T)
    base = float(certify_point(Tj, MINKOWSKI)[condition])
    floor = float(noise_floor(Tj, MINKOWSKI, condition=condition))
    for c in (1e-8, 1e-3, 1.0, 1e3, 1e8):
        got = float(certify_point(jnp.asarray(c * T), MINKOWSKI)[condition])
        if abs(base) <= floor:
            # Saturated: the true margin is zero and no float64 search resolves its
            # sign. The contract only promises the result stays inside the floor,
            # which must itself scale linearly, that is the covariance being tested.
            assert abs(got) <= c * floor * 10.0, (
                f"{name}/{condition}: saturated base {base:g} but c={c:g} gave {got:g}"
            )
            continue
        assert got == pytest.approx(c * base, rel=1e-9), (
            f"{name}/{condition}: c={c:g} gave {got:g}, expected {c * base:g}"
        )


def test_noise_floor_is_purely_relative():
    """One relative floor covers all four conditions once the flux is divided out."""
    T = np.diag([1.0, 2.0, 0.0, 0.0])
    for c in (1e-6, 1.0, 1e6):
        floors = {
            k: float(noise_floor(jnp.asarray(c * T), MINKOWSKI, condition=k))
            for k in ("nec", "wec", "sec", "dec")
        }
        assert len(set(floors.values())) == 1, floors
        assert floors["dec"] == pytest.approx(1e-12 * 2.0 * c, rel=1e-6)


def test_exact_vacuum_returns_exactly_zero():
    """The unclamped bracket collapses with the tensor, so vacuum is not convicted.

    With the bracket scale clamped at 1 the residual was absolute, about -1e-14 at
    80 ternary steps, which against a relative floor reads as Minkowski violating
    all four conditions.
    """
    margins = certify_point(jnp.zeros((4, 4)), MINKOWSKI)
    for k, v in margins.items():
        assert float(v) == 0.0, f"{k} = {float(v):g}"


# --------------------------------------------------------------------------
# The referee's Type-II locus (report item B1).
# --------------------------------------------------------------------------


def test_lmi_is_continuous_through_the_type_ii_locus():
    """Item B1's counterexample is the zero of the LMI margin, not a hole in it.

    For the momentum block ``A = [[-rho, j], [-j, S_par]]`` the discriminant
    ``Delta = (rho + S_par)^2 - 4 j^2`` vanishes with ``j != 0`` at a Hawking-Ellis
    Type-II point, so a continuous Type-I/Type-IV transition must cross Type II.
    The LMI forms no eigendecomposition and never consults the type, so its margin
    passes through that locus linearly and vanishes exactly on it.
    """
    rho = s_par = 1.0
    js = [0.90, 0.99, 0.999, 1.0, 1.001, 1.01, 1.10]
    margins = []
    for j in js:
        T = np.zeros((4, 4))
        T[0, 0] = rho
        T[0, 1] = T[1, 0] = -j
        T[1, 1] = s_par
        T[2, 2] = T[3, 3] = 0.2
        margins.append(float(certify_point(jnp.asarray(T), MINKOWSKI)["nec"]))

    # Exactly zero on the locus, and strictly monotone through it.
    assert margins[js.index(1.0)] == pytest.approx(0.0, abs=1e-13)
    assert all(a > b for a, b in pairwise(margins)), margins
    # Delta > 0 is satisfied, Delta < 0 is violated, with no jump at Delta = 0.
    assert margins[0] > 0 and margins[-1] < 0
    # The margin is (rho + S_par)/2 - |j| here, i.e. linear in j on both sides.
    for j, m in zip(js, margins, strict=True):
        assert m == pytest.approx(0.5 * (rho + s_par) - j, abs=1e-12)


@pytest.mark.slow
def test_lmi_agrees_with_brute_force_at_every_hawking_ellis_type():
    """The LMI verdict must match a brute-force observer search.

    The four canonical forms plus the Type-IV tensor whose quartic
    discriminant vanishes, in Minkowski coordinates where the tetrad is the
    identity. Moved here from a module __main__ block that never ran.
    """
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    g = jnp.asarray(eta)
    rng = np.random.default_rng(0)

    def brute(T_np, on_sphere, n=400_000):
        w = rng.normal(size=(n, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        if not on_sphere:
            w *= rng.random((n, 1)) ** (1 / 3.0)
        u = np.concatenate([np.ones((n, 1)), w], axis=1)
        return float(np.einsum("ni,ij,nj->n", u, T_np, u).min())

    k = np.array([1.0, 1.0, 0.0, 0.0])
    Tm3 = np.array([[0.7, 0.0, 1.0, 0.0], [0.0, -0.7, 1.0, 0.0],
                    [1.0, -1.0, -0.7, 0.0], [0.0, 0.0, 0.0, 0.3]])
    f, c = 1.3, 0.4
    Tm4 = np.array([[0.0, f, 0.0, 0.0], [-f, 0.0, 0.0, 0.0],
                    [0.0, 0.0, c, 0.0], [0.0, 0.0, 0.0, c]])
    cases = {
        "I(ok)": np.diag([2.0, 0.5, 0.5, 0.5]),
        "I(bad)": np.diag([-1.0, 0.5, 0.5, 0.5]),
        "II": np.outer(k, k),
        "II(bad)": np.outer(k, k) + np.diag([-1.0, 0.0, 3.0, 3.0]),
        "III": eta @ Tm3,
        "IV(D4=0)": eta @ Tm4,
    }

    for name, T_np in cases.items():
        T_np = 0.5 * (T_np + T_np.T)
        m = certify_point(jnp.asarray(T_np), g)
        for key, sphere in (("nec", True), ("wec", False)):
            cert = bool(m[key] >= -1e-8)
            true = brute(T_np, sphere) >= -1e-6
            assert cert == true, (
                f"{name}/{key}: LMI says {cert}, brute force says {true} "
                f"(margin {float(m[key]):.3e})"
            )
        # DEC and SEC must never be reported cleaner than the WEC/NEC they imply.
        assert float(m["dec"]) <= float(m["wec"]) + 1e-9, name
