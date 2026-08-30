"""Interval-arithmetic certified enclosures (paper Appendix H).

The property under test is that the interval curvature chain reproduces the
JAX pipeline exactly on a degenerate (point) box. If it does not, every
"certified" bound in the appendix is meaningless, so this is checked against the
independent autodiff implementation for both a single-component shift
(Alcubierre) and a three-component one (Rodal).
"""

import math

import jax
import jax.numpy as jnp
import mpmath
import pytest
from mpmath import iv

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions._intervalcurv import eulerian_fields_interval
from warpax.energy_conditions.enclosure import (
    METRICS,
    alcubierre_metric,
    certify_nec_deficit,
    natario_metric,
    rodal_metric,
    shape_interval,
    tail_bound,
    van_den_broeck_metric,
)
from warpax.geometry.geometry import compute_curvature_chain
from warpax.metrics import RodalMetric
from warpax.metrics.natario import NatarioMetric
from warpax.metrics.van_den_broeck import VanDenBroeckMetric

jax.config.update("jax_enable_x64", True)

_PT = [0.0, 0.62, 0.81, 0.0]


def _jax_reference(metric):
    """Eulerian decomposition in an ORTHONORMAL spatial frame, from the JAX chain.

    The orthonormalisation is the point of the test. Reading off ``T[a, i+1]`` and
    ``T[i+1, j+1]`` gives coordinate components, and comparing those against the
    interval chain's coordinate components is vacuous: both sides make the same
    mistake and the test passes while the certified objective minimises over the
    wrong sphere. That is what happened for Van den Broeck, whose slice is
    ``gamma_ij = B^2 delta_ij``. Both sides are now pushed through ``L^{-1}`` with
    ``gamma = L L^T``, so the test compares the physical quantity.
    """
    import numpy as np

    curv = compute_curvature_chain(metric, jnp.array(_PT))
    T, g_inv = curv.stress_energy, curv.metric_inv
    n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
    n_up = g_inv @ n_low
    n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
    rho = float(n_up @ (T @ n_up))
    b_cov = np.array([float(sum(n_up[a] * T[a, i + 1] for a in range(4))) for i in range(3)])
    S_cov = np.array([[float(T[i + 1, j + 1]) for j in range(3)] for i in range(3)])

    gam = np.array([[float(curv.metric[i + 1, j + 1]) for j in range(3)] for i in range(3)])
    L = np.linalg.cholesky(gam)
    Linv = np.linalg.inv(L)
    b = list(Linv @ b_cov)
    S = (Linv @ S_cov @ Linv.T).tolist()
    return rho, b, S


# Must be the benchmark instance. R_tilde was 0.6 here and in run_enclosures.py while
# every other script fixes 1.0, so this test compared the interval and JAX
# transcriptions of a spacetime the paper never reports, they agreed with each
# other and said nothing about the certified bracket in tables/enclosures.tex.
_VDB = dict(v_s=0.5, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)


# Each entry pairs an interval-algebra transcription with the JAX metric it must
# reproduce. Natario exercises a three-component vortical shift; Van den Broeck
# exercises the conformal branch of ``_assemble`` (gamma_ij = B^2 delta_ij), which
# the flat-slice drives never touch.
@pytest.mark.parametrize(
    "name,builder,reference",
    [
        (
            "Alcubierre",
            lambda: alcubierre_metric(0.5, 1.0, 8.0),
            lambda: AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
        ),
        (
            "Rodal",
            lambda: rodal_metric(0.5, 1.0, 8.0),
            lambda: RodalMetric(v_s=0.5, R=1.0, sigma=8.0),
        ),
        (
            "Natario",
            lambda: natario_metric(0.5, 1.0, 8.0),
            lambda: NatarioMetric(v_s=0.5, R=1.0, sigma=8.0),
        ),
        ("VanDenBroeck", lambda: van_den_broeck_metric(**_VDB), lambda: VanDenBroeckMetric(**_VDB)),
    ],
)
def test_interval_chain_matches_jax_at_a_point(name, builder, reference):
    mpmath.mp.prec = 80
    box = [iv.mpf([v, v]) for v in _PT]
    rho_iv, b_iv, S_iv = eulerian_fields_interval(builder(), box)
    rho_j, b_j, S_j = _jax_reference(reference())

    # The property that makes the appendix's bounds mean anything is CONTAINMENT:
    # the JAX value must lie inside the interval enclosure. The pad absorbs the
    # rounding a degenerate box still accumulates; measured, the worst excess is
    # 2.7e-17 (Rodal), 1.5e-16 (Alcubierre), 1.0e-15 (Natario), so 1e-12 is three
    # orders of headroom. It used to be 1e-9, which was loose enough to hide a
    # genuine transcription mismatch: the interval Rodal regularised the direction
    # divisor with 1e-60 where the JAX metric uses 1e-12, so the two were
    # different spacetimes agreeing to within the pad.
    def contains(c, v, pad=1e-12):
        return float(mpmath.mpf(c.a)) - pad <= v <= float(mpmath.mpf(c.b)) + pad

    def width(c):
        return float(mpmath.mpf(c.b)) - float(mpmath.mpf(c.a))

    assert contains(rho_iv, rho_j), f"{name}: rho {rho_j} outside {rho_iv}"
    for i in range(3):
        assert contains(b_iv[i], b_j[i]), f"{name}: b[{i}] outside enclosure"
    for i in range(3):
        for j in range(3):
            assert contains(S_iv[i][j], S_j[i][j]), f"{name}: S[{i}][{j}] outside"

    # And on a point box the enclosure must be tight, or branch-and-bound can
    # never converge no matter how far it subdivides.
    assert width(rho_iv) < 1e-9, f"{name}: rho enclosure too wide at a point"


def test_rodal_momentum_vanishes():
    """Irrotational shift carries no Eulerian momentum, so N = min_i(rho + p_i)."""
    mpmath.mp.prec = 80
    box = [iv.mpf([v, v]) for v in _PT]
    _, b_iv, _ = eulerian_fields_interval(rodal_metric(0.5, 1.0, 8.0), box)
    assert max(abs(float(mpmath.mpf(c.a))) for c in b_iv) < 1e-10


@pytest.mark.slow
def test_enclosure_brackets_are_ordered():
    enc = certify_nec_deficit(
        alcubierre_metric(0.5, 1.0, 8.0),
        shape_interval(1.0, 8.0),
        x_range=(-3.0, 3.0),
        s_range=(0.0, 3.0),
        tol=5e-2,
        max_boxes=1500,
    )
    assert enc.lower <= enc.upper


def test_conformal_slice_null_deficit_uses_the_physical_sphere():
    """Van den Broeck: the sign of N flips if the spatial frame is not orthonormal.

    Regression for a defect that survived the pointwise cross-check above because
    that check compared coordinate components on both sides. At this point the
    coordinate objective returns -0.0350, a certified null-energy violation,
    while the physical one returns +0.0706. Only the second is the null deficit.
    """
    import numpy as np

    mpmath.mp.prec = 80
    kw = dict(v_s=0.5, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=1.0, sigma_B=8.0)
    box = [iv.mpf([0, 0]), iv.mpf([1, 1]), iv.mpf([0, 0]), iv.mpf([0, 0])]
    rho_iv, b_iv, S_iv = eulerian_fields_interval(van_den_broeck_metric(**kw), box)

    mid = lambda c: 0.5 * (float(mpmath.mpf(c.a)) + float(mpmath.mpf(c.b)))
    rho = mid(rho_iv)
    b = np.array([mid(c) for c in b_iv])
    S = np.array([[mid(S_iv[i][j]) for j in range(3)] for i in range(3)])

    v = np.random.default_rng(0).normal(size=(200_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    n_min = (rho + 2 * v @ b + np.einsum("ni,ij,nj->n", v, S, v)).min()
    assert n_min > 0.05, f"expected the physical deficit ~ +0.0706, got {n_min}"


# ---------------------------------------------------------------------------
# The centered (mean-value) bound
#
# ``enclosure._make_objective`` tightens its lower bound with a mean-value form
# built from jet gradients. That form is the difference between a branch and bound
# that closes and one that exhausts its budget, and it is also the one place where a
# sign or index error would silently produce a bound that is tighter than the truth
# , which would make every "certified" number in the paper wrong in the dangerous
# direction. These tests check containment directly rather than trusting the algebra.
# ---------------------------------------------------------------------------

_ENCLOSURE_METRICS = [
    ("Alcubierre", lambda: alcubierre_metric(0.5, 1.0, 8.0), (-0.003, 0.932)),
    ("Rodal", lambda: rodal_metric(0.5, 1.0, 8.0), (-0.971, 0.003)),
    ("Natario", lambda: natario_metric(0.5, 1.0, 8.0), (0.003, 1.011)),
    ("VanDenBroeck", lambda: van_den_broeck_metric(**_VDB), (-0.003, 0.929)),
]


def _centered_fields(metric, cx, cy, h):
    """``(rho, b, S)`` over the box by the mean-value form, as enclosure.py builds it."""
    box = [iv.mpf([0, 0]), iv.mpf([cx - h, cx + h]), iv.mpf([cy - h, cy + h]), iv.mpf([0, 0])]
    ctr = [iv.mpf([0, 0]), iv.mpf([cx, cx]), iv.mpf([cy, cy]), iv.mpf([0, 0])]
    jf = eulerian_fields_interval(metric, box, jet=True)
    cf = eulerian_fields_interval(metric, ctr)
    dx, dy = box[1] - ctr[1], box[2] - ctr[2]
    cen = lambda j, c: c + j.d[1] * dx + j.d[2] * dy
    return (
        cen(jf[0], cf[0]),
        [cen(jf[1][i], cf[1][i]) for i in range(3)],
        [[cen(jf[2][i][j], cf[2][i][j]) for j in range(3)] for i in range(3)],
    )


@pytest.mark.parametrize("name,builder,centre", _ENCLOSURE_METRICS)
def test_centered_form_encloses_every_interior_point(name, builder, centre):
    """Sampled true values must lie inside the centered enclosure, for every box.

    This is the containment property. A wrong gradient index, a missed cross term or
    a sign slip in the jet ring shows up here immediately, because the enclosure
    stops covering values the chain itself produces.
    """
    import numpy as np

    mpmath.mp.prec = 80
    metric = builder()
    cx, cy = centre
    rng = np.random.default_rng(20260818)
    pad = 1e-9

    def lo(c):
        return float(mpmath.mpf(c.a))

    def hi(c):
        return float(mpmath.mpf(c.b))

    for h in (1e-2, 1e-3, 1e-4):
        rho_k, b_k, S_k = _centered_fields(metric, cx, cy, h)
        for px, py in rng.uniform(-h, h, size=(6, 2)):
            pt = [
                iv.mpf([0, 0]),
                iv.mpf([cx + px, cx + px]),
                iv.mpf([cy + py, cy + py]),
                iv.mpf([0, 0]),
            ]
            rho_p, b_p, S_p = eulerian_fields_interval(metric, pt)
            assert lo(rho_k) - pad <= lo(rho_p) and hi(rho_p) <= hi(rho_k) + pad, (
                f"{name} h={h}: rho {rho_p} escapes centered {rho_k}"
            )
            for i in range(3):
                assert lo(b_k[i]) - pad <= lo(b_p[i]) and hi(b_p[i]) <= hi(b_k[i]) + pad
                for j in range(3):
                    assert (
                        lo(S_k[i][j]) - pad <= lo(S_p[i][j])
                        and hi(S_p[i][j]) <= hi(S_k[i][j]) + pad
                    )


@pytest.mark.parametrize("name,builder,centre", _ENCLOSURE_METRICS)
def test_jet_ring_reproduces_the_plain_interval_values(name, builder, centre):
    """The jet ring must return the same value enclosure as the plain one.

    Both are rigorous outward-rounded enclosures of the same quantity, so neither can
    be unsound; what this guards against is a jet path that has drifted into
    computing something *else*. The two agree to a couple of ulps rather than bit for
    bit, because a handful of operations reassociate, so the check is relative and
    tight rather than exact.
    """
    mpmath.mp.prec = 80
    cx, cy = centre
    h = 1e-3
    box = [iv.mpf([0, 0]), iv.mpf([cx - h, cx + h]), iv.mpf([cy - h, cy + h]), iv.mpf([0, 0])]
    metric = builder()
    r0, b0, S0 = eulerian_fields_interval(metric, box)
    r1, b1, S1 = eulerian_fields_interval(metric, box, jet=True)

    def same(a, b, rtol=1e-14):
        for x, y in ((a.a, b.a), (a.b, b.b)):
            x, y = float(mpmath.mpf(x)), float(mpmath.mpf(y))
            if abs(x - y) > rtol * max(1.0, abs(x), abs(y)):
                return False
        return True

    assert same(r0, r1.v), f"{name}: jet value enclosure differs for rho"
    for i in range(3):
        assert same(b0[i], b1[i].v), f"{name}: jet value enclosure differs for b[{i}]"
        for j in range(3):
            assert same(S0[i][j], S1[i][j].v), f"{name}: differs for S[{i}][{j}]"


def test_jet_gradient_matches_autodiff():
    """The jet gradient of ``rho_n`` must match an independent JAX derivative.

    The jet ring is new code; ``jax.grad`` of the existing curvature chain is not.
    Agreement at a point is the cheapest independent check that the extra derivative
    the centered form leans on is the derivative it claims to be.
    """
    import numpy as np

    mpmath.mp.prec = 100
    pt = (0.62, 0.81)

    def rho_jax(x, y):
        curv = compute_curvature_chain(
            AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
            jnp.array([0.0, x, y, 0.0]),
        )
        n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
        n_up = curv.metric_inv @ n_low
        n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
        return n_up @ (curv.stress_energy @ n_up)

    gx, gy = jax.grad(rho_jax, argnums=(0, 1))(pt[0], pt[1])
    box = [iv.mpf([0, 0]), iv.mpf([pt[0], pt[0]]), iv.mpf([pt[1], pt[1]]), iv.mpf([0, 0])]
    rho_j, _, _ = eulerian_fields_interval(alcubierre_metric(0.5, 1.0, 8.0), box, jet=True)
    mid = lambda c: 0.5 * (float(mpmath.mpf(c.a)) + float(mpmath.mpf(c.b)))
    assert np.isclose(mid(rho_j.d[1]), float(gx), rtol=1e-7, atol=1e-9)
    assert np.isclose(mid(rho_j.d[2]), float(gy), rtol=1e-7, atol=1e-9)


@pytest.mark.slow
def test_certified_lower_never_exceeds_a_brute_force_minimum():
    """The certified bracket must contain the true minimum on the searched region.

    A tighter bound that is not sound would be fatal, and it would not be caught by
    any test that only checks ``lower <= upper``. This samples the objective over the
    same region the branch and bound searches and asserts the certificate is below
    every sampled value.
    """
    import numpy as np

    mpmath.mp.prec = 80
    enc = certify_nec_deficit(
        alcubierre_metric(0.5, 1.0, 8.0),
        shape_interval(1.0, 8.0),
        x_range=(-3.0, 3.0),
        s_range=(0.0, 3.0),
        tol=5e-2,
        max_boxes=4000,
    )
    rng = np.random.default_rng(7)
    metric = alcubierre_metric(0.5, 1.0, 8.0)
    shape = shape_interval(1.0, 8.0)
    v = rng.normal(size=(4000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mid = lambda c: 0.5 * (float(mpmath.mpf(c.a)) + float(mpmath.mpf(c.b)))
    checked = 0
    for x, s in zip(rng.uniform(-3.0, 3.0, 1200), rng.uniform(0.0, 3.0, 1200), strict=True):
        f = shape(iv.mpf([x, x]), iv.mpf([s, s]))
        if not (0.1 <= mid(f) <= 0.9):
            continue
        try:
            rho_iv, b_iv, S_iv = eulerian_fields_interval(
                metric, [iv.mpf([0, 0]), iv.mpf([x, x]), iv.mpf([s, s]), iv.mpf([0, 0])]
            )
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
        rho = mid(rho_iv)
        b = np.array([mid(c) for c in b_iv])
        S = np.array([[mid(S_iv[i][j]) for j in range(3)] for i in range(3)])
        n_min = (rho + 2 * v @ b + np.einsum("ni,ij,nj->n", v, S, v)).min()
        assert enc.lower <= n_min + 1e-9, (
            f"certified lower {enc.lower} exceeds a sampled value {n_min} "
            f"at (x, s) = ({x}, {s}), the bound is NOT an enclosure"
        )
        checked += 1
    assert checked > 20, "the sampler found too few wall points to be a real check"


# ---------------------------------------------------------------------------
# Soundness regressions found by an external audit of the interval pipeline.
#
# Each of these three was a place where the implementation of a valid inequality
# was not itself rigorous, so the returned "lower bound" could sit ABOVE the true
# minimum. None was caught by the containment or brute-force tests above, because
# those exercise the paper's four metrics at moderate coefficient scales, where the
# outward padding happened to absorb the error. A bound that is only accidentally
# sound is not certified, so the counterexamples are pinned here.
# ---------------------------------------------------------------------------


def _set_prec(bits=60):
    """Set BOTH precisions.

    ``mpmath.mp.prec`` and ``mpmath.iv.prec`` are independent, and
    ``certify_nec_deficit`` sets ``iv.prec`` itself, so a test that sets only
    ``mp.prec`` inherits whatever the previously executed test left behind, and its
    result depends on collection order.
    """
    mpmath.mp.prec = 200
    iv.prec = bits


def test_shape_interval_encloses_the_true_shape_value():
    """The wall mask must be an enclosure; it decides which boxes are discarded.

    The earlier implementation assembled the radius from round-to-nearest
    ``math.sqrt`` calls and installed the results as interval endpoints. At the
    point below it returned an interval that did not contain the true value, so a
    box holding the minimiser could be dropped and the certified lower bound raised.
    """
    _set_prec()
    x, s = 0.950030236708907, 0.5338202611110032
    got = shape_interval(1.0, 8.0)(iv.mpf([x, x]), iv.mpf([s, s]))

    mpmath.mp.prec = 300
    r = mpmath.sqrt(mpmath.mpf(x) ** 2 + mpmath.mpf(s) ** 2)
    truth = (mpmath.tanh(8 * (r + 1)) - mpmath.tanh(8 * (r - 1))) / (2 * mpmath.tanh(mpmath.mpf(8)))
    assert mpmath.mpf(got.a) <= truth <= mpmath.mpf(got.b), (
        f"shape_interval {got} does not contain {truth}"
    )


def test_inv4_keeps_a_derivative_through_a_zero_valued_pivot_factor():
    """A zero VALUE with a nonzero derivative must not be eliminated as zero.

    ``_inv4`` used to skip the elimination step when the factor's value interval was
    exactly zero. For a jet that is the wrong test: the value can vanish while the
    derivative does not, and skipping drops the derivative silently.
    """
    from warpax.energy_conditions import _jet
    from warpax.energy_conditions._intervalcurv import _inv4

    _set_prec()
    t = _jet.seed(iv.mpf([0, 0]), 1)
    one, zero = _jet.constant(1), _jet.constant(0)
    # M(t) = [[1, t], [t, 1]] (+) I_2, so d(M^-1)_01/dt = -1 at t = 0.
    M = [
        [one, t, zero, zero],
        [t, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, one],
    ]
    d01 = _inv4(M)[0][1].d[1]
    assert mpmath.mpf(d01.a) <= -1 <= mpmath.mpf(d01.b), (
        f"d(M^-1)_01/dt enclosure {d01} lost the true value -1"
    )


def test_decoupled_bound_stays_below_the_truth_under_heavy_cancellation():
    """The decoupled bound must not exceed the true deficit at any coefficient scale.

    It used to be evaluated in binary64 from round-to-nearest endpoints, with
    ``lambda_min`` read off ``numpy.linalg.eigvalsh`` of the midpoint matrix, which
    carries no certified error bound. Every one of those errors moves the result
    upward, and the bound is max'd against the verified LMI value, so an upward
    error wins. At the coefficients below it returned roughly six times the true
    deficit, above an achieved upper bound, hence not a lower bound at all.
    """
    import numpy as np

    from warpax.energy_conditions.enclosure import _lo

    _set_prec()
    scale = 1e100
    rho_f = 1.6131529359720847e100
    S_f = np.array(
        [
            [-0.10626933759389247, 0.8805710554015718, 0.1753514443711059],
            [0.8805710554015718, -0.4760328068765869, -1.02175725466781],
            [0.1753514443711059, -1.02175725466781, 0.4374447732538044],
        ]
    )
    # b = 0, so the true deficit is exactly rho + lambda_min(S).
    truth = rho_f + float(np.linalg.eigvalsh(S_f)[0]) * scale

    S = [[iv.mpf([float(S_f[i][j]) * scale] * 2) for j in range(3)] for i in range(3)]
    lam = None
    for i in range(3):
        row = iv.mpf([0, 0])
        for j in range(3):
            if j != i:
                row = row + abs(S[i][j])
        cand = S[i][i] - row
        lam = (
            cand
            if lam is None
            else iv.mpf(
                [
                    min(mpmath.mpf(lam.a), mpmath.mpf(cand.a)),
                    min(mpmath.mpf(lam.b), mpmath.mpf(cand.b)),
                ]
            )
        )
    bound = _lo(iv.mpf([rho_f, rho_f]) + lam)
    assert bound <= truth, f"decoupled bound {bound:.6e} exceeds the true deficit {truth:.6e}"


def test_exterior_holds_no_wall_point_in_any_direction():
    """The tail exclusion must certify band disjointness, not compare deficits.

    ``_make_objective`` carries the same wall mask as the search, so on the exterior
    it takes the masked branch and never computes a deficit: the old ``tail_bound``
    returned ``+inf`` and the ``tail > upper`` test that consumed it was vacuously
    true. The wall band is a condition on ``f`` alone, and ``f`` sees position only
    through ``r``, so one interval evaluation covers every direction, including
    ``x < -3``, which an outer annulus stated in ``x`` alone never touched.
    """
    mpmath.mp.prec = 60
    iv.prec = 60
    metric = METRICS["Alcubierre"](0.5, 1.0, 8.0)
    shape = shape_interval(1.0, 8.0)
    for x_box, s_box in (
        ((3.0, 30.0), (0.0, 30.0)),
        ((-30.0, -3.0), (0.0, 30.0)),
        ((-30.0, 30.0), (3.0, 30.0)),
    ):
        f_lo, f_hi, excluded = tail_bound(metric, shape, x_box, s_box)
        assert excluded, (x_box, s_box, f_lo, f_hi)
        assert f_hi < 0.1, f_hi
        # And it must be a real enclosure, not an infinity standing in for one.
        assert math.isfinite(f_lo) and math.isfinite(f_hi)
