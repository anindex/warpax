"""The exact certificates must agree with the float LMI wherever the LMI is decisive."""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions.certificate import (
    _CONDITIONS,
    certify,
    condition_matrix,
    find_multiplier,
    is_psd_exact,
    to_exact,
    verify,
)
from warpax.energy_conditions.slemma import certify_point, noise_floor

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
MINKOWSKI = jnp.asarray(ETA)
CONDITIONS = ("nec", "wec", "sec", "dec")


def _sym(A):
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


_certify_batch = jax.jit(jax.vmap(certify_point, in_axes=(0, None)))

# One draw set, shared by all four conditions, with the float side of the
# comparison taken in a single batched trace.
_rng = np.random.default_rng(7)
_A = _rng.normal(size=(24, 4, 4))
_DRAWS = jnp.asarray(0.5 * (_A + np.swapaxes(_A, 1, 2)))
_noise_floor_batch = {
    c: jax.jit(jax.vmap(lambda T, g, c=c: noise_floor(T, g, condition=c), in_axes=(0, None)))(
        _DRAWS, MINKOWSKI
    )
    for c in CONDITIONS
}


def test_is_psd_exact_matches_numpy_on_random_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = _sym(rng.normal(size=(4, 4)))
        want = bool(np.linalg.eigvalsh(A)[0] >= -1e-12)
        got = is_psd_exact(to_exact(A))
        if abs(np.linalg.eigvalsh(A)[0]) > 1e-9:  # skip the boundary
            assert got == want, A


def test_is_psd_exact_on_a_singular_psd_matrix():
    """A rank-deficient PSD matrix defeats an unpivoted LDL^T; principal minors do not."""
    A = np.diag([1.0, 0.0, 2.0, 0.0])
    assert is_psd_exact(to_exact(A))
    A[0, 1] = A[1, 0] = 1.0  # now indefinite: the 2x2 minor is -1
    assert not is_psd_exact(to_exact(A))


def test_psd_is_congruence_invariant_so_no_tetrad_is_needed():
    """The whole certificate rests on this: PSD of a form is basis-free.

    ``M(sigma) = That + sigma eta = e (T + sigma g) e^T`` for the tetrad ``e``, so
    certifying in coordinates certifies in every orthonormal frame, which matters
    because an orthonormal tetrad of a rational metric is not a rational object.
    """
    rng = np.random.default_rng(3)
    for _ in range(50):
        A = _sym(rng.normal(size=(4, 4)))
        P = rng.normal(size=(4, 4))
        if abs(np.linalg.det(P)) < 1e-3:
            continue
        assert is_psd_exact(to_exact(A)) == is_psd_exact(to_exact(P.T @ A @ P))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_certificates_agree_with_the_float_lmi(condition):
    """Every decisive float verdict must be backed by an exact certificate of the
    same sign, and every certificate must re-verify from scratch.

    ``certify`` runs an exact rational search, about a second per tensor, so the
    draw count is what keeps this affordable. The float margins come from one
    batched trace rather than 24 eager ones.
    """
    margins = np.asarray(_certify_batch(_DRAWS, MINKOWSKI)[condition])
    floors = np.asarray(_noise_floor_batch[condition])
    decided = 0
    for T, margin, floor in zip(np.asarray(_DRAWS), margins, floors, strict=True):
        cert = certify(T, ETA, condition)
        assert verify(cert, T, ETA), (condition, T, cert)
        if margin > floor:
            assert cert["kind"] == "satisfied", (margin, cert)
            decided += 1
        elif margin < -floor:
            assert cert["kind"] == "violated", (margin, cert)
            decided += 1
    assert decided > 12, f"only {decided} decisive cases, fixture is not exercising much"


def test_a_forged_certificate_is_rejected():
    """verify() must be a real check, not a rubber stamp."""
    T = np.diag([1.0, -3.0, 0.0, 0.0])  # NEC fails
    good = certify(T, ETA, "nec")
    assert good["kind"] == "violated" and verify(good, T, ETA)

    forged = dict(good)
    forged["kind"] = "satisfied"
    forged["sigma"] = {"nec": [0, 1]}
    forged.pop("witness_pair", None)
    assert not verify(forged, T, ETA)

    # And a witness pair that does not annihilate g must be rejected, because a
    # nonzero <g, X> leaves a multiplier free to absorb it. Here k is made timelike
    # while the partner keeps weight zero, so <g, X> = alpha g(k,k) < 0: the pair
    # still shows T is negative somewhere, but only on the causal cone, which is the
    # WEC and not the NEC.
    bad = dict(good)
    bad["witness_pair"] = [[[2, 1], [1, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]]]
    assert not verify(bad, T, ETA)


def test_the_tensor_is_checked_too_not_only_the_certificate():
    """An auditor supplies both arguments, so both are untrusted.

    A nonsymmetric T has principal minors that are not the elementary symmetric
    functions of any real spectrum, so is_psd_exact says nothing about it. This one has
    minor sums 4, 16, 24, 11, every one positive, and so passed the PSD test at
    sigma = 0 and carried a "satisfied" NEC certificate, while the null vector
    k = (1,-1,0,0) gives T(k,k) = -7.
    """
    T_forged = np.array(
        [[1.0, 10.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    k = np.array([1.0, -1.0, 0.0, 0.0])
    assert k @ ETA @ k == 0.0 and k @ T_forged @ k < 0.0
    stamp = {"condition": "nec", "kind": "satisfied", "sigma": {"nec": [0, 1]}}
    assert not verify(stamp, T_forged, ETA)

    # A symmetric tensor against a metric that is not Lorentzian is refused as well:
    # the S-lemma's multiplier is free in sign because the null cone is a real cone,
    # and a Euclidean g has none.
    T_ok = np.diag([3.0, 1.0, 1.0, 1.0])
    good = certify(T_ok, ETA, "nec")
    assert verify(good, T_ok, ETA)
    assert not verify(good, T_ok, np.eye(4))
    assert not verify(good, T_ok, np.diag([-1.0, -1.0, 1.0, 1.0]))
    assert not verify(good, T_ok, np.diag([0.0, 1.0, 1.0, 1.0]))


def test_referee_item_b1_type_ii_locus_is_certified_exactly():
    """The Delta = 0 Type-II point of report item B1 gets a definite exact verdict.

    This is the point the referee constructed to show Types I and IV are not
    exhaustive. The certificate never forms an eigendecomposition and never asks what
    the algebraic type is, so the locus is not special to it.
    """
    T = np.array(
        [[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.2]]
    )
    kinds = {}
    for c in CONDITIONS:
        cert = certify(T, ETA, c)
        assert verify(cert, T, ETA), (c, cert)
        assert cert["kind"] != "saturated", (c, cert)
        kinds[c] = cert["kind"]
    assert kinds["nec"] == "satisfied"  # the margin is exactly zero, and attained
    assert kinds["dec"] == "violated"


def test_referee_item_b2_type_iv_counterexample_is_certified_violating():
    """B2's Lorentz-self-adjoint Type-IV tensor with vanishing quartic discriminant."""
    A = np.array(
        [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.3]]
    )
    T = ETA @ A
    assert np.allclose(T, T.T)
    for c in CONDITIONS:
        cert = certify(T, ETA, c)
        assert cert["kind"] == "violated", (c, cert)
        assert verify(cert, T, ETA)


def test_multiplier_is_rational_and_small_where_it_can_be():
    """A certificate a reader can retype is worth more than one they cannot."""
    T = np.diag([2.0, 0.5, 0.5, 0.5])
    s = find_multiplier(to_exact(T), to_exact(ETA), "wec", 0.0)
    assert s is not None and s.denominator <= 10**6
    assert is_psd_exact(
        [
            [a + s * b for a, b in zip(r1, r2, strict=True)]
            for r1, r2 in zip(
                condition_matrix(to_exact(T), to_exact(ETA), "wec"), to_exact(ETA), strict=True
            )
        ]
    )


def test_certificate_survives_a_curved_metric():
    """Nothing above assumes g is Minkowski; the form is certified in coordinates."""
    g = np.diag([-4.0, 0.25, 9.0, 1.0])
    T = np.diag([3.0, 0.5, 0.5, 0.5])
    for c in CONDITIONS:
        cert = certify(T, g, c)
        assert verify(cert, T, g), (c, cert)
        margin = float(certify_point(jnp.asarray(T), jnp.asarray(g))[c])
        floor = float(noise_floor(jnp.asarray(T), jnp.asarray(g), condition=c))
        if margin > floor:
            assert cert["kind"] == "satisfied", (c, margin, cert)
        elif margin < -floor:
            assert cert["kind"] == "violated", (c, margin, cert)


def test_exact_arithmetic_only_no_float_leaks():
    """Every number inside a certificate is an integer pair."""
    T = np.diag([1.0, -3.0, 0.0, 0.0])
    for c in CONDITIONS:
        cert = certify(T, ETA, c)
        blob = repr(cert)
        assert "." not in blob.replace("nec", "").replace("sec", ""), blob
        for key in ("sigma", "witness", "weights", "witness_pair"):
            if key not in cert:
                continue
            flat = cert[key].values() if isinstance(cert[key], dict) else cert[key]
            for item in flat:
                pairs = item if isinstance(item[0], list) else [item]
                for num, den in pairs:
                    assert isinstance(num, int) and isinstance(den, int)
                    assert Fraction(num, den) is not None


def test_every_canonical_form_certifies_and_reverifies():
    """Certify and re-verify the canonical Hawking-Ellis forms exactly.

    Moved here from a module __main__ block that never ran.
    """
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    cases = {
        "perfect fluid, all hold": np.diag([2.0, 0.5, 0.5, 0.5]),
        "DEC fails (rho < |p|)": np.diag([1.0, 2.0, 0.0, 0.0]),
        "NEC fails": np.diag([1.0, -3.0, 0.0, 0.0]),
        # Lorentz-self-adjoint, Type IV, vanishing quartic discriminant.
        "Type IV (D4=0)": eta
        @ np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.3],
            ]
        ),
        # The Type-II locus, Delta = 0 with j != 0.
        "Type II (locus)": np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.2],
            ]
        ),
        "vacuum": np.zeros((4, 4)),
    }
    for name, T in cases.items():
        assert np.allclose(T, T.T), name
        for c in _CONDITIONS:
            cert = certify(T, eta, c)
            assert verify(cert, T, eta), f"{name}/{c}: certificate did not verify"
