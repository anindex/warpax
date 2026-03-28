"""Tests for the mpmath 50-digit Hawking-Ellis classification verifier.

The float64 :func:`classify_hawking_ellis` uses a two-tier tolerance
(``tol * scale`` absolute OR ``imag_rtol * unclamped_scale`` relative).
At certain scales this coarse filter can misclassify a genuine Type-IV
spectrum as Type I. The mpmath path evaluates eigenvalues at 50-digit
precision and applies the same tolerances to the true imaginary parts.

Synthetic tests below pin:

1. a known Type-IV block-diagonal ``T^a_b`` that float64 misclassifies
   as Type I, verifying that the mpmath classifier correctly returns
   Type IV,
2. a perfect-fluid Type I point, verifying mpmath agrees with float64,
3. the :func:`verify_classification_at_points` audit function on a
   mixed batch, checking flip-rate accounting.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.classification_mpmath import (
    classify_hawking_ellis_mpmath,
    eigenvalues_mpmath,
    verify_classification_at_points,
)


MINKOWSKI = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _type_iv_block_diag(imag: float = 2.0e-5) -> jnp.ndarray:
    """Return a T^a_b with a known complex eigenvalue pair ``1 ± i*imag``.

    The upper 2x2 block is the companion matrix ``[[1, -imag], [imag, 1]]``;
    the lower 2x2 block is ``diag(0.5, -0.3)``. The spectrum is therefore
    ``{1 + i*imag, 1 - i*imag, 0.5, -0.3}`` -- unambiguously Type IV.
    """
    return jnp.array(
        [
            [1.0, -imag, 0.0, 0.0],
            [imag, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, -0.3],
        ]
    )


def _perfect_fluid(rho: float = 1.0, p: float = 0.3) -> jnp.ndarray:
    """Return a perfect-fluid T^a_b at rest: diag(-rho, p, p, p)."""
    return jnp.diag(jnp.array([-rho, p, p, p]))


class TestMpmathEigenvalues:
    """Pin the high-precision eigenvalue path against hand-verified spectra."""

    def test_block_diag_spectrum(self) -> None:
        T = np.asarray(_type_iv_block_diag(imag=2.0e-5))
        evs = eigenvalues_mpmath(T, precision=50)

        imag_parts = sorted(float(abs(ev.imag)) for ev in evs)
        assert imag_parts[0] < 1.0e-45
        assert imag_parts[1] < 1.0e-45
        assert abs(imag_parts[2] - 2.0e-5) < 1.0e-20
        assert abs(imag_parts[3] - 2.0e-5) < 1.0e-20

    def test_perfect_fluid_spectrum_is_real(self) -> None:
        T = np.asarray(_perfect_fluid(rho=1.0, p=0.3))
        evs = eigenvalues_mpmath(T, precision=50)
        for ev in evs:
            assert abs(float(ev.imag)) < 1.0e-45


class TestMpmathClassifier:
    """Pin the 50-digit classifier verdicts and contrast with float64."""

    def test_type_iv_that_float64_misclassifies(self) -> None:
        T = _type_iv_block_diag(imag=2.0e-5)

        float64_result = classify_hawking_ellis(T, MINKOWSKI)
        assert int(float64_result.he_type) == 1

        mp_result = classify_hawking_ellis_mpmath(
            np.asarray(T), np.asarray(MINKOWSKI), precision=50
        )
        assert mp_result["he_type"] == 4
        assert mp_result["max_imag_abs"] > 1.0e-6

    def test_perfect_fluid_mpmath_agrees_with_float64(self) -> None:
        T = _perfect_fluid(rho=1.0, p=0.3)

        float64_result = classify_hawking_ellis(T, MINKOWSKI)
        mp_result = classify_hawking_ellis_mpmath(
            np.asarray(T), np.asarray(MINKOWSKI), precision=50
        )

        assert int(float64_result.he_type) == 1
        assert mp_result["he_type"] == 1

    def test_type_i_fluid_non_minkowski_g(self) -> None:
        """mpmath path: Type-I fluid under non-Minkowski g -> Type I.

        Mirrors the float64 contract test
        (TestCausalBasisFix::test_type_i_non_minkowski_g) at 50-digit
        precision. Pre-fix the absolute ``tol`` test in
        :func:`_causal_counts` gives n_timelike=0 at WarpShell-scale
        ``|g_{ij}|/|g_{00}| ~ 10``, forcing Type-II fallthrough.
        Post-fix the relative-sign test rescues the timelike eigenvector
        and the verdict is Type I (matching the physics).
        """
        # Same fixture as the float64 contract test
        T_mixed = np.diag(np.array([-1.0, 0.3, 0.3, 0.3]))
        g_ab = np.array([
            [-0.12, 0.05, 0.05, 0.0],
            [0.05,   1.5, 0.3,  0.2],
            [0.05,   0.3, 1.5,  0.1],
            [0.0,    0.2, 0.1,  1.5],
        ])

        mp_result = classify_hawking_ellis_mpmath(T_mixed, g_ab, precision=50)
        assert mp_result["he_type"] == 1, (
            f"Expected Type I -- Type-I fluid spectrum under non-Minkowski g; "
            f"got he_type={mp_result['he_type']} "
            f"(eigenvalues_real={mp_result['eigenvalues_real']}, "
            f"eigenvalues_imag={mp_result['eigenvalues_imag']})"
        )


class TestVerifyClassificationAtPoints:
    """Pin the audit function's flip-rate accounting."""

    def test_batch_flip_rate_is_correct(self) -> None:
        T_flip = np.asarray(_type_iv_block_diag(imag=2.0e-5))
        T_fluid = np.asarray(_perfect_fluid(rho=1.0, p=0.3))

        # Two points: one Type-I-in-float64 (actually Type IV), one genuine Type I.
        T_batch = np.stack([T_flip, T_fluid], axis=0)
        g_batch = np.stack([np.asarray(MINKOWSKI)] * 2, axis=0)
        float64_types = np.array([1, 1], dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_types, precision=50
        )

        assert report["n_points"] == 2
        assert report["n_flips"] == 1
        assert report["flip_indices"].tolist() == [0]
        # Flipped point's mpmath verdict is Type IV.
        assert report["mpmath_he_types"][0] == 4
        assert report["mpmath_he_types"][1] == 1
        assert report["flip_rate"] == 0.5

    def test_empty_batch_returns_zero_flip_rate(self) -> None:
        T_batch = np.zeros((0, 4, 4))
        g_batch = np.zeros((0, 4, 4))
        float64_types = np.zeros((0,), dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_types, precision=50
        )

        assert report["n_points"] == 0
        assert report["n_flips"] == 0
        assert report["flip_rate"] == 0.0

    def test_batch_exposes_cond_v_per_point(self) -> None:
        """batch verify exposes cond_V_per_point + uncertain_mask."""
        T_batch = np.stack([
            np.asarray(_perfect_fluid(rho=1.0, p=0.3)),
            np.asarray(_perfect_fluid(rho=2.0, p=0.5)),
        ])
        g_batch = np.stack([np.asarray(MINKOWSKI), np.asarray(MINKOWSKI)])
        float64_he = np.array([1, 1], dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_he, precision=50
        )

        assert "cond_V_per_point" in report
        assert "uncertain_mask" in report
        assert report["cond_V_per_point"].shape == (2,)
        assert report["uncertain_mask"].shape == (2,)
        assert report["uncertain_mask"].dtype == np.bool_
        # Both clean fluids should be certain
        assert report["uncertain_mask"].any() == False


# ---------------------------------------------------------------------------
# Bauer-Fike eigenvector-matrix condition number diagnostic
# ---------------------------------------------------------------------------


class TestCondV:
    """Bauer-Fike sensitivity diagnostic.

    - test_well_conditioned_flags_certain: clean perfect-fluid -> cond(V) small,
      uncertain=False.
    - test_jordan_defective_flags_uncertain: near-Jordan synthesised input ->
      cond(V) ~ inf (or 10^10+), uncertain=True.

    Threshold: cond(V) > 10**(precision/2) per Demmel 1997 Thm 4.4 (Bauer-Fike).
    """

    def test_well_conditioned_flags_certain(self) -> None:
        """Perfect fluid under Minkowski -> uncertain=False, cond_V well-bounded."""
        T = np.asarray(_perfect_fluid(rho=1.0, p=0.3))
        g = np.asarray(MINKOWSKI)

        result = classify_hawking_ellis_mpmath(T, g, precision=50)

        assert "cond_V" in result, "missing 'cond_V' key in return dict"
        assert "uncertain" in result, "missing 'uncertain' key in return dict"

        assert result["uncertain"] is False, (
            f"Expected uncertain=False for clean perfect fluid; "
            f"got uncertain={result['uncertain']}, cond_V={result['cond_V']}"
        )
        # Threshold for precision=50 is 10^25; clean fluid should be well below.
        assert result["cond_V"] < 10 ** 10, (
            f"Expected cond_V < 1e10 for clean perfect fluid; "
            f"got cond_V={result['cond_V']}"
        )
        # Existing keys still present (additivity contract)
        for k in ("he_type", "all_real", "near_vacuum", "n_timelike",
                  "n_null", "n_unique", "max_imag_abs", "max_real_abs",
                  "eigenvalues_real", "eigenvalues_imag", "precision"):
            assert k in result, f"existing key {k!r} dropped from return dict"

    def test_jordan_defective_flags_uncertain(self) -> None:
        """Exact Jordan synthesised T -> uncertain=True (cond_V exceeds 10^25).

        The top-left 2x2 is an exact Jordan block: eigenvalue 1.0 with
        algebraic multiplicity 2 and a SINGLE eigenvector (the (1,0) entry
        is exactly 0.0). ``mpmath.eig`` on a defective matrix returns a
        near-singular eigenvector column at the defective eigenvalue, so
        ``sigma_min(V) ~ 10**(-precision)`` and ``cond_V ~ 10**precision``
        - comfortably above the half-digit threshold ``10**(precision/2)``.

        Note: the plan author's original fixture used a ``(1,0)`` perturbation
        of ``1e-20`` expecting ``cond_V > 10**25``; empirically that yields
        ``cond_V ~ 10**10`` (sqrt of the perturbation), BELOW the half-digit
        threshold at ``precision=50``. An exact Jordan block is the
        physically-meaningful fixture for the Bauer-Fike diagnostic.
        """
        # EXACT Jordan block in top-left 2x2: (1,0) entry is 0.0.
        T = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, -0.3],
        ])
        g = np.asarray(MINKOWSKI)

        result = classify_hawking_ellis_mpmath(T, g, precision=50)

        assert "cond_V" in result
        assert "uncertain" in result

        # cond(V) should be very large (ill-conditioned eigenvector matrix)
        assert result["uncertain"] is True, (
            f"Expected uncertain=True for exact-Jordan T; "
            f"got uncertain={result['uncertain']}, cond_V={result['cond_V']}"
        )
        # cond_V should exceed the half-digit threshold 10^(50/2) = 10^25
        assert result["cond_V"] > 10 ** 25 or result["cond_V"] == float("inf"), (
            f"Expected cond_V > 10^25 (or inf); got cond_V={result['cond_V']}"
        )
