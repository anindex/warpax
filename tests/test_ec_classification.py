"""Tests for Hawking-Ellis classification and eigenvalue-based EC checks.

Validates against analytically constructed test tensors (NOT WarpFactory
comparison).  Tolerance tiers:
1e-12 for direct eigenvalue extraction, 1e-10 for accumulated chain.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions.classification import (
    classify_hawking_ellis,
    classify_mixed_tensor,
)
from warpax.energy_conditions.eigenvalue_checks import (
    check_all,
    check_dec,
    check_nec,
    check_sec,
    check_wec,
)
from warpax.energy_conditions.types import ClassificationResult

# Minkowski metric
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# ============================================================================
# Classification tests
# ============================================================================


class TestTypeIClassification:
    """Type I: one timelike eigenvector, three spacelike, all eigenvalues real."""

    def test_perfect_fluid_dust(self):
        """T^a_b = diag(-rho, p, p, p) for dust (rho=1, p=0)."""
        rho, p = 1.0, 0.0
        T_mixed = jnp.diag(jnp.array([-rho, p, p, p]))

        result = classify_hawking_ellis(T_mixed, ETA)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), rho, atol=1e-12)
        np.testing.assert_allclose(result.pressures, jnp.array([p, p, p]), atol=1e-12)

    def test_perfect_fluid_with_pressure(self):
        """T^a_b = diag(-rho, p, p, p) for matter with rho=2, p=0.5."""
        rho, p = 2.0, 0.5
        T_mixed = jnp.diag(jnp.array([-rho, p, p, p]))

        result = classify_hawking_ellis(T_mixed, ETA)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), rho, atol=1e-12)
        np.testing.assert_allclose(result.pressures, jnp.array([p, p, p]), atol=1e-12)

    def test_anisotropic_pressures(self):
        """T^a_b = diag(-rho, p1, p2, p3) with distinct pressures."""
        rho = 3.0
        p1, p2, p3 = 0.1, 0.5, 1.2
        T_mixed = jnp.diag(jnp.array([-rho, p1, p2, p3]))

        result = classify_hawking_ellis(T_mixed, ETA)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), rho, atol=1e-12)
        # Pressures are sorted
        np.testing.assert_allclose(
            result.pressures, jnp.array([p1, p2, p3]), atol=1e-12
        )

    def test_near_degenerate(self):
        """Near-degenerate Type I: pressures differ by ~1e-12."""
        rho = 1.0
        T_mixed = jnp.diag(jnp.array([-rho, 0.5, 0.5 + 1e-12, 0.5 - 1e-12]))

        result = classify_hawking_ellis(T_mixed, ETA)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), rho, atol=1e-10)

    def test_large_eigenvalue_type_i(self):
        """Large eigenvalues (~1e11) with numerical noise still classify as Type I."""
        T_mixed = jnp.diag(jnp.array([-1e11, 0.5e11, 0.3e11, 0.1e11]))
        result = classify_hawking_ellis(T_mixed, ETA)
        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), 1e11, rtol=1e-6)

    def test_eigenvalues_dtype_float64(self):
        """Eigenvalues should be float64 (not complex128 leaking through)."""
        T_mixed = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        result = classify_hawking_ellis(T_mixed, ETA)

        assert result.eigenvalues.dtype == jnp.float64
        assert result.rho.dtype == jnp.float64
        assert result.pressures.dtype == jnp.float64


class TestTypeIVClassification:
    """Type IV: complex eigenvalue pair."""

    def test_complex_eigenvalues(self):
        """T^a_b with a 2x2 rotation block producing complex conjugate eigenvalues."""
        # Build T^a_b with a rotation subblock in the (1,2) plane
        # Eigenvalues of [[0, -1], [1, 0]] are +/- i
        T_mixed = jnp.array([
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0, -1.0, 0.0],
            [ 0.0,  1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 0.5],
        ])

        result = classify_hawking_ellis(T_mixed, ETA)

        assert int(result.he_type) == 4
        # rho and pressures should be NaN for non-Type-I
        assert jnp.isnan(result.rho)
        assert jnp.all(jnp.isnan(result.pressures))


class TestTypeIIClassification:
    """Type II: one null eigenvector (non-diagonalizable), degenerate eigenvalue."""

    def test_null_eigenvector(self):
        """Construct T^a_b with a null eigenvector in Minkowski.

        Use a Jordan-block-like structure that has a null eigenvector.
        T^a_b = diag(0, 0, 0, 0) + epsilon * (null outer product) gives
        a degenerate tensor.  For a cleaner construction:

        Take T^a_b with a null eigenvector k = (1, 1, 0, 0):
        k is null: eta_{ab} k^a k^b = -1 + 1 = 0.
        Build T so that T^a_b k^b = lambda k^a for some lambda,
        but T is NOT diagonalizable (has a Jordan block).
        """
        # A 4x4 matrix with eigenvalues {0, 0, 1, 2} where the two zeros
        # share a Jordan block. The eigenvectors of the zero eigenvalue
        # include a null vector in Minkowski.
        # Simpler approach: build a matrix with null eigenvector explicitly.
        #
        # T^a_b = lambda * delta^a_b + mu * k^a l_b
        # where k = (1,1,0,0) is null, l_b = eta_{ba} l^a with l = (1,-1,0,0)
        # also null. Then T k = lambda k (eigenvector), and if T has non-trivial
        # Jordan structure we get Type II.
        #
        # Let's use a direct construction where (1,1,0,0) is a null eigenvector
        # with eigenvalue 0, and the other eigenvalues are distinct and real.
        k = jnp.array([1.0, 1.0, 0.0, 0.0])  # null in Minkowski
        # l_b = eta_{ab} k^a = (-1, 1, 0, 0)
        l = ETA @ k  # = [-1, 1, 0, 0]

        # T^a_b = diag(1, 1, 2, 3) + eps * k^a l_b  (Jordan perturbation)
        # This has eigenvalues close to {1, 1, 2, 3} but the eigenvector
        # structure is perturbed. The null vector k is an approximate eigenvector.
        #
        # Actually, for exact classification, construct T directly with known
        # eigenvectors including a null one.
        # T^a_b that has a null eigenvector:
        # Use T = 0 * |k><eta k| + 1 * |e2><e2| + 2 * |e3><e3| + 0 * ...
        # The challenge is getting exactly one null eigenvector.
        #
        # Simplest Type II construction: a matrix that has eigenvalue 0 with
        # geometric multiplicity 1 but algebraic multiplicity 2, plus its
        # eigenvalue-0 eigenvector being null.

        # Jordan block approach: define T in a basis where (1,1,0,0) is one
        # of the basis vectors.
        # Basis: e0=(1,1,0,0)/sqrt(2), e1=(1,-1,0,0)/sqrt(2), e2=(0,0,1,0), e3=(0,0,0,1)
        # In this basis, set T to have a Jordan block for the first eigenvalue.
        P = jnp.array([
            [1.0,  1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ]) / jnp.sqrt(2.0)
        P = P.at[2, 2].set(1.0)
        P = P.at[3, 3].set(1.0)

        # Jordan form: eigenvalue 0 with 2x2 block, plus 1 and 2
        J = jnp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ])

        T_mixed = P @ J @ jnp.linalg.inv(P)
        result = classify_hawking_ellis(T_mixed, ETA)

        # Should be Type II: has null eigenvector, eigenvalues {0, 0, 1, 2}
        # (not all degenerate, so not Type III)
        assert int(result.he_type) == 2
        assert jnp.isnan(result.rho)
        assert jnp.all(jnp.isnan(result.pressures))


class TestTypeIIIClassification:
    """Type III: all eigenvalues equal AND a null eigenvector.

    Type III is the rarest Hawking-Ellis type (no known classical source
    produces it).  Numerically constructing a true Type III tensor is
    inherently fragile because ``jnp.linalg.eig`` on a defective (Jordan
    block) matrix splits degenerate eigenvalues by O(1e-8) in float64 and
    returns perturbed eigenvectors whose causal character is near-null but
    not exactly null at the default tolerance.

    We test the classification logic path by using a tolerance large enough
    to absorb the numerical perturbation from the Jordan block construction.
    """

    def test_maximally_degenerate_null_with_relaxed_tol(self):
        """Jordan block with all-equal eigenvalue + null eigenvector = Type III.

        With ``tol=1e-6`` the classifier treats the O(1e-8) eigenvalue split
        from the Jordan block as degenerate and the O(1e-8) causal character
        as null, correctly identifying Type III.
        """
        lam = 1.0
        P = jnp.array([
            [1.0,  1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ]) / jnp.sqrt(2.0)
        P = P.at[2, 2].set(1.0)
        P = P.at[3, 3].set(1.0)

        # Jordan form: eigenvalue lam with 2x2 block, plus two more lam
        J = jnp.array([
            [lam, 1.0, 0.0, 0.0],
            [0.0, lam, 0.0, 0.0],
            [0.0, 0.0, lam, 0.0],
            [0.0, 0.0, 0.0, lam],
        ])

        T_mixed = P @ J @ jnp.linalg.inv(P)

        # Relaxed tolerance absorbs the O(1e-8) numerical perturbation
        # from eig's handling of the defective matrix.
        result = classify_hawking_ellis(T_mixed, ETA, tol=1e-6)

        assert int(result.he_type) == 3
        assert jnp.isnan(result.rho)
        assert jnp.all(jnp.isnan(result.pressures))

    def test_type_iii_vs_type_i_at_default_tol(self):
        """At the default tol=1e-10, the Jordan block is classified as Type I.

        This documents the expected behavior: numerical eigendecomposition
        cannot detect non-diagonalizability at machine precision, so the
        near-degenerate Jordan block appears as a legitimate Type I tensor
        with nearly-equal pressures.  This is physically correct the
        tensor IS Type I to the precision we can measure.
        """
        lam = 1.0
        P = jnp.array([
            [1.0,  1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ]) / jnp.sqrt(2.0)
        P = P.at[2, 2].set(1.0)
        P = P.at[3, 3].set(1.0)

        J = jnp.array([
            [lam, 1.0, 0.0, 0.0],
            [0.0, lam, 0.0, 0.0],
            [0.0, 0.0, lam, 0.0],
            [0.0, 0.0, 0.0, lam],
        ])

        T_mixed = P @ J @ jnp.linalg.inv(P)
        result = classify_hawking_ellis(T_mixed, ETA)  # default tol=1e-10

        # At default tolerance, eig's perturbation makes this look like Type I
        assert int(result.he_type) == 1


class TestJITAndVmap:
    """Verify JIT compilation and vmap batching."""

    def test_jit_compilation(self):
        """jax.jit(classify_hawking_ellis) runs without ConcretizationTypeError."""
        T_mixed = jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5]))
        classify_jit = jax.jit(classify_hawking_ellis)

        result = classify_jit(T_mixed, ETA)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), 1.0, atol=1e-12)

    def test_vmap_batch(self):
        """jax.vmap(classify_hawking_ellis) works on a batch of (N, 4, 4) tensors."""
        # Build a batch of 3 Type I tensors
        T_batch = jnp.stack([
            jnp.diag(jnp.array([-1.0, 0.5, 0.5, 0.5])),
            jnp.diag(jnp.array([-2.0, 0.1, 0.2, 0.3])),
            jnp.diag(jnp.array([-0.5, 1.0, 1.0, 1.0])),
        ])
        g_batch = jnp.broadcast_to(ETA, (3, 4, 4))

        classify_vmap = jax.vmap(classify_hawking_ellis)
        results = classify_vmap(T_batch, g_batch)

        # All should be Type I
        np.testing.assert_array_equal(np.array(results.he_type), [1, 1, 1])
        np.testing.assert_allclose(
            np.array(results.rho), [1.0, 2.0, 0.5], atol=1e-12
        )

    def test_classify_mixed_tensor(self):
        """classify_mixed_tensor raises index and then classifies."""
        # For a diagonal T_{ab} in Minkowski, T^a_b = g^{ac} T_{cb}
        # With g^{ab} = diag(-1, 1, 1, 1) and T_{ab} = diag(rho, p, p, p),
        # T^a_b = diag(-rho, p, p, p)
        rho, p = 2.0, 0.3
        T_ab = jnp.diag(jnp.array([rho, p, p, p]))
        g_inv = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

        result = classify_mixed_tensor(T_ab, ETA, g_inv)

        assert int(result.he_type) == 1
        np.testing.assert_allclose(float(result.rho), rho, atol=1e-12)


# ============================================================================
# Eigenvalue EC check tests
# ============================================================================


class TestECChecksDust:
    """Dust (rho > 0, p = 0): all conditions satisfied."""

    def test_dust_all_satisfied(self):
        """rho=1, p=[0,0,0] satisfies all four ECs."""
        rho = jnp.float64(1.0)
        pressures = jnp.array([0.0, 0.0, 0.0])

        nec, wec, sec, dec = check_all(rho, pressures)

        assert float(nec) >= 0.0  # min(1+0) = 1
        assert float(wec) >= 0.0  # min(1, 1+0) = 1
        assert float(sec) >= 0.0  # min(1+0, 1+0) = 1
        assert float(dec) >= 0.0  # min(1-0) = 1

    def test_dust_exact_margins(self):
        """Verify exact margin values for dust."""
        rho = jnp.float64(2.0)
        pressures = jnp.array([0.0, 0.0, 0.0])

        np.testing.assert_allclose(float(check_nec(rho, pressures)), 2.0, atol=1e-12)
        np.testing.assert_allclose(float(check_wec(rho, pressures)), 2.0, atol=1e-12)
        np.testing.assert_allclose(float(check_sec(rho, pressures)), 2.0, atol=1e-12)
        np.testing.assert_allclose(float(check_dec(rho, pressures)), 2.0, atol=1e-12)


class TestNegativeEnergyDensity:
    """rho < 0: WEC violated, NEC may be satisfied."""

    def test_negative_rho_wec_violation(self):
        """rho=-1, p=[0,0,0]: WEC margin = -1 (violated)."""
        rho = jnp.float64(-1.0)
        pressures = jnp.array([0.0, 0.0, 0.0])

        wec_margin = check_wec(rho, pressures)
        assert float(wec_margin) < 0.0
        np.testing.assert_allclose(float(wec_margin), -1.0, atol=1e-12)

    def test_negative_rho_nec_boundary(self):
        """rho=-1, p=[0,0,0]: NEC margin = -1 (violated)."""
        rho = jnp.float64(-1.0)
        pressures = jnp.array([0.0, 0.0, 0.0])

        nec_margin = check_nec(rho, pressures)
        np.testing.assert_allclose(float(nec_margin), -1.0, atol=1e-12)


class TestNECViolation:
    """NEC violation: rho + p_i < 0."""

    def test_nec_violated(self):
        """rho=1, p=[-2, 0, 0]: NEC margin = 1+(-2) = -1."""
        rho = jnp.float64(1.0)
        pressures = jnp.array([-2.0, 0.0, 0.0])

        nec_margin = check_nec(rho, pressures)
        np.testing.assert_allclose(float(nec_margin), -1.0, atol=1e-12)


class TestDECViolation:
    """DEC violation: rho < |p_i|."""

    def test_dec_violated(self):
        """rho=1, p=[2, 0, 0]: DEC margin = 1-2 = -1."""
        rho = jnp.float64(1.0)
        pressures = jnp.array([2.0, 0.0, 0.0])

        dec_margin = check_dec(rho, pressures)
        np.testing.assert_allclose(float(dec_margin), -1.0, atol=1e-12)


class TestSECViolation:
    """SEC violation: rho + sum(p_i) < 0."""

    def test_sec_violated(self):
        """rho=1, p=[-0.5, -0.5, -0.5]: SEC trace margin = 1+(-1.5) = -0.5."""
        rho = jnp.float64(1.0)
        pressures = jnp.array([-0.5, -0.5, -0.5])

        sec_margin = check_sec(rho, pressures)
        np.testing.assert_allclose(float(sec_margin), -0.5, atol=1e-12)


class TestVacuum:
    """Vacuum (all zero): boundary case, not violated per convention."""

    def test_vacuum_margins_zero(self):
        """rho=0, p=[0,0,0]: all margins = 0."""
        rho = jnp.float64(0.0)
        pressures = jnp.array([0.0, 0.0, 0.0])

        nec, wec, sec, dec = check_all(rho, pressures)

        np.testing.assert_allclose(float(nec), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(wec), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(sec), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(dec), 0.0, atol=1e-12)


class TestECVmapGrid:
    """Verify vectorized EC checks across a batch via vmap."""

    def test_vmap_over_batch(self):
        """vmap(check_all) works over (N,) rho and (N, 3) pressures."""
        batch_rho = jnp.array([1.0, -1.0, 0.0, 1.0])
        batch_p = jnp.array([
            [0.0, 0.0, 0.0],    # dust: all satisfied
            [0.0, 0.0, 0.0],    # negative rho: WEC violated
            [0.0, 0.0, 0.0],    # vacuum: boundary
            [-2.0, 0.0, 0.0],   # NEC violated
        ])

        vmap_all = jax.vmap(check_all)
        nec, wec, sec, dec = vmap_all(batch_rho, batch_p)

        # Point 0 (dust): all >= 0
        assert float(nec[0]) >= 0.0
        assert float(wec[0]) >= 0.0

        # Point 1 (negative rho): WEC < 0
        assert float(wec[1]) < 0.0

        # Point 2 (vacuum): all == 0
        np.testing.assert_allclose(float(nec[2]), 0.0, atol=1e-12)

        # Point 3 (NEC violation): NEC < 0
        assert float(nec[3]) < 0.0
        np.testing.assert_allclose(float(nec[3]), -1.0, atol=1e-12)

    def test_margin_dtype_float64(self):
        """EC margins should be float64."""
        rho = jnp.float64(1.0)
        pressures = jnp.array([0.5, 0.3, 0.2])

        nec, wec, sec, dec = check_all(rho, pressures)

        assert nec.dtype == jnp.float64
        assert wec.dtype == jnp.float64
        assert sec.dtype == jnp.float64
        assert dec.dtype == jnp.float64


# ============================================================================
# Scale-aware imaginary tolerance tests (root cause fix)
# ============================================================================


class TestScaleAwareImaginaryTolerance:
    """Verify scale-aware imaginary-part check prevents spurious Type IV."""

    def test_large_eigenvalues_tiny_relative_imag_is_type_i(self):
        """Moderate eigenvalues (~1e3) with |Im|/scale ~ 1e-11 -> Type I.

        This is the root cause scenario: eigenvalues have absolute imaginary
        parts ~1e-8, which exceeds the old absolute tolerance 1e-10, but
        relative to scale (~1e3) they are ~1e-11, well below tolerance.
        """
        # Build T_mixed with known real eigenvalues, then add tiny imaginary
        # perturbation via a near-antisymmetric off-diagonal term.
        # Direct approach: use a diagonal with large eigenvalues.
        T_mixed = jnp.diag(jnp.array([-1e3, 5e2, 3e2, 1e2]))

        # Add a tiny off-diagonal perturbation that produces ~1e-8 imaginary parts
        # when eigendecomposed. A 2x2 rotation block with angle epsilon
        # gives eigenvalues with imaginary part ~ epsilon.
        eps = 1e-8
        T_mixed = T_mixed.at[1, 2].set(eps)
        T_mixed = T_mixed.at[2, 1].set(-eps)

        result = classify_hawking_ellis(T_mixed, ETA)

        # With scale-aware tolerance: |Im| ~ 1e-8, scale ~ 1e3, ratio ~ 1e-11 < tol
        # -> should be Type I (not Type IV)
        assert int(result.he_type) == 1, (
            f"Expected Type I but got Type {int(result.he_type)}. "
            f"Scale-aware tolerance should classify this as real."
        )

    def test_genuine_type_iv_large_imaginary(self):
        """T^a_b with genuinely complex eigenvalues -> still Type IV."""
        T_mixed = jnp.array([
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0, -1.0, 0.0],
            [ 0.0,  1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 0.5],
        ])

        result = classify_hawking_ellis(T_mixed, ETA)

        # Eigenvalues include +/- i, which are large relative to scale ~ 1
        assert int(result.he_type) == 4, (
            f"Expected Type IV but got Type {int(result.he_type)}"
        )

    def test_eigenvalues_imag_field_present(self):
        """ClassificationResult has eigenvalues_imag field."""
        T_mixed = jnp.diag(jnp.array([-1.0, 0.5, 0.3, 0.1]))
        result = classify_hawking_ellis(T_mixed, ETA)

        assert hasattr(result, "eigenvalues_imag"), (
            "ClassificationResult missing eigenvalues_imag field"
        )
        assert result.eigenvalues_imag.shape == (4,)

    def test_eigenvalues_imag_near_zero_for_diagonal(self):
        """For a diagonal matrix, imaginary parts should be near zero."""
        T_mixed = jnp.diag(jnp.array([-2.0, 1.0, 0.5, 0.3]))
        result = classify_hawking_ellis(T_mixed, ETA)

        np.testing.assert_allclose(
            result.eigenvalues_imag, jnp.zeros(4), atol=1e-14
        )


# ============================================================================
# Type II null dust benchmark (validates non-Type-I optimizer pathway)
# ============================================================================


class TestTypeIINullDustBenchmark:
    """Null dust T_ab = Phi^2 k_a k_b is Type II by construction.

    For a null vector k^a with k_a k^a = 0, the stress-energy
    T_ab = Phi^2 k_a k_b has:
    - NEC: T_ab k^a k^b = Phi^2 (k_a k^a)^2 = 0 (saturated)
    - WEC depends on observer: T_ab u^a u^b = Phi^2 (k_a u^a)^2 >= 0
    """

    def test_null_dust_classification(self):
        """Null dust classifies as Type II (or I depending on numerics)."""
        # k = (1, 1, 0, 0) is null in Minkowski: eta_ab k^a k^b = -1 + 1 = 0
        k_up = jnp.array([1.0, 1.0, 0.0, 0.0])
        k_down = ETA @ k_up  # k_a = (-1, 1, 0, 0)

        Phi_sq = 1.0
        T_ab = Phi_sq * jnp.outer(k_down, k_down)

        # T^a_b = g^{ac} T_{cb}
        g_inv = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T_mixed = g_inv @ T_ab

        result = classify_hawking_ellis(T_mixed, ETA)

        # Null dust has eigenvalues {0, 0, 0, 0} with a null eigenvector
        # Numerically it may classify as Type I (near-vacuum) or Type II
        # depending on tolerance, but NOT Type IV
        assert int(result.he_type) != 4, (
            f"Null dust should not be Type IV, got Type {int(result.he_type)}"
        )

    def test_null_dust_nec_saturation(self):
        """NEC is saturated (margin = 0) for null dust along k."""
        k_up = jnp.array([1.0, 1.0, 0.0, 0.0])
        k_down = ETA @ k_up

        Phi_sq = 1.0
        T_ab = Phi_sq * jnp.outer(k_down, k_down)

        # T_ab k^a k^b = Phi^2 (k_a k^a)^2 = 0
        nec_val = float(jnp.einsum("a,ab,b->", k_up, T_ab, k_up))
        np.testing.assert_allclose(nec_val, 0.0, atol=1e-12)

    def test_null_dust_wec_nonnegative(self):
        """WEC is satisfied for null dust: T_ab u^a u^b >= 0 for any timelike u."""
        k_up = jnp.array([1.0, 1.0, 0.0, 0.0])
        k_down = ETA @ k_up

        Phi_sq = 2.0
        T_ab = Phi_sq * jnp.outer(k_down, k_down)

        # Check for several observers
        for u in [
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # Eulerian
            jnp.array([jnp.cosh(1.0), jnp.sinh(1.0), 0.0, 0.0]),  # boosted x
            jnp.array([jnp.cosh(2.0), 0.0, jnp.sinh(2.0), 0.0]),  # boosted y
        ]:
            wec_val = float(jnp.einsum("a,ab,b->", u, T_ab, u))
            assert wec_val >= -1e-12, (
                f"WEC violated for null dust with u={u}: {wec_val}"
            )
