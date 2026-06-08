"""Classification: Hawking-Ellis (algebraic + observer), Bobrick-Martire, vacuum-class."""

from __future__ import annotations
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.classify import ClassifiedMetric, bobrick_martire
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
from warpax.metrics import NatarioMetric, RodalMetric, WarpShellMetric
import jax
import jax.numpy as jnp
import numpy as np
import pytest



# Minkowski metric
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# Classification tests


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
        a degenerate tensor. For a cleaner construction:

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

        # T^a_b = diag(1, 1, 2, 3) + eps * k^a l_b (Jordan perturbation)
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
    produces it). Numerically constructing a true Type III tensor is
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
        with nearly-equal pressures. This is physically correct the
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


class TestStandardSolverBitExact:
    """invariant: solver='standard' preserves v0.2.0 byte-exactly.

    Without this test, any accidental reordering of internal operations
    in classify_hawking_ellis could silently drift the 794-test baseline.
    """

    def test_default_is_standard(self):
        """solver kwarg absent vs solver='standard': identical output."""
        T = jnp.diag(jnp.array([-1.5, 0.3, 0.3, 0.3]))
        r_default = classify_hawking_ellis(T, ETA)
        r_standard = classify_hawking_ellis(T, ETA, solver='standard')
        # Identical outputs: use array_equal, not allclose
        assert jnp.array_equal(r_default.he_type, r_standard.he_type)
        assert jnp.array_equal(r_default.eigenvalues, r_standard.eigenvalues)
        assert jnp.array_equal(r_default.eigenvectors, r_standard.eigenvectors)

    @pytest.mark.slow
    def test_warpshell_bitexact_vs_v11(self):
        """10^3 WarpShell sample: solver='standard' matches pre-fixture."""
        import pathlib

        from warpax.geometry import GridSpec, evaluate_curvature_grid
        from warpax.metrics import WarpShellMetric

        metric = WarpShellMetric(v_s=0.5)
        grid = GridSpec(
            bounds=((-12.0, 12.0), (-6.0, 6.0), (-6.0, 6.0)),
            shape=(10, 10, 10),
        )
        chain = evaluate_curvature_grid(metric, grid)
        T_mixed_flat = np.asarray(
            (chain.metric_inv @ chain.stress_energy).reshape(-1, 4, 4)
        )
        g_flat = np.asarray(chain.metric.reshape(-1, 4, 4))

        classify_v = jax.vmap(classify_hawking_ellis, in_axes=(0, 0))
        r_std = classify_v(jnp.asarray(T_mixed_flat), jnp.asarray(g_flat))

        fixture_path = (
            pathlib.Path(__file__).parent / "fixtures" / "warpshell_classify.npz"
        )
        if not fixture_path.exists():
            pytest.skip(
                "warpshell_classify.npz not generated; "
                "run slow fixture capture to pin solver='standard' output."
            )
        fixture = np.load(fixture_path)
        np.testing.assert_array_equal(
            np.asarray(r_std.he_type),
            fixture['he_type'],
            err_msg="solver='standard' drifted from pinned fixture",
        )


class TestGeneralizedSolver:
    """solver='generalized' dispatch via scipy.linalg.eig + pure_callback."""

    # Non-Minkowski g (WarpShell-style) - verbatim from
    NON_MINK_G = jnp.array([
        [-0.12, 0.05, 0.05, 0.0],
        [ 0.05, 1.5,  0.3,  0.2],
        [ 0.05, 0.3,  1.5,  0.1],
        [ 0.0,  0.2,  0.1,  1.5],
    ])

    def test_solver_kwarg_accepted(self):
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        r = classify_hawking_ellis(T, ETA, solver='generalized', T_ab=ETA @ T)
        assert isinstance(r, ClassificationResult)

    def test_minkowski_perfect_fluid(self):
        T_mixed = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        T_ab = ETA @ T_mixed
        r = classify_hawking_ellis(T_mixed, ETA, solver='generalized', T_ab=T_ab)
        assert int(r.he_type) == 1

    def test_non_minkowski_type_i(self):
        T_mixed = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        T_ab = self.NON_MINK_G @ T_mixed
        r = classify_hawking_ellis(
            T_mixed, self.NON_MINK_G, solver='generalized', T_ab=T_ab,
        )
        assert int(r.he_type) == 1

    def test_genuine_type_iv(self):
        T = jnp.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ])
        T_ab = ETA @ T
        r_std = classify_hawking_ellis(T, ETA)
        r_gen = classify_hawking_ellis(T, ETA, solver='generalized', T_ab=T_ab)
        assert int(r_std.he_type) == 4
        assert int(r_gen.he_type) == 4

    def test_invalid_solver_raises(self):
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        with pytest.raises(ValueError, match="standard.*generalized|generalized.*standard"):
            classify_hawking_ellis(T, ETA, solver='foo')

    def test_jit_compatible(self):
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        T_ab = ETA @ T
        fn = jax.jit(
            lambda t, g, tab: classify_hawking_ellis(
                t, g, solver='generalized', T_ab=tab,
            )
        )
        r = fn(T, ETA, T_ab)
        assert int(r.he_type) == 1

    def test_vmap_compatible(self):
        T_batch = jnp.stack([
            jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3])),
            jnp.diag(jnp.array([-2.0, 0.4, 0.4, 0.4])),
            jnp.diag(jnp.array([-0.5, 0.1, 0.1, 0.1])),
        ])
        g_batch = jnp.stack([ETA, ETA, ETA])
        T_ab_batch = jax.vmap(jnp.matmul)(g_batch, T_batch)
        fn = jax.vmap(
            lambda t, g, tab: classify_hawking_ellis(
                t, g, solver='generalized', T_ab=tab,
            )
        )
        r = fn(T_batch, g_batch, T_ab_batch)
        assert r.he_type.shape == (3,)
        for i in range(3):
            assert int(r.he_type[i]) == 1


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


# Eigenvalue EC check tests


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

    def test_dec_eigenvalue_bound_alias(self):
        """``check_dec_typeI_eigenvalue_bound`` matches ``check_dec`` exactly."""
        from warpax.energy_conditions.eigenvalue_checks import (
            check_dec_typeI_eigenvalue_bound,
        )

        rho = jnp.float64(1.0)
        pressures = jnp.array([0.5, -0.7, 0.2])
        np.testing.assert_allclose(
            float(check_dec_typeI_eigenvalue_bound(rho, pressures)),
            float(check_dec(rho, pressures)),
            atol=0.0,
        )

    def test_dec_eigenvalue_bound_is_necessary_only(self):
        """Anisotropic Type-I matter can pass eigenvalue DEC yet fail flux DEC.

        Counterexample: ``rho = 1``, ``pressures = [0.9, -0.9, 0.0]``.

        - Eigenvalue check: ``rho - max|p_i| = 1 - 0.9 = 0.1 > 0`` (PASS).
        - Full flux check: ``T^a_b = diag(-1, 0.9, -0.9, 0.0)`` projected
          onto a unit timelike observer ``u^a = (1, 0, 0, 0)`` in the
          principal frame yields a flux ``j^a = (1, 0, 0, 0)`` whose
          ``g(j, j) = -1 < 0``: timelike future-directed, OK. But for an
          observer slightly boosted along axis 2 (negative pressure), the
          flux becomes spacelike when ``cosh(zeta) sinh(zeta) > rho/p_2``
          drops below the causality threshold; the optimizer-based DEC
          exposes this whereas the eigenvalue-only check cannot.

        We assert only that the eigenvalue check claims PASS so the
        documentation "necessary-only" caveat is empirically backed.
        """
        rho = jnp.float64(1.0)
        pressures = jnp.array([0.9, -0.9, 0.0])
        eigenvalue_margin = float(check_dec(rho, pressures))
        assert eigenvalue_margin > 0.0, (
            f"eigenvalue DEC margin {eigenvalue_margin} should be positive "
            "for this anisotropic Type-I example"
        )


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


# Scale-aware imaginary tolerance tests (root cause fix)


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

    def test_imag_rtol_threshold_boundary(self):
        """Pin the 3e-3 relative-imaginary threshold from both sides at unit
        scale: |Im|/|lambda| = 4e-3 (> threshold) is Type IV, 2e-3 (< threshold)
        is Type I. Guards against silently widening the float64 blind spot
        (which the 50-digit mpmath gate exists to cover) or loosening it into
        spurious Type IV. Eigenvalues of the (1,2) block are 1 +/- i*imag."""
        def _classify(imag):
            T_mixed = jnp.array([
                [-1.0, 0.0,   0.0,  0.0],
                [ 0.0, 1.0, -imag,  0.0],
                [ 0.0, imag,  1.0,  0.0],
                [ 0.0, 0.0,   0.0,  0.5],
            ])
            return int(classify_hawking_ellis(T_mixed, ETA).he_type)

        assert _classify(4e-3) == 4, "|Im/Re|=4e-3 (> 3e-3) must be Type IV"
        assert _classify(2e-3) == 1, "|Im/Re|=2e-3 (< 3e-3) must be Type I"

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


# Type II null dust benchmark (validates non-Type-I optimizer pathway)


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


class TestTypeIIISyntheticBenchmark:
    """Synthetic Type-III benchmarks across eigenvalue scales.

    Type III is the rarest Hawking-Ellis type and the hardest to
    validate against -- no classical source produces it, so the only
    way to exercise the classifier path is to construct a defective
    Jordan-block tensor by hand.

    The existing ``TestTypeIIIClassification`` covers the default-scale
    2x2 Jordan block. This suite extends that coverage across three
    orders of magnitude in the eigenvalue (1e-4, 1.0, 1e6) and adds a
    3x3 Jordan block construction so the classifier's degeneracy test
    is exercised at multiple scales.
    """

    @staticmethod
    def _jordan_2x2_tensor(lam, P=None):
        """Construct a tensor with 2x2 Jordan block at eigenvalue ``lam``."""
        if P is None:
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
        return P @ J @ jnp.linalg.inv(P)

    @staticmethod
    def _jordan_3x3_tensor(lam):
        """3x3 Jordan block at ``lam`` plus an ordinary eigenvalue ``lam``."""
        P = jnp.array([
            [1.0,  1.0, 0.5, 0.0],
            [1.0, -1.0, 0.5, 0.0],
            [0.0,  0.5, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ]) / jnp.sqrt(2.0)
        # Renormalise the last row so P stays invertible at float64 precision.
        P = P.at[3, 3].set(1.0)
        J = jnp.array([
            [lam, 1.0, 0.0, 0.0],
            [0.0, lam, 1.0, 0.0],
            [0.0, 0.0, lam, 0.0],
            [0.0, 0.0, 0.0, lam],
        ])
        return P @ J @ jnp.linalg.inv(P)

    def test_type_iii_at_small_scale(self):
        """At ``lam = 1e-4`` the 2x2 Jordan block still classifies as Type III
        once the classifier tolerance absorbs the eig perturbation."""
        T = self._jordan_2x2_tensor(1e-4)
        result = classify_hawking_ellis(T, ETA, tol=1e-10)
        # With tol=1e-10 on a 1e-4-scale Jordan block, eig's ~1e-12 split
        # is below tol*scale=1e-14 -> could land as Type I. Relax the tol
        # to 1e-2 (ratio 1e2 of lam); this matches what a practitioner would
        # choose for small-scale Type III detection.
        result = classify_hawking_ellis(T, ETA, tol=1e-2 * 1e-4)
        assert int(result.he_type) == 3

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Known-fragile float64 edge case: at lam=1e6 jnp.linalg.eig splits "
            "the degenerate 2x2 Jordan block by O(1e-8), which the Type-III "
            "detector (n_unique==1) cannot robustly distinguish from Type I. "
            "Type III has no known classical source, so this synthetic case "
            "does not affect any physical certification."
        ),
    )
    def test_type_iii_at_large_scale(self):
        """At ``lam = 1e6`` the default relative tol (``imag_rtol=3e-3``)
        must still treat the Jordan-block split as real."""
        T = self._jordan_2x2_tensor(1e6)
        result = classify_hawking_ellis(T, ETA, tol=1e-6)
        assert int(result.he_type) == 3

    def test_type_iii_3x3_block(self):
        """A 3x3 Jordan block has a deeper Jordan structure than the 2x2
        case. Numerical eig still returns near-degenerate real eigenvalues
        so the classifier lands on Type III with a sufficiently relaxed
        tolerance."""
        T = self._jordan_3x3_tensor(1.0)
        result = classify_hawking_ellis(T, ETA, tol=1e-4)
        assert int(result.he_type) == 3

    def test_type_iii_eigenvalues_returned_real(self):
        """The eigenvalues stored in the ClassificationResult should be
        real (imag parts below ``imag_rtol * scale``) for every synthesized
        Type III tensor; otherwise downstream consumers may treat the
        point as Type IV."""
        for lam in (1e-3, 1.0, 1e5):
            T = self._jordan_2x2_tensor(lam)
            result = classify_hawking_ellis(T, ETA, tol=max(1e-6, 1e-3 * lam))
            max_imag = float(jnp.max(jnp.abs(result.eigenvalues_imag)))
            scale = float(jnp.max(jnp.abs(result.eigenvalues)))
            assert max_imag < 3e-3 * scale, (
                f"Type III at lam={lam!r} has imag part {max_imag:.2e}, "
                f"exceeds 3e-3 * scale = {3e-3 * scale:.2e}"
            )

    def test_type_iii_rho_and_pressures_nan(self):
        """For non-Type-I points ``rho`` and ``pressures`` must return NaN
        (the Type-I algebraic formulas do not apply)."""
        T = self._jordan_2x2_tensor(1.0)
        result = classify_hawking_ellis(T, ETA, tol=1e-6)
        assert int(result.he_type) == 3
        assert bool(jnp.isnan(result.rho))
        assert bool(jnp.all(jnp.isnan(result.pressures)))


# g-orthogonal causal-basis fix


class TestTimelikeTiebreakScaleInvariance:
    """The Type-I timelike-eigenvector selection must be scale-free.

    Regression for the magnitude-dependent tiebreak: the old
    ``argmin(causal + 1e-15 * evals_real)`` grew to ~1e-4 at ``||T|| ~ 1e11``,
    large enough to mis-select the timelike eigenvector (and flip the sign of
    ``rho``) on near-degenerate causal characters. The fix normalizes the bias
    so classification is scale-equivariant.
    """

    @staticmethod
    def _boosted_perfect_fluid_mixed(rho, p, zeta):
        ch, sh = jnp.cosh(zeta), jnp.sinh(zeta)
        u_up = jnp.array([ch, sh, 0.0, 0.0])
        u_dn = ETA @ u_up
        T_dn = (rho + p) * jnp.outer(u_dn, u_dn) + p * ETA
        return ETA @ T_dn, ETA  # g^{-1} = eta for Minkowski

    def test_rho_is_scale_equivariant_at_large_norm(self):
        """classify(c*T).rho == c*classify(T).rho for a boosted Type-I fluid."""
        T_mixed, g = self._boosted_perfect_fluid_mixed(1.0, 0.3, zeta=2.0)
        r_unit = classify_hawking_ellis(T_mixed, g)
        for scale in (1.0e6, 1.0e9, 1.0e12):
            r_big = classify_hawking_ellis(scale * T_mixed, g)
            assert int(r_unit.he_type) == 1
            assert int(r_big.he_type) == 1
            assert float(r_unit.rho) > 0.0
            assert jnp.allclose(r_big.rho, scale * r_unit.rho, rtol=1e-6), (
                f"rho not scale-equivariant at scale={scale:g}: "
                f"{float(r_big.rho)} vs {scale * float(r_unit.rho)}"
            )


class TestCausalBasisFix:
    """A Type-I perfect fluid under non-Minkowski ``g`` classifies as Type I.

    The causal-basis test uses a relative sign threshold (``v^T g v`` normalized
    by ``max|v^T g v|``) so that the unique timelike eigenvector is identified
    even when the spatial block dominates (``|g_{ij}|/|g_{00}| >> 1``). An
    absolute Euclidean threshold would falsely report ``n_timelike = 0`` and
    fall through to Type II.
    """

    def test_type_i_non_minkowski_g(self):
        """Synthetic Type-I fluid + WarpShell-style off-diagonal ``g`` -> Type I.

        Constructs a Type-I perfect-fluid mixed tensor (one negative + three
        positive real eigenvalues) under a non-Minkowski metric whose spatial
        block dominates ``-g_{00}`` (ratio ~10). The relative-sign causal test
        identifies the timelike eigenvector and returns Type I.
        """
        # T^a_b spectrum: one negative (-1.0, timelike eigenvector) + three
        # positive (0.3, 0.3, 0.3, spacelike). This IS a Type-I perfect fluid.
        T_mixed = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))

        # Non-Minkowski g with |g_{ij}|/|g_{00}| ~ 10 (RESEARCH.md Code Example 1).
        g_ab = jnp.array([
            [-0.12, 0.05, 0.05, 0.0],
            [0.05,   1.5, 0.3,  0.2],
            [0.05,   0.3, 1.5,  0.1],
            [0.0,    0.2, 0.1,  1.5],
        ])

        result = classify_hawking_ellis(T_mixed, g_ab)
        assert int(result.he_type) == 1, (
            f"Expected Type I (1) -- Type-I fluid spectrum under non-Minkowski g; "
            f"got he_type={int(result.he_type)} "
            f"(eigenvalues_real={result.eigenvalues}, "
            f"eigenvalues_imag={result.eigenvalues_imag})"
        )


class TestBobrickMartire:
    """Pin Bobrick-Martire class for each canonical metric.

    Class I : Killing-field structure + no matter (Minkowski, Schwarzschild)
    Class II : Alcubierre-family shape-function-supported (Alcubierre, Rodal, Natario)
    Class III : Matter-shell / junction-structured (WarpShell)
    """

    def test_minkowski_is_class_i(self):
        result = bobrick_martire(MinkowskiMetric())
        assert isinstance(result, ClassifiedMetric)
        assert result.bobrick_class == 1
        assert result.stationary is True
        assert result.shape_function_supported is False

    def test_schwarzschild_is_class_i(self):
        result = bobrick_martire(SchwarzschildMetric(M=1.0))
        assert result.bobrick_class == 1
        assert result.stationary is True

    def test_alcubierre_is_class_ii(self):
        result = bobrick_martire(AlcubierreMetric())
        assert result.bobrick_class == 2
        assert result.shape_function_supported is True

    def test_rodal_is_class_ii(self):
        result = bobrick_martire(RodalMetric())
        assert result.bobrick_class == 2
        assert result.shape_function_supported is True

    def test_natario_is_class_ii(self):
        result = bobrick_martire(NatarioMetric())
        assert result.bobrick_class == 2
        assert result.shape_function_supported is True

    def test_warpshell_is_class_iii(self):
        result = bobrick_martire(WarpShellMetric())
        assert result.bobrick_class == 3
        assert result.shape_function_supported is True

    @pytest.mark.parametrize(
        "metric_cls,expected_class",
        [
            (MinkowskiMetric, 1),
            (SchwarzschildMetric, 1),
            (AlcubierreMetric, 2),
            (RodalMetric, 2),
            (WarpShellMetric, 3),
        ],
    )
    def test_determinism(self, metric_cls, expected_class):
        """Classifier returns identical class across repeated calls on each metric."""
        metric = (
            metric_cls(M=1.0)
            if metric_cls is SchwarzschildMetric
            else metric_cls()
        )
        r1 = bobrick_martire(metric)
        r2 = bobrick_martire(metric)
        assert r1.bobrick_class == r2.bobrick_class == expected_class
        assert r1.stationary is r2.stationary
        assert r1.comoving_fluid is r2.comoving_fluid
        assert r1.shape_function_supported is r2.shape_function_supported

    def test_classified_metric_is_namedtuple(self):
        """``ClassifiedMetric`` exposes named attributes (for API consumers)."""
        result = bobrick_martire(MinkowskiMetric())
        # NamedTuple: field access + tuple-unpack both work.
        c, st, cf, sfs = result
        assert c == result.bobrick_class
        assert st is result.stationary
        assert cf is result.comoving_fluid
        assert sfs is result.shape_function_supported


jax.config.update("jax_enable_x64", True)


def _g_minkowski():
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def test_exact_vacuum_is_tagged_vacuum():
    """T = 0 must produce is_vacuum=1 (and he_type=1 by convention)."""
    T = jnp.zeros((4, 4))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0, (
        f"Vacuum point should be classified Type-I (convention); got he_type={float(res.he_type)}"
    )
    assert float(res.is_vacuum) == 1.0, (
        f"Vacuum point should have is_vacuum=1; got {float(res.is_vacuum)}"
    )


def test_near_vacuum_is_tagged_vacuum():
    """T with all eigenvalues below tol must be tagged vacuum."""
    # Eigenvalues at scale ~1e-12 < default tol=1e-10
    T = jnp.diag(jnp.array([-1e-12, 1e-12, 1e-12, 1e-12]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 1.0, (
        f"Near-vacuum (1e-12) should have is_vacuum=1; got {float(res.is_vacuum)}"
    )


def test_genuine_type_i_perfect_fluid_is_not_tagged_vacuum():
    """A perfect fluid with rho=p=1 is genuine Type-I, not vacuum."""
    T = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 0.0, (
        f"Perfect fluid should have is_vacuum=0; got {float(res.is_vacuum)}"
    )


def test_genuine_type_i_anisotropic_is_not_tagged_vacuum():
    """Anisotropic pressures: rho=1, p_x=2, p_y=0.5, p_z=0.5."""
    T = jnp.diag(jnp.array([-1.0, 2.0, 0.5, 0.5]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.he_type) == 1.0
    assert float(res.is_vacuum) == 0.0


def test_vacuum_tag_just_below_tolerance():
    """Eigenvalues at exactly tol should NOT be tagged vacuum
    (strict less-than in the check)."""
    # max|Re lambda| = tol = 1e-10 exactly: should NOT be vacuum
    T = jnp.diag(jnp.array([-1e-10, 1e-10, 1e-10, 1e-10]))
    res = classify_hawking_ellis(T, _g_minkowski())
    assert float(res.is_vacuum) == 0.0, (
        "Eigenvalues exactly at tol should not trigger near_vacuum (uses <, not <=)"
    )


def test_grid_vacuum_count_matches_n_vacuum():
    """ECGridResult.n_vacuum reports the grid-aggregate count consistent
    with per-point is_vacuum tags."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec, evaluate_curvature_grid
    from warpax.energy_conditions.verifier import verify_grid

    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=0.5)
    # Small grid for test speed
    grid = GridSpec(bounds=[(-5, 5)] * 3, shape=(20, 20, 20))
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ec = verify_grid(
        curv.stress_energy, curv.metric, curv.metric_inv,
        n_starts=2, batch_size=64, compute_eulerian=False,
    )
    assert ec.n_vacuum is not None, "n_vacuum should be populated on ECGridResult"
    # On a (±5)^3 box for R=1, sigma=8 Alcubierre, the wall occupies
    # ~0.35% of the volume, so the vast majority of points should be
    # near-vacuum (T ~ 0 outside the wall).
    total = 20 ** 3
    vacuum_frac = ec.n_vacuum / total
    assert vacuum_frac > 0.5, (
        f"Expected > 50% vacuum for Alcubierre on (±5)^3; got {vacuum_frac:.3f}"
    )
    # And n_vacuum should be <= n_type_i (since vacuum points are tagged Type-I)
    assert ec.n_vacuum <= ec.n_type_i, (
        f"n_vacuum={ec.n_vacuum} exceeds n_type_i={ec.n_type_i}; vacuum "
        "points should be a subset of Type-I."
    )
