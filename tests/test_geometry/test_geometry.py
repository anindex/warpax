"""Tests for autodiff geometry chain.

Validates every tensor in the curvature chain against SymPy symbolic ground truth
at specific coordinate points, confirming machine-precision agreement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import sympy as sp

from warpax.geometry.geometry import (
    CurvatureResult,
    christoffel_symbols,
    compute_curvature_chain,
    einstein_tensor,
    ricci_scalar,
    ricci_tensor,
    riemann_tensor,
    stress_energy_tensor,
)
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.benchmarks.alcubierre import (
    AlcubierreMetric,
    eulerian_energy_density,
    _shape_function,
)
from warpax.geometry.metric import SymbolicMetric


# ---------------------------------------------------------------------------
# Inline SymPy symbolic geometry (replaces legacy imports)
# ---------------------------------------------------------------------------


def _christoffel_symbolic(sm: SymbolicMetric) -> sp.Array:
    """Compute Christoffel symbols Gamma^l_{mn} symbolically."""
    g = sm.g
    g_inv = sm.g_inv
    coords = sm.coords
    dim = 4
    result = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                val = sp.Rational(0)
                for sigma in range(dim):
                    val += sp.Rational(1, 2) * g_inv[lam, sigma] * (
                        sp.diff(g[sigma, mu], coords[nu])
                        + sp.diff(g[sigma, nu], coords[mu])
                        - sp.diff(g[mu, nu], coords[sigma])
                    )
                result[lam, mu, nu] = val
    return sp.Array(result)


def _riemann_symbolic(sm: SymbolicMetric) -> sp.Array:
    """Compute Riemann tensor R^l_{mnr} symbolically."""
    coords = sm.coords
    dim = 4
    gamma = _christoffel_symbolic(sm)
    result = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                for rho in range(dim):
                    val = (
                        sp.diff(gamma[lam, mu, rho], coords[nu])
                        - sp.diff(gamma[lam, mu, nu], coords[rho])
                    )
                    for sigma in range(dim):
                        val += (
                            gamma[lam, nu, sigma] * gamma[sigma, mu, rho]
                            - gamma[lam, rho, sigma] * gamma[sigma, mu, nu]
                        )
                    result[lam, mu, nu, rho] = val
    return sp.Array(result)


# ---------------------------------------------------------------------------
# Helper: evaluate a SymPy tensor at a specific coordinate point
# ---------------------------------------------------------------------------


def _eval_sympy_at_point(sympy_array, symbolic_metric, param_subs, coord_vals):
    """Evaluate a SymPy tensor array at a specific coordinate point.

    Parameters
    ----------
    sympy_array : sp.Array | sp.Matrix | sp.Expr
        The SymPy expression(s) to evaluate.
    symbolic_metric : SymbolicMetric
        Provides coordinate symbols via ``.coords``.
    param_subs : dict
        Parameter substitutions (e.g. ``{M: 1.0}``).
    coord_vals : tuple | list
        Numeric coordinate values to substitute.

    Returns
    -------
    np.ndarray
        Numeric result with dtype float64.
    """
    expr = sympy_array
    if param_subs:
        expr = expr.subs(param_subs)
    coord_subs = dict(zip(symbolic_metric.coords, coord_vals))
    expr = expr.subs(coord_subs)
    return np.array(expr.tolist(), dtype=np.float64)


# =========================================================================
# 1. Minkowski (flat) all curvature quantities must be exactly zero
# =========================================================================


class TestMinkowski:
    """Flat Minkowski spacetime: every curvature tensor is identically zero."""

    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 2.0, 3.0])

    def test_christoffel_zero(self):
        gamma = christoffel_symbols(self.metric, self.coords)
        npt.assert_allclose(gamma, 0.0, atol=1e-15)

    def test_riemann_zero(self):
        R = riemann_tensor(self.metric, self.coords)
        npt.assert_allclose(R, 0.0, atol=1e-15)

    def test_ricci_zero(self):
        R_abcd = riemann_tensor(self.metric, self.coords)
        Ric = ricci_tensor(R_abcd)
        npt.assert_allclose(Ric, 0.0, atol=1e-15)

    def test_ricci_scalar_zero(self):
        R_abcd = riemann_tensor(self.metric, self.coords)
        Ric = ricci_tensor(R_abcd)
        g_inv = jnp.linalg.inv(self.metric(self.coords))
        R_sc = ricci_scalar(g_inv, Ric)
        npt.assert_allclose(R_sc, 0.0, atol=1e-15)

    def test_einstein_zero(self):
        R_abcd = riemann_tensor(self.metric, self.coords)
        Ric = ricci_tensor(R_abcd)
        g = self.metric(self.coords)
        g_inv = jnp.linalg.inv(g)
        R_sc = ricci_scalar(g_inv, Ric)
        G = einstein_tensor(Ric, R_sc, g)
        npt.assert_allclose(G, 0.0, atol=1e-15)

    def test_stress_energy_zero(self):
        R_abcd = riemann_tensor(self.metric, self.coords)
        Ric = ricci_tensor(R_abcd)
        g = self.metric(self.coords)
        g_inv = jnp.linalg.inv(g)
        R_sc = ricci_scalar(g_inv, Ric)
        G = einstein_tensor(Ric, R_sc, g)
        T = stress_energy_tensor(G)
        npt.assert_allclose(T, 0.0, atol=1e-15)

    def test_full_chain_zero(self):
        result = compute_curvature_chain(self.metric, self.coords)
        npt.assert_allclose(result.christoffel, 0.0, atol=1e-15)
        npt.assert_allclose(result.riemann, 0.0, atol=1e-15)
        npt.assert_allclose(result.ricci, 0.0, atol=1e-15)
        npt.assert_allclose(result.ricci_scalar, 0.0, atol=1e-15)
        npt.assert_allclose(result.einstein, 0.0, atol=1e-15)
        npt.assert_allclose(result.stress_energy, 0.0, atol=1e-15)


# =========================================================================
# 2. Schwarzschild Christoffel symbols vs SymPy ground truth
# =========================================================================


class TestSchwarzschildChristoffel:
    """Compare autodiff Christoffel symbols against SymPy at a specific point."""

    # Evaluation point: r_iso = sqrt(3^2 + 4^2 + 0^2) = 5.0 (well outside horizon)
    jax_coords = jnp.array([0.0, 3.0, 4.0, 0.0])
    coord_vals = (0.0, 3.0, 4.0, 0.0)

    @pytest.fixture(autouse=True, scope="class")
    def _setup_sympy(self, request):
        """Compute SymPy Christoffel symbols once for the class."""
        metric = SchwarzschildMetric(M=1.0)
        sm = metric.symbolic()
        gamma_sym = _christoffel_symbolic(sm)

        M_sym = sp.Symbol("M", positive=True)
        t, x, y, z = sm.coords
        gamma_num = np.array(
            gamma_sym.subs(M_sym, 1.0)
            .subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0})
            .tolist(),
            dtype=np.float64,
        )
        request.cls.gamma_sympy = gamma_num
        request.cls.sm = sm

    def test_christoffel_sympy_match(self):
        """All 64 Christoffel components match SymPy ground truth."""
        metric = SchwarzschildMetric(M=1.0)
        gamma_ad = christoffel_symbols(metric, self.jax_coords)
        npt.assert_allclose(
            np.array(gamma_ad),
            self.gamma_sympy,
            atol=1e-12,
            err_msg="Autodiff Christoffel symbols do not match SymPy",
        )

    def test_christoffel_lower_index_symmetry(self):
        """Gamma^l_{mn} == Gamma^l_{nm} (symmetry in lower indices)."""
        metric = SchwarzschildMetric(M=1.0)
        gamma = christoffel_symbols(metric, self.jax_coords)
        # Swap lower indices (axes 1 and 2)
        npt.assert_allclose(
            np.array(gamma),
            np.array(jnp.swapaxes(gamma, 1, 2)),
            atol=1e-15,
            err_msg="Christoffel symbols not symmetric in lower indices",
        )


# =========================================================================
# 3. Schwarzschild Riemann tensor vs SymPy ground truth
# =========================================================================


class TestSchwarzschildRiemann:
    """Compare autodiff Riemann tensor against SymPy."""

    jax_coords = jnp.array([0.0, 3.0, 4.0, 0.0])
    coord_vals = (0.0, 3.0, 4.0, 0.0)

    @pytest.fixture(autouse=True, scope="class")
    def _setup_sympy(self, request):
        """Compute SymPy Riemann tensor once for the class (may take 10-30s)."""
        metric = SchwarzschildMetric(M=1.0)
        sm = metric.symbolic()
        riemann_sym = _riemann_symbolic(sm)

        M_sym = sp.Symbol("M", positive=True)
        t, x, y, z = sm.coords
        riemann_num = np.array(
            riemann_sym.subs(M_sym, 1.0)
            .subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0})
            .tolist(),
            dtype=np.float64,
        )
        request.cls.riemann_sympy = riemann_num
        request.cls.sm = sm

    def test_riemann_sympy_match(self):
        """All 256 Riemann components match SymPy ground truth."""
        metric = SchwarzschildMetric(M=1.0)
        R_ad = riemann_tensor(metric, self.jax_coords)
        npt.assert_allclose(
            np.array(R_ad),
            self.riemann_sympy,
            atol=1e-12,
            err_msg="Autodiff Riemann tensor does not match SymPy",
        )

    def test_riemann_antisymmetry(self):
        """R^l_{mnr} == -R^l_{mrn} (antisymmetry in last two indices)."""
        metric = SchwarzschildMetric(M=1.0)
        R = riemann_tensor(metric, self.jax_coords)
        # Swap last two indices (axes 2 and 3)
        R_swapped = jnp.swapaxes(R, 2, 3)
        npt.assert_allclose(
            np.array(R),
            -np.array(R_swapped),
            atol=1e-14,
            err_msg="Riemann tensor not antisymmetric in last two indices",
        )


# =========================================================================
# 4. Schwarzschild full chain vacuum solution (Ricci-flat)
# =========================================================================


class TestSchwarzschildFullChain:
    """Schwarzschild is a vacuum solution: R_mn=0, R=0, G_mn=0, T_mn=0."""

    metric = SchwarzschildMetric(M=1.0)
    jax_coords = jnp.array([0.0, 3.0, 4.0, 0.0])

    @pytest.fixture(autouse=True, scope="class")
    def _setup_sympy(self, request):
        """Compute SymPy Christoffel for full-chain reference."""
        sm = self.metric.symbolic()
        gamma_sym = _christoffel_symbolic(sm)
        M_sym = sp.Symbol("M", positive=True)
        t, x, y, z = sm.coords
        gamma_num = np.array(
            gamma_sym.subs(M_sym, 1.0)
            .subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0})
            .tolist(),
            dtype=np.float64,
        )
        request.cls.gamma_sympy = gamma_num

    def test_ricci_sympy_match(self):
        """Schwarzschild Ricci tensor is zero (vacuum)."""
        R_abcd = riemann_tensor(self.metric, self.jax_coords)
        Ric = ricci_tensor(R_abcd)
        npt.assert_allclose(
            np.array(Ric),
            0.0,
            atol=1e-10,
            err_msg="Schwarzschild Ricci tensor not zero",
        )

    def test_ricci_scalar_zero(self):
        """Schwarzschild Ricci scalar is zero (vacuum)."""
        R_abcd = riemann_tensor(self.metric, self.jax_coords)
        Ric = ricci_tensor(R_abcd)
        g_inv = jnp.linalg.inv(self.metric(self.jax_coords))
        R_sc = ricci_scalar(g_inv, Ric)
        npt.assert_allclose(
            float(R_sc),
            0.0,
            atol=1e-10,
            err_msg="Schwarzschild Ricci scalar not zero",
        )

    def test_einstein_zero(self):
        """Schwarzschild Einstein tensor is zero (vacuum)."""
        R_abcd = riemann_tensor(self.metric, self.jax_coords)
        Ric = ricci_tensor(R_abcd)
        g = self.metric(self.jax_coords)
        g_inv = jnp.linalg.inv(g)
        R_sc = ricci_scalar(g_inv, Ric)
        G = einstein_tensor(Ric, R_sc, g)
        npt.assert_allclose(
            np.array(G),
            0.0,
            atol=1e-10,
            err_msg="Schwarzschild Einstein tensor not zero",
        )

    def test_stress_energy_zero(self):
        """Schwarzschild stress-energy tensor is zero (vacuum)."""
        R_abcd = riemann_tensor(self.metric, self.jax_coords)
        Ric = ricci_tensor(R_abcd)
        g = self.metric(self.jax_coords)
        g_inv = jnp.linalg.inv(g)
        R_sc = ricci_scalar(g_inv, Ric)
        G = einstein_tensor(Ric, R_sc, g)
        T = stress_energy_tensor(G)
        npt.assert_allclose(
            np.array(T),
            0.0,
            atol=1e-10,
            err_msg="Schwarzschild stress-energy tensor not zero",
        )

    def test_full_chain_vacuum(self):
        """Full curvature chain: metric shape correct, Christoffel matches SymPy,
        Ricci/Einstein/T all zero."""
        result = compute_curvature_chain(self.metric, self.jax_coords)

        # Metric shape
        assert result.metric.shape == (4, 4)
        assert result.metric_inv.shape == (4, 4)

        # Christoffel matches SymPy
        npt.assert_allclose(
            np.array(result.christoffel),
            self.gamma_sympy,
            atol=1e-12,
            err_msg="Full chain Christoffel mismatch with SymPy",
        )

        # Vacuum: everything downstream is zero
        npt.assert_allclose(np.array(result.ricci), 0.0, atol=1e-10)
        npt.assert_allclose(float(result.ricci_scalar), 0.0, atol=1e-10)
        npt.assert_allclose(np.array(result.einstein), 0.0, atol=1e-10)
        npt.assert_allclose(np.array(result.stress_energy), 0.0, atol=1e-10)


# =========================================================================
# 5. Alcubierre warp drive non-trivial stress-energy, WEC violation
# =========================================================================


class TestAlcubierre:
    """Alcubierre metric: non-flat with WEC-violating stress-energy."""

    # Near bubble wall: r_s = sqrt(0.8^2 + 0.5^2 + 0^2) ~ 0.94
    # With R=1.0, sigma=8.0 this is close to the bubble wall where df/dr_s is large
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
    jax_coords = jnp.array([0.0, 0.8, 0.5, 0.0])

    def test_christoffel_nonzero(self):
        """Near the bubble wall, Christoffel symbols are non-trivial."""
        gamma = christoffel_symbols(self.metric, self.jax_coords)
        max_abs = float(jnp.max(jnp.abs(gamma)))
        assert max_abs > 0.01, (
            f"Alcubierre Christoffel max |gamma| = {max_abs}, expected > 0.01"
        )

    def test_christoffel_lower_symmetry(self):
        """Gamma^l_{mn} == Gamma^l_{nm} (symmetry in lower indices)."""
        gamma = christoffel_symbols(self.metric, self.jax_coords)
        npt.assert_allclose(
            np.array(gamma),
            np.array(jnp.swapaxes(gamma, 1, 2)),
            atol=1e-15,
            err_msg="Alcubierre Christoffel symbols not symmetric in lower indices",
        )

    def test_stress_energy_nonzero_and_wec_violation(self):
        """Stress-energy is non-zero and energy density is negative (WEC violation).

        The Eulerian energy density for Alcubierre is:
            rho = -(v_s^2 / 32pi) (df/dr_s)^2 (y^2 + z^2) / r_s^2

        This is always <= 0, confirming WEC violation.

        The Eulerian observer has unit normal n^a = (1/alpha, -beta^i/alpha).
        For Alcubierre with alpha=1 and beta^x = -v_s*f:
            n^a = (1, v_s*f, 0, 0)
        Energy density rho = T_{ab} n^a n^b.
        """
        result = compute_curvature_chain(self.metric, self.jax_coords)

        # T must be non-zero
        T_max = float(jnp.max(jnp.abs(result.stress_energy)))
        assert T_max > 1e-6, (
            f"Alcubierre T_{'{mn}'} max = {T_max}, expected non-zero"
        )

        # Compute analytical Eulerian energy density at the same spatial point
        _, x_val, y_val, z_val = self.jax_coords
        rho_analytical = eulerian_energy_density(
            x_val, y_val, z_val, v_s=0.5, R=1.0, sigma=8.0, x_s=0.0
        )

        # Construct Eulerian normal vector n^a = (1/alpha, -beta^i/alpha)
        # For Alcubierre: alpha=1, shift = (-v_s*f, 0, 0)
        # So n^a = (1, v_s*f, 0, 0)
        r_s = jnp.sqrt(x_val**2 + y_val**2 + z_val**2)
        f = _shape_function(r_s, 1.0, 8.0)
        n_up = jnp.array([1.0, 0.5 * f, 0.0, 0.0])

        # Eulerian energy density: rho = T_{ab} n^a n^b
        T = result.stress_energy
        rho_autodiff = jnp.einsum("ab,a,b->", T, n_up, n_up)

        # Both should be negative (WEC violation)
        assert float(rho_analytical) < 0, (
            f"Analytical energy density should be negative, got {float(rho_analytical)}"
        )
        assert float(rho_autodiff) < 0, (
            f"Autodiff energy density should be negative, got {float(rho_autodiff)}"
        )

        # Both should agree on sign and magnitude
        assert np.sign(float(rho_analytical)) == np.sign(float(rho_autodiff)), (
            "Analytical and autodiff energy densities disagree on sign"
        )
        npt.assert_allclose(
            float(rho_autodiff),
            float(rho_analytical),
            rtol=1e-6,
            err_msg="Autodiff and analytical energy densities disagree",
        )


# =========================================================================
# 6. JIT compilability
# =========================================================================


class TestJITCompilability:
    """Verify all geometry functions work under jax.jit."""

    metric = SchwarzschildMetric(M=1.0)
    coords = jnp.array([0.0, 3.0, 4.0, 0.0])

    def test_christoffel_jit(self):
        """jax.jit(christoffel_symbols) matches eager evaluation."""
        eager = christoffel_symbols(self.metric, self.coords)
        jitted = jax.jit(christoffel_symbols)(self.metric, self.coords)
        npt.assert_allclose(
            np.array(eager),
            np.array(jitted),
            atol=1e-15,
            err_msg="JIT Christoffel does not match eager",
        )

    def test_riemann_jit(self):
        """jax.jit(riemann_tensor) matches eager evaluation."""
        eager = riemann_tensor(self.metric, self.coords)
        jitted = jax.jit(riemann_tensor)(self.metric, self.coords)
        npt.assert_allclose(
            np.array(eager),
            np.array(jitted),
            atol=1e-15,
            err_msg="JIT Riemann does not match eager",
        )

    def test_full_chain_jit(self):
        """jax.jit(compute_curvature_chain) matches eager evaluation."""
        eager = compute_curvature_chain(self.metric, self.coords)
        jitted = jax.jit(compute_curvature_chain)(self.metric, self.coords)

        for field_name in CurvatureResult._fields:
            npt.assert_allclose(
                np.array(getattr(eager, field_name)),
                np.array(getattr(jitted, field_name)),
                atol=1e-15,
                err_msg=f"JIT full chain mismatch for {field_name}",
            )


# =========================================================================
# 7. CurvatureResult NamedTuple structure
# =========================================================================


class TestCurvatureResult:
    """Verify CurvatureResult structure and JAX pytree compatibility."""

    def test_namedtuple_fields(self):
        """CurvatureResult has 8 expected fields."""
        expected_fields = (
            "metric",
            "metric_inv",
            "christoffel",
            "riemann",
            "ricci",
            "ricci_scalar",
            "einstein",
            "stress_energy",
        )
        assert CurvatureResult._fields == expected_fields

    def test_is_pytree(self):
        """CurvatureResult is a valid JAX pytree with 8 array leaves."""
        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        result = compute_curvature_chain(metric, coords)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 8, (
            f"Expected 8 pytree leaves, got {len(leaves)}"
        )
        # All leaves should be JAX arrays
        for leaf in leaves:
            assert hasattr(leaf, "shape"), f"Leaf {type(leaf)} is not an array"
