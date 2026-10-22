"""Geometry: curvature chain, ADM/metric specs, transitions, type contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import sympy as sp

from warpax.benchmarks.alcubierre import (
    AlcubierreMetric,
    alcubierre_symbolic,
    eulerian_energy_density,
)
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.geometry.geometry import (
    CurvatureResult,
    christoffel_symbols,
    compute_curvature_chain,
    riemann_tensor,
)
from warpax.geometry.metric import SymbolicMetric, adm_to_full_metric
from warpax.geometry.transitions import smoothstep, smoothstep_c1, smoothstep_c2
from warpax.geometry.types import GridSpec, TensorField
from warpax.metrics._common import alcubierre_shape


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
                    val += (
                        sp.Rational(1, 2)
                        * g_inv[lam, sigma]
                        * (
                            sp.diff(g[sigma, mu], coords[nu])
                            + sp.diff(g[sigma, nu], coords[mu])
                            - sp.diff(g[mu, nu], coords[sigma])
                        )
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
                    val = sp.diff(gamma[lam, mu, rho], coords[nu]) - sp.diff(
                        gamma[lam, mu, nu], coords[rho]
                    )
                    for sigma in range(dim):
                        val += (
                            gamma[lam, nu, sigma] * gamma[sigma, mu, rho]
                            - gamma[lam, rho, sigma] * gamma[sigma, mu, nu]
                        )
                    result[lam, mu, nu, rho] = val
    return sp.Array(result)


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
    coord_subs = dict(zip(symbolic_metric.coords, coord_vals, strict=True))
    expr = expr.subs(coord_subs)
    return np.array(expr.tolist(), dtype=np.float64)


class TestMinkowski:
    """Flat Minkowski spacetime: every curvature tensor is identically zero."""

    metric = MinkowskiMetric()
    coords = jnp.array([0.0, 1.0, 2.0, 3.0])

    def test_full_chain_zero(self):
        """Single flatness sentinel: every tensor in the chain must vanish."""
        result = compute_curvature_chain(self.metric, self.coords)
        npt.assert_allclose(result.christoffel, 0.0, atol=1e-15)
        npt.assert_allclose(result.riemann, 0.0, atol=1e-15)
        npt.assert_allclose(result.ricci, 0.0, atol=1e-15)
        npt.assert_allclose(result.ricci_scalar, 0.0, atol=1e-15)
        npt.assert_allclose(result.einstein, 0.0, atol=1e-15)
        npt.assert_allclose(result.stress_energy, 0.0, atol=1e-15)


class TestSchwarzschildChristoffel:
    """Compare autodiff Christoffel symbols against SymPy at a specific point.

    The evaluation point has r_iso = sqrt(3^2 + 4^2 + 0^2) = 5.0, well
    outside the horizon.
    """

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
            gamma_sym.subs(M_sym, 1.0).subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0}).tolist(),
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
        npt.assert_allclose(
            np.array(gamma),
            np.array(jnp.swapaxes(gamma, 1, 2)),
            atol=1e-15,
            err_msg="Christoffel symbols not symmetric in lower indices",
        )


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
            riemann_sym.subs(M_sym, 1.0).subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0}).tolist(),
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
            gamma_sym.subs(M_sym, 1.0).subs({t: 0.0, x: 3.0, y: 4.0, z: 0.0}).tolist(),
            dtype=np.float64,
        )
        request.cls.gamma_sympy = gamma_num

    def test_full_chain_vacuum(self):
        """Full curvature chain: metric shape correct, Christoffel matches SymPy,
        Ricci/Einstein/T all zero."""
        result = compute_curvature_chain(self.metric, self.jax_coords)

        assert result.metric.shape == (4, 4)
        assert result.metric_inv.shape == (4, 4)

        npt.assert_allclose(
            np.array(result.christoffel),
            self.gamma_sympy,
            atol=1e-12,
            err_msg="Full chain Christoffel mismatch with SymPy",
        )

        npt.assert_allclose(np.array(result.ricci), 0.0, atol=1e-10)
        npt.assert_allclose(float(result.ricci_scalar), 0.0, atol=1e-10)
        npt.assert_allclose(np.array(result.einstein), 0.0, atol=1e-10)
        npt.assert_allclose(np.array(result.stress_energy), 0.0, atol=1e-10)


class TestAlcubierre:
    """Alcubierre metric: non-flat with WEC-violating stress-energy.

    The evaluation point sits near the bubble wall, r_s = sqrt(0.8^2 + 0.5^2)
    ~ 0.94 with R=1.0 and sigma=8.0, where df/dr_s is large.
    """

    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
    jax_coords = jnp.array([0.0, 0.8, 0.5, 0.0])

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

        T_max = float(jnp.max(jnp.abs(result.stress_energy)))
        assert T_max > 1e-6, f"Alcubierre T_{'{mn}'} max = {T_max}, expected non-zero"

        _, x_val, y_val, z_val = self.jax_coords
        rho_analytical = eulerian_energy_density(
            x_val, y_val, z_val, v_s=0.5, R=1.0, sigma=8.0, x_s=0.0
        )

        r_s = jnp.sqrt(x_val**2 + y_val**2 + z_val**2)
        f = alcubierre_shape(r_s, 1.0, 8.0)
        n_up = jnp.array([1.0, 0.5 * f, 0.0, 0.0])

        T = result.stress_energy
        rho_autodiff = jnp.einsum("ab,a,b->", T, n_up, n_up)

        assert float(rho_analytical) < 0, (
            f"Analytical energy density should be negative, got {float(rho_analytical)}"
        )
        assert float(rho_autodiff) < 0, (
            f"Autodiff energy density should be negative, got {float(rho_autodiff)}"
        )

        assert np.sign(float(rho_analytical)) == np.sign(float(rho_autodiff)), (
            "Analytical and autodiff energy densities disagree on sign"
        )
        npt.assert_allclose(
            float(rho_autodiff),
            float(rho_analytical),
            rtol=1e-6,
            err_msg="Autodiff and analytical energy densities disagree",
        )


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


class TestCurvatureResult:
    """Verify CurvatureResult structure and JAX pytree compatibility."""

    def test_is_pytree(self):
        """CurvatureResult is a valid JAX pytree with 8 array leaves."""
        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        result = compute_curvature_chain(metric, coords)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 8, f"Expected 8 pytree leaves, got {len(leaves)}"
        for leaf in leaves:
            assert hasattr(leaf, "shape"), f"Leaf {type(leaf)} is not an array"


class TestADMToFullMetric:
    """Tests for the ADM -> full metric reconstruction."""

    def test_adm_to_full_metric_flat(self):
        """adm_to_full_metric(1, [0,0,0], eye(3)) == diag(-1,1,1,1)."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.0, 0.0, 0.0])
        gamma = jnp.eye(3)

        g = adm_to_full_metric(alpha, beta_up, gamma)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, expected, atol=1e-15)

    def test_adm_to_full_metric_shift(self):
        """adm_to_full_metric with nonzero shift, verify g_0i = beta_down_i.

        With alpha=1, beta^i = (0.5, 0, 0), gamma = I: beta_i = (0.5, 0, 0),
        g_00 = -(alpha^2 - beta_i beta^i) = -0.75, g_01 = 0.5, g_02 = g_03 = 0,
        and the spatial block equals gamma.
        """
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.5, 0.0, 0.0])
        gamma = jnp.eye(3)

        g = adm_to_full_metric(alpha, beta_up, gamma)
        assert jnp.isclose(g[0, 0], -0.75, atol=1e-15)
        assert jnp.isclose(g[0, 1], 0.5, atol=1e-15)
        assert jnp.isclose(g[1, 0], 0.5, atol=1e-15)
        assert jnp.isclose(g[0, 2], 0.0, atol=1e-15)
        assert jnp.isclose(g[0, 3], 0.0, atol=1e-15)
        assert jnp.allclose(g[1:, 1:], gamma, atol=1e-15)

    def test_adm_to_full_metric_jit(self):
        """Run through jax.jit, verify same result."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.3, -0.1, 0.2])
        gamma = jnp.eye(3)

        g_eager = adm_to_full_metric(alpha, beta_up, gamma)
        g_jit = jax.jit(adm_to_full_metric)(alpha, beta_up, gamma)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)


class TestSymbolicMetric:
    """Tests for the SymbolicMetric class."""

    def test_symbolic_metric_inverse(self):
        """Verify g * g_inv = identity (symbolically)."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.diag(-1, 1, 1, 1)
        sm = SymbolicMetric([t, x, y, z], g)
        product = sp.simplify(sm.g * sm.g_inv)
        assert product == sp.eye(4)

    def test_symbolic_metric_invalid_coords(self):
        """Verify ValueError for wrong number of coordinates."""
        x, y, z = sp.symbols("x y z")
        g = sp.diag(-1, 1, 1, 1)
        import pytest

        with pytest.raises(ValueError, match="4 coordinate symbols"):
            SymbolicMetric([x, y, z], g)

    def test_symbolic_metric_invalid_shape(self):
        """Verify ValueError for wrong matrix shape."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.diag(-1, 1, 1)
        import pytest

        with pytest.raises(ValueError, match="\\(4, 4\\)"):
            SymbolicMetric([t, x, y, z], g)


class TestSmoothstepTransitions:
    """Tests for C1/C2 smoothstep functions."""

    def test_smoothstep_c1_boundary_values(self):
        """C1 cubic: f(0)=0, f(1)=1, f(0.5)=0.5."""
        assert jnp.isclose(smoothstep_c1(jnp.array(0.0)), 0.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c1(jnp.array(1.0)), 1.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c1(jnp.array(0.5)), 0.5, atol=1e-15)

    def test_smoothstep_c2_boundary_values(self):
        """C2 quintic: f(0)=0, f(1)=1, f(0.5)=0.5."""
        assert jnp.isclose(smoothstep_c2(jnp.array(0.0)), 0.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c2(jnp.array(1.0)), 1.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c2(jnp.array(0.5)), 0.5, atol=1e-15)

    def test_smoothstep_c1_first_derivative_zero(self):
        """C1 cubic: f'(0)=0, f'(1)=0; the derivative of 3t^2 - 2t^3 is 6t - 6t^2."""
        grad_fn = jax.grad(lambda t: smoothstep_c1(t))
        assert jnp.isclose(grad_fn(jnp.array(0.0)), 0.0, atol=1e-10)
        assert jnp.isclose(grad_fn(jnp.array(1.0)), 0.0, atol=1e-10)

    def test_smoothstep_c2_first_derivative_zero(self):
        """C2 quintic: f'(0)=0, f'(1)=0."""
        grad_fn = jax.grad(lambda t: smoothstep_c2(t))
        assert jnp.isclose(grad_fn(jnp.array(0.0)), 0.0, atol=1e-10)
        assert jnp.isclose(grad_fn(jnp.array(1.0)), 0.0, atol=1e-10)

    def test_smoothstep_c2_second_derivative_zero(self):
        """C2 quintic: f''(0)=0, f''(1)=0.

        This is THE defining property of C2 smoothness. The second
        derivative 120t^3 - 180t^2 + 60t vanishes at both endpoints.
        """
        grad2_fn = jax.grad(jax.grad(lambda t: smoothstep_c2(t)))
        assert jnp.isclose(grad2_fn(jnp.array(0.0)), 0.0, atol=1e-10), (
            f"C2 f''(0) = {grad2_fn(jnp.array(0.0))}, expected 0.0"
        )
        assert jnp.isclose(grad2_fn(jnp.array(1.0)), 0.0, atol=1e-10), (
            f"C2 f''(1) = {grad2_fn(jnp.array(1.0))}, expected 0.0"
        )

    def test_smoothstep_c1_second_derivative_nonzero(self):
        """C1 cubic: f''(eps) is large (NOT near zero confirms C1 is not C2).

        The second derivative of 3t^2 - 2t^3 is 6 - 12t, which gives
        f''(0)=6 analytically. At the exact clip boundary, JAX autodiff
        returns a modified value due to the clip gradient convention, so
        we evaluate at a point just inside [0, 1]: f''(0.01) = 6 - 12*0.01 = 5.88.
        """
        grad2_fn = jax.grad(jax.grad(lambda t: smoothstep_c1(t)))
        f2_near_0 = grad2_fn(jnp.array(0.01))
        assert jnp.isclose(f2_near_0, 5.88, atol=0.01), (
            f"C1 f''(0.01) = {f2_near_0}, expected ~5.88"
        )
        assert not jnp.isclose(f2_near_0, 0.0, atol=1.0), "C1 f''(0.01) should NOT be near zero"

    def test_smoothstep_dispatch(self):
        """smoothstep(t, order=N) matches the corresponding function."""
        t = jnp.linspace(0.0, 1.0, 50)
        assert jnp.allclose(smoothstep(t, order=1), smoothstep_c1(t), atol=1e-15)
        assert jnp.allclose(smoothstep(t, order=2), smoothstep_c2(t), atol=1e-15)

    def test_smoothstep_invalid_order(self):
        """smoothstep(t, order=3) raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported smoothstep order"):
            smoothstep(jnp.array(0.5), order=3)

    def test_smoothstep_clipping(self):
        """Values outside [0,1] are clipped: f(-0.5)=0, f(1.5)=1."""
        for fn in [smoothstep_c1, smoothstep_c2]:
            assert jnp.isclose(fn(jnp.array(-0.5)), 0.0, atol=1e-15), (
                f"{fn.__name__}(-0.5) should be 0.0"
            )
            assert jnp.isclose(fn(jnp.array(1.5)), 1.0, atol=1e-15), (
                f"{fn.__name__}(1.5) should be 1.0"
            )

    def test_smoothstep_monotonic(self):
        """Both C1 and C2 are monotonically non-decreasing on [0, 1]."""
        t = jnp.linspace(0.0, 1.0, 100)
        for fn in [smoothstep_c1, smoothstep_c2]:
            vals = fn(t)
            diffs = jnp.diff(vals)
            assert jnp.all(diffs >= -1e-15), (
                f"{fn.__name__} is not monotonic: min diff = {jnp.min(diffs)}"
            )


class TestTensorField:
    """Tests for the TensorField Equinox module."""

    def test_tensorfield_default_index_positions(self):
        """Create with empty index_positions, verify default is 'd' * rank."""
        components = jnp.zeros((4, 4, 4))
        tf = TensorField(components=components, rank=3)
        assert tf.index_positions == "ddd"

    def test_tensorfield_jit(self):
        """Pass TensorField through jax.jit and verify components survive."""
        components = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        tf = TensorField(components=components, rank=2, index_positions="dd")

        @jax.jit
        def identity(field: TensorField) -> TensorField:
            return field

        result = identity(tf)
        assert jnp.allclose(result.components, components)
        assert result.rank == 2
        assert result.index_positions == "dd"

    def test_tensorfield_invalid_rank(self):
        """Verify ValueError when index_positions length != rank."""
        with pytest.raises(ValueError, match="index_positions length"):
            TensorField(
                components=jnp.zeros((4, 4)),
                rank=2,
                index_positions="ddd",
            )

    def test_tensorfield_grid_shape(self):
        """Verify grid_shape property for a field on a grid."""
        components = jnp.zeros((10, 10, 4, 4))
        tf = TensorField(components=components, rank=2)
        assert tf.grid_shape == (10, 10)
        assert tf.tensor_shape == (4, 4)


class TestGridSpec:
    """Tests for the GridSpec Equinox module."""

    def test_gridspec_spacing(self):
        """Verify spacing computation matches expected values."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (0.0, 4.0), (-3.0, 3.0)],
            shape=(11, 5, 7),
        )
        spacing = grid.spacing
        assert abs(spacing[0] - 0.2) < 1e-14
        assert abs(spacing[1] - 1.0) < 1e-14
        assert abs(spacing[2] - 1.0) < 1e-14

    def test_gridspec_axes(self):
        """Verify axes are jnp arrays with correct dtype (float64) and length."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        axes = grid.axes
        assert len(axes) == 3
        for i, (ax, n) in enumerate(zip(axes, grid.shape, strict=True)):
            assert isinstance(ax, jax.Array)
            assert ax.dtype == jnp.float64
            assert ax.shape == (n,)

    def test_gridspec_meshgrid(self):
        """Verify meshgrid returns JAX arrays with correct shapes."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        mg = grid.meshgrid
        assert len(mg) == 3
        for arr in mg:
            assert isinstance(arr, jax.Array)
            assert arr.shape == (5, 10, 15)
            assert arr.dtype == jnp.float64

    def test_gridspec_coordinate_fields(self):
        """Verify 4D coordinate fields [t=0, x, y, z]."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        fields = grid.coordinate_fields
        assert len(fields) == 4
        assert jnp.allclose(fields[0], 0.0)
        mg = grid.meshgrid
        for f, m in zip(fields[1:], mg, strict=True):
            assert jnp.allclose(f, m)
        for f in fields:
            assert f.dtype == jnp.float64


class TestGridEvaluation:
    """Grid curvature is a reshape of the point chain; t and bounds are data."""

    @pytest.mark.parametrize("t", [0.0, 0.3])
    @pytest.mark.parametrize("batch_size", [None, 50])
    def test_grid_eval_matches_point_eval(self, t, batch_size):
        from jax import lax

        from warpax.geometry.grid import _make_point_fn, evaluate_curvature_grid

        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-2.0, 2.0)] * 3, shape=(6, 6, 6))
        got = evaluate_curvature_grid(metric, grid, t=t, batch_size=batch_size)

        @eqx.filter_jit
        def old_fused_path(metric_fn, grid_spec, t):
            X, Y, Z = grid_spec.meshgrid
            coords = jnp.stack([jnp.full_like(X, t), X, Y, Z], axis=-1).reshape(-1, 4)
            point_fn = _make_point_fn(metric_fn, True)
            if batch_size is not None:
                return lax.map(point_fn, coords, batch_size=batch_size)
            return jax.vmap(point_fn)(coords)

        ref = old_fused_path(metric, grid, t)
        for name in ref._fields:
            r = getattr(ref, name)
            npt.assert_allclose(
                getattr(got, name), r.reshape(6, 6, 6, *r.shape[1:]), rtol=1e-12, atol=0.0
            )

    def test_time_and_bounds_do_not_retrace(self, monkeypatch):
        from warpax.geometry import grid as grid_mod

        inner = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        traces = []
        make_point_fn = grid_mod._make_point_fn

        def counting_make_point_fn(*args, **kwargs):
            traces.append(1)
            return make_point_fn(*args, **kwargs)

        monkeypatch.setattr(grid_mod, "_make_point_fn", counting_make_point_fn)

        def metric_fn(c):
            return inner(c)

        for t in (0.0, 0.1, 0.2):
            grid_mod.evaluate_curvature_grid(
                metric_fn, GridSpec(bounds=[(-2.0, 2.0)] * 3, shape=(4, 4, 4)), t=t
            )
        for hi in (1.0, 1.5, 2.5):
            grid_mod.evaluate_curvature_grid(
                metric_fn, GridSpec(bounds=[(-hi, hi)] * 3, shape=(4, 4, 4))
            )
        assert len(traces) == 1, f"point function traced {len(traces)} times"

    def test_bounds_are_immutable_tuples(self):
        grid = GridSpec(bounds=[[-1, 1], (0.0, 4.0), np.array([-3.0, 3.0])], shape=(3, 3, 3))
        assert grid.bounds == ((-1.0, 1.0), (0.0, 4.0), (-3.0, 3.0))
        assert isinstance(grid.bounds, tuple)
        assert all(isinstance(b, tuple) for b in grid.bounds)
        assert all(type(v) is float for b in grid.bounds for v in b)
        with pytest.raises(TypeError):
            grid.bounds[0][0] = 5.0
        assert not hasattr(grid, "volume_weights")

    def test_stress_energy_kernel_matches_full_chain(self):
        """The three-field kernel equals the full-chain grid path certify used before."""
        from warpax.geometry.grid import (
            _stress_energy_points,
            build_coord_batch,
            evaluate_curvature_grid,
        )
        from warpax.grids import wall_clustered

        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = wall_clustered(metric, [(-3.0, 3.0)] * 3, (8, 8, 8), wall_radius=1.0)
        coords = build_coord_batch(grid, t=0.0)
        T, g, gi = _stress_energy_points(metric, coords, batch_size=100)
        ref = evaluate_curvature_grid(metric, grid, batch_size=100)
        npt.assert_allclose(T.reshape(8, 8, 8, 4, 4), ref.stress_energy, rtol=1e-12, atol=0.0)
        npt.assert_allclose(g.reshape(8, 8, 8, 4, 4), ref.metric, rtol=1e-12, atol=0.0)
        npt.assert_allclose(gi.reshape(8, 8, 8, 4, 4), ref.metric_inv, rtol=1e-12, atol=0.0)

    def test_time_does_not_recompile_coordinate_builder(self, caplog):
        """Three time slices compile the coordinate builder once and the curvature kernel once.

        ``t`` enters the compiled builder as a float64 array, so its cache is
        keyed on the grid spec alone; expected compile counts are 1 and 1.
        """
        from warpax.geometry.grid import evaluate_curvature_grid

        inner = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)

        def metric_fn(c):
            return inner(c)

        grid = GridSpec(bounds=[(-2.0, 2.0)] * 3, shape=(4, 4, 4))
        jax.clear_caches()
        jax.config.update("jax_log_compiles", True)
        try:
            with caplog.at_level("INFO", logger="jax"):
                for t in (0.0, 0.1, 0.2):
                    evaluate_curvature_grid(metric_fn, grid, t=t)
        finally:
            jax.config.update("jax_log_compiles", False)
        messages = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("Compiling")
        ]
        assert sum("_build_coord_batch" in m for m in messages) == 1, messages
        assert sum("evaluate_curvature_points" in m for m in messages) == 1, messages

    def test_grid_eval_at_nonzero_time_matches_points(self):
        """Grid results at t=0.3 equal the point-wise chain on the same coordinates (rtol 1e-12)."""
        from warpax.geometry.grid import build_coord_batch, evaluate_curvature_grid

        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-2.0, 2.0)] * 3, shape=(5, 5, 5))
        coords = build_coord_batch(grid, t=0.3)
        npt.assert_array_equal(coords[:, 0], 0.3)
        got = evaluate_curvature_grid(metric, grid, t=0.3)
        ref = jax.vmap(lambda c: compute_curvature_chain(metric, c))(coords)
        for name in ref._fields:
            r = getattr(ref, name)
            npt.assert_allclose(
                getattr(got, name), r.reshape(5, 5, 5, *r.shape[1:]), rtol=1e-12, atol=0.0
            )


def test_shared_christoffel_matches_spatial_einsum():
    """One Christoffel helper serves the 4D chain and every 3D spatial-metric site."""
    from warpax.geometry.geometry import christoffel_from_jacobian

    metric = SchwarzschildMetric(M=1.0)
    coords = jnp.array([0.0, 3.0, 0.7, -0.4])

    def gamma_fn(xyz):
        return metric(jnp.concatenate([coords[:1], xyz]))[1:, 1:]

    g_inv = jnp.linalg.inv(gamma_fn(coords[1:]))
    dg = jax.jacfwd(gamma_fn)(coords[1:])
    inline = 0.5 * (
        jnp.einsum("lk,jki->lij", g_inv, dg)
        + jnp.einsum("lk,ikj->lij", g_inv, dg)
        - jnp.einsum("lk,ijk->lij", g_inv, dg)
    )
    npt.assert_allclose(christoffel_from_jacobian(g_inv, dg), inline, rtol=1e-12, atol=0)

    g4 = metric(coords)
    npt.assert_allclose(
        christoffel_from_jacobian(jnp.linalg.inv(g4), jax.jacfwd(metric)(coords)),
        christoffel_symbols(metric, coords),
        rtol=1e-12,
        atol=0,
    )


def test_spatial_ricci_scalar_matches_inline_riemann_assembly():
    """The ADM Hamiltonian residual reuses the geometry Riemann assembly."""
    from warpax.constraints.residuals import _spatial_ricci_scalar
    from warpax.geometry.geometry import christoffel_from_jacobian

    coords = jnp.array([0.0, 0.3, 0.7, -0.4])

    def metric(c):
        psi = 1.0 + c[1] ** 2 + 0.5 * c[2] * c[3]
        return jnp.diag(jnp.array([-1.0, psi**4, psi**4, psi**4]))

    def gamma_fn(xyz):
        return metric(jnp.concatenate([coords[:1], xyz]))[1:, 1:]

    def christoffel_fn(xyz):
        return christoffel_from_jacobian(jnp.linalg.inv(gamma_fn(xyz)), jax.jacfwd(gamma_fn)(xyz))

    xyz = coords[1:]
    gamma = christoffel_fn(xyz)
    dgamma = jax.jacfwd(christoffel_fn)(xyz)
    riemann = (
        jnp.swapaxes(dgamma, 2, 3)
        - dgamma
        + jnp.einsum("lsn,smr->lmnr", gamma, gamma)
        - jnp.einsum("lsr,smn->lmnr", gamma, gamma)
    )
    inline = jnp.einsum("ij,ij->", jnp.linalg.inv(gamma_fn(xyz)), jnp.einsum("kikj->ij", riemann))
    assert float(jnp.abs(inline)) > 1e-6
    npt.assert_allclose(_spatial_ricci_scalar(metric, coords), inline, rtol=1e-12, atol=0)


def test_alcubierre_symbolic_shared_by_class_and_module():
    assert AlcubierreMetric().symbolic().g == alcubierre_symbolic().g
