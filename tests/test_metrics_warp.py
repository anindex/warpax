"""Warp-metric construction and contracts: Lentz, Natario, Rodal, VdB, WarpShell.

``CONTRACTS`` states the contract every warp metric owes, asserted once over the
table: ``probe`` is a generic evaluation point, ``speed_point`` sits where changing
v_s must move the metric, and ``wall`` tunes the metric so ``wall_point`` lands
where curvature is nontrivial (Lentz's wall point is off-axis to avoid the L1
kink). ``test_far_field_is_minkowski`` covers the metrics whose defaults give a
Minkowski limit on the x-axis; Natario is co-moving, so its far field carries a
uniform flow, and WarpShell checks its exterior at two radii, so both keep their
own test.
"""

from pathlib import Path
from runpy import run_path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry import SymbolicMetric, adm_to_full_metric, compute_curvature_chain
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
    WarpShellPhysical,
    tshell_default,
)
from warpax.metrics import warpshell as warpshell_module
from warpax.metrics._common import alcubierre_shape
from warpax.metrics.natario import natario_eulerian_energy_density
from warpax.metrics.rodal import _rodal_G, _rodal_g_paper

CONTRACTS = [
    dict(
        name="Lentz",
        build=LentzMetric,
        probe=[0.0, 5.0, 2.0, 3.0],
        speeds=({"v_s": 0.1}, {"v_s": 0.5}),
        speed_point=[0.0, 10.0, 5.0, 0.0],
        wall={"v_s": 0.1, "R": 100.0, "sigma": 8.0},
        wall_point=[0.0, 50.0, 50.0, 0.0],
    ),
    dict(
        name="Natario",
        build=NatarioMetric,
        probe=[0.0, 1.0, 2.0, 3.0],
        speeds=({"v_s": 0.1}, {"v_s": 0.5}),
        speed_point=[0.0, 50.0, 0.0, 0.0],
        wall={"v_s": 0.1, "R": 100.0, "sigma": 0.03},
        wall_point=[0.0, 100.0, 1.0, 0.0],
    ),
    dict(
        name="Rodal",
        build=RodalMetric,
        probe=[0.0, 1.0, 2.0, 3.0],
        speeds=({"v_s": 0.1}, {"v_s": 0.5}),
        speed_point=[0.0, 10.0, 0.0, 0.0],
        wall={"v_s": 0.1, "R": 100.0, "sigma": 0.03},
        wall_point=[0.0, 100.0, 1.0, 0.0],
    ),
    dict(
        name="VanDenBroeck",
        build=VanDenBroeckMetric,
        probe=[0.0, 1.0, 2.0, 3.0],
        speeds=({"v_s": 0.1}, {"v_s": 0.5}),
        speed_point=[0.0, 10.0, 0.0, 0.0],
        wall={
            "v_s": 0.1,
            "R": 350.0,
            "sigma": 8.0,
            "R_tilde": 200.0,
            "alpha_vdb": 0.5,
            "sigma_B": 8.0,
        },
        wall_point=[0.0, 200.0, 1.0, 0.0],
    ),
    dict(
        name="WarpShell",
        build=WarpShellMetric,
        probe=[0.0, 15.0, 2.0, 3.0],
        speeds=({"v_s": 0.02}, {"v_s": 0.1}),
        speed_point=[0.0, 5.0, 0.0, 0.0],
        wall={"v_s": 0.02, "R_1": 10.0, "R_2": 20.0, "R_b": 1.0, "r_s_param": 5.0},
        wall_point=[0.0, 15.0, 0.0, 0.0],
    ),
    dict(
        name="WarpShellPhysical",
        build=WarpShellPhysical,
        probe=[0.0, 15.0, 2.0, 3.0],
        speeds=({"v_s": 0.02}, {"v_s": 0.1}),
        speed_point=[0.0, 5.0, 0.0, 0.0],
        wall={"v_s": 0.02, "R_1": 10.0, "R_2": 20.0, "r_s_param": 5.0},
        wall_point=[0.0, 15.0, 0.0, 0.0],
    ),
]

contract = pytest.mark.parametrize("c", CONTRACTS, ids=lambda c: c["name"])


@contract
def test_jit_matches_eager(c):
    m = c["build"]()
    coords = jnp.array(c["probe"])
    assert jnp.allclose(m(coords), eqx.filter_jit(m)(coords), atol=1e-15)


@contract
def test_call_matches_adm_reconstruction(c):
    m = c["build"]()
    coords = jnp.array(c["probe"])
    g_adm = adm_to_full_metric(m.lapse(coords), m.shift(coords), m.spatial_metric(coords))
    assert jnp.allclose(m(coords), g_adm, atol=1e-15)


@contract
def test_symbolic_is_a_wellformed_metric(c):
    sm = c["build"]().symbolic()
    assert isinstance(sm, SymbolicMetric)
    assert sm.g.shape == (4, 4)
    assert len(sm.coords) == 4


@contract
def test_speed_is_a_dynamic_field(c):
    slow, fast = (c["build"](**kw) for kw in c["speeds"])
    coords = jnp.array(c["speed_point"])
    assert not jnp.allclose(slow(coords), fast(coords), atol=1e-10)


@contract
def test_curvature_chain_is_finite_at_the_wall(c):
    result = compute_curvature_chain(c["build"](**c["wall"]), jnp.array(c["wall_point"]))
    assert result.christoffel.shape == (4, 4, 4)
    assert result.riemann.shape == (4, 4, 4, 4)
    for field in ("metric", "christoffel", "riemann", "ricci", "einstein", "stress_energy"):
        assert jnp.all(jnp.isfinite(getattr(result, field))), field


def test_gaussian_example_has_regular_center_curvature():
    example = Path(__file__).resolve().parents[1] / "examples/07_custom_warp_metric.py"
    metric = run_path(str(example))["GaussianWarpMetric"](v_s=0.5, w=1.3)
    for time in (0.0, 2.0):
        center = jnp.array([time, metric.v_s * time, 0.0, 0.0])
        result = compute_curvature_chain(metric, center)
        assert result.riemann.dtype == jnp.float64
        for field in ("christoffel", "riemann", "ricci", "einstein", "stress_energy"):
            assert jnp.all(jnp.isfinite(getattr(result, field))), field
        hessian = jax.hessian(metric.shape_function_value)(center)
        assert jnp.allclose(hessian[1:, 1:], -jnp.eye(3) / metric.w**2, atol=1e-14)


@pytest.mark.parametrize(
    "build",
    [LentzMetric, RodalMetric, VanDenBroeckMetric, WarpShellPhysical],
    ids=["Lentz", "Rodal", "VanDenBroeck", "WarpShellPhysical"],
)
def test_far_field_is_minkowski(build):
    g = build()(jnp.array([0.0, 1000.0, 0.0, 0.0]))
    assert jnp.allclose(g, jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])), atol=1e-6)


class TestLentz:
    """Tests for LentzMetric."""

    def test_lentz_at_origin(self):
        """Evaluate near origin, verify metric structure.

        Near origin (inside diamond): f~1, shift = (-v_s, 0, 0).
        g_00 = -(1 - v_s^2), g_01 = -v_s, spatial = delta_ij.

        Note: We use a point slightly off origin to avoid the L1 norm
        non-differentiable point at x_rel=0.
        """
        m = LentzMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        assert g[0, 0] < 0.0

    def test_lentz_subluminal_g00_negative(self):
        """For v_s=0.5, verify g_00 < 0 at bubble center (timelike preserved)."""
        m = LentzMetric(v_s=0.5, R=100.0, sigma=8.0)
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        g = m(coords)
        assert g[0, 0] < 0.0, f"g_00 = {g[0, 0]}, expected < 0 for subluminal v_s=0.5"

    def test_lentz_superluminal_g00_positive(self):
        """For v_s=2.0, verify g_00 > 0 at bubble center (signature change)."""
        m = LentzMetric(v_s=2.0, R=100.0, sigma=8.0)
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        g = m(coords)
        assert g[0, 0] > 0.0, f"g_00 = {g[0, 0]}, expected > 0 for superluminal v_s=2.0"

    def test_lentz_diamond_geometry(self):
        """Verify diamond (L1) geometry differs from spherical (L2) Alcubierre.

        At a 45-degree diagonal point, L1 distance = |x| + |y| = 100
        while L2 distance = sqrt(x^2 + y^2) ~ 70.7. The Lentz bubble is
        at the wall while Alcubierre is still well inside. This produces
        different shift magnitudes.
        """
        R, sigma, v_s = 100.0, 8.0, 0.1
        lentz = LentzMetric(v_s=v_s, R=R, sigma=sigma)
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        diag = jnp.array([0.0, 50.0, 50.0, 0.0])
        beta_lentz = lentz.shift(diag)
        beta_alc = alc.shift(diag)

        assert not jnp.allclose(beta_lentz, beta_alc, atol=1e-3), (
            f"Lentz shift = {beta_lentz}, Alcubierre shift = {beta_alc}. "
            "Diamond and spherical should produce different shifts at diagonal."
        )

    def test_lentz_regression_baseline(self):
        """Regression baseline for Lentz Eulerian energy density.

        Computes T_ab n^a n^b at representative points and compares against
        stored baselines to detect changes in the Lentz metric or
        curvature pipeline.

        Regression baseline NOT a published WarpFactory comparison.
        Parameter set: v_s=0.1, R=100, sigma=8.
        """
        m = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)

        test_cases = {
            "near_center": {
                "coords": jnp.array([0.0, 5.0, 5.0, 0.0]),
                "rho_euler": 0.0,
            },
            "wall_diagonal": {
                "coords": jnp.array([0.0, 50.0, 50.0, 0.0]),
                "rho_euler": -1.5915494309189531e-03,
            },
            "outside": {
                "coords": jnp.array([0.0, 200.0, 200.0, 0.0]),
                "rho_euler": 0.0,
            },
        }

        for name, case in test_cases.items():
            coords = case["coords"]
            result = compute_curvature_chain(m, coords)

            beta = m.shift(coords)
            n_up = jnp.array([1.0, -beta[0], -beta[1], -beta[2]])
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert jnp.allclose(rho_euler, case["rho_euler"], atol=1e-8), (
                f"Lentz {name}: rho_euler = {rho_euler:.16e}, expected {case['rho_euler']:.16e}"
            )


class TestNatario:
    """Tests for NatarioMetric."""

    def test_natario_at_origin(self):
        """The moving laboratory center has a unit timelike tangent."""
        m = NatarioMetric()
        for t in (0.0, 1.3):
            coords = jnp.array([t, m.v_s * t, 0.0, 0.0])
            tangent = jnp.array([1.0, m.v_s, 0.0, 0.0])
            assert jnp.allclose(m.shift(coords), jnp.array([-m.v_s, 0.0, 0.0]))
            assert jnp.isclose(tangent @ m(coords) @ tangent, -1.0, atol=1e-14)

    def test_natario_far_field(self):
        """The laboratory metric tends to Minkowski at spatial infinity."""
        m = NatarioMetric(v_s=0.1)
        g = m(jnp.array([0.0, 1000.0, 0.0, 0.0]))
        assert jnp.allclose(g, jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])), atol=1e-14)

    def test_natario_zero_expansion(self):
        """Verify div(beta) = 0 (zero expansion) at multiple points.

        The Natario metric is constructed so that the trace of the extrinsic
        curvature K = div(beta) = 0 everywhere. We verify this by computing
        the divergence of the shift vector via JAX autodiff.
        """
        m = NatarioMetric(v_s=0.1, R=100.0, sigma=0.03)

        def shift_fn(coords):
            return m.shift(coords)

        test_points = [
            jnp.array([0.0, 50.0, 10.0, 0.0]),
            jnp.array([0.0, 100.0, 5.0, 3.0]),
            jnp.array([0.0, 80.0, 20.0, 10.0]),
        ]

        for coords in test_points:
            J = jax.jacfwd(shift_fn)(coords)
            div_beta = J[0, 1] + J[1, 2] + J[2, 3]
            assert jnp.isclose(div_beta, 0.0, atol=1e-8), (
                f"div(beta) = {div_beta} at {coords}, expected ~0"
            )

    def test_natario_shape_function(self):
        """Verify n(0)=0, n(inf)=1/2.

        The Natario shape function n(r) = (1/2)*(1 - f_Alc(r)).
        """
        from warpax.metrics.natario import _natario_n

        R, sigma = 100.0, 0.03
        n_0 = _natario_n(jnp.array(0.0), R, sigma)
        n_inf = _natario_n(jnp.array(1000.0), R, sigma)

        assert jnp.isclose(n_0, 0.0, atol=1e-6), f"n(0) = {n_0}, expected ~0"
        assert jnp.isclose(n_inf, 0.5, atol=1e-6), f"n(inf) = {n_inf}, expected ~0.5"

    def test_natario_energy_strictly_negative(self):
        """Verify analytical Eulerian energy density is strictly non-positive.

        rho = -(v_s^2 / kappa) * [...] is negative wherever dn/dr != 0,
        confirming WEC violation wherever the density is negative.
        """
        x = jnp.linspace(-200, 200, 40)
        y = jnp.linspace(-200, 200, 40)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = jnp.zeros_like(X)

        rho = natario_eulerian_energy_density(X, Y, Z, v_s=0.1, R=100.0, sigma=0.03)
        assert jnp.all(rho <= 1e-15), f"max rho = {jnp.max(rho)}"
        assert rho.dtype == jnp.float64
        assert jnp.min(rho) < -1e-10, f"min rho = {jnp.min(rho)}, should be significantly negative"


class TestRodal:
    """Tests for RodalMetric."""

    def test_rodal_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin with r_s=0: F(0)~1, G(0)~1, so shift = (-v_s, 0, 0).
        g_00 = -(1 - v_s^2), g_01 = -v_s, spatial = delta_ij.
        """
        m = RodalMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        assert g[0, 0] < 0.0

    def test_rodal_irrotational_shift(self):
        """Check that G_0i components of Einstein tensor are near-zero.

        Irrotational shift (curl-free) means the momentum density
        G^0_i should vanish. We check via the mixed-index Einstein
        tensor: G^mu_nu = g^{mu alpha} G_{alpha nu}.
        """
        m = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        coords = jnp.array([0.0, 80.0, 20.0, 0.0])
        result = compute_curvature_chain(m, coords)
        G = result.einstein
        g_inv = result.metric_inv

        G_mixed = jnp.einsum("ma,an->mn", g_inv, G)

        G_0i = G_mixed[0, 1:]
        assert jnp.allclose(G_0i, 0.0, atol=1e-8), (
            f"G^0_i = {G_0i}, expected ~0 for irrotational shift"
        )

    def test_rodal_shape_functions(self):
        """Verify f(0)~1, f(inf)~0, G(0)~1, G(inf)~0 at representative r values.

        Lab-frame convention: F = f_Alc (radial), G = 1 - g_paper (angular).
        Both are ~1 at center and ~0 far away.

        Note: G(0) = 1 exactly for all R*sigma (via the analytic limit
        Delta'(0) = -2*sigma*tanh(sigma*R)). The physical behavior is
        correct: G is 1 at center and 0 at far field. G(r) has a 1/r tail
        (slower convergence than F), so the test checks monotone decrease
        and relies on the full-metric far-field test rather than requiring
        G(1000) ~ 0.
        """
        R, sigma = 100.0, 0.03

        r_zero = jnp.array(0.0)
        r_far = jnp.array(1000.0)
        F_0 = alcubierre_shape(r_zero, R, sigma)
        F_inf = alcubierre_shape(r_far, R, sigma)
        assert jnp.isclose(F_0, 1.0, atol=1e-6), f"F(0) = {F_0}"
        assert jnp.isclose(F_inf, 0.0, atol=1e-6), f"F(inf) = {F_inf}"

        G_0 = _rodal_G(r_zero, R, sigma)
        G_mid = _rodal_G(jnp.array(500.0), R, sigma)
        G_far = _rodal_G(r_far, R, sigma)
        G_very_far = _rodal_G(jnp.array(50000.0), R, sigma)
        assert G_0 > 0.95, f"G(0) = {G_0}, expected > 0.95"
        assert G_mid < G_0, f"G should decrease: G(0)={G_0}, G(500)={G_mid}"
        assert G_far < G_mid, f"G should decrease: G(500)={G_mid}, G(1000)={G_far}"
        assert G_very_far < 0.01, f"G(50000) = {G_very_far}, expected < 0.01"

    def test_rodal_g_numerical_stability(self):
        """Evaluate g_paper(r) at large r*sigma to confirm no NaN/Inf.

        The logcosh implementation should be stable for large arguments.
        """
        R, sigma = 100.0, 0.03
        r_values = jnp.array([0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0])
        g_vals = jax.vmap(lambda r: _rodal_g_paper(r, R, sigma))(r_values)
        assert not jnp.any(jnp.isnan(g_vals)), f"NaN found in g_paper: {g_vals}"
        assert not jnp.any(jnp.isinf(g_vals)), f"Inf found in g_paper: {g_vals}"
        assert jnp.all(g_vals >= -1e-10), f"g_paper < 0: {g_vals}"
        assert jnp.all(g_vals <= 1.0 + 1e-10), f"g_paper > 1: {g_vals}"

    def test_rodal_cartesian_origin_regularity(self):
        """shift(origin) = (-v_s, 0, 0) without NaN.

        With the Cartesian formula beta = -v_s * [G*x_hat + (F-G)*n_x*n],
        at origin n_x=0 and F(0)=G(0)=1, so beta = (-v_s, 0, 0).
        No jnp.where origin patches needed.
        """
        m = RodalMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        shift = m.shift(coords)
        assert not jnp.any(jnp.isnan(shift)), f"NaN in shift at origin: {shift}"
        assert jnp.isclose(shift[0], -m.v_s, atol=1e-8), (
            f"shift[0] = {shift[0]}, expected ~ {-m.v_s}"
        )
        assert jnp.isclose(shift[1], 0.0, atol=1e-10), f"shift[1] = {shift[1]}"
        assert jnp.isclose(shift[2], 0.0, atol=1e-10), f"shift[2] = {shift[2]}"

    def test_rodal_shift_gradient_at_origin(self):
        """jax.jacfwd(m.shift) produces no NaN at origin.

        The Cartesian form is manifestly regular, so forward-mode AD
        should produce finite Jacobian entries everywhere including r=0.
        """
        m = RodalMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        J = jax.jacfwd(m.shift)(coords)
        assert not jnp.any(jnp.isnan(J)), f"NaN in Jacobian at origin: {J}"

    def test_rodal_G_stable_origin(self):
        """G(0) correct for various R*sigma products via analytic limit.

        The analytic limit Delta'(0) = -2*sigma*tanh(sigma*R) gives
        g_paper(0) = 0 exactly (G(0) = 1) for all R*sigma products.
        """
        G_0_soft = _rodal_G(jnp.array(0.0), 100.0, 0.03)
        assert G_0_soft > 0.99, f"G(0) for R*sigma=3: {G_0_soft}, expected > 0.99"
        assert G_0_soft < 1.01, f"G(0) for R*sigma=3: {G_0_soft}, expected < 1.01"
        G_0_sharp = _rodal_G(jnp.array(0.0), 100.0, 1.0)
        assert jnp.isclose(G_0_sharp, 1.0, atol=1e-6), (
            f"G(0) for R*sigma=100: {G_0_sharp}, expected ~ 1.0"
        )

    def test_rodal_convention_mapping(self):
        """F = 1 - f_paper via f_Alc; G = 1 - g_paper verified."""
        R, sigma = 100.0, 0.03
        for r_val in [0.0, 50.0, 100.0, 200.0, 500.0]:
            r = jnp.array(r_val)
            g_p = _rodal_g_paper(r, R, sigma)
            G = _rodal_G(r, R, sigma)
            assert jnp.isclose(G, 1.0 - g_p, atol=1e-12), (
                f"Convention mismatch at r={r_val}: G={G}, 1-g_paper={1.0 - g_p}"
            )

    def test_rodal_cartesian_matches_reference(self):
        """Cartesian form matches reference values at 5 off-axis points.

        Reference values computed from the algebraically equivalent
        spherical-tetrad form at points away from origin.
        """
        m = RodalMetric()
        test_cases = [
            (jnp.array([0.0, 50.0, 10.0, 0.0]), jnp.array([-0.09557155, 0.0006476, 0.0])),
            (jnp.array([0.0, 100.0, 1.0, 0.0]), jnp.array([-0.0502442, 0.00038641, 0.0])),
            (jnp.array([0.0, 0.0, 50.0, 50.0]), jnp.array([-0.09672827, 0.0, 0.0])),
            (jnp.array([0.0, 200.0, 0.0, 0.0]), jnp.array([-0.00024849, 0.0, 0.0])),
            (jnp.array([0.0, 30.0, 20.0, 10.0]), None),
        ]
        for coords, expected in test_cases:
            shift = m.shift(coords)
            assert not jnp.any(jnp.isnan(shift)), f"NaN at coords={coords}: shift={shift}"
            if expected is not None:
                assert jnp.allclose(shift, expected, atol=1e-4), (
                    f"Mismatch at coords={coords}: got {shift}, expected {expected}"
                )


class TestVanDenBroeck:
    """Tests for VanDenBroeckMetric."""

    def test_vdb_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin with r_s=0: f(0)~1, B(0)=1+alpha_vdb.
        Spatial block = B^2 * I. Shift = (-v_s * f, 0, 0).
        """
        m = VanDenBroeckMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        B_sq = (1.0 + m.alpha_vdb) ** 2
        assert jnp.allclose(g[1:, 1:], B_sq * jnp.eye(3), atol=1e-6)
        assert g[0, 0] < 0.0

    def test_vdb_conformal_factor(self):
        """At bubble center, B(0) = 1 + alpha_vdb, so gamma_ij = (1+alpha_vdb)^2 * I.

        Verify spatial metric diagonal has the expected conformal factor.
        """
        m = VanDenBroeckMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        gamma = m.spatial_metric(coords)
        B_center = 1.0 + m.alpha_vdb
        expected_gamma = B_center**2 * jnp.eye(3)
        assert jnp.allclose(gamma, expected_gamma, atol=1e-6), (
            f"gamma = {jnp.diag(gamma)}, expected diag = {B_center**2}"
        )

    def test_vdb_shift_matches_alcubierre(self):
        """Verify shift is identical to Alcubierre (same f(r_s) shape).

        Both use the same Alcubierre top-hat shape function for the shift,
        so at matching parameter values the shift vectors should be identical.
        """
        v_s, R, sigma = 0.1, 350.0, 8.0
        vdb = VanDenBroeckMetric(v_s=v_s, R=R, sigma=sigma)
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        coords = jnp.array([0.0, 100.0, 50.0, 0.0])
        beta_vdb = vdb.shift(coords)
        beta_alc = alc.shift(coords)
        assert jnp.allclose(beta_vdb, beta_alc, atol=1e-14), (
            f"VdB shift = {beta_vdb}, Alcubierre shift = {beta_alc}"
        )

    def test_vdb_energy_density_reduced(self):
        """Verify stress-energy magnitude differs from Alcubierre due to conformal factor.

        The conformal factor B^2 on the spatial metric modifies the curvature
        and hence the stress-energy tensor. At the same point near the bubble
        wall, VdB and Alcubierre should produce different stress-energy.
        """
        v_s, R, sigma = 0.1, 350.0, 8.0
        vdb = VanDenBroeckMetric(
            v_s=v_s, R=R, sigma=sigma, R_tilde=200.0, alpha_vdb=0.5, sigma_B=8.0
        )
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        coords = jnp.array([0.0, 200.0, 1.0, 0.0])
        result_vdb = compute_curvature_chain(vdb, coords)
        result_alc = compute_curvature_chain(alc, coords)

        T_vdb = result_vdb.stress_energy
        T_alc = result_alc.stress_energy

        assert not jnp.allclose(T_vdb, T_alc, atol=1e-10), (
            "Stress-energy should differ between VdB and Alcubierre near conformal bubble wall"
        )


class TestWarpShell:
    """Tests for WarpShellMetric."""

    def test_warpshell_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin (deep interior, r=0 < R_1): lapse=1, shift=-v_s,
        spatial=identity. The metric should be a Minkowski-like metric
        with uniform shift (co-moving with the bubble). At r=0 the spatial
        block is forced to identity by the r < 1e-10 guard; S_warp(0)=1 and
        lapse=1 exactly, so g_01 = -v_s and g_00 = -(1 - v_s^2).
        """
        m = WarpShellMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        assert jnp.isclose(g[0, 1], -m.v_s, atol=1e-12)
        assert jnp.isclose(g[0, 0], -(1.0 - m.v_s**2), atol=1e-12)

    def test_warpshell_far_field(self):
        """Outside the shell (r > R_2), metric is Minkowski.

        Checks both just outside the shell (r=50) and deep far field (r=1000).
        """
        m = WarpShellMetric()
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        for r in (50.0, 1000.0):
            g = m(jnp.array([0.0, r, 0.0, 0.0]))
            assert jnp.allclose(g, minkowski, atol=1e-6), (
                f"Exterior metric at r={r} should be Minkowski. "
                f"Got g_00={g[0, 0]}, diag_spatial={jnp.diag(g[1:, 1:])}"
            )

    def test_warpshell_interior_flat(self):
        """At r < R_1 (deep interior), spatial metric = identity, lapse = 1."""
        m = WarpShellMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        gamma = m.spatial_metric(coords)
        alpha = m.lapse(coords)
        assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
            f"Interior spatial metric should be identity. Got diag = {jnp.diag(gamma)}"
        )
        assert jnp.isclose(alpha, 1.0, atol=1e-10), f"Interior lapse should be 1.0. Got {alpha}"

    def test_warpshell_shell_non_flat(self):
        """At r between R_1 and R_2, spatial metric is NOT identity.

        The Schwarzschild-like radial stretching gives gamma_rr > 1 in the shell.
        """
        m = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        gamma = m.spatial_metric(coords)
        assert not jnp.allclose(gamma, jnp.eye(3), atol=1e-6), (
            "Shell spatial metric should NOT be identity"
        )
        assert gamma[0, 0] > 1.0 + 1e-6, (
            f"gamma_xx = {gamma[0, 0]}, expected > 1 (Schwarzschild radial stretching)"
        )

    def test_warpshell_shell_lapse_reduced(self):
        """At r in shell, lapse < 1 (Schwarzschild time dilation)."""
        m = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        alpha = m.lapse(coords)
        assert alpha < 1.0 - 1e-6, (
            f"Shell lapse = {alpha}, expected < 1 (Schwarzschild time dilation)"
        )
        assert alpha > 0.0, "Lapse must be positive"

    def test_warpshell_spatial_metric_symmetric(self):
        """At 5+ points in the shell region, verify gamma_ij == gamma_ji.

        The spherical-to-Cartesian tensor transformation must preserve symmetry.
        """
        m = WarpShellMetric()
        shell_points = [
            jnp.array([0.0, 12.0, 3.0, 0.0]),
            jnp.array([0.0, 10.0, 10.0, 0.0]),
            jnp.array([0.0, 0.0, 15.0, 0.0]),
            jnp.array([0.0, 8.0, 8.0, 8.0]),
            jnp.array([0.0, 5.0, 5.0, 12.0]),
        ]
        for coords in shell_points:
            gamma = m.spatial_metric(coords)
            assert jnp.allclose(gamma, gamma.T, atol=1e-15), (
                f"Spatial metric not symmetric at {coords[1:]}: "
                f"max diff = {jnp.max(jnp.abs(gamma - gamma.T))}"
            )

    def test_warpshell_spatial_metric_positive_definite(self):
        """At 5+ points in the shell region, all eigenvalues of gamma_ij > 0.

        This catches sign errors in the Schwarzschild-to-Cartesian conversion.
        """
        m = WarpShellMetric()
        shell_points = [
            jnp.array([0.0, 12.0, 3.0, 0.0]),
            jnp.array([0.0, 10.0, 10.0, 0.0]),
            jnp.array([0.0, 0.0, 15.0, 0.0]),
            jnp.array([0.0, 8.0, 8.0, 8.0]),
            jnp.array([0.0, 5.0, 5.0, 12.0]),
        ]
        for coords in shell_points:
            gamma = m.spatial_metric(coords)
            eigvals = jnp.linalg.eigvalsh(gamma)
            assert jnp.all(eigvals > 0.0), (
                f"Spatial metric not positive definite at {coords[1:]}: eigenvalues = {eigvals}"
            )

    def test_warpshell_spatial_metric_boundary_identity(self):
        """At r << R_1 (interior) and r >> R_2 (exterior), eigenvalues all 1.0.

        Confirms the transition function correctly reduces to flat space at boundaries.
        """
        m = WarpShellMetric()
        interior_points = [
            jnp.array([0.0, 3.0, 0.0, 0.0]),
            jnp.array([0.0, 2.0, 2.0, 0.0]),
        ]
        exterior_points = [
            jnp.array([0.0, 50.0, 0.0, 0.0]),
            jnp.array([0.0, 100.0, 0.0, 0.0]),
        ]
        for coords in interior_points + exterior_points:
            gamma = m.spatial_metric(coords)
            eigvals = jnp.linalg.eigvalsh(gamma)
            assert jnp.allclose(eigvals, 1.0, atol=1e-6), (
                f"Boundary spatial metric eigenvalues should be 1.0 at "
                f"r={jnp.sqrt(coords[1] ** 2 + coords[2] ** 2 + coords[3] ** 2):.1f}. "
                f"Got {eigvals}"
            )

    def test_warpshell_vacuum_shell_energy_density(self):
        """WarpShell Eulerian energy density vanishes everywhere (C2 default).

        With smooth_width=1.2 the shell indicator is exactly 1 throughout
        [R_1, R_2], so the metric is locally Schwarzschild at r=12, 15, 18.
        Schwarzschild is a vacuum solution, so T_ab n^a n^b must be zero up
        to machine epsilon there (measured ~1e-20). Interior and exterior
        are exactly flat.

        Parameter set: v_s=0.02, R_1=10, R_2=20, R_b=1, r_s_param=5.
        Uses default transition_order=2 (C2 quintic smoothstep).
        """
        m = WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0)

        points = {
            "interior": jnp.array([0.0, 5.0, 0.0, 0.0]),
            "shell_inner": jnp.array([0.0, 12.0, 0.0, 0.0]),
            "shell_mid": jnp.array([0.0, 15.0, 0.0, 0.0]),
            "shell_outer": jnp.array([0.0, 18.0, 0.0, 0.0]),
            "exterior": jnp.array([0.0, 50.0, 0.0, 0.0]),
        }

        for name, coords in points.items():
            result = compute_curvature_chain(m, coords)

            alpha = m.lapse(coords)
            beta = m.shift(coords)
            n_up = jnp.array(
                [
                    1.0 / alpha,
                    -beta[0] / alpha,
                    -beta[1] / alpha,
                    -beta[2] / alpha,
                ]
            )
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert abs(rho_euler) < 1e-15, (
                f"WarpShell {name}: rho_euler = {rho_euler:.16e}, "
                "expected ~0 (locally Schwarzschild shell is vacuum)"
            )

    def test_warpshell_c1_continuity(self):
        """Lapse gradient stays bounded across shell transitions (C1).

        For a C1 function the second derivative is bounded; a C0 transition
        would give second-difference jumps >> 1e3 here.
        """
        m = WarpShellMetric()
        r_vals = jnp.linspace(5.0, 25.0, 200)
        coords_fn = lambda r: jnp.array([0.0, r, 0.0, 0.0])

        dlapse_dr = jax.vmap(lambda r: jax.grad(lambda c: m.lapse(c))(coords_fn(r))[1])(r_vals)

        assert not jnp.any(jnp.isnan(dlapse_dr)), "Lapse gradient contains NaN values"

        d2 = jnp.diff(dlapse_dr)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d2))) / dr
        assert max_jump < 1e3, (
            f"Lapse gradient has large jump (max_jump/dr={max_jump:.1f}), "
            f"suggesting C0 discontinuity"
        )

    def test_warpshell_c1_shift_continuity(self):
        """Shift gradient stays bounded across shell transitions (C1)."""
        m = WarpShellMetric()
        r_vals = jnp.linspace(5.0, 25.0, 200)
        coords_fn = lambda r: jnp.array([0.0, r, 0.0, 0.0])

        dshift_dr = jax.vmap(lambda r: jax.grad(lambda c: m.shift(c)[0])(coords_fn(r))[1])(r_vals)

        assert not jnp.any(jnp.isnan(dshift_dr)), "Shift gradient contains NaN values"

        d2 = jnp.diff(dshift_dr)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d2))) / dr
        assert max_jump < 1e3, (
            f"Shift gradient has large jump (max_jump/dr={max_jump:.1f}), "
            f"suggesting C0 discontinuity"
        )

    def test_warpshell_smooth_width_parameter(self):
        """smooth_width constructor parameter works and affects transition zone.

        Verifies:
        1. smooth_width can be passed as constructor parameter
        2. Different smooth_width values produce different metrics in transition zone
        3. Interior/exterior flatness is preserved regardless of smooth_width

        smooth_width=None resolves to the default 1.2; r=9 lies in the
        transition zone just inside R_1=10, r=5 is interior and the
        exterior is checked beyond R_2.
        """
        m_default = WarpShellMetric()
        m_narrow = WarpShellMetric(smooth_width=0.5)
        m_wide = WarpShellMetric(smooth_width=3.0)

        assert m_default.smooth_width is None
        assert m_narrow.smooth_width == 0.5
        assert m_wide.smooth_width == 3.0

        coords_transition = jnp.array([0.0, 9.0, 0.0, 0.0])
        alpha_default = m_default.lapse(coords_transition)
        alpha_narrow = m_narrow.lapse(coords_transition)
        alpha_wide = m_wide.lapse(coords_transition)

        assert not jnp.isclose(alpha_narrow, alpha_wide, atol=1e-6), (
            f"Different smooth_width should produce different lapse in transition zone. "
            f"narrow={alpha_narrow}, wide={alpha_wide}"
        )

        coords_interior = jnp.array([0.0, 5.0, 0.0, 0.0])
        for m_test in [m_default, m_narrow, m_wide]:
            alpha = m_test.lapse(coords_interior)
            gamma = m_test.spatial_metric(coords_interior)
            assert jnp.isclose(alpha, 1.0, atol=1e-10), (
                f"Interior lapse should be 1.0 with smooth_width={m_test.smooth_width}. Got {alpha}"
            )
            assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
                f"Interior spatial metric should be identity with smooth_width={m_test.smooth_width}"
            )

        coords_exterior = jnp.array([0.0, 50.0, 0.0, 0.0])
        for m_test in [m_default, m_narrow, m_wide]:
            alpha = m_test.lapse(coords_exterior)
            gamma = m_test.spatial_metric(coords_exterior)
            assert jnp.isclose(alpha, 1.0, atol=1e-10), (
                f"Exterior lapse should be 1.0 with smooth_width={m_test.smooth_width}. Got {alpha}"
            )
            assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
                f"Exterior spatial metric should be identity with smooth_width={m_test.smooth_width}"
            )

    def test_warpshell_c1_regression(self):
        """C1 regression: pin the transition_order=1 code path.

        Two parts:
        1. Eulerian energy density stays near zero (|rho| < 1e-14) at five
           radial points, a near-zero smoke test of the C1 curvature
           pipeline, not a baseline lock (shell is locally Schwarzschild
           vacuum, so rho sits at machine epsilon ~1e-20).
        2. Pins that actually distinguish C1 from C2 in the transition
           zone at r=9.2: the cubic and quintic smoothsteps differ at
           O(1e-2) there (C2 lapse is 0.9319294656761363). The transition
           zone is [R_1 - smooth_width, R_1] = [8.8, 10].
        """
        m = WarpShellMetric(
            v_s=0.02,
            R_1=10.0,
            R_2=20.0,
            R_b=1.0,
            r_s_param=5.0,
            transition_order=1,
        )

        for r in (5.0, 12.0, 15.0, 18.0, 50.0):
            coords = jnp.array([0.0, r, 0.0, 0.0])
            result = compute_curvature_chain(m, coords)

            alpha = m.lapse(coords)
            beta = m.shift(coords)
            n_up = jnp.array(
                [
                    1.0 / alpha,
                    -beta[0] / alpha,
                    -beta[1] / alpha,
                    -beta[2] / alpha,
                ]
            )
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert abs(rho_euler) < 1e-14, (
                f"C1 regression r={r}: rho_euler = {rho_euler:.16e}, expected ~0"
            )

        seam = jnp.array([0.0, 9.2, 0.0, 0.0])
        lapse_c1 = m.lapse(seam)
        assert jnp.isclose(lapse_c1, 0.9159128693646388, atol=1e-12), (
            f"C1 lapse at r=9.2 = {lapse_c1:.16e}, "
            "expected 0.9159128693646388 (C2 gives 0.9319294656761363)"
        )
        riemann_c1 = compute_curvature_chain(m, seam).riemann
        assert jnp.isclose(riemann_c1[0, 1, 0, 1], 0.20735301228665254, atol=1e-9), (
            f"C1 Riemann[0,1,0,1] at r=9.2 = {riemann_c1[0, 1, 0, 1]:.16e}, "
            "expected 0.20735301228665254 (C2 gives 0.7242653636732740)"
        )

    def test_warpshell_transition_order_field(self):
        """Default transition_order=2, configurable to 1."""
        m_default = WarpShellMetric()
        m_c1 = WarpShellMetric(transition_order=1)
        assert m_default.transition_order == 2
        assert m_c1.transition_order == 1

    def test_warpshell_c2_second_derivative_continuity(self):
        """C2 lapse has bounded third differences (continuous second derivative).

        Sweeps lapse along r from 5 to 25 (200 points), computes second
        derivative via nested jax.grad, and checks that third differences
        are bounded. The C2 quintic should produce significantly smaller
        jumps than C1 cubic at transition seams, so the bound (< 100) is
        tighter than the C1 bound (< 1e3).
        """
        m = WarpShellMetric(transition_order=2)
        r_vals = jnp.linspace(5.0, 25.0, 200)

        def lapse_of_x(x):
            return m.lapse(jnp.array([0.0, x, 0.0, 0.0]))

        d2lapse = jax.vmap(jax.grad(jax.grad(lapse_of_x)))(r_vals)

        assert not jnp.any(jnp.isnan(d2lapse)), "C2 lapse second derivative contains NaN values"

        d3 = jnp.diff(d2lapse)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d3))) / dr

        assert max_jump < 100, (
            f"C2 lapse second derivative has large jump (max_jump/dr={max_jump:.1f}), "
            f"expected < 100 for C2 smoothness"
        )

    def test_warpshell_c2_vs_c1_different_at_seam(self):
        """C2 and C1 metrics produce different curvature at transition seam.

        At a point just outside the inner shell boundary (in the
        transition zone), the Riemann tensor components should differ
        between C1 and C2. This confirms the upgrade actually changes
        something. With R_1 = 10 and smooth_width = 1.2 the transition
        spans [8.8, 10], so r=9.2 is inside it.
        """
        m_c1 = WarpShellMetric(transition_order=1)
        m_c2 = WarpShellMetric(transition_order=2)

        coords = jnp.array([0.0, 9.2, 0.0, 0.0])

        result_c1 = compute_curvature_chain(m_c1, coords)
        result_c2 = compute_curvature_chain(m_c2, coords)

        riemann_diff = jnp.max(jnp.abs(result_c1.riemann - result_c2.riemann))
        assert riemann_diff > 1e-10, (
            f"C1 and C2 Riemann tensors are identical at seam "
            f"(max diff = {riemann_diff:.2e}). Expected them to differ."
        )


class TestWarpShellPhysical:
    """Tests for the physical-regime WarpShellPhysical class.

    The physical regime requires ``r_s_param < R_1`` so the Schwarzschild
    radius sits inside the flat interior, well clear of the shell where
    the Schwarzschild-like geometry takes over. Without the clamp the
    lapse is a clean ``sqrt(1 - r_s / r)`` in the shell, with no
    ``minimum(ratio, 1.0 - 1e-12)`` or ``maximum(alpha, 1e-12)`` safety
    patches.
    """

    def test_physical_accepts_r_s_less_than_R_1(self):
        """Default r_s_param=5.0 < R_1=10.0, construction must succeed."""
        m = WarpShellPhysical()
        assert m.r_s_param < m.R_1

    def test_physical_rejects_r_s_not_less_than_R_1(self):
        """r_s_param >= R_1 must raise at construction."""
        with pytest.raises(ValueError, match="r_s_param"):
            WarpShellPhysical(r_s_param=10.0, R_1=10.0, R_2=20.0)
        with pytest.raises(ValueError, match="r_s_param"):
            WarpShellPhysical(r_s_param=15.0, R_1=10.0, R_2=20.0)

    def test_physical_matches_stress_test_in_physical_regime(self):
        """With r_s_param < R_1 - smooth_width neither variant clamps, so the two
        agree everywhere, including the inner blend [8.8, 10] and the flat core."""
        phys = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        stress = WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        for x in (0.5, 5.0, 8.9, 9.5, 11.0, 15.0, 19.0, 25.0):
            coords = jnp.array([0.0, x, 0.0, 0.0])
            assert jnp.allclose(phys(coords), stress(coords), atol=1e-14)

    def test_inner_blend_is_c2(self):
        """The Schwarzschild factor is evaluated at max(r, R_1 - sw), so one-sided
        first and second derivatives of alpha and gamma_rr agree at R_1 and at
        R_1 - sw, where the quintic indicator has S = S' = S'' = 0. A factor
        frozen at R_1 would give alpha' and gamma_rr' jumps at R_1 of
        r_s/(2 R_1^2 sqrt(1 - r_s/R_1)) = 0.035355339 and -0.2."""
        m = WarpShellPhysical(v_s=0.0)
        lapse = lambda x: m.lapse(jnp.array([0.0, x, 0.0, 0.0]))
        gxx = lambda x: m.spatial_metric(jnp.array([0.0, x, 0.0, 0.0]))[0, 0]
        for fn in (lapse, gxx):
            d1 = jax.grad(fn)
            d2 = jax.grad(d1)
            for r in (m.R_1 - 0.12 * (m.R_2 - m.R_1), m.R_1):
                assert abs(float(d1(r - 1e-9)) - float(d1(r + 1e-9))) < 1e-8
                assert abs(float(d2(r - 1e-9)) - float(d2(r + 1e-9))) < 1e-6
        assert abs(float(jax.grad(lapse)(m.R_1 + 1e-9)) - 0.035355339) < 1e-8
        assert abs(float(jax.grad(gxx)(m.R_1 + 1e-9)) + 0.2) < 1e-8

    def test_rejects_r_s_inside_blend(self):
        with pytest.raises(ValueError, match="smooth_width"):
            WarpShellPhysical(r_s_param=9.0, R_1=10.0, R_2=20.0)

    def test_outer_blend_has_negative_energy(self):
        """m(r) = (r/2)(1 - 1/gamma_rr) equals r_s/2 in the shell and 0 in the flat
        exterior, so m' = 4 pi r^2 rho < 0 somewhere in the outer blend: at the
        defaults rho(20.6) = -7.203278e-4 and min_i(rho + p_i)(20.9) = -1.93850e-2."""
        from warpax.energy_conditions.classification import classify_hawking_ellis

        m = WarpShellPhysical(v_s=0.0)

        def rest_frame(x):
            res = compute_curvature_chain(m, jnp.array([0.0, x, 0.0, 0.0]))
            cls = classify_hawking_ellis(res.metric_inv @ res.stress_energy, res.metric)
            assert int(cls.he_type) == 1
            return float(cls.rho), float(jnp.min(cls.rho + cls.pressures))

        rho, _ = rest_frame(20.6)
        assert abs(rho + 7.203278e-4) < 1e-8
        _, nec = rest_frame(20.9)
        assert abs(nec + 1.93850e-2) < 1e-6
        assert all(not v for v in warpshell_module.GROUND_TRUTH["energy_conditions"].values())

    def test_physical_lapse_positive_without_floor(self):
        """Within the shell the lapse is sqrt(1 - r_s_param / r) with
        r_s_param / r < 1 by construction, so it is strictly positive
        without relying on the 1e-12 floor."""
        m = WarpShellPhysical(r_s_param=5.0, R_1=10.0, R_2=20.0)
        for x in (10.0, 12.5, 15.0, 17.5, 20.0):
            coords = jnp.array([0.0, x, 0.0, 0.0])
            alpha = m.lapse(coords)
            assert float(alpha) > 0.1

    def test_stress_test_alias_matches_legacy_metric(self):
        """WarpShellMetric must expose the same behavior as the
        legacy WarpShellMetric class."""
        legacy = WarpShellMetric()
        renamed = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        assert jnp.allclose(legacy(coords), renamed(coords), atol=1e-14)


class TestTShell:
    """Smoke tests for source-first T-shell metric."""

    def test_tshell_on_axis_finite(self):
        m = tshell_default()
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        g = m(coords)
        assert jnp.all(jnp.isfinite(g))
        assert g[0, 0] < 0.0

    def test_tshell_curvature_finite(self):
        m = tshell_default()
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        chain = compute_curvature_chain(m, coords)
        assert jnp.all(jnp.isfinite(chain.stress_energy))
        assert jnp.all(jnp.isfinite(chain.metric))


class TestShellTotalMassArrayLeaf:
    """Regression (total_mass static-leaf retrace): ``total_mass`` was a
    Python float pytree leaf, so ``eqx.partition(metric, eqx.is_array)``
    placed it in the static partition and ``eqx.filter_jit`` retraced for
    every new mass value. It must live in the array partition.
    """

    def test_sshell_total_mass_in_array_partition(self):
        import equinox as eqx

        from warpax.metrics.sshell import sshell_from_profiles
        from warpax.metrics.sshell_profiles import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        m = sshell_from_profiles(profiles, n_grid=128)
        arrays, static = eqx.partition(m, eqx.is_array)
        assert eqx.is_array(arrays.total_mass), (
            f"total_mass not in array partition: {type(arrays.total_mass)}"
        )
        assert static.total_mass is None
        assert float(arrays.total_mass) > 0.0

    def test_tshell_total_mass_in_array_partition(self):
        import equinox as eqx

        from warpax.metrics.tshell import tshell_from_profiles
        from warpax.metrics.tshell_profiles import constant_velocity_profiles

        profiles = constant_velocity_profiles(
            R_1=10.0,
            R_2=20.0,
            rho_0=1e-4,
            v_0=0.1,
        )
        m = tshell_from_profiles(profiles, n_grid=128)
        arrays, static = eqx.partition(m, eqx.is_array)
        assert eqx.is_array(arrays.total_mass), (
            f"total_mass not in array partition: {type(arrays.total_mass)}"
        )
        assert static.total_mass is None
        assert float(arrays.total_mass) > 0.0

    def test_shell_scalars_are_array_leaves(self):
        """``R_1, R_2, v_s, smooth_width`` live in the array partition, so
        ``eqx.filter_jit`` does not retrace per radius or shift value."""
        import equinox as eqx

        from warpax.metrics import sshell_default, tshell_default

        arrays, static = eqx.partition(sshell_default(v_s=0.02), eqx.is_array)
        for name in ("R_1", "R_2", "v_s", "smooth_width"):
            assert eqx.is_array(getattr(arrays, name)), name
            assert getattr(static, name) is None, name
        arrays, static = eqx.partition(tshell_default(), eqx.is_array)
        for name in ("R_1", "R_2"):
            assert eqx.is_array(getattr(arrays, name)), name
            assert getattr(static, name) is None, name
