"""Independent checks of metric conventions, center limits, and smooth joins."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from warpax.geometry.metric import adm_to_full_metric
from warpax.metrics.natario import NatarioMetric, _natario_dn_dr, _natario_n
from warpax.metrics.rodal import RodalMetric


def test_shared_shape_cartesian_center_and_join():
    """The even profile has an isotropic Hessian and no artificial central source."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import compute_curvature_chain
    from warpax.metrics._common import alcubierre_shape

    for sigma in (1.0, 8.0):
        profile = lambda x: alcubierre_shape(jnp.sqrt(x @ x + 1e-60), 1.0, sigma)
        expected = -2 * sigma**2 / np.cosh(sigma) ** 2
        np.testing.assert_allclose(
            jax.hessian(profile)(jnp.zeros(3)), expected * np.eye(3), rtol=1e-12, atol=1e-15
        )
        # Both exact forms must agree in value and derivatives at their join.
        outer = lambda r: (
            (jnp.tanh(sigma * (r + 1)) - jnp.tanh(sigma * (r - 1))) / (2 * jnp.tanh(sigma))
        )
        actual = lambda r: alcubierre_shape(r, 1.0, sigma)
        for _ in range(3):
            for r in (0.499 / sigma, 0.501 / sigma):
                np.testing.assert_allclose(actual(r), outer(r), rtol=1e-8, atol=1e-14)
            actual, outer = jax.grad(actual), jax.grad(outer)
    metric = AlcubierreMetric()
    center = compute_curvature_chain(metric, jnp.zeros(4)).stress_energy
    nearby = compute_curvature_chain(metric, jnp.array([0.0, 1e-6, 0.0, 0.0])).stress_energy
    np.testing.assert_allclose(center, nearby, rtol=1e-8, atol=1e-14)


def test_rodal_origin_derivatives_and_irrotationality():
    """Check derivatives, not only finiteness, at the removable origin and join."""
    import mpmath as mp

    from warpax.metrics.rodal import _rodal_G

    R, sigma, v = 2.0, 1.5, 0.5
    metric = RodalMetric(R=R, sigma=sigma, v_s=v)
    h = -(sigma**2) / (3 * np.cosh(sigma * R) ** 2)
    expected = np.zeros((3, 3, 3))
    for a in range(3):
        for i in range(3):
            for j in range(3):
                expected[a, i, j] = (
                    -2 * v * h * ((a == 0) * (i == j) + (i == 0) * (a == j) + (j == 0) * (a == i))
                )
    actual = jax.jacfwd(jax.jacfwd(metric.shift))(jnp.zeros(4))[:, 1:, 1:]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-16)
    with mp.workdps(60):
        exact = lambda r: (
            (mp.log(mp.cosh(sigma * (r + R))) - mp.log(mp.cosh(sigma * (r - R))))
            / (2 * sigma * r * mp.tanh(sigma * R))
        )
        numerical = lambda r: _rodal_G(r, R, sigma)
        for order in range(3):
            for r in (1e-8, 0.0099 / sigma, 0.0101 / sigma):
                np.testing.assert_allclose(
                    numerical(r), float(mp.diff(exact, mp.mpf(r), order)), rtol=2e-6, atol=1e-12
                )
            numerical = jax.grad(numerical)
    for r in (0.0, 1e-8, 1e-6, 0.0099 / sigma, 0.0101 / sigma, 2.0):
        dshift = jax.jacfwd(metric.shift)(jnp.array([0.0, r, r / 2, r / 3]))[:, 1:]
        np.testing.assert_allclose(dshift, dshift.T, atol=2e-12)


def test_rodal_full_symbolic_and_rest_lab_tensor_agreement():
    metric = RodalMetric(v_s=0.6, R=2.0, sigma=1.5)
    symbolic = metric.symbolic()
    parameters = {
        s: float(getattr(metric, str(s)))
        for s in symbolic.g.free_symbols
        if str(s) in ("v_s", "R", "sigma")
    }
    evaluate = sp.lambdify(symbolic.coords, symbolic.g.subs(parameters), "numpy")
    v, R, sigma = float(metric.v_s), float(metric.R), float(metric.sigma)
    jacobian = np.eye(4)
    jacobian[1, 0] = -v
    for point in ([0.4, 2.1, 0.7, -0.3], [0.9, -3.0, 0.4, 0.6]):
        t, x, y, z = point
        displacement = np.array([x - v * t, y, z])
        r = np.linalg.norm(displacement)
        n = displacement / r
        f = 1 - (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (2 * np.tanh(sigma * R))
        # Source Eq. (42), with its half-angle denominator.
        g = (
            2 * r * sigma * np.sinh(R * sigma)
            + np.cosh(R * sigma) * np.log(np.cosh(sigma * (r - R)) / np.cosh(sigma * (r + R)))
        ) / (4 * r * sigma * np.sinh(R * sigma / 2) * np.cosh(R * sigma / 2))
        X = -v * (g * np.array([1.0, 0.0, 0.0]) + (f - g) * n[0] * n)
        rest = adm_to_full_metric(jnp.array(1.0), -jnp.asarray(X), jnp.eye(3))
        ideal = evaluate(*point)
        np.testing.assert_allclose(ideal, jacobian.T @ rest @ jacobian, atol=2e-14)
        np.testing.assert_allclose(metric(jnp.asarray(point)), ideal, rtol=1e-11, atol=1e-12)
        assert abs(ideal[0, 2]) > 1e-3 and abs(ideal[0, 3]) > 1e-3
    tangent = jnp.array([1.0, v, 0.0, 0.0])
    center = jnp.array([1.2, v * 1.2, 0.0, 0.0])
    np.testing.assert_allclose(tangent @ metric(center) @ tangent, -1.0, atol=2e-15)


def test_natario_rest_lab_tensor_and_symbolic_agreement():
    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    symbolic = metric.symbolic()
    parameters = {
        s: float(getattr(metric, str(s)))
        for s in symbolic.g.free_symbols
        if str(s) in ("v_s", "R", "sigma")
    }
    evaluate = sp.lambdify(symbolic.coords, symbolic.g.subs(parameters), "numpy")
    jacobian = np.eye(4)
    jacobian[1, 0] = -float(metric.v_s)  # d xi = d x - v_s d t
    for point in ([0.0, 0.62, 0.81, 0.2], [1.3, 1.1, -0.4, 0.7]):
        t, x, y, z = point
        dx = x - float(metric.v_s) * t
        r = jnp.sqrt(dx**2 + y**2 + z**2)
        n, dn = _natario_n(r, metric.R, metric.sigma), _natario_dn_dr(r, metric.R, metric.sigma)
        # Natario's published X in rest coordinates; beta_rest=-X.
        X = metric.v_s * jnp.array(
            [-(2 * n + dn * (y * y + z * z) / r), dn * dx * y / r, dn * dx * z / r]
        )
        rest = adm_to_full_metric(jnp.array(1.0), -X, jnp.eye(3))
        lab = np.asarray(metric(jnp.array(point)))
        np.testing.assert_allclose(lab, jacobian.T @ rest @ jacobian, atol=2e-14)
        np.testing.assert_allclose(lab, evaluate(*point), atol=2e-14)
        derivative = jax.jacfwd(metric.shift)(jnp.array(point))
        assert abs(float(jnp.trace(derivative[:, 1:]))) < 2e-13


def test_fuchs_covariant_adm_symbolic_and_stationary_agreement():
    from warpax.metrics import fuchs_default

    metric = fuchs_default(n_grid=256)
    symbolic = metric.symbolic()
    sym_v = next(s for s in symbolic.g.free_symbols if str(s) == "v_s")
    modules = {
        "a": lambda r: float(metric._potentials(jnp.array(r))[0]),
        "b": lambda r: float(metric._potentials(jnp.array(r))[1]),
        "S_warp": lambda r: float(metric.shape_function_value(jnp.array([0.0, r, 0.0, 0.0]))),
    }
    evaluate = sp.lambdify(symbolic.coords, symbolic.g.subs(sym_v, metric.v_s), [modules, "numpy"])
    for point in (
        [0.0, 0.0, 0.0, 0.0],
        [0.7, 12.0, 8.0, 3.0],
        [1.2, 19.0, -2.0, 1.0],
        [0.0, 35.0, 2.0, 0.0],
    ):
        coords = jnp.array(point)
        g = np.asarray(metric(coords))
        a, _ = metric._potentials(jnp.linalg.norm(coords[1:]))
        assert g[0, 0] == -np.exp(2 * float(a))
        assert g[0, 2] == g[0, 3] == 0.0
        assert g[0, 1] == -metric.v_s * metric.shape_function_value(coords)
        reconstructed = adm_to_full_metric(
            metric.lapse(coords), metric.shift(coords), metric.spatial_metric(coords)
        )
        np.testing.assert_allclose(g, reconstructed, atol=2e-15)
        assert np.sum(np.linalg.eigvalsh(g) < 0) == 1
        np.testing.assert_allclose(jax.jacfwd(metric)(coords)[:, :, 0], 0.0, atol=0.0)
        if np.linalg.norm(point[1:]) > 0:
            np.testing.assert_allclose(g, evaluate(*point), atol=2e-14)


def test_fuchs_c2_joins_and_origin_curvature():
    from warpax.geometry.geometry import compute_curvature_chain
    from warpax.metrics import fuchs_default
    from warpax.metrics.fuchs_construction import _fuchs_shift_transition

    metric = fuchs_default(n_grid=128)
    potentials = lambda r: jnp.stack(metric._potentials(r))
    # Include every interior spline knot, every fixed blend endpoint and the
    # former buffered clamps, so the check detects C1-only interpolation.
    knots = jnp.concatenate(
        (metric._r_grid[1:-1], jnp.array([2.5, 5.0, 10.0, 11.0, 19.0, 20.0, 25.0, 30.0]))
    )
    derivative = potentials
    for _ in range(3):
        values = jax.jit(jax.vmap(derivative))
        np.testing.assert_allclose(values(knots - 1e-7), values(knots + 1e-7), atol=3e-8)
        derivative = jax.jacfwd(derivative)
    transition = lambda r: _fuchs_shift_transition(r, 10.0, 20.0, 1.0)
    assert 0.0 < float(transition(jnp.array(19.0))) < 1.0
    assert 0.0 < float(transition(jnp.array(11.0))) < 1.0
    for r in (10.0, 20.0):
        assert float(jax.grad(transition)(jnp.array(r))) == 0.0
        assert float(jax.grad(jax.grad(transition))(jnp.array(r))) == 0.0
    curvature = compute_curvature_chain(metric, jnp.zeros(4))
    assert np.all(np.isfinite(curvature.stress_energy))
    np.testing.assert_allclose(curvature.stress_energy, 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "knots,potentials,message",
    [
        ([3.0, 10.0, 20.0, 30.0], [0.0] * 4, "C2 inner join"),
        ([0.0, 10.0, 10.0, 30.0], [0.0] * 4, "strictly increasing"),
        ([0.0, 10.0, 20.0, 30.0], [0.0, np.nan, 0.0, 0.0], "finite potentials"),
    ],
)
def test_fuchs_rejects_profiles_that_break_spline_regularity(knots, potentials, message):
    from warpax.metrics.fuchs_construction import FuchsMetric

    with pytest.raises(ValueError, match=message):
        FuchsMetric(
            _r_grid=jnp.array(knots),
            _a_grid=jnp.array(potentials),
            _b_grid=jnp.zeros(4),
            v_s=0.02,
            R_1=10.0,
            R_2=20.0,
            R_b=1.0,
            total_mass=1.0,
        )


def test_natario_center_hessian_is_the_analytic_limit():
    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    coefficient = float(metric.sigma**2 / (2 * jnp.cosh(metric.sigma * metric.R) ** 2))
    hessian = np.asarray(jax.jacfwd(jax.jacfwd(metric.shift))(jnp.zeros(4)))[:, 1:, 1:]
    expected = np.zeros((3, 3, 3))
    expected[0] = np.diag([4.0, 8.0, 8.0]) * float(metric.v_s) * coefficient
    expected[1, 0, 1] = expected[1, 1, 0] = -2 * float(metric.v_s) * coefficient
    expected[2, 0, 2] = expected[2, 2, 0] = -2 * float(metric.v_s) * coefficient
    np.testing.assert_allclose(hessian, expected, rtol=1e-13, atol=1e-20)
    for x in (1e-12, 1e-9, 1e-6):
        np.testing.assert_allclose(
            jax.jacfwd(jax.jacfwd(metric.shift))(jnp.array([0.0, x, 0.0, 0.0]))[:, 1:, 1:],
            expected,
            rtol=1e-8,
            atol=1e-15,
        )


def test_natario_density_helper_center_near_center_and_wall():
    from warpax.geometry.geometry import compute_curvature_chain
    from warpax.metrics.natario import natario_eulerian_energy_density

    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    for point in (
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1e-13, 2e-13, -1e-13],
        [0.0, 1e-6, 2e-6, 0.0],
        [0.7, 0.8, 0.8, 0.2],
    ):
        coords = jnp.array(point)
        density = natario_eulerian_energy_density(
            *coords[1:], v_s=metric.v_s, R=metric.R, sigma=metric.sigma, t=coords[0]
        )
        derivative = jax.jacfwd(metric.shift)(coords)[:, 1:]
        K = (derivative + derivative.T) / 2
        hamiltonian = (jnp.trace(K) ** 2 - jnp.sum(K * K)) / (16 * jnp.pi)
        np.testing.assert_allclose(density, hamiltonian, rtol=1e-11, atol=1e-50)
        chain = compute_curvature_chain(metric, coords)
        normal = jnp.concatenate((jnp.ones(1), -metric.shift(coords)))
        contracted = normal @ chain.stress_energy @ normal
        np.testing.assert_allclose(density, contracted, rtol=1e-11, atol=1e-17)
