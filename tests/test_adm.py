"""ADM 3+1 split: kinematics, constraints, superluminal handling."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.adm import adm_mass, adm_mass_richardson, asymptotic_flatness_report, falloff_check
from warpax.geometry.metric import MetricSpecification


def test_adm_mass_schwarzschild():
    """ADM mass of Schwarzschild should equal its parameter M."""
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    metric = SchwarzschildMetric()
    M = adm_mass(metric, r_surface=100.0, n_theta=16, n_phi=32)
    assert jnp.abs(M - 1.0) < 0.05, f"Expected ~1.0, got {M}"


def test_adm_mass_minkowski():
    """ADM mass of Minkowski should be zero."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    M = adm_mass(metric, r_surface=100.0, n_theta=8, n_phi=16)
    assert jnp.abs(M) < 1e-10, f"Expected ~0, got {M}"


def test_falloff_check_minkowski():
    """Minkowski has exact flat falloff."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    result = falloff_check(metric, r_test=100.0, expected_order=1)
    assert result["g_tt"] is True
    assert result["g_xx"] is True


def test_falloff_check_schwarzschild():
    """Schwarzschild should show O(1/r) falloff."""
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    metric = SchwarzschildMetric()
    result = falloff_check(metric, r_test=200.0, expected_order=1)
    assert result["g_tt"] is True
    assert result["g_xx"] is True


class TestADMSplit:
    """Verify (alpha, beta, gamma, K) extraction from full spacetime metrics."""

    def test_minkowski_decomposition(self):
        """Minkowski: alpha=1, beta=0, gamma=I, K=0."""
        from warpax.benchmarks import MinkowskiMetric
        from warpax.geometry import adm_split

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.allclose(adm.lapse, 1.0, atol=1e-14)
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.spatial_metric, jnp.eye(3), atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-10)

    def test_schwarzschild_decomposition(self):
        """Schwarzschild isotropic: alpha = (1-M/2r)/(1+M/2r), beta=0, K=0."""
        from warpax.benchmarks import SchwarzschildMetric
        from warpax.geometry import adm_split

        metric = SchwarzschildMetric(M=1.0)
        r_iso = 10.0
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        adm = adm_split(metric, coords)

        ratio = 1.0 / (2.0 * r_iso)
        expected_lapse = (1.0 - ratio) / (1.0 + ratio)

        assert jnp.allclose(adm.lapse, expected_lapse, rtol=1e-12), (
            f"Lapse: got {adm.lapse}, expected {expected_lapse}"
        )
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-8), (
            f"Static K must vanish, got max|K|={jnp.max(jnp.abs(adm.extrinsic_curvature))}"
        )

    def test_warpshell_nonzero_shift(self):
        """WarpShell interior has beta^x = -v_s."""
        from warpax.geometry import adm_split
        from warpax.metrics import WarpShellPhysical

        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.abs(adm.shift_upper[0] - (-0.02)) < 1e-6, (
            f"Interior shift should be -v_s=-0.02, got {adm.shift_upper[0]}"
        )
        assert jnp.allclose(adm.shift_upper[1:], 0.0, atol=1e-10)

    def test_warpshell_shell_region(self):
        """WarpShell transition region produces finite K_{ij}."""
        from warpax.geometry import adm_split
        from warpax.metrics import WarpShellPhysical

        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        K_max = jnp.max(jnp.abs(adm.extrinsic_curvature))
        assert K_max < 100, f"K should be finite, got max|K|={K_max}"


class TestConstraints:
    """Verify Hamiltonian and momentum constraint evaluations."""

    def test_minkowski_hamiltonian(self):
        """Minkowski vacuum: H = 0."""
        from warpax.constraints import hamiltonian_constraint

        gamma = jnp.eye(3)
        K = jnp.zeros((3, 3))
        H = hamiltonian_constraint(gamma, K, jnp.array(0.0), R=jnp.array(0.0))
        assert jnp.abs(H) < 1e-14, f"H should be 0 for Minkowski, got {H}"

    def test_pure_K_trace(self):
        """Flat space with K=delta_{ij}: H = K^2 - K_{ij}K^{ij} = 9-3 = 6."""
        from warpax.constraints import hamiltonian_constraint

        gamma = jnp.eye(3)
        K = jnp.eye(3)
        H = hamiltonian_constraint(gamma, K, jnp.array(0.0), R=jnp.array(0.0))
        assert jnp.allclose(H, 6.0, atol=1e-12), f"Expected H=6, got {H}"

    def test_normalized_residuals_minkowski(self):
        """Minkowski: eps_H ~0, eps_M ~0."""
        from warpax.benchmarks import MinkowskiMetric
        from warpax.constraints import normalized_residuals

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        result = normalized_residuals(metric, coords)

        assert result["epsilon_H"] < 1e-10, f"eps_H={result['epsilon_H']}"

    def test_normalized_residuals_schwarzschild(self):
        """Schwarzschild vacuum: eps_H ~0."""
        from warpax.benchmarks import SchwarzschildMetric
        from warpax.constraints import normalized_residuals

        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 10.0, 0.0, 0.0])
        result = normalized_residuals(metric, coords)

        assert result["epsilon_H"] < 1e-6, f"eps_H={result['epsilon_H']}"

    def test_momentum_constraint_minkowski(self):
        """Minkowski: M_i = 0."""
        from warpax.benchmarks.minkowski import MinkowskiMetric
        from warpax.constraints import momentum_constraint

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
        gamma = jnp.eye(3)
        K = jnp.zeros((3, 3))
        M_i = momentum_constraint(
            gamma,
            K,
            jnp.zeros(3),
            metric_fn=metric,
            coords=coords,
        )
        assert jnp.allclose(M_i, 0.0, atol=1e-14)

    def test_sshell_hamiltonian_residual_vanishes_with_sources(self):
        """H = R + K^2 - K_ij K^ij - 16 pi E with the prescribed E = rho(r).

        Inside the shell the vacuum-source residual equals 16 pi rho_0 ~ 5e-3;
        with ``metric.sources`` it drops to the solver's discretization error.
        """
        from warpax.constraints import normalized_residuals
        from warpax.metrics import sshell_default

        metric = sshell_default()
        for r in (11.0, 15.0, 19.0):
            coords = jnp.array([0.0, r, 0.0, 0.0])
            E, S_i = metric.sources(jnp.asarray(r))
            vacuum = float(jnp.abs(normalized_residuals(metric, coords)["H_raw"]))
            sourced = float(
                jnp.abs(
                    normalized_residuals(metric, coords, energy_density=E, momentum_density=S_i)[
                        "H_raw"
                    ]
                )
            )
            assert sourced < 1e-5, f"r={r}: |H|={sourced}"
            assert sourced * 10.0 < vacuum, f"r={r}: sourced {sourced} vs vacuum {vacuum}"

    def test_residual_helper_matches_eager(self):
        """Jitted ``mean_squared_constraint_residual`` equals the eager vmap mean."""
        import jax

        from warpax.constraints.residuals import (
            mean_squared_constraint_residual,
            squared_constraint_residual,
        )
        from warpax.metrics import sshell_default

        metric = sshell_default()
        r_probes = jnp.linspace(10.0, 20.0, 5)
        eager = jnp.mean(jax.vmap(lambda r: squared_constraint_residual(metric, r))(r_probes))
        jitted = mean_squared_constraint_residual(metric, r_probes)
        assert float(jnp.abs(jitted - eager)) <= 1e-12 * float(jnp.abs(eager))


from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import LentzMetric, NatarioMetric, RodalMetric, VanDenBroeckMetric

WARP_METRICS_UNIT_LAPSE = [
    ("alcubierre", AlcubierreMetric(R=1.0, sigma=8.0)),
    ("natario", NatarioMetric(R=1.0, sigma=8.0)),
    ("vdb", VanDenBroeckMetric(R=1.0, sigma=8.0)),
    ("lentz", LentzMetric(R=100.0, sigma=8.0)),
    ("rodal", RodalMetric(R=100.0, sigma=0.03)),
]


def _eval_at(metric, v_s, x):
    """Evaluate the 4x4 metric at (t=0, x, 0.01, 0), rebuilding the metric with
    the given v_s (dataclasses.replace or eqx.tree_at)."""
    import dataclasses

    if dataclasses.is_dataclass(metric):
        metric = dataclasses.replace(metric, v_s=v_s)
    else:
        import equinox as eqx

        metric = eqx.tree_at(lambda m: m.v_s, metric, replace=v_s)
    coords = jnp.array([0.0, float(x), 0.01, 0.0])
    g = metric(coords)
    return g


@pytest.mark.parametrize("name,metric", WARP_METRICS_UNIT_LAPSE)
@pytest.mark.parametrize("v_s", [0.5, 0.99, 1.0, 1.5])
def test_det_g_stays_negative_at_all_v_s(name, metric, v_s):
    """det(g) = -1 (or close to it) for unit-lapse warp metrics at all v_s.

    This is the textbook ADM identity det(g) = -alpha^2 * det(gamma); for
    alpha = 1 and gamma_ij = delta_ij, det(g) = -1 identically. A signature
    flip would require det(g) -> 0 at v_s = 1; we confirm it does not.

    Note: Rodal and VdB have non-trivial gamma_ij (Rodal has a non-flat
    spatial metric in the lab frame; VdB has a conformal factor) so we
    test that det(g) stays strictly negative rather than exactly -1.
    """
    x_samples = (
        [0.5, 1.0, 1.5] if name in ("alcubierre", "natario", "vdb") else [50.0, 100.0, 150.0]
    )
    for x in x_samples:
        g = _eval_at(metric, v_s, x)
        det_g = float(jnp.linalg.det(g))
        assert det_g < 0.0, (
            f"{name} at v_s={v_s}, x={x}: det(g)={det_g} is non-negative; "
            "this would indicate a signature change. The ADM identity "
            "det(g) = -alpha^2 det(gamma) requires det(g) < 0 for any "
            "Lorentzian spacetime."
        )


@pytest.mark.parametrize("v_s", [0.5, 1.0, 1.5])
def test_alcubierre_det_g_is_exactly_minus_one(v_s):
    """For Alcubierre specifically (unit lapse, flat spatial), det(g) = -1
    at every spacetime point, independent of v_s."""
    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=v_s)
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]:
        g = _eval_at(metric, v_s, x)
        det_g = float(jnp.linalg.det(g))
        assert abs(det_g + 1.0) < 1e-10, (
            f"Alcubierre at v_s={v_s}, x={x}: det(g)={det_g} != -1. "
            "For unit lapse and Euclidean spatial metric we must have "
            "det(g) = -1 exactly."
        )


@pytest.mark.parametrize("v_s", [0.5, 1.0, 1.5])
def test_alcubierre_spatial_metric_is_positive_definite(v_s):
    """For Alcubierre, gamma_ij = delta_ij at every point, so it is
    positive-definite (Cholesky-decomposable) at all v_s."""
    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=v_s)
    for x in [0.0, 1.0, 5.0]:
        g = _eval_at(metric, v_s, x)
        gamma = g[1:, 1:]
        try:
            L = jnp.linalg.cholesky(gamma)
            assert jnp.allclose(L @ L.T, gamma, atol=1e-12), (
                f"Cholesky reconstruction failed at v_s={v_s}, x={x}"
            )
        except Exception as exc:
            pytest.fail(
                f"Alcubierre spatial metric not positive-definite at v_s={v_s}, x={x}: {exc}"
            )


@pytest.mark.parametrize("v_s", [0.99, 1.0, 1.5])
def test_alcubierre_g00_can_be_non_negative_at_superluminal(v_s):
    """At v_s >= 1, g_00 = -1 + v_s^2 f^2 can be >= 0 in the bubble wall;
    this is the ergoregion-like coordinate feature.

    This test documents the actual behavior so any future change that
    silently restricts v_s would fail this assertion. At the center
    f ~ 1, so g_00 ~ -1 + v_s^2: positive for v_s > 1, ~ v_s^2 - 1 for
    v_s < 1 (the eval point sits at y=0.01 so f is slightly below 1, ~1e-8
    slack), and zero at v_s = 1. Either way det(g) must stay at -1.
    """
    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=v_s)
    g_center = _eval_at(metric, v_s, 0.0)
    g00 = float(g_center[0, 0])
    if v_s > 1.0:
        assert g00 > 0.0, (
            f"At v_s={v_s} we expect g_00 > 0 in the bubble interior (f ~ 1); got g_00={g00}."
        )
    elif v_s < 1.0:
        assert g00 < 0.0, f"At v_s={v_s} < 1 we expect g_00 < 0 at the center; got {g00}."
        assert abs(g00 - (v_s**2 - 1.0)) < 1e-6, (
            f"At v_s={v_s}, g_00={g00} should be ~ v_s^2 - 1 = {v_s**2 - 1.0}."
        )
    else:
        assert abs(g00) < 1e-6, f"At v_s=1 we expect g_00 ~ 0 at the center; got {g00}."
    det_g = float(jnp.linalg.det(g_center))
    assert abs(det_g + 1.0) < 1e-10, (
        f"At v_s={v_s}, det(g)={det_g}, not -1 (signature is preserved)."
    )


def test_alcubierre_no_nan_in_curvature_chain_at_superluminal():
    """The curvature chain stays finite at v_s = 1.5 (no autodiff NaN
    when g_00 changes sign)."""
    from warpax.geometry import compute_curvature_chain

    metric = AlcubierreMetric(R=100.0, sigma=8.0, v_s=1.5)
    coords = jnp.array([0.0, 100.0, 0.01, 0.0])
    curv = compute_curvature_chain(metric, coords)
    assert not jnp.any(jnp.isnan(curv.stress_energy)), (
        "Stress-energy is NaN at v_s=1.5; the autodiff chain has a bug."
    )
    assert not jnp.any(jnp.isnan(curv.metric)), "Metric is NaN at v_s=1.5."


class _RadialPerturbation(MetricSpecification):
    """gamma_ij = delta_ij + (B - 1) n_i n_j with B = 1 + 2/r + 2/r^2, lapse 1, no shift."""

    def __call__(self, coords):
        x = coords[1:]
        r = jnp.sqrt(jnp.sum(x**2))
        n = x / r
        B = 1.0 + 2.0 / r + 2.0 / r**2
        gamma = jnp.eye(3) + (B - 1.0) * jnp.outer(n, n)
        g = jnp.zeros((4, 4)).at[0, 0].set(-1.0)
        return g.at[1:, 1:].set(gamma)

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "RadialPerturbation"


class _LogDivergent(MetricSpecification):
    """gamma_ij = f delta_ij with f = 1 + 1e-14 log r; the surface mass is M(r) = -5e-15 r."""

    def __call__(self, coords):
        r = jnp.sqrt(jnp.sum(coords[1:] ** 2))
        f = 1.0 + 1e-14 * jnp.log(r)
        return jnp.diag(jnp.array([-1.0, f, f, f]))

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "LogDivergent"


class _AnisotropicYY(MetricSpecification):
    """Minkowski except gamma_yy = 1 + y^2 / (1 + r^2), which tends to 2 along +-y."""

    def __call__(self, coords):
        x = coords[1:]
        r2 = jnp.sum(x**2)
        g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        return g.at[2, 2].set(1.0 + x[1] ** 2 / (1.0 + r2))

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "AnisotropicYY"


def test_adm_richardson_non_geometric_radii():
    """For gamma_ij = delta_ij + h n_i n_j the ADM integrand is 2h/r, so M(r) = r h / 2 = 1 + 1/r
    exactly (quadrature is exact for a constant integrand). Radii (10, 20, 50) give
    (1.1, 1.05, 1.02); the order equation is solved by p = 1 and the extrapolated mass is
    (1.02*50 - 1.05*20)/30 = 1."""
    result = adm_mass_richardson(_RadialPerturbation(), radii=[10.0, 20.0, 50.0])
    assert jnp.allclose(jnp.array(result["M_values"]), jnp.array([1.1, 1.05, 1.02]), atol=1e-10)
    assert abs(result["convergence_order"] - 1.0) < 1e-6
    assert abs(result["M_extrapolated"] - 1.0) < 1e-6
    assert result["valid"] is True


def test_adm_richardson_two_radii_and_converged():
    """Two radii keep the linear extrapolation with order NaN; a flat metric has
    |M1 - M2| <= 1e-12 so the order is reported inf and valid is True."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    two = adm_mass_richardson(_RadialPerturbation(), radii=[10.0, 20.0])
    assert two["convergence_order"] != two["convergence_order"]
    assert abs(two["M_extrapolated"] - 1.0) < 1e-6
    assert two["valid"] is True

    flat = adm_mass_richardson(MinkowskiMetric(), radii=[10.0, 20.0, 50.0], n_theta=8, n_phi=16)
    assert flat["convergence_order"] == float("inf")
    assert flat["valid"] is True
    assert abs(flat["M_extrapolated"]) < 1e-10


def test_adm_richardson_divergent_small_mass():
    """For gamma_ij = f delta_ij the ADM integrand is -2 f'(r) r / r = -2e-14 / r, so
    M(r) = -5e-15 r: (-2.5e-13, -5e-13, -1e-12) at radii (50, 100, 200). The pair gaps
    grow (2.5e-13 then 5e-13) and no p in [0.1, 8] solves the order equation, so the
    result is not valid and the extrapolation is NaN."""
    result = adm_mass_richardson(_LogDivergent(), radii=[50.0, 100.0, 200.0], n_theta=4, n_phi=8)
    assert jnp.allclose(
        jnp.array(result["M_values"]), jnp.array([-2.5e-13, -5e-13, -1e-12]), rtol=1e-6
    )
    assert result["valid"] is False
    assert result["M_extrapolated"] != result["M_extrapolated"]


def test_falloff_check_multi_direction():
    """gamma_yy = 1 + y^2/(1 + r^2) is exactly flat along +-x and +-z but tends to 2 along +-y
    (deviation 0.996 at r=50 vs 0.999 at r=100, measured order ~0), so g_yy must be False
    while the other nine components stay True."""
    result = falloff_check(_AnisotropicYY(), r_test=100.0, expected_order=1)
    assert result["g_yy"] is False
    assert all(v is True for k, v in result.items() if k != "g_yy")
    assert len(result) == 10


def test_asymptotic_flatness_report_multi_direction():
    """The same metric is reported not asymptotically flat, with g_yy failing and its
    worst-direction measured order near 0; Schwarzschild passes every component."""
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    report = asymptotic_flatness_report(_AnisotropicYY(), radii=[50.0, 100.0, 200.0])
    assert report["is_asymptotically_flat"] is False
    assert report["components"]["g_yy"]["passed"] is False
    assert abs(report["components"]["g_yy"]["measured_order"]) < 0.05
    assert report["components"]["g_xx"]["passed"] is True

    schw = asymptotic_flatness_report(SchwarzschildMetric(), radii=[50.0, 100.0, 200.0])
    assert schw["is_asymptotically_flat"] is True
    assert len(schw["components"]) == 10


def test_asymptotic_flatness_report_diagonal_and_shift_views():
    """'diagonal' holds the g_tt, g_xx, g_yy, g_zz entries of 'components' and 'shift' the
    g_tx, g_ty, g_tz entries, for Minkowski (every order inf) and Schwarzschild (order 1)."""
    from warpax.benchmarks.minkowski import MinkowskiMetric
    from warpax.benchmarks.schwarzschild import SchwarzschildMetric

    for metric in (MinkowskiMetric(), SchwarzschildMetric()):
        report = asymptotic_flatness_report(metric, radii=[50.0, 100.0, 200.0])
        assert list(report["diagonal"]) == ["g_tt", "g_xx", "g_yy", "g_zz"]
        assert list(report["shift"]) == ["g_tx", "g_ty", "g_tz"]
        for name in [*report["diagonal"], *report["shift"]]:
            view = report["diagonal"].get(name) or report["shift"][name]
            assert view == report["components"][name]
            assert view["passed"] is True
