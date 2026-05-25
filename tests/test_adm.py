"""ADM 3+1 split: kinematics, constraints, superluminal handling."""

from __future__ import annotations
from warpax.adm import adm_mass, falloff_check
import jax
import jax.numpy as jnp
import pytest



jax.config.update("jax_enable_x64", True)


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


jax.config.update("jax_enable_x64", True)


# 3+1 ADM decomposition

class TestADMSplit:
    """Verify (alpha, beta, gamma, K) extraction from full spacetime metrics."""

    def test_minkowski_decomposition(self):
        """Minkowski: alpha=1, beta=0, gamma=I, K=0."""
        from warpax.geometry import adm_split
        from warpax.benchmarks import MinkowskiMetric

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.allclose(adm.lapse, 1.0, atol=1e-14)
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.spatial_metric, jnp.eye(3), atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-10)

    def test_schwarzschild_decomposition(self):
        """Schwarzschild isotropic: alpha = (1-M/2r)/(1+M/2r), beta=0, K=0."""
        from warpax.geometry import adm_split
        from warpax.benchmarks import SchwarzschildMetric

        metric = SchwarzschildMetric(M=1.0)
        r_iso = 10.0
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        adm = adm_split(metric, coords)

        ratio = 1.0 / (2.0 * r_iso)
        expected_lapse = (1.0 - ratio) / (1.0 + ratio)

        assert jnp.allclose(adm.lapse, expected_lapse, rtol=1e-12), \
            f"Lapse: got {adm.lapse}, expected {expected_lapse}"
        assert jnp.allclose(adm.shift_upper, 0.0, atol=1e-14)
        assert jnp.allclose(adm.extrinsic_curvature, 0.0, atol=1e-8), \
            f"Static K must vanish, got max|K|={jnp.max(jnp.abs(adm.extrinsic_curvature))}"

    def test_warpshell_nonzero_shift(self):
        """WarpShell interior has beta^x = -v_s."""
        from warpax.geometry import adm_split
        from warpax.metrics import WarpShellPhysical

        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])
        adm = adm_split(metric, coords)

        assert jnp.abs(adm.shift_upper[0] - (-0.02)) < 1e-6, \
            f"Interior shift should be -v_s=-0.02, got {adm.shift_upper[0]}"
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


# Constraint residuals

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
        from warpax.constraints import normalized_residuals
        from warpax.benchmarks import MinkowskiMetric

        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])
        result = normalized_residuals(metric, coords)

        assert result["epsilon_H"] < 1e-10, f"eps_H={result['epsilon_H']}"

    def test_normalized_residuals_schwarzschild(self):
        """Schwarzschild vacuum: eps_H ~0."""
        from warpax.constraints import normalized_residuals
        from warpax.benchmarks import SchwarzschildMetric

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
            gamma, K, jnp.zeros(3), metric_fn=metric, coords=coords,
        )
        assert jnp.allclose(M_i, 0.0, atol=1e-14)


jax.config.update("jax_enable_x64", True)

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
    """Evaluate the 4x4 metric at (t=0, x, 0.01, 0)."""
    # Set v_s by rebuilding the metric (all warp metrics expose v_s)
    # We use eqx.tree_at-style replacement by direct attribute override
    # since these are dataclass-style Equinox modules.
    import dataclasses
    if dataclasses.is_dataclass(metric):
        metric = dataclasses.replace(metric, v_s=v_s)
    else:
        # Equinox modules: use eqx.tree_at
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
    # Sample on the wall radius and a couple of nearby points
    x_samples = [0.5, 1.0, 1.5] if name in ("alcubierre", "natario", "vdb") else [50.0, 100.0, 150.0]
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
        # Cholesky succeeds iff gamma is positive definite
        try:
            L = jnp.linalg.cholesky(gamma)
            # Verify L L^T = gamma
            assert jnp.allclose(L @ L.T, gamma, atol=1e-12), (
                f"Cholesky reconstruction failed at v_s={v_s}, x={x}"
            )
        except Exception as exc:
            pytest.fail(
                f"Alcubierre spatial metric not positive-definite at "
                f"v_s={v_s}, x={x}: {exc}"
            )


@pytest.mark.parametrize("v_s", [0.99, 1.0, 1.5])
def test_alcubierre_g00_can_be_non_negative_at_superluminal(v_s):
    """At v_s >= 1, g_00 = -1 + v_s^2 f^2 can be >= 0 in the bubble wall;
    this is the ergoregion-like coordinate feature.

    This test documents the actual behavior so any future change that
    silently restricts v_s would fail this assertion.
    """
    metric = AlcubierreMetric(R=1.0, sigma=8.0, v_s=v_s)
    # f(r_s ~ 0) ~ 1, so g_00 ~ -1 + v_s^2
    g_center = _eval_at(metric, v_s, 0.0)
    g00 = float(g_center[0, 0])
    if v_s > 1.0:
        # g_00 = -1 + v_s^2 * f(0)^2 ~ -1 + v_s^2 > 0
        assert g00 > 0.0, (
            f"At v_s={v_s} we expect g_00 > 0 in the bubble interior "
            f"(f ~ 1); got g_00={g00}."
        )
    # Either way, det(g) must stay at -1
    det_g = float(jnp.linalg.det(g_center))
    assert abs(det_g + 1.0) < 1e-10, (
        f"At v_s={v_s}, det(g)={det_g}, not -1 (signature is preserved)."
    )


def test_alcubierre_no_nan_in_curvature_chain_at_superluminal():
    """The curvature chain stays finite at v_s = 1.5 (no autodiff NaN
    when g_00 changes sign)."""
    from warpax.geometry import compute_curvature_chain
    metric = AlcubierreMetric(R=100.0, sigma=8.0, v_s=1.5)
    # Sample at a point near the wall where g_00 changes sign
    coords = jnp.array([0.0, 100.0, 0.01, 0.0])
    curv = compute_curvature_chain(metric, coords)
    assert not jnp.any(jnp.isnan(curv.stress_energy)), (
        "Stress-energy is NaN at v_s=1.5; the autodiff chain has a bug."
    )
    assert not jnp.any(jnp.isnan(curv.metric)), (
        "Metric is NaN at v_s=1.5."
    )
