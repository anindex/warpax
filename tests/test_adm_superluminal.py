"""ADM/signature regression tests for v_s >= 1.

For unit-lapse, flat-spatial-metric warp drives, det(g) = -alpha^2 det(gamma)
= -1 at all velocities; the appearance of g_00 >= 0 at and beyond v_s = 1 is
an ergoregion-like coordinate feature, not a signature change. These tests
pin that behavior so any future regression that reintroduces a
sqrt(-g_00)-style bug fails fast.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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
