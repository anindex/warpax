"""Shift-vector kinematic decomposition (expansion / shear / vorticity).

Analytic checks on synthetic shift fields plus physics checks on the Rodal
(irrotational) and Natario (zero-expansion) warp drives.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.analysis import (
    compute_shift_kinematics,
    compute_shift_kinematics_grid,
    rotationality,
)


def _flat_metric_with_shift(beta_fn):
    """Build a metric_fn with unit lapse, flat spatial slice, shift beta_fn(x,y,z)."""

    def metric_fn(coords):
        b = beta_fn(coords[1], coords[2], coords[3])
        g = jnp.eye(4).at[0, 0].set(-1.0)
        g = g.at[0, 1:].set(b).at[1:, 0].set(b)
        return g

    return metric_fn


def test_irrotational_gradient_shift_has_zero_vorticity():
    m = _flat_metric_with_shift(lambda x, y, z: jnp.array([2 * x, 2 * y, 0.0]))
    theta, sigma_sq, omega_sq = compute_shift_kinematics(m, jnp.array([0.0, 0.3, -0.4, 0.2]))
    assert float(omega_sq) < 1e-20
    assert float(theta) == pytest.approx(4.0, abs=1e-9)
    assert float(rotationality(theta, sigma_sq, omega_sq)) < 1e-12


def test_pure_rotation_shift_matches_analytic_curl():
    c = 0.7
    m = _flat_metric_with_shift(lambda x, y, z: jnp.array([-c * y, c * x, 0.0]))
    theta, sigma_sq, omega_sq = compute_shift_kinematics(m, jnp.array([0.0, 0.1, 0.2, 0.3]))
    assert float(theta) == pytest.approx(0.0, abs=1e-9)
    assert float(sigma_sq) == pytest.approx(0.0, abs=1e-12)
    assert float(omega_sq) == pytest.approx(2.0 * c**2, rel=1e-6)
    assert float(rotationality(theta, sigma_sq, omega_sq)) == pytest.approx(1.0, abs=1e-9)


def test_rodal_shift_is_irrotational():
    from warpax.metrics import RodalMetric

    metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
    _, sigma_sq, omega_sq = compute_shift_kinematics(metric, jnp.array([0.0, 1.0, 0.3, 0.0]))
    assert float(omega_sq) < 1e-10 * (float(sigma_sq) + 1e-30) + 1e-18


def test_natario_zero_expansion_but_rotational():
    from warpax.metrics import NatarioMetric

    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    theta, _, omega_sq = compute_shift_kinematics(metric, jnp.array([0.0, 1.0, 0.4, 0.0]))
    assert abs(float(theta)) < 1e-8
    assert float(omega_sq) > 1e-3


def test_grid_matches_pointwise():
    """Rigid rotation with c = 0.5: omega^2 = 2 c^2 = 0.5 everywhere on the grid."""
    from warpax.geometry.types import GridSpec

    m = _flat_metric_with_shift(lambda x, y, z: jnp.array([-0.5 * y, 0.5 * x, 0.0]))
    spec = GridSpec(bounds=[(-1.0, 1.0)] * 3, shape=(5, 5, 5))
    theta_g, sigma_g, omega_g = compute_shift_kinematics_grid(m, spec, t=0.0)
    assert theta_g.shape == (5, 5, 5)
    assert float(jnp.max(jnp.abs(omega_g - 0.5))) < 1e-6


def test_kinematic_grid_jit_matches_eager():
    """Jitted grid kernels reproduce the eager pointwise vmap, with and without batching.

    Tolerance is rtol 1e-12 plus an atol of 1e-12 times the field scale, so
    entries that are roundoff of an exact zero do not fail on relative error.
    """
    import jax
    import numpy as np

    from warpax.analysis import compute_kinematic_scalars, compute_kinematic_scalars_grid
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry.grid import build_coord_batch
    from warpax.geometry.types import GridSpec

    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    spec = GridSpec(bounds=[(-1.5, 1.5)] * 3, shape=(6, 6, 6))
    coords = build_coord_batch(spec, t=0.0)
    with jax.disable_jit():
        want_kin = jax.vmap(lambda c: jnp.stack(compute_kinematic_scalars(metric, c)))(coords)
        want_shift = jax.vmap(lambda c: jnp.stack(compute_shift_kinematics(metric, c)))(coords)
    for batch_size in (None, 50):
        got_kin = compute_kinematic_scalars_grid(metric, spec, batch_size=batch_size)
        got_shift = compute_shift_kinematics_grid(metric, spec, batch_size=batch_size)
        for got, want in ((got_kin, want_kin), (got_shift, want_shift)):
            for k in range(3):
                w = np.asarray(want[:, k])
                np.testing.assert_allclose(
                    np.asarray(got[k]).ravel(), w, rtol=1e-12, atol=1e-12 * np.max(np.abs(w))
                )
