"""Shift-vector kinematic decomposition (expansion / shear / vorticity).

Analytic checks on synthetic shift fields plus physics checks on the Rodal
(irrotational) and Natario (zero-expansion) warp drives.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from warpax.analysis import (  # noqa: E402
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
    # beta = grad(x^2 + y^2) = (2x, 2y, 0): pure gradient, so curl is zero.
    m = _flat_metric_with_shift(lambda x, y, z: jnp.array([2 * x, 2 * y, 0.0]))
    theta, sigma_sq, omega_sq = compute_shift_kinematics(m, jnp.array([0.0, 0.3, -0.4, 0.2]))
    assert float(omega_sq) < 1e-20
    assert float(theta) == pytest.approx(4.0, abs=1e-9)  # div = 2 + 2
    assert float(rotationality(theta, sigma_sq, omega_sq)) < 1e-12


def test_pure_rotation_shift_matches_analytic_curl():
    # beta = c * (-y, x, 0): rigid rotation. theta = 0, sigma^2 = 0, omega^2 = 2 c^2.
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
    # A wall point (R_b = 1, so r ~ 1 is the active wall).
    _, sigma_sq, omega_sq = compute_shift_kinematics(metric, jnp.array([0.0, 1.0, 0.3, 0.0]))
    # Irrotational by construction: vorticity is negligible vs the shear scale.
    assert float(omega_sq) < 1e-10 * (float(sigma_sq) + 1e-30) + 1e-18


def test_natario_zero_expansion_but_rotational():
    from warpax.metrics import NatarioMetric

    metric = NatarioMetric(v_s=0.5, R=1.0, sigma=8.0)
    theta, _, omega_sq = compute_shift_kinematics(metric, jnp.array([0.0, 1.0, 0.4, 0.0]))
    assert abs(float(theta)) < 1e-8           # zero-expansion drive
    assert float(omega_sq) > 1e-3             # yet carries vorticity


def test_grid_matches_pointwise():
    from warpax.geometry.types import GridSpec

    m = _flat_metric_with_shift(lambda x, y, z: jnp.array([-0.5 * y, 0.5 * x, 0.0]))
    spec = GridSpec(bounds=[(-1.0, 1.0)] * 3, shape=(5, 5, 5))
    theta_g, sigma_g, omega_g = compute_shift_kinematics_grid(m, spec, t=0.0)
    assert theta_g.shape == (5, 5, 5)
    # Rigid rotation: omega^2 = 2 * 0.25 = 0.5 everywhere.
    assert float(jnp.max(jnp.abs(omega_g - 0.5))) < 1e-6
