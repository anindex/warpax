"""Parity smoke test pinning the legacy Fuchs analytical path.

``_FuchsAnalytical`` (the constant-density Schwarzschild-shell ansatz)
and the canonical Gaussian-smoothed ``FuchsMetric`` differ in detail
(``rho_smoothed`` is no longer piecewise constant), but the total shell
mass and the asymptotic Schwarzschild parameters must agree at the
paper-default parameters. This locks the legacy path against silent
drift.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from warpax.metrics import fuchs_default
from warpax.metrics._fuchs_legacy import (
    _FuchsAnalytical,
    fuchs_shell_profiles,
)


def test_legacy_analytical_total_mass_matches_paper_default():
    """``_FuchsAnalytical`` mass profile sums to r_s / 2 at the paper defaults."""
    metric = _FuchsAnalytical(
        v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0, transition_order=2,
    )
    profiles = metric.shell_profiles()
    assert profiles.total_mass == 2.5
    # At r = R_2 the cumulative mass equals total mass.
    m_outer = float(profiles.cumulative_mass(jnp.float64(20.0)))
    assert abs(m_outer - profiles.total_mass) < 1e-12


def test_legacy_shell_profiles_factory_agrees_with_canonical_total_mass():
    """Legacy and canonical totals agree to ~1 % (the canonical path smooths off a sliver of mass)."""
    profiles = fuchs_shell_profiles(R_1=10.0, R_2=20.0, r_s_param=5.0)
    canonical = fuchs_default()
    rel = abs(profiles.total_mass - canonical.total_mass) / profiles.total_mass
    assert rel < 1e-2, f"legacy vs canonical mass drift {rel:.4f} exceeds 1 %"


def test_legacy_metric_evaluates_finite_outside_shell():
    """Sanity: the legacy metric remains finite outside the shell (no NaN in g_ab)."""
    metric = _FuchsAnalytical(
        v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0, transition_order=2,
    )
    coords = jnp.array([0.0, 30.0, 0.0, 0.0])  # exterior probe
    g = np.asarray(metric(coords))
    assert np.all(np.isfinite(g))
    # Lapse must be positive, signature must be (-+++) outside the shell.
    assert g[0, 0] < 0.0
    assert g[1, 1] > 0.0 and g[2, 2] > 0.0 and g[3, 3] > 0.0
