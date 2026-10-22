r"""Tests for the Newman-Penrose peeling-falloff extractor (warpax.bondi.peeling).

Verifies the asymptotic structure underlying the model-independent
no-reactionless-steering theorem (T1-GR): along outgoing null cones the five NP
Weyl scalars peel as ``Psi_n ~ r^{-(5-n)}``.  The robust, universal certificate is
the Coulombic ``Psi_2 ~ r^{-3}`` (slope -3), confirmed on three independent exact
spacetimes:

  * Schwarzschild (static vacuum), ``Psi_2`` peels, all other scalars at floor.
  * Kinnersley photon rocket (accelerating + radiating), ``Psi_2`` peels and the
    radiative ``Psi_4`` sits at the pipeline floor (the Damour dipole is
    gravitational-wave-silent).
  * Vaidya monopole (isotropic mass loss), ``Psi_2`` peels.

The radiative ``Psi_4 ~ r^{-1}`` tail (slope -1) is exhibited on a
deliberately non-silent linearized transverse-traceless wave, the positive
control proving the instrument reads real radiative peeling, not a pipeline
artifact.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.bondi import peeling_slopes
from warpax.bondi.peeling import _null_tetrad

from ._helpers import ETA, krt_cone, make_krt


def _schw_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([0.0, r * c, r * s, 0.0])


_WAVE_A, _WAVE_OM = 1e-3, 1.0


def _wave_metric(X):
    """Linearized outgoing TT wave hp = A cos(om(t-r))/r in the (y,z) plane (axis=1)."""
    t, x, y, z = X
    r = jnp.sqrt(x**2 + y**2 + z**2)
    hp = _WAVE_A * jnp.cos(_WAVE_OM * (t - r)) / r
    g = ETA.at[2, 2].add(hp)
    g = g.at[3, 3].add(-hp)
    return g


def _wave_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([r, r * c, r * s, 0.0])


def test_schwarzschild_psi2_peels_minus3():
    """Static vacuum: Psi_2 ~ r^-3; all other scalars below the resolution floor."""
    schw = SchwarzschildMetric(M=1.0)
    res = peeling_slopes(schw.__call__, _schw_cone, radii=(40.0, 80.0, 160.0, 320.0, 640.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.06
    for n in (0, 1, 3, 4):
        assert not res.above_floor[n]


def test_krt_psi2_peels_and_gw_silent():
    """Kinnersley rocket: Psi_2 ~ r^-3; Psi_4 is tetrad leakage ~ r^-5, no r^-1 tail.

    The Eulerian g-null tetrad is tilted O(m/r) from the Kerr-Schild principal
    direction, so ``Psi_4 ~ 0.14 (m/r)^2 Psi_2 ~ r^-5`` (fitted slope -4.96);
    GW silence is the absence of the radiative slope -1.
    """
    metric_fn, _ = make_krt(G0=0.10, scan_length=80)
    res = peeling_slopes(metric_fn, krt_cone, radii=(20.0, 40.0, 80.0, 160.0, 320.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.02
    assert res.slopes[4] < -4.5
    assert res.ricci_max < 1e-7


def test_vaidya_psi2_peels():
    """Vaidya monopole (isotropic mass loss): Psi_2 ~ r^-3."""
    metric_fn, _ = make_krt(G0=1e-8, scan_length=80)
    res = peeling_slopes(metric_fn, krt_cone, radii=(20.0, 40.0, 80.0, 160.0, 320.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.02


def test_wave_psi4_peels_minus1():
    """Linearized TT wave: radiative Psi_4 ~ r^-1 (the positive control)."""
    res = peeling_slopes(_wave_metric, _wave_cone, radii=(50.0, 100.0, 200.0, 400.0, 800.0))
    assert res.above_floor[4]
    assert abs(res.slopes[4] - (-1.0)) < 0.05


def test_null_tetrad_is_g_null_on_diagonal_metric():
    """On g = diag(-1,2,3,4) the tetrad satisfies g(l,l)=g(n,n)=g(m,m)=0, g(l,n)=-1, g(m,mbar)=1.

    ``compute_orthonormal_tetrad`` rescales each coordinate leg by ``1/sqrt(g_ii)``,
    so the Minkowski algebra of ``l = e0 + nhat``, ``n = (e0 - nhat)/2``,
    ``m = (e_theta + i e_phi)/sqrt2`` carries over exactly (to 1e-12).
    """
    g = np.diag([-1.0, 2.0, 3.0, 4.0])
    c, s = np.cos(0.9), np.sin(0.9)
    for axis in (1, 2, 3):
        l, n, m = _null_tetrad(g, c, s, axis)
        dot = lambda a, b: a @ g @ b
        assert abs(dot(l, l)) < 1e-12
        assert abs(dot(n, n)) < 1e-12
        assert abs(dot(m, m)) < 1e-12
        assert abs(dot(l, n) + 1.0) < 1e-12
        assert abs(dot(m, np.conj(m)) - 1.0) < 1e-12
        assert abs(dot(l, m)) < 1e-12
        assert abs(dot(n, m)) < 1e-12
    l, n, m = _null_tetrad(np.asarray(ETA), c, s, 1)
    np.testing.assert_allclose(l, [1.0, c, s, 0.0], atol=1e-15)
    np.testing.assert_allclose(n, [0.5, -0.5 * c, -0.5 * s, 0.0], atol=1e-15)
    np.testing.assert_allclose(m, [0.0, -s, c, 1j] / np.sqrt(2.0), atol=1e-15)
