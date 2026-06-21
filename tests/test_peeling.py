r"""Tests for the Newman--Penrose peeling-falloff extractor (warpax.bondi.peeling).

Verifies the asymptotic structure underlying the model-independent
no-reactionless-steering theorem (T1-GR): along outgoing null cones the five NP
Weyl scalars peel as ``Psi_n ~ r^{-(5-n)}``.  The robust, universal certificate is
the Coulombic ``Psi_2 ~ r^{-3}`` (slope -3), confirmed on three independent exact
spacetimes:

  * Schwarzschild (static vacuum) -- ``Psi_2`` peels, all other scalars at floor.
  * Kinnersley photon rocket (accelerating + radiating) -- ``Psi_2`` peels and the
    radiative ``Psi_4`` sits at the pipeline floor (the Damour dipole is
    gravitational-wave-silent).
  * Vaidya monopole (isotropic mass loss) -- ``Psi_2`` peels.

The genuine radiative ``Psi_4 ~ r^{-1}`` tail (slope -1) is exhibited on a
deliberately non-silent linearized transverse-traceless wave -- the positive
control proving the instrument reads real radiative peeling, not a pipeline
artifact.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.bondi import peeling_slopes

_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _make_krt(G0, M0=1.0, M1=0.05, OM=0.5):
    """Exact accelerating Kerr-Schild photon rocket; G0 -> 0 gives Vaidya.

    Identical to ``tests/test_bondi.py._make_krt`` except the retarded-``u``
    ``jax.lax.scan`` length is raised 12 -> 80 to converge at the large peeling
    radii (r up to a few hundred M).
    """

    def worldline(u):
        sx = jnp.sinh(G0 * u) / G0 if G0 != 0 else u
        xx = (jnp.cosh(G0 * u) - 1.0) / G0 if G0 != 0 else 0.0
        return jnp.array([sx, xx, 0.0, 0.0])

    def m_of(u):
        return M0 + M1 * jnp.sin(OM * u)

    @jax.custom_jvp
    def retarded_u(X):
        def body(u, _):
            z = worldline(u)
            zdot = jax.jacfwd(worldline)(u)
            D = X - z
            r = -(_ETA @ zdot) @ D
            F = -D[0] ** 2 + D[1] ** 2 + D[2] ** 2 + D[3] ** 2
            return u - F / (2.0 * r), None

        u0 = X[0] - jnp.sqrt(X[1] ** 2 + X[2] ** 2 + X[3] ** 2)
        u, _ = jax.lax.scan(body, u0, None, length=80)
        return u

    @retarded_u.defjvp
    def _jvp(primals, tangents):
        (X,), (dX,) = primals, tangents
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(_ETA @ zdot) @ D
        l_dn = _ETA @ (D / r)
        return u, -(l_dn @ dX)

    def metric_fn(X):
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(_ETA @ zdot) @ D
        l_up = D / r
        l_dn = _ETA @ l_up
        return _ETA + (2.0 * m_of(u) / r) * jnp.outer(l_dn, l_dn)

    return metric_fn, m_of


def _krt_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([r, r * c, r * s, 0.0])


def _schw_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([0.0, r * c, r * s, 0.0])


# linearized outgoing TT wave: hp = A cos(om(t-r))/r in the (y,z) plane (axis=1).
_WAVE_A, _WAVE_OM = 1e-3, 1.0


def _wave_metric(X):
    t, x, y, z = X
    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    hp = _WAVE_A * jnp.cos(_WAVE_OM * (t - r)) / r
    g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g = g.at[2, 2].add(hp)
    g = g.at[3, 3].add(-hp)
    return g


def _wave_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([r, r * c, r * s, 0.0])


def test_schwarzschild_psi2_peels_minus3():
    """Static vacuum: Psi_2 ~ r^-3; all other scalars below the resolution floor."""
    schw = SchwarzschildMetric(M=1.0)
    res = peeling_slopes(schw.__call__, _schw_cone,
                         radii=(40.0, 80.0, 160.0, 320.0, 640.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.06
    for n in (0, 1, 3, 4):
        assert not res.above_floor[n]


def test_krt_psi2_peels_and_gw_silent():
    """Kinnersley rocket: Psi_2 ~ r^-3 and Psi_4 at floor (GW-silent dipole)."""
    metric_fn, _ = _make_krt(G0=0.10)
    res = peeling_slopes(metric_fn, _krt_cone,
                         radii=(20.0, 40.0, 80.0, 160.0, 320.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.02
    assert not res.above_floor[4]          # GW-silent: no radiative Psi_4 tail
    assert res.ricci_max < 1e-7            # exact solution


def test_vaidya_psi2_peels():
    """Vaidya monopole (isotropic mass loss): Psi_2 ~ r^-3."""
    metric_fn, _ = _make_krt(G0=1e-8)      # G0 -> 0: spherical Vaidya
    res = peeling_slopes(metric_fn, _krt_cone,
                         radii=(20.0, 40.0, 80.0, 160.0, 320.0))
    assert abs(res.slopes[2] - (-3.0)) < 0.02


def test_wave_psi4_peels_minus1():
    """Linearized TT wave: genuine radiative Psi_4 ~ r^-1 (the positive control)."""
    res = peeling_slopes(_wave_metric, _wave_cone,
                         radii=(50.0, 100.0, 200.0, 400.0, 800.0))
    assert res.above_floor[4]
    assert abs(res.slopes[4] - (-1.0)) < 0.05
