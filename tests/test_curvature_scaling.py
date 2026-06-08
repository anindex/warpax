"""Curvature-invariant v_s scaling (K11): sentinels and fit recovery.

Sentinels guard the physics: Minkowski has vanishing curvature invariants, and
the Schwarzschild Kretschmann scalar reproduces the closed form
``K = 48 M^2 / r^6`` (the areal radius is recovered from the isotropic
coordinate used by ``SchwarzschildMetric``). The fit test pins the log-log
power-law extraction used to report the universal exponents.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import MinkowskiMetric, SchwarzschildMetric
from warpax.geometry import (
    compute_curvature_chain,
    compute_invariants,
    kretschmann_scalar,
)

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_script(name):
    path = os.path.join(_SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSentinels:
    def test_minkowski_invariants_zero(self):
        res = compute_curvature_chain(MinkowskiMetric(), jnp.array([0.0, 3.0, 1.0, 0.0]))
        K, R2, W2, CP = compute_invariants(res)
        for v in (K, R2, W2, CP):
            assert float(jnp.abs(v)) < 1e-12

    def test_schwarzschild_kretschmann_closed_form(self):
        M, r_iso = 1.0, 5.0
        res = compute_curvature_chain(
            SchwarzschildMetric(M=M), jnp.array([0.0, r_iso, 0.0, 0.0])
        )
        K = float(kretschmann_scalar(res.riemann, res.metric, res.metric_inv))
        # Isotropic -> areal radius: r = r_iso (1 + M/2 r_iso)^2.
        r_areal = r_iso * (1.0 + M / (2.0 * r_iso)) ** 2
        K_exact = 48.0 * M**2 / r_areal**6
        assert abs(K - K_exact) / K_exact < 1e-6

    def test_schwarzschild_weyl_equals_kretschmann(self):
        """Schwarzschild is Ricci-flat: Ricci^2 = 0 and C^2 = K exactly.
        Guards the sign of the Weyl term in the Gauss-Bonnet identity."""
        res = compute_curvature_chain(
            SchwarzschildMetric(M=1.0), jnp.array([0.0, 5.0, 0.0, 0.0])
        )
        K, R2, W2, _ = compute_invariants(res)
        assert float(jnp.abs(R2)) < 1e-8                       # Ricci-flat
        assert abs(float(W2) - float(K)) / float(K) < 1e-6     # C^2 = K

    def test_schwarzschild_chern_pontryagin_zero(self):
        """A static, parity-even spacetime has vanishing Chern-Pontryagin.
        Guards the Levi-Civita permutation tables / Hodge-dual contraction."""
        res = compute_curvature_chain(
            SchwarzschildMetric(M=1.0), jnp.array([0.0, 5.0, 0.0, 0.0])
        )
        _, _, _, CP = compute_invariants(res)
        assert float(jnp.abs(CP)) < 1e-8

    def test_conformally_flat_weyl_zero(self):
        """g = Omega(x)^2 eta is conformally flat: C^2 = 0 exactly, while
        K and Ricci^2 are nonzero. This is the only sentinel exercising the
        three-term cancellation in C^2 = K - 2 R_ab R^ab + R^2/3 with every
        term contributing (Minkowski has all three zero; Schwarzschild has
        R^2 = R = 0)."""
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

        def conformal(coords):
            x, y, z = coords[1], coords[2], coords[3]
            omega2 = 1.0 + 0.3 * jnp.exp(-(x**2 + y**2 + z**2) / 4.0)
            return omega2 * eta

        res = compute_curvature_chain(conformal, jnp.array([0.0, 1.0, 0.0, 0.0]))
        K, R2, W2, _ = compute_invariants(res)
        assert float(jnp.abs(W2)) < 1e-8           # conformally flat -> Weyl = 0
        assert float(jnp.abs(K)) > 1e-4            # but K nonzero ...
        assert float(jnp.abs(R2)) > 1e-4           # ... and Ricci^2 nonzero


class TestFit:
    def test_power_law_recovers_exponent(self):
        mod = _load_script("run_curvature_scaling.py")
        rows = [
            {"metric": "X", "v_s": v, "weyl_squared_max": 5.0 * v**2}
            for v in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        fit = mod.fit_power_law(rows, "X", "weyl_squared")
        assert abs(fit["q"] - 2.0) < 1e-6
        assert abs(fit["A"] - 5.0) < 1e-6
        assert fit["r_squared"] > 0.9999

    def test_run_point_finite_positive(self):
        mod = _load_script("run_curvature_scaling.py")
        r = mod.run_point("Alcubierre", 0.5, 16)
        for key in ("kretschmann_max", "weyl_squared_max", "ricci_squared_max"):
            assert np.isfinite(r[key]) and r[key] > 0.0

    def test_cached_exponents_regression(self):
        """If the cached sweep exists, the irrotational/vortical exponent split
        (Rodal ~ v_s^4, Natario ~ v_s^2) must hold."""
        import json

        path = os.path.join(_SCRIPTS, "..", "results", "curvature_scaling.json")
        if not os.path.exists(path):
            pytest.skip("results/curvature_scaling.json not present")
        fits = json.load(open(path))["fits"]
        assert abs(fits["Rodal"]["weyl_squared"]["q"] - 4.0) < 0.2
        assert abs(fits["Natário"]["weyl_squared"]["q"] - 2.0) < 0.2
