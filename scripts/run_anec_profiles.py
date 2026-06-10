"""ANEC line integrals along an off-axis null ray.

Integrates the ANEC integrand T_ab k^a k^b along an axial null ray
(slightly off-axis at y=1e-3 to avoid the Natario-class shape-function
coordinate singularity at y=z=0) for Alcubierre, Fuchs, S-shell, T-shell.

The integration path is a coordinate null ray
``x^mu(lambda) = x_0^mu + lambda k^mu_0`` with affine parameter advancing in
coordinates rather than an integrated metric null geodesic.  For
asymptotically flat regions the deviation from the true null geodesic is
small; near the smoothed-tail interior or the Alcubierre bubble wall the
deviation is a known systematic.  A supplementary geodesic-integrated
cross-check is in ``run_anec_geodesic_check.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

from _json_io import dump_json

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec, _anec_integrand_at_point


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "anec"


def _axial_null_geodesic_factory(y_offset=1e-3, x_start=-30.0, c=1.0):
    """Return an axial null ray x = x_start + c*lambda with given y_offset."""
    def geo(affine):
        return jnp.stack([
            jnp.array(affine),
            jnp.array(x_start + affine * c),
            jnp.array(y_offset),
            jnp.array(0.0),
        ])
    return geo


def _axial_null_geodesic(affine, *, x_start=-30.0, c=1.0, y_offset=1e-3):
    """Coordinate null ray x = x_start + c*lambda with y_offset off-axis."""
    return jnp.stack([
        jnp.array(affine),
        jnp.array(x_start + affine * c),
        jnp.array(y_offset),
        jnp.array(0.0),
    ])


def _evaluate(name, metric, *, bounds=(-30.0, 30.0), n=512, y_offset=1e-3):
    print(f"\n--- {name} (y={y_offset}) ---")
    geo = _axial_null_geodesic_factory(y_offset=y_offset, x_start=bounds[0])
    lams = jnp.linspace(*bounds, n)
    pos = jax.vmap(geo)(lams)
    res = anec(metric, geo, tangent_norm="renormalized",
               n_samples=n, affine_bounds=bounds)
    vel = jax.vmap(jax.jacfwd(geo))(lams)
    integrand = jax.vmap(
        lambda c, u: _anec_integrand_at_point(metric, c, u, "renormalized")
    )(pos, vel)
    print(f"  ANEC = {float(res.line_integral):+.5e}")
    print(f"  integrand [min, max] = [{float(integrand.min()):+.3e}, {float(integrand.max()):+.3e}]")
    return {
        "name": name,
        "line_integral": float(res.line_integral),
        "lambda": np.array(lams, dtype=float),
        "x": np.array(pos[:, 1], dtype=float),
        "integrand": np.array(integrand, dtype=float),
        "y_offset": float(y_offset),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from warpax.benchmarks import AlcubierreMetric
    from warpax.metrics.fuchs_construction import fuchs_default
    from warpax.metrics.sshell import sshell_default
    from warpax.metrics.tshell import tshell_default
    from warpax.metrics.tshell_profiles import constant_velocity_profiles
    from warpax.metrics.tshell import tshell_from_profiles
    from warpax.optimization.sweep import _rho_from_compactness

    # high-compactness T-shell at the binding-failure corner of the
    # phase diagram (C=0.20, dR/R_2=0.80, v_0=0.1). Tests whether the worst
    # pointwise margin at -2.78e-2 propagates to averaged ANEC violation.
    R_2 = 20.0
    high_C, high_dR = 0.20, 0.80
    R_1_high = R_2 * (1.0 - high_dR)  # = 4.0
    rho_0_high = _rho_from_compactness(high_C, R_1_high, R_2)
    high_profiles = constant_velocity_profiles(
        R_1=R_1_high, R_2=R_2, rho_0=rho_0_high, v_0=0.1,
    )
    tshell_highC = tshell_from_profiles(high_profiles)

    metrics = [
        ("Alcubierre",     AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)),
        ("Fuchs",          fuchs_default()),
        ("S-shell",        sshell_default()),
        ("T-shell",        tshell_default()),
        ("T-shell-highC",  tshell_highC),
    ]
    summary = {}
    for name, m in metrics:
        r = _evaluate(name, m)
        np.savez(
            OUTPUT_DIR / f"anec_{name.lower().replace('-', '_')}.npz",
            lam=r["lambda"], x=r["x"], integrand=r["integrand"],
            line_integral=r["line_integral"],
            y_offset=r["y_offset"],
        )
        summary[name] = r["line_integral"]

    # y-offset robustness scan for Fuchs (spherical, expected y-invariant)
    # and Alcubierre (Natario-class, expected y-sensitive near origin).
    print("\n=== y-offset robustness scan ===")
    y_grid = [1e-3, 1e-2, 1e-1, 1.0]
    y_scan = {"Fuchs": [], "Alcubierre": []}
    fuchs_m = fuchs_default()
    alc_m = AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
    for y in y_grid:
        rf = _evaluate(f"Fuchs(y={y})", fuchs_m, y_offset=y)
        ra = _evaluate(f"Alcubierre(y={y})", alc_m, y_offset=y)
        y_scan["Fuchs"].append({"y": y, "line_integral": rf["line_integral"]})
        y_scan["Alcubierre"].append({"y": y, "line_integral": ra["line_integral"]})

    dump_json({
        "anec_line_integrals": summary,
        "y_offset_scan": y_scan,
        "notes": {
            "method": (
                "Coordinate null ray with affine straight-line "
                "parameterization in coordinates (not a "
                "metric-integrated geodesic).  See "
                "run_anec_geodesic_check.py for the geodesic-integrated "
                "supplementary cross-check."
            ),
            "high_C_tshell": {
                "compactness": high_C, "thickness_ratio": high_dR,
                "R_1": R_1_high, "R_2": R_2, "rho_0": rho_0_high,
                "v_0": 0.1,
            },
            "interpretation": (
                "high-C T-shell tests whether pointwise binding-corner DEC "
                "violation flips averaged ANEC sign; y-scan tests that the "
                "off-axis ray choice is not a coordinate artifact for "
                "spherically symmetric shells."
            ),
        },
    }, OUTPUT_DIR / "summary.json")
    print(f"\n  -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
