"""Supplementary geodesic-integrated ANEC cross-check.

Integrates the actual null geodesic of each warp metric (via
:func:`warpax.geodesics.null_ic` + :func:`warpax.geodesics.integrate_geodesic`)
and evaluates the ANEC line integral along that integrated trajectory, rather
than along the coordinate null ray used in
:mod:`run_anec_profiles`.

For the source-prescribed shells (Fuchs, S-shell, T-shell), the integrated
null geodesic preserves ``g_ab k^a k^b = 0`` to better than 1e-6 along the
path and yields ANEC line integrals that are sign-consistent with the
coordinate-ray diagnostic (all positive).  For the Alcubierre baseline the
strong shift bubble drives the trajectory away from the null cone within the
Diffrax adaptive RK tolerance budget, so the Alcubierre line integral
reported here is a numerical placeholder rather than a defensible
geodesic-integrated result and is flagged accordingly in the output.

This script writes ``warpax/output/anec/geodesic_check.json`` and does NOT
overwrite the coordinate-ray ``summary.json`` consumed by
``source_warp/paper/verify_claims.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.averaged.anec import anec
from warpax.geodesics import integrate_geodesic, null_ic


OUTPUT_DIR = Path("warpax/output/anec")


def _integrate_null_geodesic(
    metric, x_start=-30.0, x_end=30.0, y_offset=1e-3, num_points=512,
):
    x0 = jnp.array([0.0, x_start, y_offset, 0.0], dtype=jnp.float64)
    spatial = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    x0, k0 = null_ic(metric, x0, spatial)
    span = float(x_end - x_start)
    return integrate_geodesic(
        metric, x0, k0, tau_span=(0.0, span), num_points=num_points,
    )


def _evaluate(name, metric, *, bounds=(-30.0, 30.0), n=512, y_offset=1e-3):
    print(f"\n--- {name} (y={y_offset}) ---")
    geo = _integrate_null_geodesic(
        metric, x_start=bounds[0], x_end=bounds[1],
        y_offset=y_offset, num_points=n,
    )
    res = anec(metric, geo, tangent_norm="renormalized")
    g_kk = jax.vmap(
        lambda c, u: jnp.einsum("ab,a,b->", metric(c), u, u)
    )(geo.positions, geo.velocities)
    max_abs_g_kk = float(jnp.max(jnp.abs(g_kk)))
    null_preserved = max_abs_g_kk < 1e-4
    flag = "" if null_preserved else " [NULL PRESERVATION POOR]"
    print(f"  ANEC = {float(res.line_integral):+.5e}{flag}")
    print(f"  max |g(k, k)| along path = {max_abs_g_kk:.3e}")
    return {
        "name": name,
        "line_integral": float(res.line_integral),
        "max_abs_g_kk": max_abs_g_kk,
        "null_preserved": null_preserved,
        "geodesic_complete": bool(res.geodesic_complete),
        "termination_reason": res.termination_reason,
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

    R_2 = 20.0
    high_C, high_dR = 0.20, 0.80
    R_1_high = R_2 * (1.0 - high_dR)
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
        summary[name] = _evaluate(name, m)

    out_path = OUTPUT_DIR / "geodesic_check.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": (
                "Metric-integrated null geodesic: null IC solved via "
                "warpax.geodesics.null_ic; geodesic integrated by "
                "warpax.geodesics.integrate_geodesic (Diffrax Tsit5, "
                "rtol=atol=1e-10). g(k, k) drift along integrated "
                "trajectory is reported alongside each line integral "
                "as a numerical-null-preservation diagnostic; entries "
                "with null_preserved=false should be read as a numerical "
                "limit rather than a converged ANEC value."
            ),
            "results": summary,
        }, f, indent=2)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
