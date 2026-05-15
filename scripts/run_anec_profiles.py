"""ANEC line integrals along an off-axis null geodesic.

Integrates the ANEC integrand T_ab k^a k^b along an axial null ray
(slightly off-axis at y=1e-3 to avoid the Natario-class shape-function
coordinate singularity at y=z=0) for Alcubierre, Fuchs, S-shell, T-shell.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec, _anec_integrand_at_point


OUTPUT_DIR = Path("warpax/output/anec")


def _axial_null_geodesic(affine, *, x_start=-30.0, c=1.0, y_offset=1e-3):
    """Coordinate null ray x = x_start + c*lambda with y_offset off-axis."""
    return jnp.stack([
        jnp.array(affine),
        jnp.array(x_start + affine * c),
        jnp.array(y_offset),
        jnp.array(0.0),
    ])


def _evaluate(name, metric, *, bounds=(-30.0, 30.0), n=512):
    print(f"\n--- {name} ---")
    lams = jnp.linspace(*bounds, n)
    pos = jax.vmap(_axial_null_geodesic)(lams)
    res = anec(metric, _axial_null_geodesic, tangent_norm="renormalized",
               n_samples=n, affine_bounds=bounds)
    vel = jax.vmap(jax.jacfwd(_axial_null_geodesic))(lams)
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
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from warpax.benchmarks import AlcubierreMetric
    from warpax.metrics.fuchs_construction import fuchs_default
    from warpax.metrics.sshell import sshell_default
    from warpax.metrics.tshell import tshell_default

    metrics = [
        ("Alcubierre", AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)),
        ("Fuchs",      fuchs_default()),
        ("S-shell",    sshell_default()),
        ("T-shell",    tshell_default()),
    ]
    summary = {}
    for name, m in metrics:
        r = _evaluate(name, m)
        np.savez(
            OUTPUT_DIR / f"anec_{name.lower().replace('-', '_')}.npz",
            lam=r["lambda"], x=r["x"], integrand=r["integrand"],
            line_integral=r["line_integral"],
        )
        summary[name] = r["line_integral"]
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({"anec_line_integrals": summary}, f, indent=2)
    print(f"\n  -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
