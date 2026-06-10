"""T-shell EC-margin ablation across bubble velocity v_0.

Holds (R_1, R_2, rho_0) fixed and sweeps v_0 in {0.01, 0.05, 0.1, 0.2}.
A v_0-invariant boundary DEC margin indicates the failure is geometric
(driven by the smooth source-vacuum transition); a strongly v_0-dependent
margin would indicate a kinematic (shift-driven) origin.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from _json_io import dump_json

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.constraints.residuals import normalized_residuals
from warpax.energy_conditions import verify_point
from warpax.geometry import compute_curvature_chain
from warpax.metrics.tshell import tshell_default


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "v0_ablation.json"

R_PROBES = [10.0, 10.2, 12.5, 15.0, 17.5, 19.8]
V0_GRID = [0.0, 0.01, 0.05, 0.1, 0.2]  # 0.0 = static-limit endpoint


def _eval(metric, n_starts=8):
    eps_H_max = 0.0
    worst_int = float("inf")
    worst_R1 = float("inf")
    for r in R_PROBES:
        coords = jnp.array([0.0, float(r), 0.0, 0.0])
        cr = normalized_residuals(metric, coords)
        eH = float(cr["epsilon_H"])
        if eH == eH:
            eps_H_max = max(eps_H_max, eH)
        curv = compute_curvature_chain(metric, coords)
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv,
                          n_starts=n_starts)
        m = min(float(ec.nec_margin), float(ec.wec_margin), float(ec.dec_margin))
        if 10.2 <= r <= 19.8:
            worst_int = min(worst_int, m)
        if abs(r - 10.0) < 1e-9:
            worst_R1 = min(worst_R1, m)
    return {"eps_H_max": eps_H_max,
            "worst_interior_margin": worst_int,
            "worst_R1_boundary_margin": worst_R1}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("T-shell v_0 ablation (R_1=10, R_2=20, rho_0=1e-4)")
    out = []
    for v0 in V0_GRID:
        t0 = time.time()
        rec = {"v_0": v0, **_eval(tshell_default(v_0=float(v0)))}
        rec["elapsed_s"] = time.time() - t0
        out.append(rec)
        print(f"  v_0={v0}: eps_H_max={rec['eps_H_max']:.3e}  "
              f"interior={rec['worst_interior_margin']:+.3e}  "
              f"r=R_1={rec['worst_R1_boundary_margin']:+.3e}  ({rec['elapsed_s']:.1f}s)")
    dump_json({"results": out}, OUTPUT)
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
