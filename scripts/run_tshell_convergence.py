"""Grid-convergence check for the T-shell worst observer-robust EC margin.

Evaluates a small set of representative phase-diagram points across
``n_grid`` resolutions {256, 512, 1024}. Reports the worst margin at
each resolution so the *sign* (feasibility verdict) can be checked
for grid stability.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)

from warpax.optimization.sweep import _evaluate_point, _rho_from_compactness


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "convergence_tshell.json"

POINTS = [
    {"label": "low-C near-feasible",  "compactness": 0.01, "thickness_ratio": 0.336},
    {"label": "high-C bulk violation", "compactness": 0.20, "thickness_ratio": 0.80},
]
RESOLUTIONS = [256, 512, 1024]


def _evaluate(pt, n_grid, R_2=20.0):
    R_1 = R_2 * (1.0 - pt["thickness_ratio"])
    rho_0 = _rho_from_compactness(pt["compactness"], R_1, R_2)
    t0 = time.time()
    out = _evaluate_point(
        ansatz="tshell",
        R_1=R_1, R_2=R_2, rho_0=rho_0,
        n_density=4, n_velocity=4,
        n_grid=n_grid, n_probes=15, n_ec_starts=8,
    )
    return {
        "n_grid": n_grid,
        "worst_ec_margin": float(out["worst_ec_margin"]),
        "ec_feasible": bool(out["ec_feasible"]),
        "constraint_residual": float(out["constraint_residual"]),
        "elapsed_s": time.time() - t0,
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for pt in POINTS:
        print(f"\n=== {pt['label']} (C={pt['compactness']}, dR/R={pt['thickness_ratio']}) ===")
        runs = []
        for N in RESOLUTIONS:
            print(f"  n_grid={N} ...", flush=True)
            r = _evaluate(pt, n_grid=N)
            print(f"    worst margin = {r['worst_ec_margin']:+.4e}  feasible={r['ec_feasible']}  ({r['elapsed_s']:.1f}s)")
            runs.append(r)
        signs = ["neg" if r["worst_ec_margin"] < 0 else "pos" for r in runs]
        out.append({**pt, "by_resolution": runs,
                    "sign_stable": all(s == signs[0] for s in signs)})

    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
