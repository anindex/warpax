"""Lentz L1 vs L2 distance comparison.

Runs the SAME radial EC verification (identical probes, n_starts) for the
Lentz diamond soliton with distance="L1" (piecewise-linear Manhattan, the
published WarpFactory construction) and distance="L2" (Euclidean/spherical,
C^infty away from the floor). Goal: show the enormous L1 boundary
curvature/EC spikes (NEC ~ -1e16, DEC ~ -1e36) are an artifact of the
L1 piecewise-linear distance, not intrinsic.

Probe setup is taken verbatim from run_lentz.py / _radial_sweep.py:
sweep along +x at coords=[0, r, 0, 0], r in [0.5, 200], 50 points, n_starts=16.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _radial_sweep import radial_sweep

import json

from warpax.metrics import LentzMetric


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "lentz_distance_comparison.json"

R_RANGE = (0.5, 200.0)
N_SWEEP = 50
N_STARTS = 16


def worst_point(per_point, key):
    """Return (r, margin) for the most-negative robust margin of `key`."""
    worst = min(per_point, key=lambda p: p["ec_robust"][key])
    return {"r": worst["r"], "margin": worst["ec_robust"][key]}


def run_one(distance):
    m = LentzMetric(distance=distance)
    result = radial_sweep(
        m, r_range=R_RANGE, n_sweep=N_SWEEP, n_starts=N_STARTS, progress=False
    )
    s = result["summary"]
    pp = result["per_point"]
    summary = {
        "params": {"v_s": m.v_s, "R": m.R, "sigma": m.sigma, "distance": distance},
        "he_type_census": s["he_type_census"],
        "violated_robust": s["violated_robust"],
        "violated_eulerian": s["violated_eulerian"],
        "min_margins_robust": s["min_margins_robust"],
        "worst_locations": {
            k: worst_point(pp, k) for k in ("nec", "wec", "sec", "dec")
        },
        "elapsed_s": result["elapsed_s"],
    }
    return {"summary": summary, "per_point": pp}


def main():
    out = {
        "probe": {
            "r_range": list(R_RANGE),
            "n_sweep": N_SWEEP,
            "n_starts": N_STARTS,
            "axis": "+x (coords=[0,r,0,0])",
            "metric": "LentzMetric (defaults v_s/R/sigma; only distance differs)",
        },
        "L1": run_one("L1"),
        "L2": run_one("L2"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Compact stdout table
    for d in ("L1", "L2"):
        s = out[d]["summary"]
        c = s["he_type_census"]
        mm = s["min_margins_robust"]
        v = s["violated_robust"]
        nviol = sum(1 for k in ("nec", "wec", "sec", "dec") if mm[k] < 0)
        print(
            f"{d}: HE I={c['1']} II={c['2']} III={c['3']} IV={c['4']} | "
            f"NEC={mm['nec']:.3e} WEC={mm['wec']:.3e} DEC={mm['dec']:.3e} | "
            f"viol nec/wec/sec/dec={v['nec']}/{v['wec']}/{v['sec']}/{v['dec']}"
        )
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
