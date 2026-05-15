"""Radial verification of the canonical Gaussian-smoothed Fuchs shell.

Verifies ``fuchs_default()`` (the full 5-step iterative-smoothing
construction) over a 50-point radial sweep, reporting HE classification,
observer-robust and Eulerian EC margins, constraint residuals, and the
interior-vs-exterior breakdown of EC violations.

Usage:
    python -m scripts.run_fuchs_canonical  # from warpax/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _radial_sweep import radial_sweep, save_json  # noqa: E402

from warpax.metrics import fuchs_default  # noqa: E402


OUTPUT = Path("warpax/output/fuchs_canonical.json")


def main():
    metric = fuchs_default()
    print(f"Fuchs (Gaussian-smoothed): R_1={metric.R_1}, R_2={metric.R_2}, v_s={metric.v_s}")
    result = radial_sweep(metric, r_range=(0.5, 40.0), n_sweep=50, n_starts=16)

    R_1, R_2 = metric.R_1, metric.R_2
    interior = [p for p in result["per_point"] if R_1 <= p["r"] <= R_2]
    int_viol = sum(
        1 for p in interior
        if any(p["ec_robust"][k] < 0 for k in ("nec", "wec", "dec"))
    )
    result["summary"]["interior_count"] = len(interior)
    result["summary"]["interior_violations"] = int_viol

    s = result["summary"]
    print(f"  HE census: {s['he_type_census']}")
    print(f"  Robust violations (full sweep): {s['violated_robust']}")
    print(f"  Interior violations [{R_1}, {R_2}]: {int_viol}/{len(interior)}")
    print(f"  max eps_H = {s['max_epsilon_H']:.4e}, max eps_M = {s['max_epsilon_M']:.4e}")
    save_json(OUTPUT, result)


if __name__ == "__main__":
    main()
