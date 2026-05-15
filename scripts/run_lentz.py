"""Radial verification of the Lentz diamond soliton.

Sweeps ``LentzMetric()`` over r in [0.5, 200] (R=100 default) and
reports the EC and constraint diagnostics. The on-axis NaN in
``rho_perp = sqrt(y**2 + z**2)`` is resolved by the safe-sqrt floor
in ``metrics/lentz.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _radial_sweep import radial_sweep, save_json

from warpax.metrics import LentzMetric


OUTPUT = Path("warpax/output/lentz_verification.json")


def main():
    m = LentzMetric()
    print(f"Lentz: v_s={m.v_s}, R={m.R}, sigma={m.sigma}")
    result = radial_sweep(m, r_range=(0.5, 200.0), n_sweep=50, n_starts=16)
    s = result["summary"]
    print(f"  HE census: {s['he_type_census']}")
    print(f"  Robust violations: {s['violated_robust']}")
    print(f"  Eulerian violations: {s['violated_eulerian']}")
    print(f"  max eps_H: {s['max_epsilon_H']:.3e}")
    print(f"  worst margins: {s['min_margins_robust']}")
    save_json(OUTPUT, result)


if __name__ == "__main__":
    main()
