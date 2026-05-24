"""Cross-class radial verification: Alcubierre, Natario, Van den Broeck.

Applies the 5-criterion admissibility check to the three foundational
Natario-class warp metrics at a common scale (v_s=0.1, R=20, sigma=2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _radial_sweep import radial_sweep, save_json

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics.natario import NatarioMetric
from warpax.metrics.van_den_broeck import VanDenBroeckMetric


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "landscape.json"


def _verify(name, metric, **kw):
    print(f"\n=== {name} ===")
    r = radial_sweep(metric, **kw)
    s = r["summary"]
    print(f"  Robust violations: {s['violated_robust']}")
    print(f"  Eulerian violations: {s['violated_eulerian']}")
    print(f"  HE census: {s['he_type_census']}")
    return {"name": name, **r}


def main():
    out = {}
    out["Alcubierre"] = _verify(
        "Alcubierre", AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
    )
    out["Natario"] = _verify(
        "Natario", NatarioMetric(v_s=0.1, R=20.0, sigma=2.0)
    )
    out["VanDenBroeck"] = _verify(
        "Van den Broeck",
        VanDenBroeckMetric(v_s=0.1, R=20.0, sigma=2.0,
                            R_tilde=10.0, alpha_vdb=0.5, sigma_B=2.0),
    )
    save_json(OUTPUT, out)


if __name__ == "__main__":
    main()
