"""Standalone Rodal admissibility verification, persisted for paper claim-checking.

Regenerates the Rodal (arXiv:2512.18008) irrotational-shift evaluation and writes a
dedicated artifact (``warpax/output/rodal_verification.json``) so the per-probe
NEC/WEC/DEC counts quoted in the CQG paper (9/50 NEC, 46/50 DEC under multi-observer
BFGS certification) are independently reproducible and machine-checkable.

Reuses ``evaluate_rodal()`` from ``verify_proposals.py`` (RodalMetric v_s=0.1, R=100,
sigma=0.03; 50-point radial sweep; n_starts=16) rather than duplicating the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from verify_proposals import evaluate_rodal  # sibling script on sys.path[0]


def main() -> None:
    report = evaluate_rodal()
    ec = report["criteria"]["D_ec_margins"]
    n_probes = len(report["per_point"])
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": "RodalMetric(v_s=0.1, R=100.0, sigma=0.03)",
        "reference": "Rodal 2026, arXiv:2512.18008",
        "n_probes": n_probes,
        "n_starts": 16,
        "violated": ec["violated"],
        "min_margins": ec["min_margins"],
        "he_type_census": report["he_type_census"],
        "report": report,
    }
    out_path = Path(__file__).resolve().parents[1] / "output" / "rodal_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    v = ec["violated"]
    print(f"\nRodal verification persisted: {out_path}")
    print(f"  probes={n_probes}  NEC={v['nec']}  WEC={v['wec']}  "
          f"DEC={v['dec']}  SEC={v['sec']}")


if __name__ == "__main__":
    main()
