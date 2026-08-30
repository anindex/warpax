"""Verdicts certified from the metric, at representative wall points.

The exact rational certificates of ``energy_conditions.certificate`` are exact
relative to the float64 components they are handed. This closes the remaining
step by running the interval curvature chain and the interval LDL^T multiplier
test at chosen points, so the verdict is about the spacetime rather than about a
snapshot of its stress-energy.

A handful of points per drive, not a grid: the interval chain costs seconds per
point, and the claim it supports is qualitative, that the type-free decision
survives being taken back to the metric, not a new census.

Outputs
-------
- results/interval_lmi_spotcheck.json
- ../warpax_arxiv/tables/interval_lmi_spotcheck.tex
"""

from __future__ import annotations

import argparse
import os

from _json_io import dump_json
from _json_io import write_table as write_tex_table

from warpax.energy_conditions.enclosure import (
    alcubierre_metric,
    natario_metric,
    rodal_metric,
    van_den_broeck_metric,
)
from warpax.energy_conditions.interval_lmi import certify_point_from_metric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
VDB_KW = dict(R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)

BUILDERS = {
    "Alcubierre": lambda: alcubierre_metric(V_S, R_B, SIGMA),
    "Natário": lambda: natario_metric(V_S, R_B, SIGMA),
    "Van den Broeck": lambda: van_den_broeck_metric(V_S, R_B, SIGMA, **VDB_KW),
    "Rodal": lambda: rodal_metric(V_S, R_B, SIGMA),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]

# (x, s) on the axisymmetry-reduced half-plane, all inside the f in [0.1, 0.9]
# band at these parameters. The axis point and the deep off-axis one are chosen so
# that both verdicts are exercised: Van den Broeck's Type-I remainder reaches the
# axis, and its Type-IV region does not.
POINTS = [("on axis", 1.0, 0.0), ("wall, $s=0.35$", 0.9, 0.35), ("wall, $s=0.5$", 0.95, 0.5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prec", type=int, default=80)
    ap.add_argument("--metrics", nargs="+", default=ORDER)
    args = ap.parse_args()

    rows = []
    print("=" * 72)
    print(f"INTERVAL LMI SPOT CHECK (v_s={V_S}, R_b={R_B}, sigma={SIGMA}, prec={args.prec} bits)")
    print("=" * 72)
    for name in args.metrics:
        metric = BUILDERS[name]()
        for label, x, s in POINTS:
            r = certify_point_from_metric(metric, (0.0, x, s, 0.0), prec=args.prec)
            r["metric"] = name
            r["label"] = label
            rows.append(r)
            width = r["nec_upper"] - r["nec_lower"]
            print(
                f"  {name:>15s} {label:>9s}  NEC in "
                f"[{r['nec_lower']:+.6f}, {r['nec_upper']:+.6f}]  "
                f"width {width:.1e}  -> {r['nec']}",
                flush=True,
            )

    dump_json(
        {
            "config": vars(args),
            "params": {"v_s": V_S, "R_b": R_B, "sigma": SIGMA},
            "points": POINTS,
            "rows": rows,
        },
        os.path.join(RESULTS_DIR, "interval_lmi_spotcheck.json"),
    )
    write_table(rows, os.path.join(TABLES_DIR, "interval_lmi_spotcheck.tex"))


def write_table(rows, out_path):
    # The two endpoints agree to the six decimals a table can carry, so printing
    # them alone shows a reader two identical numbers and asks them to take the
    # enclosure on trust. The measured width is the whole content of the row.
    lines = [
        r"\begin{tabular}{@{}l l rr r c@{}}",
        r"  \toprule",
        r"  Metric & wall point & $\mathcal{N}_{\rm lower}$ & $\mathcal{N}_{\rm upper}$"
        r" & width & verdict \\",
        r"  \midrule",
    ]
    for r in rows:
        width = r["nec_upper"] - r["nec_lower"]
        lines.append(
            f"  {r['metric']} & {r['label']} & "
            f"${r['nec_lower']:+.6f}$ & ${r['nec_upper']:+.6f}$ & "
            f"${width:.1e}".replace("e-", r"\times10^{-").replace("e+", r"\times10^{")
            + f"}}$ & {r['nec']} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path,
        lines,
        script="scripts/run_interval_lmi_spotcheck.py",
        sources="results/interval_lmi_spotcheck.json",
    )
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
